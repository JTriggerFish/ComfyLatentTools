import comfy
from ..core import latent_filters as lf

# import torch


class LatentNormalizedLanczosResize:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ["LATENT"],
                "vae": ["VAE"],
                "size_multiplier": (
                    "FLOAT",
                    {"default": 2.0, "min": 0.1, "max": 4.0, "step": 0.5},
                ),
                "soft_clamp_outliers": (["enable", "disable"], {"default": "enable"}),
                "outlier_quantile": (
                    "FLOAT",
                    {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "outlier_clamp_slope": (
                    "FLOAT",
                    {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.1},
                ),
                "add_latent_noise": (["enable", "disable"], {"default": "disable"}),
                "latent_noise_std": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01},
                ),
                "latent_noise_scale": (
                    "FLOAT",
                    {"default": 0.1, "min": 0.01, "max": 10.0},
                ),
                "add_latent_upscale_with_weight": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "transform"
    CATEGORY = "image/upscaling"
    OUTPUT_NODE = False

    @classmethod
    def help(cls):
        return (
            "Upscale an image or latent using a normalized Lanczos resizer that tries to match the distribution "
            "of the input latent, optionally adding noise."
        )

    def transform(
        self,
        latent,
        vae,
        size_multiplier: float = 2.0,
        soft_clamp_outliers: str = "enable",
        outlier_quantile: float = 0.01,
        outlier_clamp_slope: float = 0.1,
        add_latent_noise: str = "disable",
        latent_noise_std: float = 0.1,
        latent_noise_scale: float = 1.0,
        add_latent_upscale_with_weight: float = 0.0,
    ):
        samples = latent["samples"]

        soft_clamp_outliers = (
            (soft_clamp_outliers == "enable")
            if isinstance(soft_clamp_outliers, str)
            else soft_clamp_outliers
        )
        add_latent_noise = (
            (add_latent_noise == "enable")
            if isinstance(add_latent_noise, str)
            else add_latent_noise
        )

        if soft_clamp_outliers:
            samples = lf.huberize_quantile(
                samples, outlier_quantile, 1 - outlier_quantile, outlier_clamp_slope
            )

        image = vae.decode(samples)

        width = round(image.shape[2] * size_multiplier)
        height = round(image.shape[1] * size_multiplier)
        image_upscaled = comfy.utils.common_upscale(
            image.movedim(-1, 1),
            width,
            height,
            "lanczos",
            "disabled",
        ).movedim(1, -1)

        latent_of_upscaled = vae.encode(image_upscaled)
        if add_latent_upscale_with_weight > 0.0:
            a = add_latent_upscale_with_weight
            h, w = latent_of_upscaled.shape[2:]
            up_latent = lf.latent_upscale(samples, h, w)
            latent_of_upscaled = (1 - a) * latent_of_upscaled + a * up_latent

        matched_latent = lf.moment_match(samples, latent_of_upscaled)

        if add_latent_noise:
            matched_latent = lf.add_correlated_gaussian_noise(
                matched_latent, latent_noise_scale, latent_noise_std
            )

        return ({"samples": matched_latent},)


NODE_CLASS_MAPPINGS = {
    "LatentNormalizedLanczosResize": LatentNormalizedLanczosResize,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LatentNormalizedLanczosResize": "Latent Normalized Lanczos Resize(LNLR)"
}
