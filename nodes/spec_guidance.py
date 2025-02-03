import kornia.geometry.transform

from comfy.model_patcher import ModelPatcher, set_model_options_patch_replace
from comfy.samplers import calc_cond_batch
from comfy.ldm.modules.attention import optimized_attention
from ..core import guidance as guidance
from ..core import latent_filters as lf


class DownsampledLatentGuidance:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "downsample_factor": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 1.0,
                        "max": 4.0,
                        "step": 0.1,
                        "round": 0.01,
                    },
                ),
                "guidance_weight": (
                    "FLOAT",
                    {
                        "default": 1.5,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "round": 0.01,
                    },
                ),
                "apply_rescaling_to_guidance": ("BOOLEAN", {"default": True}),
                "rescaling_fraction": (
                    "FLOAT",
                    {
                        "default": 0.7,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "round": 0.0001,
                    },
                ),
                "noise_fraction_start": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "round": False,
                    },
                ),
                "noise_fraction_end": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "round": False,
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model_patches/unet"
    OUTPUT_NODE = False

    @classmethod
    def help(cls):
        return "Use guidance extracted from a downscaled latent space"

    def patch(
        self,
        model: ModelPatcher,
        downsample_factor: float = 2.0,
        guidance_weight: float = 1.5,
        apply_rescaling_to_guidance: bool = True,
        rescaling_fraction: float = 0.7,
        noise_fraction_start: float = 1.0,
        noise_fraction_end: float = 0.0,
    ):

        def cfg_function(args):
            """Rescaled CFG+PAG"""
            model = args["model"]
            cond_pred = args["cond_denoised"]  # x_0
            uncond_pred = args["uncond_denoised"]
            cond_scale = args["cond_scale"]
            cond = args["cond"]
            uncond = args["uncond"]
            sigma = args["sigma"]
            model_options = args["model_options"].copy()
            x = args["input"]

            noise_fracs = model_options["transformer_options"]["sample_sigmas"] ** 2
            current_frac = sigma**2 / noise_fracs[0]

            if (noise_fraction_end <= current_frac) and (
                current_frac <= noise_fraction_start
            ):
                x_downsampled = lf.downsample_latent(x, downsample_factor)

                downsampled_cond, downsampled_uncond = calc_cond_batch(
                    model, [cond, uncond], x_downsampled, sigma, model_options
                )
                downsampled_cond = kornia.geometry.transform.rescale(
                    downsampled_cond,
                    float(downsample_factor),
                    interpolation="bilinear",
                    antialias=True,
                )
                downsampled_uncond = kornia.geometry.transform.rescale(
                    downsampled_uncond,
                    float(downsample_factor),
                    interpolation="bilinear",
                    antialias=True,
                )
            else:
                downsampled_cond, downsampled_uncond = cond_pred, uncond_pred

            pred_final = guidance.guidance_combine_and_scale(
                x,
                sigma,
                cond_pred,
                uncond_pred,
                cond_scale,
                downsampled_cond,
                downsampled_uncond,
                guidance_weight,
                scaling_method=guidance.GuidanceScalingMethod.V_SPACE_RESCALE,
                apply_rescaling_to_guidance=apply_rescaling_to_guidance,
                rescaling_fraction=rescaling_fraction,
            )
            return pred_final

        m = model.clone()
        m.set_model_sampler_post_cfg_function(cfg_function)
        return (m,)


class DownsampledAttentionGuidance:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "downsample_factor": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 1.0,
                        "max": 4.0,
                        "step": 0.1,
                        "round": 0.01,
                    },
                ),
                "guidance_weight": (
                    "FLOAT",
                    {
                        "default": 1.5,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "round": 0.01,
                    },
                ),
                "apply_rescaling_to_guidance": ("BOOLEAN", {"default": True}),
                "rescaling_fraction": (
                    "FLOAT",
                    {
                        "default": 0.7,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "round": 0.0001,
                    },
                ),
                "noise_fraction_start": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "round": False,
                    },
                ),
                "noise_fraction_end": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "round": False,
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model_patches/unet"
    OUTPUT_NODE = False

    @classmethod
    def help(cls):
        return (
            "Use guidance from downsampled keys and queries in the attention mechanism"
        )

    def patch(
        self,
        model: ModelPatcher,
        downsample_factor: float = 2.0,
        guidance_weight: float = 1.5,
        apply_rescaling_to_guidance: bool = True,
        rescaling_fraction: float = 0.7,
        noise_fraction_start: float = 1.0,
        noise_fraction_end: float = 0.0,
    ):

        def cfg_function(args):
            """Rescaled CFG+PAG"""
            model = args["model"]
            cond_pred = args["cond_denoised"]  # x_0
            uncond_pred = args["uncond_denoised"]
            cond_scale = args["cond_scale"]
            cond = args["cond"]
            uncond = args["uncond"]
            sigma = args["sigma"]
            model_options = args["model_options"].copy()
            x = args["input"]

            noise_fracs = model_options["transformer_options"]["sample_sigmas"] ** 2
            current_frac = sigma**2 / noise_fracs[0]

            if (noise_fraction_end <= current_frac) and (
                current_frac <= noise_fraction_start
            ):
                blocks = [("middle", 0, None)]
                # blocks = (
                #     # [("input", i, None) for i in range(4)]
                #     # + [("middle", 0, None)]
                #     # + [("output", i, None) for i in range(4)]
                # )

                for block in blocks:
                    layer, number, index = block
                    model_options = set_model_options_patch_replace(
                        model_options,
                        guidance.downsampled_attention_wrapper(
                            optimized_attention,
                            downsample_factor=downsample_factor,
                        ),
                        "attn1",
                        layer,
                        number,
                        index,
                    )

                downsampled_cond_pred, downsampled_uncond_pred = calc_cond_batch(
                    model, [cond, uncond], x, sigma, model_options
                )
            else:
                downsampled_cond_pred = cond_pred
                downsampled_uncond_pred = uncond_pred

            pred_final = guidance.guidance_combine_and_scale(
                x,
                sigma,
                cond_pred,
                uncond_pred,
                cond_scale,
                downsampled_cond_pred,
                downsampled_uncond_pred,
                # cond_pred,
                guidance_weight,
                scaling_method=GuidanceScalingMethod.V_SPACE_RESCALE,
                apply_rescaling_to_guidance=apply_rescaling_to_guidance,
                rescaling_fraction=rescaling_fraction,
            )
            return pred_final

        m = model.clone()
        m.set_model_sampler_post_cfg_function(cfg_function)
        return (m,)


NODE_CLASS_MAPPINGS = {
    "DownsampledLatentGuidance": DownsampledLatentGuidance,
    "DownsampledAttentionGuidance": DownsampledAttentionGuidance,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DownsampledLatentGuidance": "Downsampled Latent Guidance",
    "DownsampledAttentionGuidance": "Downsampled Attention Guidance",
}
