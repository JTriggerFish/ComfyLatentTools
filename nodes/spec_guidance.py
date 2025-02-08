import kornia.geometry.transform

from comfy.model_patcher import ModelPatcher, set_model_options_patch_replace
from comfy.samplers import calc_cond_batch
import comfy.utils
from comfy.ldm.modules.attention import optimized_attention
from ..core import guidance as guidance
from ..core import utils as utils
from ..core import latent_filters as lf
import torch


class DownsampledLatentGuidance:
    rescaling_methods = [k.value for k in guidance.GuidanceScalingMethod]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "reference_latent": ("LATENT",),
                "downscaling_factor": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 0.0,
                        "max": 8.0,
                        "step": 0.01,
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
                "apply_rescaling_to_alternate_guidance": (
                    "BOOLEAN",
                    {"default": False},
                ),
                "rescaling_method ": (
                    s.rescaling_methods,
                    {"default": "None"},
                ),
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
                "unet_block": (["input", "middle", "output"], {"default": "middle"}),
                "unet_block_id": ("INT", {"default": 0}),
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
            "optional": {
                "unet_block_list": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model_patches/unet"
    OUTPUT_NODE = False

    @classmethod
    def help(cls):
        return "!EXPERIMENTAL! use guidance extracted from a downscaled latent space to improve high-resolution image synthesis."

    def patch(
        self,
        model: ModelPatcher,
        reference_latent: dict,
        downsample_factor: float = 2.0,
        guidance_weight: float = 1.5,
        apply_rescaling_to_alternate_guidance: bool = False,
        rescaling_method: str = "None",
        rescaling_fraction: float = 0.7,
        unet_block: str = "middle",
        unet_block_id: int = 0,
        noise_fraction_start: float = 1.0,
        noise_fraction_end: float = 0.0,
        unet_block_list: str = "",
    ):

        if unet_block_list:
            blocks = utils.parse_unet_blocks(model, unet_block_list)
        else:
            blocks = [(unet_block, unet_block_id, None)]

        model_sampling = model.get_model_object("model_sampling")
        sigma_start = model_sampling.percent_to_sigma(1 - noise_fraction_start)
        sigma_end = model_sampling.percent_to_sigma(1 - noise_fraction_end)
        rescaling_method = guidance.GuidanceScalingMethod(rescaling_method)

        reference_latent = reference_latent["samples"]

        def cfg_function(args):
            """Rescaled CFG+PAG"""
            model = args["model"]
            cond_pred = args["cond_denoised"]  # x_0
            uncond_pred = args["uncond_denoised"]
            cond_scale = args["cond_scale"]
            cond = args["cond"]
            uncond = args["uncond"]
            sigma = args["sigma"]
            model_options = args["model_options"]
            x = args["input"]

            sigmas = model_options["transformer_options"]["sample_sigmas"]
            current_frac = sigma**2 / sigmas[0] ** 2

            # attention_kq_copy_fn = guidance.copy_attention_keys_queries_wrapper()

            if (
                (current_frac > noise_fraction_start)
                or (current_frac <= noise_fraction_end)
                or guidance_weight == 0
            ):
                # Standard CFG
                return uncond_pred + cond_scale * (cond_pred - uncond_pred)
            else:
                # sigma_filter = 2.61
                # x_filtered = lf.gaussian_blur_2d(
                #     x, lf.gaussian_kernel_size_for_img(sigma_filter, x), sigma_filter
                # )

                ref_latent = reference_latent.to(device=cond_pred.device)
                noise = sigma.to(device=ref_latent.device) * torch.randn_like(
                    ref_latent, device=ref_latent.device
                )
                x_reference = ref_latent + noise

                # x_downscaled = kornia.geometry.transform.rescale(
                #     x,
                #     1 / downsample_factor,
                #     interpolation="area",
                #     antialias=False,
                # )
                (cond_down,) = calc_cond_batch(
                    model,
                    [cond],
                    # x_downscaled,
                    x_reference,
                    sigma,
                    model_options,
                )
                cond_down = kornia.geometry.transform.rescale(
                    cond_down,
                    float(downsample_factor),
                    interpolation="nearest",
                    antialias=False,
                )
                attention_fn = guidance.random_rotation_wrapper(1.0, 1.0)
                # attention_fn = guidance.fuzzy_attention_wrapper(1.0, 0.5)
                # attention_fn = guidance.affine_attention_transform_wrapper(1.0, -100, 0)

                model_options_perturbed = guidance.patch_attention_in_model_blocks(
                    model_options, attention_fn, blocks
                )

                (cond_down_peturbed,) = calc_cond_batch(
                    model,
                    [cond],
                    # x_downscaled,
                    x_reference,
                    sigma,
                    model_options_perturbed,
                )
                # uncond_down = kornia.geometry.transform.rescale(
                #     uncond_down,
                #     float(downsample_factor),
                #     interpolation="bicubic",
                #     antialias=False,
                # )
                cond_down_peturbed = kornia.geometry.transform.rescale(
                    cond_down_peturbed,
                    float(downsample_factor),
                    interpolation="nearest",
                    antialias=False,
                )

                # return lf.mix_fft_phase_amplitude(
                #     uncond_pred + cond_scale * (cond_pred - uncond_pred),
                #     cond_pred + 1 * (cond_down - cond_down_peturbed),
                #     guidance_weight,
                # )
                final_pred = (
                    uncond_pred
                    + cond_scale * (cond_pred - uncond_pred)
                    + guidance_weight * (cond_down - cond_down_peturbed)
                )
                moment_matched = lf.moment_match(cond_pred, final_pred)
                return (
                    final_pred * (1 - rescaling_fraction)
                    + moment_matched * rescaling_fraction
                )

        m = model.clone()
        m.set_model_sampler_post_cfg_function(cfg_function)
        return (m,)


NODE_CLASS_MAPPINGS = {
    "DownsampledLatentGuidance": DownsampledLatentGuidance,
    # "DownsampledAttentionGuidance": DownsampledAttentionGuidance,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DownsampledLatentGuidance": "Downsampled Latent Guidance",
    # "DownsampledAttentionGuidance": "Downsampled Attention Guidance",
}
