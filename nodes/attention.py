import torch
from comfy.model_patcher import ModelPatcher, set_model_options_patch_replace
from comfy.samplers import calc_cond_batch
from comfy.ldm.modules.attention import optimized_attention
from ..core import guidance as guidance
from ..core import latent_filters as lf
from ..core.combinations import (
    v_space_normalised_combination,
    pred_space_normalised_combination,
    snf_combination,
    AlternateConditionalComparator,
)


class RescaledPAG:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "pag_weight": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "round": 0.01,
                    },
                ),
                "apply_rescaling_to_pag": ("BOOLEAN", {"default": True}),
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
        return "Implementation of Perturbed Attention Guidance ( PAG ) that also incorporates the Rescale CFG approach"

    def patch(
        self,
        model: ModelPatcher,
        pag_weight: float = 2.0,
        apply_rescaling_to_pag: bool = True,
        rescaling_fraction: float = 0.7,
        unet_block: str = "middle",
        unet_block_id: int = 0,
        noise_fraction_start: float = 1.0,
        noise_fraction_end: float = 0.0,
        unet_block_list: str = "",
    ):

        def cfg_function(args):
            """Rescaled CFG+PAG"""
            model = args["model"]
            cond_pred = args["cond_denoised"]  # x_0
            uncond_pred = args["uncond_denoised"]
            cond_scale = args["cond_scale"]
            cond = args["cond"]
            sigma = args["sigma"]
            model_options = args["model_options"].copy()
            x = args["input"]

            noise_fracs = model_options["transformer_options"]["sample_sigmas"] ** 2
            current_frac = sigma**2 / noise_fracs[0]

            if (noise_fraction_end <= current_frac) and (
                current_frac <= noise_fraction_start
            ):
                if unet_block_list:
                    blocks = guidance.parse_unet_blocks(model, unet_block_list)
                else:
                    blocks = [(unet_block, unet_block_id, None)]

                for block in blocks:
                    layer, number, index = block
                    model_options = set_model_options_patch_replace(
                        model_options,
                        guidance.pag_perturbed_attention,
                        "attn1",
                        layer,
                        number,
                        index,
                    )

                (pag_cond_pred,) = calc_cond_batch(
                    model, [cond], x, sigma, model_options
                )
            else:
                pag_cond_pred = cond_pred

            pred_final = v_space_normalised_combination(
                x,
                sigma,
                cond_pred,
                uncond_pred,
                cond_scale,
                pag_cond_pred,
                None,
                pag_weight,
                rescaling_fraction,
                apply_rescaling_to_pag,
                AlternateConditionalComparator.BASE_COND,
            )

            return pred_final

        m = model.clone()
        m.set_model_sampler_post_cfg_function(cfg_function)
        return (m,)


class RescaledSEG:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "seg_sigma": (
                    "FLOAT",
                    {
                        "default": 1000.0,
                        "min": 0.0,
                        "max": 9999.0,
                        "step": 5.0,
                        "round": 0.01,
                    },
                ),
                "seg_weight": (
                    "FLOAT",
                    {
                        "default": 1.5,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "round": 0.01,
                    },
                ),
                "apply_rescaling_to_seg": ("BOOLEAN", {"default": False}),
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
        return "Implementation of Smoothed Energy Guidance ( SEG ) that also incorporates the Rescale CFG approach"

    def patch(
        self,
        model: ModelPatcher,
        seg_sigma: float = 1000.0,
        seg_weight: float = 1.5,
        apply_rescaling_to_seg: bool = True,
        rescaling_fraction: float = 0.7,
        unet_block: str = "middle",
        unet_block_id: int = 0,
        noise_fraction_start: float = 1.0,
        noise_fraction_end: float = 0.0,
        unet_block_list: str = "",
    ):

        def cfg_function(args):
            """Rescaled CFG+PAG"""
            model = args["model"]
            cond_pred = args["cond_denoised"]  # x_0
            uncond_pred = args["uncond_denoised"]
            cond_scale = args["cond_scale"]
            cond = args["cond"]
            sigma = args["sigma"]
            model_options = args["model_options"].copy()
            x = args["input"]

            noise_fracs = model_options["transformer_options"]["sample_sigmas"] ** 2
            current_frac = sigma**2 / noise_fracs[0]

            if (noise_fraction_end <= current_frac) and (
                current_frac <= noise_fraction_start
            ):
                if unet_block_list:
                    blocks = guidance.parse_unet_blocks(model, unet_block_list)
                else:
                    blocks = [(unet_block, unet_block_id, None)]
                    # blocks = (
                    #     # [("input", i, None) for i in range(4)]
                    #     # + [("middle", 0, None)]
                    #     # + [("output", i, None) for i in range(4)]
                    # )

                for block in blocks:
                    layer, number, index = block
                    model_options = set_model_options_patch_replace(
                        model_options,
                        guidance.seg_attention_wrapper(
                            optimized_attention,
                            blur_sigma=seg_sigma,
                        ),
                        "attn1",
                        layer,
                        number,
                        index,
                    )

                (seg_cond_pred,) = calc_cond_batch(
                    model, [cond], x, sigma, model_options
                )
            else:
                seg_cond_pred = cond_pred

            pred_final = v_space_normalised_combination(
                x,
                sigma,
                cond_pred,
                uncond_pred,
                cond_scale,
                seg_cond_pred,
                None,
                seg_weight,
                rescaling_fraction,
                apply_rescaling_to_seg,
                AlternateConditionalComparator.BASE_COND,
            )
            # pred_final = uncond_pred + snf_combination(
            #     cond_pred,
            #     uncond_pred,
            #     seg_cond_pred,
            #     None,
            #     cond_scale,
            #     seg_weight,
            #     AlternateConditionalComparator.BASE_COND,
            # )
            return pred_final

        m = model.clone()
        m.set_model_sampler_post_cfg_function(cfg_function)
        return (m,)


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
            else:
                downsampled_cond, downsampled_uncond = cond_pred, uncond_pred

            pred_final = v_space_normalised_combination(
                x,
                sigma,
                cond_pred,
                uncond_pred,
                cond_scale,
                downsampled_cond,
                downsampled_uncond,
                -guidance_weight,
                rescaling_fraction,
                apply_rescaling_to_guidance,
                AlternateConditionalComparator.ALTERNATE_UNCOND,
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

                (downsampled_cond_pred,) = calc_cond_batch(
                    model, [cond], x, sigma, model_options
                )
            else:
                downsampled_cond_pred = cond_pred

            pred_final = v_space_normalised_combination(
                x,
                sigma,
                cond_pred,
                uncond_pred,
                cond_scale,
                downsampled_cond_pred,
                None,
                -guidance_weight,
                rescaling_fraction,
                apply_rescaling_to_guidance,
                AlternateConditionalComparator.BASE_COND,
            )
            return pred_final

        m = model.clone()
        m.set_model_sampler_post_cfg_function(cfg_function)
        return (m,)


NODE_CLASS_MAPPINGS = {
    "RescaledPAG": RescaledPAG,
    "RescaledSEG": RescaledSEG,
    "DownsampledLatentGuidance": DownsampledLatentGuidance,
    "DownsampledAttentionGuidance": DownsampledAttentionGuidance,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RescaledPAG": "Rescaled PAG",
    "RescaledSEG": "Rescaled SEG",
    "DownsampledLatentGuidance": "Downsampled Latent Guidance",
    "DownsampledAttentionGuidance": "Downsampled Attention Guidance",
}
