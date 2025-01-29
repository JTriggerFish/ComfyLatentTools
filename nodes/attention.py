import torch
from comfy.model_patcher import ModelPatcher, set_model_options_patch_replace
from comfy.samplers import calc_cond_batch
from comfy.ldm.modules.attention import optimized_attention
from unet import parse_unet_blocks, pag_perturbed_attention, seg_attention_wrapper


# import torch


class RescaledPAG:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "pag_weight": (
                    "FLOAT",
                    {
                        "default": 1.5,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "round": 0.01,
                    },
                ),
                "rescaling_fraction": (
                    "FLOAT",
                    {
                        "default": 0.9,
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
        pag_weight: float = 3.0,
        rescaling_fraction: float = 0.7,
        unet_block: str = "middle",
        unet_block_id: int = 0,
        noise_fraction_start: float = 1.0,
        noise_fraction_end: float = 0.0,
        unet_block_list: str = "",
    ):
        """
        :param model:
        :param pag_weight:
        :param rescaling_fraction:
        :param unet_block:
        :param unet_block_id:
        :param noise_fraction_start:
        :param noise_fraction_end:
        :param unet_block_list:
        :return:
        """

        def cfg_function(args):
            """Rescaled CFG+PAG"""
            model = args["model"]
            cond_pred = args["cond_denoised"]  # x_0
            uncond_pred = args["uncond_denoised"]
            cond_scale = args["cond_scale"]
            cond = args["cond"]
            sigma_ = args["sigma"]
            model_options = args["model_options"].copy()
            sigma = sigma_.view(sigma_.shape[:1] + (1,) * (cond_pred.ndim - 1))
            x = args["input"]

            noise_fracs = model_options["transformer_options"]["sample_sigmas"] ** 2
            current_frac = sigma_**2 / noise_fracs[0]

            alpha = 1 / (sigma**2 + 1.0) ** 0.5
            sigma = sigma / (sigma**2 + 1.0) ** 0.5
            sigma_inv = 1.0 / sigma

            cond_v = (alpha * x - cond_pred) * sigma_inv
            uncond_v = (alpha * x - uncond_pred) * sigma_inv

            if (noise_fraction_end <= current_frac) and (
                current_frac <= noise_fraction_start
            ):
                if unet_block_list:
                    blocks = parse_unet_blocks(model, unet_block_list)
                else:
                    blocks = [(unet_block, unet_block_id, None)]

                for block in blocks:
                    layer, number, index = block
                    model_options = set_model_options_patch_replace(
                        model_options,
                        pag_perturbed_attention,
                        "attn1",
                        layer,
                        number,
                        index,
                    )

                (pag_cond_pred,) = calc_cond_batch(
                    model, [cond], x, sigma_, model_options
                )
                pag_cond_v = (alpha * x - pag_cond_pred) * sigma_inv
            else:
                pag_cond_v = cond_v

            v_out_unnorm = (
                uncond_v
                + cond_scale * (cond_v - uncond_v)
                - pag_weight * (pag_cond_v - cond_v)
            )
            s0 = torch.std(cond_v, dim=(1, 2, 3), keepdim=True)
            s1 = torch.std(v_out_unnorm, dim=(1, 2, 3), keepdim=True)

            v_out_normed = v_out_unnorm * s0 / s1
            v_final = v_out_normed * rescaling_fraction + v_out_unnorm * (
                1 - rescaling_fraction
            )

            pred_final = alpha * x - v_final * sigma
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
                        "default": 30.0,
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
                "rescaling_fraction": (
                    "FLOAT",
                    {
                        "default": 0.9,
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
        seg_sigma: float = 30.0,
        seg_weight: float = 3.0,
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
            sigma_ = args["sigma"]
            model_options = args["model_options"].copy()
            sigma = sigma_.view(sigma_.shape[:1] + (1,) * (cond_pred.ndim - 1))
            x = args["input"]

            noise_fracs = model_options["transformer_options"]["sample_sigmas"] ** 2
            current_frac = sigma_**2 / noise_fracs[0]

            alpha = 1 / (sigma**2 + 1.0) ** 0.5
            sigma = sigma / (sigma**2 + 1.0) ** 0.5
            sigma_inv = 1.0 / sigma

            cond_v = (alpha * x - cond_pred) * sigma_inv
            uncond_v = (alpha * x - uncond_pred) * sigma_inv

            if (noise_fraction_end <= current_frac) and (
                current_frac <= noise_fraction_start
            ):
                if unet_block_list:
                    blocks = parse_unet_blocks(model, unet_block_list)
                else:
                    blocks = [(unet_block, unet_block_id, None)]

                for block in blocks:
                    layer, number, index = block
                    model_options = set_model_options_patch_replace(
                        model_options,
                        seg_attention_wrapper(
                            optimized_attention, blur_sigma=seg_sigma
                        ),
                        "attn1",
                        layer,
                        number,
                        index,
                    )

                (pag_cond_pred,) = calc_cond_batch(
                    model, [cond], x, sigma_, model_options
                )
                pag_cond_v = (alpha * x - pag_cond_pred) * sigma_inv
            else:
                pag_cond_v = cond_v

            v_out_unnorm = (
                uncond_v
                + cond_scale * (cond_v - uncond_v)
                - seg_weight * (pag_cond_v - cond_v)
            )
            s0 = torch.std(cond_v, dim=(1, 2, 3), keepdim=True)
            s1 = torch.std(v_out_unnorm, dim=(1, 2, 3), keepdim=True)

            v_out_normed = v_out_unnorm * s0 / s1
            v_final = v_out_normed * rescaling_fraction + v_out_unnorm * (
                1 - rescaling_fraction
            )

            pred_final = alpha * x - v_final * sigma
            return pred_final

        m = model.clone()
        m.set_model_sampler_post_cfg_function(cfg_function)
        return (m,)


NODE_CLASS_MAPPINGS = {"RescaledPAG": RescaledPAG, "RescaledSEG": RescaledSEG}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RescaledPAG": "Rescaled PAG",
    "RescaledSEG": "Rescaled SEG",
}
