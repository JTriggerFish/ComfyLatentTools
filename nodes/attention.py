import torch
from comfy.model_patcher import ModelPatcher, set_model_options_patch_replace
from comfy.samplers import calc_cond_batch
from comfy.ldm.modules.attention import optimized_attention
from ..core import unet as unet


# import torch
def compute_v_space_normalised_combination(
    x: torch.Tensor,
    sigma_: float,
    cond_x: torch.Tensor,
    uncond_x: torch.Tensor,
    cond_scale: float,
    alternate_cond_x: torch.Tensor,
    alternate_weight: float,
    normalised_fraction: float,
    apply_rescaling_to_alternate: bool,
) -> torch.Tensor:
    """
    Convert from x_0 prediction space to v prediction space and combine
    with partial rescaling

    """
    # Reshape to match the dimensions of the input tensor
    sigma = sigma_.view(sigma_.shape[:1] + (1,) * (cond_x.ndim - 1))

    # Convert to variance preserving space
    alpha = 1 / (sigma**2 + 1.0) ** 0.5
    sigma = sigma * alpha
    sigma_inv = 1.0 / sigma
    x = x * alpha

    cond_v = (alpha * x - cond_x) * sigma_inv
    uncond_v = (alpha * x - uncond_x) * sigma_inv

    s0: torch.Tensor = torch.std(cond_v, dim=(1, 2, 3), keepdim=True)
    if apply_rescaling_to_alternate:
        alternate_cond_v = (alpha * x - alternate_cond_x) * sigma_inv
        v_out_unnorm: torch.Tensor = (
            uncond_v
            + cond_scale * (cond_v - uncond_v)
            - alternate_weight * (alternate_cond_v - cond_v)
        )
        s1: torch.Tensor = torch.std(v_out_unnorm, dim=(1, 2, 3), keepdim=True)
        v_out_normed: torch.Tensor = v_out_unnorm * s0 / s1
        v_final: torch.Tensor = v_out_normed * normalised_fraction + v_out_unnorm * (
            1 - normalised_fraction
        )
        # Convert back to prediction space and rescale
        # pred_final = (alpha * x - v_final * sigma) / alpha
        pred_final = x * alpha - v_final * sigma
    else:
        # Apply standard CFG rescaling and combine with alternate
        v_cfg_unnorm: torch.Tensor = uncond_v + cond_scale * (cond_v - uncond_v)
        s1: torch.Tensor = torch.std(v_cfg_unnorm, dim=(1, 2, 3), keepdim=True)
        v_cfg_normed: torch.Tensor = v_cfg_unnorm * s0 / s1
        v_cfg: torch.Tensor = v_cfg_normed * normalised_fraction + v_cfg_unnorm * (
            1 - normalised_fraction
        )
        pred_cfg = x * alpha - v_cfg * sigma
        pred_final = pred_cfg - alternate_weight * (alternate_cond_x - cond_x)

    return pred_final


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
                "apply_rescaling_to_pag": ("BOOLEAN", {"default": True}),
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
                    blocks = unet.parse_unet_blocks(model, unet_block_list)
                else:
                    blocks = [(unet_block, unet_block_id, None)]

                for block in blocks:
                    layer, number, index = block
                    model_options = set_model_options_patch_replace(
                        model_options,
                        unet.pag_perturbed_attention,
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

            pred_final = compute_v_space_normalised_combination(
                x,
                sigma,
                cond_pred,
                uncond_pred,
                cond_scale,
                pag_cond_pred,
                pag_weight,
                rescaling_fraction,
                apply_rescaling_to_pag,
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
                "apply_rescaling_to_seg": ("BOOLEAN", {"default": True}),
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

            alpha = 1 / (sigma**2 + 1.0) ** 0.5
            sigma = sigma / (sigma**2 + 1.0) ** 0.5
            sigma_inv = 1.0 / sigma

            if (noise_fraction_end <= current_frac) and (
                current_frac <= noise_fraction_start
            ):
                if unet_block_list:
                    blocks = unet.parse_unet_blocks(model, unet_block_list)
                else:
                    blocks = [(unet_block, unet_block_id, None)]

                for block in blocks:
                    layer, number, index = block
                    model_options = set_model_options_patch_replace(
                        model_options,
                        unet.seg_attention_wrapper(
                            optimized_attention, blur_sigma=seg_sigma
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

            pred_final = compute_v_space_normalised_combination(
                x,
                sigma,
                cond_pred,
                uncond_pred,
                cond_scale,
                seg_cond_pred,
                seg_weight,
                rescaling_fraction,
                apply_rescaling_to_seg,
            )
            return pred_final

        m = model.clone()
        m.set_model_sampler_post_cfg_function(cfg_function)
        return (m,)


NODE_CLASS_MAPPINGS = {"RescaledPAG": RescaledPAG, "RescaledSEG": RescaledSEG}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RescaledPAG": "Rescaled PAG",
    "RescaledSEG": "Rescaled SEG",
}
