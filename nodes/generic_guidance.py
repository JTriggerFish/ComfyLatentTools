from comfy.ldm.flux.math import attention
from comfy.model_patcher import ModelPatcher
from comfy.samplers import calc_cond_batch
from ..core import guidance as guidance
from ..core import utils as utils
import torch
import numpy as np


class GenericAttentionGuidance:
    rescaling_methods = [k.value for k in guidance.GuidanceScalingMethod]
    guidance_types = [k.value for k in guidance.GuidanceType]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "guidance_type": (
                    s.guidance_types,
                    {"default": "RandomDrop"},
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
                "param1": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -1.0,
                        "max": 1.0,
                        "step": 0.01,
                        "round": 0.01,
                    },
                ),
                "param2": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -1.0,
                        "max": 1.0,
                        "step": 0.01,
                        "round": 0.01,
                    },
                ),
                "param3": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -1.0,
                        "max": 1.0,
                        "step": 0.01,
                        "round": 0.01,
                    },
                ),
                "apply_rescaling_to_alternate_guidance": (
                    "BOOLEAN",
                    {"default": False},
                ),
                "rescaling_method ": (
                    s.rescaling_methods,
                    {"default": "VSpaceRescale"},
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
                "apply_cosine_schedule_to_guidance": (
                    "BOOLEAN",
                    {"default": False},
                ),
                "time_perturbation_std": (
                    "FLOAT",
                    {
                        "default": 0.00,
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
        return "Implementation of generic attention guidance, with different types of guidance and rescaling available"

    def patch(
        self,
        model: ModelPatcher,
        guidance_type: str = "AAT",
        guidance_weight: float = 2.0,
        param1: float = 0.0,
        param2: float = 0.0,
        param3: float = 0.0,
        apply_rescaling_to_alternate_guidance: bool = False,
        rescaling_method: str = "None",
        rescaling_fraction: float = 0.7,
        unet_block: str = "middle",
        unet_block_id: int = 0,
        noise_fraction_start: float = 1.0,
        noise_fraction_end: float = 0.0,
        apply_cosine_schedule_to_guidance: bool = False,
        time_perturbation_std: float = 0.05,
        unet_block_list: str = "",
    ):
        """

        :param model:
        :param guidance_type:
        :param guidance_weight: Recommeded around 2.0
        :param param1: 1.0 for most
        :param param2: -100 for AAT, 0 to 1 for RandomRotation
        :param param3:
        :param apply_rescaling_to_alternate_guidance:
        :param rescaling_method: Recommended : PredSpaceRescale
        :param rescaling_fraction: Recommended : 0.7 for VSpaceRescale, 0.3-0.7 for PredSpaceRescale,
            1.0 for Softmax ( there it is the temperature)
        :param unet_block:
        :param unet_block_id:
        :param noise_fraction_start:
        :param noise_fraction_end:
        :param apply_cosine_schedule_to_guidance:
        :param unet_block_list:
        :return:
        """
        if unet_block_list:
            blocks = utils.parse_unet_blocks(model, unet_block_list)
        else:
            blocks = [(unet_block, unet_block_id, None)]

        guidance_type = guidance.GuidanceType(guidance_type)
        rescaling_method = guidance.GuidanceScalingMethod(rescaling_method)

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
            current_step = torch.argmin(torch.abs(sigmas - sigma)).item()
            total_steps = len(sigmas)

            # guidance_schedule_weight = 1 - exp_linear_schedule(
            #     current_step, total_steps, alpha=-3
            # )

            if apply_cosine_schedule_to_guidance:
                cosine_schedule = 1 - 0.5 * (
                    1 + np.cos(np.pi * current_step / total_steps)
                )
                # guidance_w = guidance_weight * cosine_schedule
                guidance_w = guidance_weight * exp_linear_schedule(
                    current_step, total_steps, alpha=5
                )
                guidance_w = guidance_weight * float(
                    torch.sqrt(sigmas[0] ** 2 - sigma**2) / sigmas[0]
                )
            else:
                guidance_w = guidance_weight

            if (
                (current_frac > noise_fraction_start)
                or (current_frac <= noise_fraction_end)
                or abs(guidance_w) < 1e-6
            ):
                # Skip
                alternate_cond_pred = cond_pred
            else:
                match guidance_type:
                    case guidance.GuidanceType.SCRAMBLE:
                        attention_fn = guidance.scramble_attention_wrapper(
                            param1, param2
                        )
                    case guidance.GuidanceType.PAG:
                        attention_fn = guidance.pag_attention_wrapper()
                    case guidance.GuidanceType.SEG:
                        attention_fn = guidance.seg_attention_wrapper(param1)
                    case guidance.GuidanceType.AAT:
                        attention_fn = guidance.affine_attention_transform_wrapper(
                            param1, param2, param3
                        )
                    case guidance.GuidanceType.FUZZY:
                        attention_fn = guidance.fuzzy_attention_wrapper(param1, param2)
                    case guidance.GuidanceType.RANDOM_ROTATION:
                        attention_fn = guidance.random_rotation_wrapper(param1, param2)
                    case guidance.GuidanceType.PERMUTE:
                        attention_fn = guidance.permute_attention_wrapper(
                            param1, param2
                        )
                    case guidance.GuidanceType.VALUE_RESCALE:
                        attention_fn = guidance.value_rescale_attention_wrapper(
                            param1, param2, param3
                        )
                    case guidance.GuidanceType.RANDOM_DROP:
                        attention_fn = guidance.random_drop_wrapper(
                            param1, param2, param3
                        )
                    case guidance.GuidanceType.SVD:
                        attention_fn = guidance.value_svd_attention_wrapper(
                            max(0, int(param1)), param2
                        )
                    case guidance.GuidanceType.RANDOM_SUBSPACE:
                        attention_fn = guidance.random_subspace_projection_wrapper(
                            max(0, int(param1)), param2
                        )
                    case guidance.GuidanceType.PHASE:
                        attention_fn = guidance.fft_phase_shift_wrapper(param1, param2)
                    case _:
                        raise ValueError(f"Unsupported guidance type: {guidance_type}")

                if attention_fn is not None:
                    model_options = guidance.patch_attention_in_model_blocks(
                        model_options, attention_fn, blocks
                    )

                if time_perturbation_std > 0:
                    sigma = (
                        sigma
                        * torch.exp(
                            time_perturbation_std * torch.randn_like(sigma)
                            - 0.5 * time_perturbation_std**2
                        )
                    ).clamp(sigmas[-1], sigmas[0])
                (alternate_cond_pred,) = calc_cond_batch(
                    model, [cond], x, sigma, model_options
                )

            pred_final = guidance.guidance_combine_and_scale(
                x,
                sigma,
                cond_pred,
                uncond_pred,
                cond_scale,
                alternate_cond_pred,
                # alternate_uncond_pred,
                cond_pred,
                guidance_w,
                scaling_method=rescaling_method,
                apply_rescaling_to_guidance=apply_rescaling_to_alternate_guidance,
                rescaling_fraction=rescaling_fraction,
            )
            return pred_final

        m = model.clone()
        m.set_model_sampler_post_cfg_function(cfg_function)
        return (m,)


def exp_linear_schedule(
    step: int,
    total_steps: int,
    min_val: float = 0.0,
    max_val: float = 1.0,
    alpha: float = 0.0,
) -> float:
    """
    Returns a value in [min_val, max_val] at integer 'step' (0 <= step < total_steps).
    Interpolates between linear (alpha=0) and exponential-like shape (alpha>0).

    fraction(t; alpha) = (1 - exp(-alpha * t)) / (1 - exp(-alpha)), where t = step / (total_steps - 1).

    Args:
        step (int): Current step, 0 <= step < total_steps.
        total_steps (int): Total number of steps.
        min_val (float): The minimum output value.
        max_val (float): The maximum output value.
        alpha (float): Shape parameter.
            - alpha = 0   => purely linear (in the limit sense).
            - alpha > 0   => more exponential-like.

    Returns:
        float: Scheduled value at the given step.

    Raises:
        ValueError: If step is out of range or total_steps < 1.
    """
    if step < 0 or step >= total_steps:
        raise ValueError(f"step={step} must be in [0, {total_steps-1}].")
    if total_steps < 2:
        return float(max_val)

    t = step / (total_steps - 1)

    if alpha == 0.0:
        fraction = t
    else:
        denom = 1.0 - np.exp(-alpha)
        if abs(denom) < 1e-12:
            fraction = t
        else:
            fraction = (1.0 - np.exp(-alpha * t)) / denom

    val = min_val + fraction * (max_val - min_val)
    return float(val)


NODE_CLASS_MAPPINGS = {
    "GenericAttentionGuidance": GenericAttentionGuidance,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "GenericAttentionGuidance": "Generic Attention Guidance",
}
