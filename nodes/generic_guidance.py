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
            current_step = torch.where(sigmas == sigma)[0].item()
            total_steps = len(sigmas)

            if apply_cosine_schedule_to_guidance:
                cosine_schedule = 0.5 * (1 + np.cos(np.pi * current_step / total_steps))
                guidance_w = guidance_weight * cosine_schedule
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
                    case _:
                        raise ValueError(f"Unsupported guidance type: {guidance_type}")

                model_options = guidance.patch_attention_in_model_blocks(
                    model_options, attention_fn, blocks
                )

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


NODE_CLASS_MAPPINGS = {
    "GenericAttentionGuidance": GenericAttentionGuidance,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "GenericAttentionGuidance": "Generic Attention Guidance",
}
