from comfy.model_patcher import ModelPatcher
from comfy.samplers import calc_cond_batch
from ..core import guidance as guidance
from ..core import utils as utils


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
                        "default": -1.0,
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
        guidance_type: str = "SEG",
        guidance_weight: float = 1.5,
        param1: float = 0.0,
        param2: float = -1.0,
        param3: float = -1.0,
        apply_rescaling_to_alternate_guidance: bool = False,
        rescaling_method: str = "VSpaceRescale",
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

        guidance_type = guidance.GuidanceType(guidance_type)
        rescaling_method = guidance.GuidanceScalingMethod(rescaling_method)

        def cfg_function(args):
            """Rescaled CFG+PAG"""
            model = args["model"]
            cond_pred = args["cond_denoised"]  # x_0
            uncond_pred = args["uncond_denoised"]
            cond_scale = args["cond_scale"]
            cond = args["cond"]
            sigma = args["sigma"]
            model_options = args["model_options"]
            x = args["input"]

            sigmas = model_options["transformer_options"]["sample_sigmas"]
            current_frac = sigma**2 / sigmas[0] ** 2

            if (
                (current_frac > noise_fraction_start)
                or (current_frac <= noise_fraction_end)
                or guidance_weight == 0
            ):
                # Skip
                alternate_cond_pred = cond_pred
            else:
                match guidance_type:
                    case guidance.GuidanceType.PAG:
                        attention_fn = guidance.pag_attention_wrapper()
                    case guidance.GuidanceType.SEG:
                        attention_fn = guidance.seg_attention_wrapper(param1)
                    case guidance.GuidanceType.RANDOM_DROP:
                        attention_fn = guidance.random_drop_attention_wrapper(
                            param1, param2
                        )
                    case guidance.GuidanceType.RANDOM_DROP_DUAL:
                        attention_fn = guidance.random_drop_dual_attention_wrapper(
                            param1, param2
                        )
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
                cond_pred,
                guidance_weight,
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
