from comfy.model_patcher import ModelPatcher, set_model_options_patch_replace
from comfy.samplers import calc_cond_batch
import kornia.geometry
from torch import Tensor
import torch
import math
import enum
from comfy.ldm.modules.attention import optimized_attention

# Seems like for ComfyUI / ComfyScript, only relative imports work properly
try:
    from .latent_filters import gaussian_blur_2d, gaussian_kernel_size_for_img
    from . import utils
except ImportError:
    from latent_filters import gaussian_blur_2d, gaussian_kernel_size_for_img
    import utils


class GuidanceType(str, enum.Enum):
    SEG = "SEG"
    PAG = "PAG"
    SWG = "SWG"
    DOWNSAMPLED = "Downsampled"
    RANDOM_DROP = "RandomDrop"
    DISTANCE_WEIGHTED = "DistanceWeighted"


class GuidanceScalingMethod(str, enum.Enum):
    NONE = "None"
    PRED_SPACE_RESCALE = "PredSpaceRescale"
    V_SPACE_RESCALE = "VSpaceRescale"
    SNF = "SNF"


def plain_guidance_combine(
    cond_pred: Tensor,
    uncond_pred: Tensor,
    cfg: float,
    alternate_pred: Tensor | None,
    alternate_pred_ref: Tensor | None,
    alternate_guidance_weight: float,
) -> Tensor:
    base_guidance = uncond_pred + cfg * (cond_pred - uncond_pred)

    if alternate_pred is None:
        guidance_adj = 0
    else:
        if alternate_pred_ref is None:
            alternate_pred_ref = 0
        guidance_adj = alternate_guidance_weight * (alternate_pred_ref - alternate_pred)

    return base_guidance + guidance_adj


def snf_guidance_combine(
    cond_pred: Tensor,
    uncond_pred: Tensor,
    cfg: float,
    alternate_pred: Tensor | None,
    alternate_pred_ref: Tensor | None,
    alternate_guidance_weight: float,
) -> Tensor:
    base_guidance = uncond_pred + cfg * (cond_pred - uncond_pred)

    if alternate_pred is None:
        raise ValueError("Alternate prediction must be provided for SNF guidance.")
    else:
        if alternate_pred_ref is None:
            alternate_pred_ref = 0
        guidance_adj = alternate_guidance_weight * (alternate_pred_ref - alternate_pred)

    return utils.saliency_tensor_combination(base_guidance, guidance_adj)


def pred_rescaled_guidance_combine(
    cond_pred: Tensor,
    uncond_pred: Tensor,
    cfg: float,
    alternate_pred: Tensor | None,
    alternate_pred_ref: Tensor | None,
    alternate_guidance_weight: float,
    apply_rescaling_to_guidance: bool,
    rescaling_fraction: float,
) -> Tensor:
    base_guidance = uncond_pred + cfg * (cond_pred - uncond_pred)

    if alternate_pred is None:
        guidance_adj = 0
    else:
        if alternate_pred_ref is None:
            alternate_pred_ref = 0
        guidance_adj = alternate_guidance_weight * (alternate_pred_ref - alternate_pred)

    if apply_rescaling_to_guidance:
        guidance = base_guidance + guidance_adj
        guidance = utils.partial_rescaling(
            ref_tensor=cond_pred, z=guidance, rescaled_fraction=rescaling_fraction
        )
        return guidance
    else:
        base_guidance = utils.partial_rescaling(
            ref_tensor=cond_pred, z=base_guidance, rescaled_fraction=rescaling_fraction
        )
        return base_guidance - guidance_adj


def v_space_rescaled_guidance_combine(
    x: Tensor,
    sigma: Tensor,
    cond_pred: Tensor,
    uncond_pred: Tensor,
    cfg: float,
    alternate_pred: Tensor | None,
    alternate_pred_ref: Tensor | None,
    alternate_guidance_weight: float,
    apply_rescaling_to_guidance: bool,
    rescaling_fraction: float,
) -> Tensor:
    # Reshape sigma to match the dimensions of the input tensor
    sigma = sigma.view(sigma.shape[:1] + (1,) * (cond_pred.ndim - 1))

    # Transform predictions to V-space
    [cond, uncond, alt, alt_ref] = utils.pred_to_v(
        x,
        sigma,
        [cond_pred, uncond_pred, alternate_pred, alternate_pred_ref],
    )
    if not apply_rescaling_to_guidance:
        alt, alt_ref = alternate_pred, alternate_pred_ref

    base_guidance = uncond + cfg * (cond - uncond)

    if alt is None:
        guidance_adj = 0
    else:
        if alt_ref is None:
            alt_ref = 0
        guidance_adj = alternate_guidance_weight * (alt_ref - alt)

    if apply_rescaling_to_guidance:
        guidance = base_guidance + guidance_adj
        guidance = utils.partial_rescaling(
            ref_tensor=cond, z=guidance, rescaled_fraction=rescaling_fraction
        )
        final_pred = utils.v_to_pred(x, sigma, [guidance])[0]
    else:
        base_guidance = utils.partial_rescaling(
            ref_tensor=cond, z=base_guidance, rescaled_fraction=rescaling_fraction
        )
        final_pred = utils.v_to_pred(x, sigma, [base_guidance])[0]
        final_pred += guidance_adj

    return final_pred


def guidance_combine_and_scale(
    x: Tensor,
    sigma: Tensor,
    cond_pred: Tensor,
    uncond_pred: Tensor,
    cfg: float,
    alternate_pred: Tensor | None,
    alternate_pred_ref: Tensor | None,
    alternate_guidance_weight: float,
    scaling_method: GuidanceScalingMethod,
    apply_rescaling_to_guidance: bool,
    rescaling_fraction: float,
):
    match scaling_method:
        case GuidanceScalingMethod.NONE:
            return plain_guidance_combine(
                cond_pred,
                uncond_pred,
                cfg,
                alternate_pred,
                alternate_pred_ref,
                alternate_guidance_weight,
            )
        case GuidanceScalingMethod.PRED_SPACE_RESCALE:
            return pred_rescaled_guidance_combine(
                cond_pred,
                uncond_pred,
                cfg,
                alternate_pred,
                alternate_pred_ref,
                alternate_guidance_weight,
                apply_rescaling_to_guidance,
                rescaling_fraction,
            )
        case GuidanceScalingMethod.V_SPACE_RESCALE:
            return v_space_rescaled_guidance_combine(
                x,
                sigma,
                cond_pred,
                uncond_pred,
                cfg,
                alternate_pred,
                alternate_pred_ref,
                alternate_guidance_weight,
                apply_rescaling_to_guidance,
                rescaling_fraction,
            )
        case GuidanceScalingMethod.SNF:
            return snf_guidance_combine(
                cond_pred,
                uncond_pred,
                cfg,
                alternate_pred,
                alternate_pred_ref,
                alternate_guidance_weight,
            )
        case _:
            raise ValueError(f"Unknown guidance scaling method: {scaling_method}")


def get_all_attention_blocks_for_sdxl() -> list[tuple[str, int, int | None]]:
    return (
        [("input", i, None) for i in range(4)]
        + [("middle", 0, None)]
        + [("output", i, None) for i in range(4)]
    )


def patch_attention_in_model_blocks(
    model_options: dict,
    attention_fn: callable,
    blocks: list[tuple[str, int, int | None]],
) -> dict:
    model_options = model_options.copy()
    for block in blocks:
        layer, number, index = block
        model_options = set_model_options_patch_replace(
            model_options,
            attention_fn,
            "attn1",
            layer,
            number,
            index,
        )
    return model_options


def pag_attention_wrapper() -> callable:
    def perturbed_attention(
        q: Tensor, k: Tensor, v: Tensor, extra_options, mask=None
    ) -> Tensor:
        """Perturbed self-attention corresponding to an identity matrix replacing the attention matrix."""
        return v

    return perturbed_attention


def seg_attention_wrapper(scaled_blur_sigma: float = 0.1) -> callable:
    """
    Wraps an attention function to apply a Gaussian blur (via Kornia) on q before computing attention.

    Args:
        scaled_blur_sigma: If >= 0, apply Gaussian blur with this sigma. If < 0, replace q by the global mean.
        The sigma is scaled by the square root of the area of the image / 10, such that
        a sigma of 1 corresponds to a blur radius of 1/10th of the image size and a sigma of 10 corresponds to a blur
        radius of the entire image size.
    """

    def seg_perturbed_attention(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        extra_options,
        mask=None,
    ) -> Tensor:
        heads = extra_options["n_heads"]
        bs, area, inner_dim = q.shape

        height_orig, width_orig = extra_options["original_shape"][2:4]
        aspect_ratio = width_orig / height_orig

        # Reshape (B, area, dim) -> (B, dim, H, W)
        if aspect_ratio >= 1.0:
            height = round((area / aspect_ratio) ** 0.5)
            q = q.permute(0, 2, 1).reshape(bs, inner_dim, height, -1)
        else:
            width = round((area * aspect_ratio) ** 0.5)
            q = q.permute(0, 2, 1).reshape(bs, inner_dim, -1, width)

        blur_sigma = scaled_blur_sigma * math.sqrt(height_orig * width_orig * 64) / 10

        if blur_sigma >= 0:
            kernel_size = gaussian_kernel_size_for_img(
                blur_sigma, q, cap_at_half_smallest_dim=False
            )
            q = gaussian_blur_2d(q, kernel_size, blur_sigma)
        else:
            # Negative blur_sigma => set entire q to the mean
            q[:] = q.mean(dim=(-2, -1), keepdim=True)

        # Reshape back to (B, area, dim) for the attention function
        q = q.reshape(bs, inner_dim, -1).permute(0, 2, 1)

        return optimized_attention(q, k, v, heads=heads)

    return seg_perturbed_attention


def downsampled_attention_wrapper(
    attention: callable,
    downsample_factor: float,
) -> callable:

    def downsampled_attention(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        extra_options,
        mask=None,
    ) -> Tensor:
        heads = extra_options["n_heads"]
        bs, area, inner_dim = q.shape

        height_orig, width_orig = extra_options["original_shape"][2:4]
        aspect_ratio = width_orig / height_orig

        # Reshape (B, area, dim) -> (B, dim, H, W)
        if aspect_ratio >= 1.0:
            height = round((area / aspect_ratio) ** 0.5)
            # q = q.permute(0, 2, 1).reshape(bs, inner_dim, height, -1)
            k = k.permute(0, 2, 1).reshape(bs, inner_dim, height, -1)
            # v = v.permute(0, 2, 1).reshape(bs, inner_dim, height, -1)
        else:
            width = round((area * aspect_ratio) ** 0.5)
            # q = q.permute(0, 2, 1).reshape(bs, inner_dim, -1, width)
            k = k.permute(0, 2, 1).reshape(bs, inner_dim, -1, width)
            # v = v.permute(0, 2, 1).reshape(bs, inner_dim, -1, width)

        # q = kornia.geometry.resize(
        #     q, (new_height, new_width), interpolation="area", antialias=True
        # )
        k = kornia.geometry.rescale(
            k, 1 / downsample_factor, interpolation="area", antialias=True
        )
        k = kornia.geometry.rescale(
            k, float(downsample_factor), interpolation="nearest", antialias=True
        )
        # v = kornia.geometry.resize(
        #     v, (new_height, new_width), interpolation="area", antialias=True
        # )

        # Reshape back to (B, area, dim) for the attention function
        # q = q.reshape(bs, inner_dim, -1).permute(0, 2, 1)
        k = k.reshape(bs, inner_dim, -1).permute(0, 2, 1)
        # v = v.reshape(bs, inner_dim, -1).permute(0, 2, 1)

        res = attention(q, k, v, heads=heads)

        # Re upsample the attention map
        # res = res.permute(0, 2, 1).reshape(bs, inner_dim, new_height, new_width)
        # res = kornia.geometry.resize(
        #     res, (height, width), interpolation="area", antialias=True
        # )
        # res = res.reshape(bs, inner_dim, -1).permute(0, 2, 1)
        return res

    return downsampled_attention


def sliding_window_guidance_wrapper(
    model,
    x: Tensor,
    cond: list[dict],
    sigma: float,
    model_options: dict,
    tile_width: int,
    tile_height: int,
    tile_overlap: int,
) -> Tensor:
    b, c, h, w = x.shape
    swg_pred = torch.zeros_like(x)
    overlap = torch.zeros_like(x)

    tiles_w = math.ceil(w / (tile_width - tile_overlap))
    tiles_h = math.ceil(h / (tile_height - tile_overlap))

    for w_i in range(tiles_w):
        for h_i in range(tiles_h):
            left, right = tile_width * w_i, tile_width * (w_i + 1) + tile_overlap
            top, bottom = tile_height * h_i, tile_height * (h_i + 1) + tile_overlap

            x_window = x[:, :, top:bottom, left:right]
            if x_window.shape[-1] == 0 or x_window.shape[-2] == 0:
                continue

            swg_pred_window = calc_cond_batch(
                model, [cond], x_window, sigma, model_options
            )[0]
            swg_pred[:, :, top:bottom, left:right] += swg_pred_window

            overlap_window = torch.ones_like(swg_pred_window)
            overlap[:, :, top:bottom, left:right] += overlap_window

    swg_pred = swg_pred / overlap
    return swg_pred
