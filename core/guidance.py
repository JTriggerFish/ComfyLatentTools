import torch
import enum
import dataclasses
from .latent_filters import gaussian_blur_2d


class GuidanceCombinationMethod(str, enum.Enum):
    PLAIN = "Plain"
    PRED_SPACE_RESCALE = "PredSpaceRescale"
    V_SPACE_RESCALE = "VSpaceRescale"
    SNF = "SNF"


class AlternateConditionalComparator(str, enum.Enum):
    BASE_COND = "BaseCond"
    ALTERNATE_UNCOND = "AlternateUncond"


def plain_combination(
    cond_x,
    uncond_x,
    cfg: float,
    alternate_cond_x,
    alternate_uncond_x,
    alternate_weight: float,
    alternate_cond_comparator: AlternateConditionalComparator,
) -> torch.Tensor:
    if (
        alternate_cond_comparator == AlternateConditionalComparator.ALTERNATE_UNCOND
        and alternate_uncond_x is None
    ):
        raise ValueError(
            "Alternate uncond x is required for alternate uncond comparator"
        )
    if alternate_cond_comparator == AlternateConditionalComparator.ALTERNATE_UNCOND:
        alternate_guidance = alternate_uncond_x - alternate_cond_x
    else:
        alternate_guidance = alternate_cond_x - cond_x
    return uncond_x + cfg * (cond_x - uncond_x) - alternate_weight * alternate_guidance


def v_space_normalised_combination(
    x,
    sigma_,
    cond_x,
    uncond_x,
    cfg: float,
    alternate_cond_x,
    alternate_uncond_x,
    alternate_weight: float,
    rescaling_weight: float,
    apply_rescaling_to_alternate: bool,
    alternate_cond_comparator: AlternateConditionalComparator,
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

    s0 = torch.std(cond_v, dim=(1, 2, 3), keepdim=True)
    if apply_rescaling_to_alternate:
        alternate_cond_v = (alpha * x - alternate_cond_x) * sigma_inv
        if alternate_cond_comparator == AlternateConditionalComparator.ALTERNATE_UNCOND:
            alternate_uncond_v = (alpha * x - alternate_uncond_x) * sigma_inv
            alternate_guidance = alternate_cond_v - alternate_uncond_v
        else:
            alternate_guidance = alternate_cond_v - cond_v
        v_out_unnorm = (
            uncond_v + cfg * (cond_v - uncond_v) - alternate_weight * alternate_guidance
        )
        s1 = torch.std(v_out_unnorm, dim=(1, 2, 3), keepdim=True)
        v_out_normed = v_out_unnorm * s0 / s1
        v_final = v_out_normed * rescaling_weight + v_out_unnorm * (
            1 - rescaling_weight
        )
        # Convert back to prediction space and rescale
        pred_final = x * alpha - v_final * sigma
    else:
        # Apply standard CFG rescaling and combine with alternate
        v_cfg_unnorm = uncond_v + cfg * (cond_v - uncond_v)
        s1 = torch.std(v_cfg_unnorm, dim=(1, 2, 3), keepdim=True)
        v_cfg_normed = v_cfg_unnorm * s0 / s1
        v_cfg = v_cfg_normed * rescaling_weight + v_cfg_unnorm * (1 - rescaling_weight)
        pred_cfg = x * alpha - v_cfg * sigma
        if alternate_cond_comparator == AlternateConditionalComparator.ALTERNATE_UNCOND:
            alternate_guidance = alternate_uncond_x - alternate_cond_x
        else:
            alternate_guidance = alternate_cond_x - cond_x
        pred_final = pred_cfg - alternate_weight * alternate_guidance

    return pred_final


def pred_space_normalised_combination(
    cond_x,
    uncond_x,
    cfg: float,
    alternate_cond_x,
    alternate_uncond_x,
    alternate_weight: float,
    rescaling_weight: float,
    apply_rescaling_to_alternate: bool,
    alternate_cond_comparator: AlternateConditionalComparator,
) -> torch.Tensor:

    s0 = torch.std(cond_x, dim=(1, 2, 3), keepdim=True)
    if apply_rescaling_to_alternate:
        if alternate_cond_comparator == AlternateConditionalComparator.ALTERNATE_UNCOND:
            alternate_guidance = alternate_cond_x - alternate_uncond_x
        else:
            alternate_guidance = alternate_cond_x - cond_x
        x_out_unnorm = (
            uncond_x + cfg * (cond_x - uncond_x) - alternate_weight * alternate_guidance
        )
        s1 = torch.std(x_out_unnorm, dim=(1, 2, 3), keepdim=True)
        x_out_normed = x_out_unnorm * s0 / s1
        x_final = x_out_normed * rescaling_weight + x_out_unnorm * (
            1 - rescaling_weight
        )
    else:
        # Apply standard CFG rescaling and combine with alternate
        x_cfg_unnorm = uncond_x + cfg * (cond_x - uncond_x)
        s1 = torch.std(x_cfg_unnorm, dim=(1, 2, 3), keepdim=True)
        x_cfg_normed = x_cfg_unnorm * s0 / s1
        x_cfg = x_cfg_normed * rescaling_weight + x_cfg_unnorm * (1 - rescaling_weight)
        if alternate_cond_comparator == AlternateConditionalComparator.ALTERNATE_UNCOND:
            alternate_guidance = alternate_uncond_x - alternate_cond_x
        else:
            alternate_guidance = alternate_cond_x - cond_x
        x_final = x_cfg - alternate_weight * alternate_guidance

    return x_final


def snf_combination(
    cond_a: torch.Tensor,
    uncond_a: torch.Tensor,
    cond_b: torch.Tensor,
    uncond_b: torch.Tensor,
    weight_a: float,
    weight_b: float,
    alternate_cond_comparator: AlternateConditionalComparator,
):
    """
    From  Saliency-adaptive Noise Fusion based on High-fidelity Person-centric Subject-to-Image Synthesis (Wang et al.)
    :param cond_a:
    :param uncond_a:
    :param cond_b:
    :param uncond_b:
    :param weight_a:
    :param weight_b:
    :param alternate_cond_comparator:
    :return:
    """
    x_a = weight_a * (cond_a - uncond_a)
    if alternate_cond_comparator == AlternateConditionalComparator.ALTERNATE_UNCOND:
        x_b = weight_b * (cond_b - uncond_b)
    else:
        x_b = weight_b * (cond_b - cond_a)

    b, c, h, w = x_a.shape

    a_bar = gaussian_blur_2d(torch.abs(x_a), 3, 1.0)
    b_bar = gaussian_blur_2d(torch.abs(x_b), 3, 1.0)
    a_softmax = torch.softmax(a_bar.reshape(b * c, h * w), dim=1).reshape(b, c, h, w)
    b_softmax = torch.softmax(b_bar.reshape(b * c, h * w), dim=1).reshape(b, c, h, w)
    guidance_stacked = torch.stack([x_a, x_b], dim=0)

    ab_softmax = torch.stack([a_softmax, b_softmax], dim=0)
    argeps = torch.argmax(ab_softmax, dim=0, keepdim=True)

    # TODO : should do softmax instead of argmax
    snf = torch.gather(guidance_stacked, dim=0, index=argeps).squeeze(0)
    return snf
