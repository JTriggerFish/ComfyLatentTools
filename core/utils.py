from itertools import groupby

import torch.nn.functional as F
import torch
from .latent_filters import gaussian_blur_2d, moment_match


def pred_to_v(
    x: torch.Tensor,
    sigma: torch.Tensor,
    preds: list[torch.Tensor | None],
) -> list[torch.Tensor | None]:
    """
    Note that in comfyui, the "pred" output are actually alpha_t * E[x_{t-1}| x_t]
    therefore v_t = (pred - x) / sigma**2
    :param x:
    :param alpha:
    :param sigma:
    :param preds:
    :return:
    """
    sigma2_inv = 1.0 / sigma**2
    ret = []
    for pred in preds:
        if pred is not None:
            ret.append((pred - x) * sigma2_inv)
        else:
            ret.append(None)
    return ret


def v_to_pred(
    x: torch.Tensor,
    sigma: torch.Tensor,
    v_preds: list[torch.Tensor | None],
) -> list[torch.Tensor | None]:
    ret = []
    sigma2 = sigma**2
    for v_pred in v_preds:
        if v_pred is not None:
            ret.append(sigma2 * v_pred + x)
        else:
            ret.append(None)
    return ret


def partial_rescaling(
    ref_tensor: torch.Tensor,
    z: torch.Tensor,
    rescaled_fraction: float,
    match_mean: bool = False,
    per_channel: bool = True,
) -> torch.Tensor:
    rescaled = moment_match(
        ref_tensor, z, per_channel=per_channel, match_mean=match_mean
    )
    return rescaled_fraction * rescaled + (1 - rescaled_fraction) * z


def saliency_tensor_combination(
    x_a: torch.Tensor,
    x_b: torch.Tensor,
) -> torch.Tensor:
    """
    From  Saliency-adaptive Noise Fusion based on High-fidelity Person-centric Subject-to-Image Synthesis (Wang et al.)
    """
    assert x_a.shape == x_b.shape
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


def softmax_weighted_combination(
    x1: torch.Tensor,
    x2: torch.Tensor,
    temperature: float = 1.0,
):
    assert (
        x1.shape == x2.shape
    ), f"x1 and x2 must have the same shape; got {x1.shape} vs {x2.shape}"
    B, C, H, W = x1.shape
    # (B) Single-scalar weighting per batch element:
    # 1. Flatten each (b) => shape (B, C*H*W)
    x1_flat = x1.view(B, -1)
    x2_flat = x2.view(B, -1)
    # 2. L2 norms => shape (B,)
    n1 = x1_flat.pow(2).sum(dim=1).sqrt()
    n2 = x2_flat.pow(2).sum(dim=1).sqrt()
    # 3. For each b, do a 2-element softmax across n1[b], n2[b]
    #    => shape (B,2)
    norms = torch.stack([n1, n2], dim=1) / temperature
    weights = F.softmax(norms, dim=1)  # shape (B,2)
    alpha = weights[:, 0]  # shape (B,)
    # 4. Expand alpha => shape (B,1,1,1)
    alpha = alpha.view(B, 1, 1, 1)
    # 5. Weighted blend
    out = alpha * x1 + (1 - alpha) * x2

    return out


def parse_unet_blocks(model, unet_block_list: str):
    """
    Copied from https://github.com/pamparamm/sd-perturbed-attention/blob/master/pag_utils.py#L9
    :param model:
    :param unet_block_list:
    :return:
    """
    output: list[tuple[str, int, int | None]] = []

    # Get all Self-attention blocks
    input_blocks, middle_blocks, output_blocks = [], [], []
    for name, module in model.model.diffusion_model.named_modules():
        if module.__class__.__name__ == "CrossAttention" and name.endswith("attn1"):
            parts = name.split(".")
            block_name = parts[0]
            block_id = int(parts[1])
            if block_name.startswith("input"):
                input_blocks.append(block_id)
            elif block_name.startswith("middle"):
                middle_blocks.append(block_id - 1)
            elif block_name.startswith("output"):
                output_blocks.append(block_id)

    def group_blocks(blocks: list[int]):
        return [(i, len(list(gr))) for i, gr in groupby(blocks)]

    input_blocks, middle_blocks, output_blocks = (
        group_blocks(input_blocks),
        group_blocks(middle_blocks),
        group_blocks(output_blocks),
    )

    unet_blocks = [b.strip() for b in unet_block_list.split(",")]
    for block in unet_blocks:
        name, indices = block[0], block[1:].split(".")
        match name:
            case "d":
                layer, cur_blocks = "input", input_blocks
            case "m":
                layer, cur_blocks = "middle", middle_blocks
            case "u":
                layer, cur_blocks = "output", output_blocks
        if len(indices) >= 2:
            number, index = cur_blocks[int(indices[0])][0], int(indices[1])
            assert 0 <= index < cur_blocks[int(indices[0])][1]
        else:
            number, index = cur_blocks[int(indices[0])][0], None
        output.append((layer, number, index))

    return output
