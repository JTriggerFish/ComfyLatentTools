from itertools import groupby

import torch
from .latent_filters import gaussian_blur_2d


def pred_to_v(
    x: torch.Tensor,
    alpha: torch.Tensor,
    sigma: torch.Tensor,
    preds: list[torch.Tensor | None],
) -> list[torch.Tensor | None]:
    sigma_inv = 1.0 / sigma
    alpha_x = alpha * x
    ret = []
    for pred in preds:
        if pred is not None:
            ret.append((alpha_x - pred) * sigma_inv)
        else:
            ret.append(None)
    return ret


def v_to_pred(
    x: torch.Tensor,
    alpha: torch.Tensor,
    sigma: torch.Tensor,
    v_preds: list[torch.Tensor | None],
) -> list[torch.Tensor | None]:
    alpha_x = alpha * x
    ret = []
    for v_pred in v_preds:
        if v_pred is not None:
            ret.append(alpha_x - v_pred * sigma)
        else:
            ret.append(None)
    return ret


def partial_rescaling(
    ref_tensor: torch.Tensor,
    z: torch.Tensor,
    rescaled_fraction: float,
):
    target_std = torch.std(ref_tensor, dim=(-3, -2, -1), keepdim=True)
    std = torch.std(z, dim=(-3, -2, -1), keepdim=True)
    return z * (1 - rescaled_fraction) + target_std * z / std * rescaled_fraction


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
    for name, module in model.diffusion_model.named_modules():
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
