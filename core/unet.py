import math
from typing import Callable
from torch import Tensor
from itertools import groupby

from .latent_filters import gaussian_blur_2d


def pag_perturbed_attention(
    q: Tensor, k: Tensor, v: Tensor, extra_options, mask=None
) -> Tensor:
    """Perturbed self-attention corresponding to an identity matrix replacing the attention matrix."""
    return v


def seg_attention_wrapper(attention: Callable, blur_sigma: float = 10.0):
    def seg_perturbed_attention(
        q: Tensor, k: Tensor, v: Tensor, extra_options, mask=None
    ) -> Tensor:
        """Smoothed Energy Guidance self-attention"""
        heads = extra_options["n_heads"]
        bs, area, inner_dim = q.shape

        height_orig, width_orig = extra_options["original_shape"][2:4]
        aspect_ratio = width_orig / height_orig

        if aspect_ratio >= 1.0:
            height = round((area / aspect_ratio) ** 0.5)
            q = q.permute(0, 2, 1).reshape(bs, inner_dim, height, -1)
        else:
            width = round((area * aspect_ratio) ** 0.5)
            q = q.permute(0, 2, 1).reshape(bs, inner_dim, -1, width)

        if blur_sigma >= 0:
            kernel_size = math.ceil(6 * blur_sigma) + 1 - math.ceil(6 * blur_sigma) % 2
            q = gaussian_blur_2d(q, kernel_size, blur_sigma)
        else:
            q[:] = q.mean(dim=(-2, -1), keepdim=True)

        q = q.reshape(bs, inner_dim, -1).permute(0, 2, 1)

        return attention(q, k, v, heads=heads)

    return seg_perturbed_attention


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
