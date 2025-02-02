import kornia.filters
import numpy as np
from typing import Callable
from PIL import Image
import torchvision.transforms as transforms

import torch
from kornia.filters import GaussianBlur2d
from torch import Tensor
from torch.nn import functional as F
import math


def center_tensor(
    input_tensor: torch.Tensor,
    per_channel_shift: float = 1,
    full_tensor_shift: float = 1,
    channels: list[int] | None = None,
) -> torch.Tensor:
    """
    Recenter a tensor by subtracting the mean of each channel and the full tensor, with optional shift scaling.
    Expect a tensor of shape (B, C, H, W) and return the same shape.
    :param input_tensor:
    :param per_channel_shift:
    :param full_tensor_shift:
    :param channels:
    :return:
    """
    if channels is None:
        channels = torch.range(input_tensor.shape[1])
    for channel in channels:
        input_tensor[0, channel] -= input_tensor[0, channel].mean() * per_channel_shift
    return input_tensor - input_tensor.mean() * full_tensor_shift


def huberize_quantile(
    tensor: torch.Tensor, lowQ: float = 0.01, highQ: float = 0.99, slope: float = 0.1
) -> torch.Tensor:
    """
    Linearly soft clamps extreme values in a tensor based on quantiles,
    using a piecewise linear function:

      1) Identity (slope=1) for x in [lower_val, upper_val].
      2) Reduced slope for x < lower_val or x > upper_val.

    The slope in the tails is controlled by 'slope' (0 < slope < 1),
    ensuring we don't cut off values entirely but do compress them.

    Args:
        tensor (torch.Tensor): The input data.
        lowQ (float): Lower quantile, e.g. 0.01 (1%).
        highQ (float): Upper quantile, e.g. 0.99 (99%).
        slope (float): Slope to use beyond the low/high quantiles.
                       0 would be a hard clamp; 1 means no clamp.

    Returns:
        torch.Tensor: The transformed tensor, same shape as input.
    """
    # 1) Determine the lower and upper quantiles
    lower_val = torch.quantile(tensor, lowQ)
    upper_val = torch.quantile(tensor, highQ)

    # 2) Define a central "identity" region around the midpoint
    center = 0.5 * (lower_val + upper_val)
    half_range = 0.5 * (upper_val - lower_val)

    # 3) Shift the tensor so that 'center' is at 0
    x_shifted = tensor - center

    # 4) Piecewise function:
    #    - If x < -half_range:   y = -half_range + slope*(x + half_range)
    #    - If -half_range <= x <= half_range:  y = x
    #    - If x >  half_range:   y =  half_range + slope*(x - half_range)

    y = x_shifted.clone()  # same shape as x_shifted

    mask_lower = x_shifted < -half_range
    mask_upper = x_shifted > half_range

    # Values below the lower bound
    y[mask_lower] = -half_range + slope * (x_shifted[mask_lower] + half_range)

    # Values above the upper bound
    y[mask_upper] = half_range + slope * (x_shifted[mask_upper] - half_range)

    # 5) Shift back
    y += center

    return y


def moment_match(
    from_tensor: torch.Tensor, to_tensor: torch.Tensor, per_channel: bool = True
) -> torch.Tensor:
    """
    Match the mean and variance of a tensor to another tensor.
    :param from_tensor: The tensor to match.
    :param to_tensor: The tensor to match to.
    :return: The transformed tensor.
    """
    if not per_channel:
        from_mean = from_tensor.mean()
        from_std = from_tensor.std()
        to_mean = to_tensor.mean()
        to_std = to_tensor.std()
        return (to_tensor + to_mean - from_mean) * to_std / from_std
    else:
        ret = to_tensor.clone()
        ret += from_tensor.mean(dim=(2, 3), keepdim=True) - ret.mean(
            dim=(2, 3), keepdim=True
        )
        ret *= from_tensor.std(dim=(2, 3), keepdim=True) / ret.std(
            dim=(2, 3), keepdim=True
        )
        return ret


def add_correlated_gaussian_noise(
    latent: torch.Tensor,
    kernel_sigma: float = 1.0,
    amplitude: float = 1.0,
    noise_seed: int = 0,
) -> torch.Tensor:
    """
    Add correlated Gaussian noise to a latent tensor by:
      1) Generating white noise (normal(0,1)).
      2) Convolving that noise with a 2D Gaussian kernel.
      3) Scaling and adding it to the original latent.

    Args:
        latent (torch.Tensor): Input tensor of shape (B, C, H, W).
        kernel_sigma (float): Standard deviation for the Gaussian kernel filter (controls correlation).
        amplitude (float): Scale factor for the filtered noise before adding.

    Returns:
        torch.Tensor: Tensor of the same shape, with correlated noise added.
    """

    # 1) Generate white noise
    generator = torch.manual_seed(noise_seed)
    noise = torch.randn(
        latent.size(),
        dtype=latent.dtype,
        layout=latent.layout,
        generator=generator,
        device="cpu",
    )
    kernel_size = gaussian_kernel_size_for_img(kernel_sigma, latent)
    q = kornia.filters.GaussianBlur2d(
        (kernel_size, kernel_size), (kernel_sigma, kernel_sigma)
    )
    correlated_noise = q(noise)

    # 4) Scale and add to the original latent
    correlated_noise = amplitude * correlated_noise
    return latent + correlated_noise


def latent_upscale(
    latent: torch.Tensor,
    new_height: int,
    new_width: int,
    mode: transforms.InterpolationMode = transforms.InterpolationMode.NEAREST_EXACT,
) -> torch.Tensor:
    upscale = transforms.Resize(
        (new_height, new_width), transforms.InterpolationMode.NEAREST_EXACT
    )

    # Upscale each channel separately
    res = upscale(latent)
    return res


def downsample_latent(latent: torch.Tensor, factor: float) -> torch.Tensor:
    """
    Downscale an SDXL latent by 'factor' with a Gaussian pre-filter and area interpolation.
    Ensures final shape is divisible by 8.
    """
    B, C, H, W = latent.shape

    # 1) Pre-blur to avoid aliasing
    sigma = 0.5 * factor
    kernel_size = int(math.ceil(6 * sigma))
    if kernel_size % 2 == 0:
        kernel_size += 1
    # Build Kornia's GaussianBlur2d
    blur = kornia.filters.GaussianBlur2d(
        kernel_size=(kernel_size, kernel_size),
        sigma=(sigma, sigma),
        border_type="reflect",
        separable=True,
    )
    latent_blurred = blur(latent)  # on GPU if latent is on GPU

    # 2) Downscale
    new_h = int(H // factor)
    new_w = int(W // factor)

    # ensure multiple of 8
    new_h = new_h // 8 * 8
    new_w = new_w // 8 * 8

    # do the actual resizing
    latent_down = F.interpolate(latent_blurred, size=(new_h, new_w), mode="area")

    return latent_down


def gaussian_kernel_size_for_img(
    sigma: float,
    img: torch.Tensor,
    kernel_size_cap: int | None = None,
    cap_at_half_smallest_dim: bool = False,
) -> int:
    kernel_size = math.ceil(6 * sigma) + 1 - math.ceil(6 * sigma) % 2
    height, width = img.shape[-2:]
    smallest_dim = min(height, width)
    if cap_at_half_smallest_dim:
        max_k = smallest_dim + 1 - smallest_dim % 2
    else:
        max_k = 2 * smallest_dim - 1
    kernel_size_cap = (
        min(kernel_size_cap, max_k) if kernel_size_cap is not None else max_k
    )
    return min(kernel_size, kernel_size_cap)


def gaussian_blur_2d(img: Tensor, kernel_size: int, sigma: float) -> Tensor:
    blur = GaussianBlur2d(
        kernel_size=(kernel_size, kernel_size),
        sigma=(sigma, sigma),
    )
    im = blur(img)
    return im


def _gaussian_blur_2d_manual(img: Tensor, kernel_size: int, sigma: float) -> Tensor:
    ksize_half = (kernel_size - 1) * 0.5

    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)

    pdf = torch.exp(-0.5 * (x / sigma).pow(2))

    x_kernel = pdf / pdf.sum()
    x_kernel = x_kernel.to(device=img.device, dtype=img.dtype)

    kernel2d = torch.mm(x_kernel[:, None], x_kernel[None, :])
    kernel2d = kernel2d.expand(img.shape[-3], 1, kernel2d.shape[0], kernel2d.shape[1])

    padding = [kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2]

    img = F.pad(img, padding, mode="reflect")
    img = F.conv2d(img, kernel2d, groups=img.shape[-3])

    return img
