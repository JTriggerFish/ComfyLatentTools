import numpy as np
from typing import Callable
from PIL import Image
import torchvision.transforms as transforms

import torch
import torch.nn.functional as F


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
        ret = from_tensor.clone()
        ret += from_tensor.mean(dim=(2, 3), keepdim=True) - ret.mean(
            dim=(2, 3), keepdim=True
        )
        ret *= from_tensor.std(dim=(2, 3), keepdim=True) / ret.std(
            dim=(2, 3), keepdim=True
        )
        return ret


def add_correlated_gaussian_noise(
    latent: torch.Tensor,
    sigma: float = 1.0,
    amplitude: float = 1.0,
    max_kernel_size: int = 13,
) -> torch.Tensor:
    """
    Add correlated Gaussian noise to a latent tensor by:
      1) Generating white noise (normal(0,1)).
      2) Convolving that noise with a 2D Gaussian kernel.
      3) Scaling and adding it to the original latent.

    Args:
        latent (torch.Tensor): Input tensor of shape (B, C, H, W).
        sigma (float): Standard deviation for the Gaussian kernel filter (controls correlation).
        amplitude (float): Scale factor for the filtered noise before adding.
        max_kernel_size:  Maximum size of the Gaussian kernel (odd number).

    Returns:
        torch.Tensor: Tensor of the same shape, with correlated noise added.
    """
    assert max_kernel_size % 2 == 1, "max_kernel_size must be odd"

    # 1) Generate white noise
    noise = torch.randn_like(latent)  # same shape, ~N(0,1)

    # base size based on sigma
    kernel_size = min(2 * np.ceil(3 * sigma) + 1, max_kernel_size)

    # 2) Build a 2D Gaussian kernel
    #    We'll define an inline helper to create a normalized 2D kernel.
    def gaussian_kernel_2d(k_size, sigma_val):
        # Create coordinate grid centered at 0
        coords = torch.arange(k_size) - (k_size - 1) / 2
        xx, yy = torch.meshgrid(coords, coords, indexing="xy")
        kernel_2d = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma_val**2))
        kernel_2d = kernel_2d / kernel_2d.sum()  # normalize to sum=1
        return kernel_2d

    kernel_2d = gaussian_kernel_2d(kernel_size, sigma)  # shape: (k_size, k_size)
    kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)  # shape: (1, 1, k_size, k_size)

    # 3) Convolve noise with Gaussian kernel for each channel
    #    - We'll "repeat" the kernel for each channel and use grouped convolution
    B, C, H, W = latent.shape
    kernel_2d = kernel_2d.repeat(C, 1, 1, 1)  # shape: (C, 1, k_size, k_size)

    #    - Pad so output is the same spatial size
    pad = (kernel_size - 1) // 2
    correlated_noise = F.conv2d(noise, kernel_2d, padding=pad, groups=C)

    # 4) Scale and add to the original latent
    correlated_noise = amplitude * correlated_noise
    return latent + correlated_noise
