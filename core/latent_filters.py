import math
import kornia.filters
import torchvision.transforms as transforms

import torch
from kornia.filters import GaussianBlur2d
from torch import Tensor
from torch.nn import functional as F
import matplotlib as plt


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


def normalize_tensor(
    tensor: torch.Tensor,
    per_channel: bool = True,
    normalize_mean: bool = False,
    eps: float = 1e-8,
    target_mean: float | Tensor = 0.0,
    target_std: float | Tensor = 1.0,
) -> torch.Tensor:
    """
    Normalize a tensor to have zero mean and unit variance.
    :param tensor:
    :param per_channel:
    :param eps:
    :return:
    """
    if not per_channel:
        mean = tensor.mean()
        std = tensor.std() + eps
    else:
        mean = tensor.mean(dim=(-2, -1), keepdim=True)
        std = tensor.std(dim=(-2, -1), keepdim=True) + eps
    if normalize_mean:
        return (tensor - mean) * target_std / std + target_mean
    else:
        return (tensor - mean) * target_std / std + mean


def moment_match(
    from_tensor: torch.Tensor,
    to_tensor: torch.Tensor,
    per_channel: bool = True,
    match_mean: bool = True,
) -> torch.Tensor:
    """
    Match the mean and variance of a tensor to another tensor.
    :param from_tensor: The tensor to match.
    :param to_tensor:
    :param per_channel:
    :param match_mean:
    """
    if not per_channel:
        target_mean = from_tensor.mean()
        target_std = from_tensor.std()
    else:
        target_mean = from_tensor.mean(dim=(-2, -1), keepdim=True)
        target_std = from_tensor.std(dim=(-2, -1), keepdim=True)
    return normalize_tensor(
        to_tensor,
        per_channel,
        target_mean=target_mean,
        target_std=target_std,
        normalize_mean=match_mean,
    )


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


def mix_fft_phase_amplitude(
    x: torch.Tensor, y: torch.Tensor, y_weight: float
) -> torch.Tensor:
    """
    x, y: PyTorch tensors of shape (..., C, H, W) or (C, H, W)
          representing images in the spatial domain.

    Returns:
        A torch.Tensor of the same shape as x and y, with the amplitude from x
        and the "averaged" phase from x and y (correctly handled on the unit circle).
    """
    # Compute 2D FFT along the last two dimensions (height, width)
    assert y_weight >= 0 and y_weight <= 1
    X = torch.fft.fft2(x, dim=(-2, -1))
    Y = torch.fft.fft2(y, dim=(-2, -1))

    # Extract amplitude (magnitude) from X
    amplitude_x = torch.abs(X)

    # Extract phase (angle) from both X and Y
    phase_x = torch.angle(X)
    phase_y = torch.angle(Y)

    # Convert phases to complex exponentials on the unit circle
    # e^{i * phase} = cos(phase) + i sin(phase)
    phase_x_complex = torch.complex(torch.cos(phase_x), torch.sin(phase_x))
    phase_y_complex = torch.complex(torch.cos(phase_y), torch.sin(phase_y))

    # Sum the unit vectors to get the average direction in the complex plane
    phase_sum = (1.0 - y_weight) * phase_x_complex + y_weight * phase_y_complex

    # The correct averaged phase is the argument of the sum
    # (Dividing by 2 won't change the argument, so it's unnecessary here.)
    phase_avg = torch.angle(phase_sum)

    real_part = amplitude_x * torch.cos(phase_avg)
    imag_part = amplitude_x * torch.sin(phase_avg)
    mixed_fft = torch.complex(real_part, imag_part)

    # Inverse FFT to go back to the spatial domain
    mixed_spatial = torch.fft.ifft2(mixed_fft, dim=(-2, -1))

    return mixed_spatial.real


def downsample_latent(
    latent: torch.Tensor, factor: float, filter_sigma: float
) -> torch.Tensor:
    """
    Downscale a diffusion latent by 'factor' with a Gaussian pre-filter and area interpolation.
    The filter_sigma controls the standard deviation of the Gaussian kernel used for pre-filtering.
    """

    kernel_size = gaussian_kernel_size_for_img(
        filter_sigma, latent, cap_at_half_smallest_dim=False
    )
    filtered_latent = gaussian_blur_2d(latent, kernel_size, filter_sigma)

    latent = kornia.geometry.transform.rescale(
        filtered_latent, 1 / factor, interpolation="area", antialias=False
    )
    return latent


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


def compute_cosine_distance(
    tensorA: Tensor, tensorB: Tensor, eps=1e-8, return_similarity: bool = False
) -> Tensor:
    """
    Compute element-wise cosine distance between two tensors of shape [B, C, H, W].
    Returns a heatmap of shape [B, H, W] where each pixel is:
        1 - cos_sim = 1 - (a·b / (||a||*||b||)).
    """
    # Ensure same shape
    if tensorA.shape != tensorB.shape:
        raise ValueError("tensorA and tensorB must have the same shape.")

    # [B, C, H, W] -> (a·b) along C dimension
    dot_product = (tensorA * tensorB).sum(dim=1)  # => [B, H, W]
    normA = tensorA.norm(dim=1)  # => [B, H, W]
    normB = tensorB.norm(dim=1)  # => [B, H, W]

    cos_sim = dot_product / (normA * normB + eps)
    if return_similarity:
        return cos_sim
    cos_dist = 1.0 - cos_sim
    return cos_dist


def tokens_to_spatial(x: Tensor, aspect_ratio: float) -> Tensor:
    """
    Beshape (B, area, dim) -> (B, dim, H, W)
    :param x:
    :param aspect_ratio:
    :return:
    """
    bs, area, inner_dim = x.shape
    height_orig, width_orig = int(math.sqrt(area / aspect_ratio)), int(
        math.sqrt(area * aspect_ratio)
    )
    if aspect_ratio >= 1.0:
        height = round((area / aspect_ratio) ** 0.5)
        x = x.permute(0, 2, 1).reshape(bs, inner_dim, height, -1)
    else:
        width = round((area * aspect_ratio) ** 0.5)
        x = x.permute(0, 2, 1).reshape(bs, inner_dim, -1, width)
    return x


def spatial_to_tokens(x: Tensor) -> Tensor:
    """
    Reshape (B, dim, H, W) -> (B, area, dim)
    :param x:
    :return:
    """
    bs, inner_dim, height, width = x.shape
    return x.reshape(bs, inner_dim, -1).permute(0, 2, 1)


def compare_kqv_resolutions(
    k_hi: Tensor,
    q_hi: Tensor,
    v_hi: Tensor,  # [B, C, L_hi] each (high resolution)
    k_lo: Tensor,
    q_lo: Tensor,
    v_lo: Tensor,  # [B, C, L_lo] each (low resolution)
    aspect_ratio: float,
    upsample_mode="bicubic",
    attention_hi=None,  # [B, n_heads, L_hi, L_hi] (optional)
    attention_lo=None,  # [B, n_heads, L_lo, L_lo] (optional)
):
    """
    1. Reshape hi-res and lo-res (K,Q,V) to [B, C, H, W] using the given aspect ratio.
    2. Upsample lo-res to hi-res shape.
    3. Compute and plot cosine-distance heatmaps for K, Q, V.
    4. Optionally compare attention maps by upsampling lo-res attention to hi-res
       and plotting difference (or ratio).

    Parameters:
        k_hi, q_hi, v_hi:  [B, C, L_hi]
        k_lo, q_lo, v_lo:  [B, C, L_lo]
        aspect_ratio:      float, used to factor L into H,W
        upsample_mode:     'nearest' or 'bicubic', etc.
        attention_hi:      optional [B, n_heads, L_hi, L_hi]
        attention_lo:      optional [B, n_heads, L_lo, L_lo]
    """
    # 1. Reshape to spatial
    k_hi_2d = tokens_to_spatial(k_hi, aspect_ratio)  # [B, C, H_hi, W_hi]
    q_hi_2d = tokens_to_spatial(q_hi, aspect_ratio)
    v_hi_2d = tokens_to_spatial(v_hi, aspect_ratio)

    k_lo_2d = tokens_to_spatial(k_lo, aspect_ratio)  # [B, C, H_lo, W_lo]
    q_lo_2d = tokens_to_spatial(q_lo, aspect_ratio)
    v_lo_2d = tokens_to_spatial(v_lo, aspect_ratio)

    B, C, H_hi, W_hi = k_hi_2d.shape
    _, _, H_lo, W_lo = k_lo_2d.shape

    # 2. Upsample lo-res -> hi-res
    scale_factor_h = H_hi / H_lo
    scale_factor_w = W_hi / W_lo

    k_lo_up = F.interpolate(
        k_lo_2d, size=(H_hi, W_hi), mode=upsample_mode
    )  # [B, C, H_hi, W_hi]
    q_lo_up = F.interpolate(q_lo_2d, size=(H_hi, W_hi), mode=upsample_mode)
    v_lo_up = F.interpolate(v_lo_2d, size=(H_hi, W_hi), mode=upsample_mode)

    # 3. Compute cosine distance heatmaps for K, Q, V
    k_dist = compute_cosine_distance(k_hi_2d, k_lo_up)  # [B, H_hi, W_hi]
    q_dist = compute_cosine_distance(q_hi_2d, q_lo_up)
    v_dist = compute_cosine_distance(v_hi_2d, v_lo_up)

    # Example: Plot these heatmaps for the first item in the batch.
    # Real usage might store them or handle multiple batch elements.
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, dist_map, title in zip(
        axes, [k_dist, q_dist, v_dist], ["K Dist", "Q Dist", "V Dist"]
    ):
        ax.imshow(
            dist_map[0].cpu().detach().numpy(), cmap="magma", interpolation="nearest"
        )
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.show()

    # 4. Optionally compare attention maps
    #    attention_hi: [B, n_heads, L_hi, L_hi]
    #    attention_lo: [B, n_heads, L_lo, L_lo]
    if attention_hi is not None and attention_lo is not None:
        # We'll upsample attention_lo to [B, n_heads, L_hi, L_hi] by 2D interpolation.
        # But we must treat each head [L_lo, L_lo] like an image [1, 1, H_lo, W_lo].
        # So let's reshape to [B*n_heads, 1, L_lo_h, L_lo_w] if we want to do the same
        # aspect-based approach. However, typically L_lo_h = L_lo_w = sqrt(L_lo),
        # or you need a separate approach if the attention map isn't truly a 2D "image".

        # For simplicity, assume L_lo = H_lo * W_lo from above.
        # We'll just do naive approach: reshape attention to [B*n_heads, 1, H_lo, W_lo]
        # and upsample to [B*n_heads, 1, H_hi, W_hi], then flatten back to [B, n_heads, L_hi, L_hi].

        # 4a) Reshape hi attention to [B*n_heads, 1, H_hi, W_hi]
        #     and lo attention to [B*n_heads, 1, H_lo, W_lo]
        # We have to factor L_hi and L_lo the same way we did for K,Q,V.
        # If the aspect ratio is the same, we can reuse tokens_to_spatial or do it manually.

        # For demonstration, let's do the quick version:

        B2, n_heads, L_hi2, _ = attention_hi.shape
        B3, n_heads2, L_lo2, _ = attention_lo.shape
        if not (B2 == B3 == B) or n_heads != n_heads2:
            raise ValueError(
                "Attention shapes and batch size do not match the given Q,K,V shapes."
            )

        # Factor L_hi into (H_hi, W_hi) the same way:
        # But note that L_hi2 might be different from L_hi above if your n_heads differ or if aspect ratio differs.
        # We'll assume we used the same approach for L_hi => H_hi, W_hi.
        # Let's do it carefully:
        H_hi_attn, W_hi_attn = find_hw_for_aspect_ratio(L_hi2, aspect_ratio)
        if H_hi_attn * W_hi_attn != L_hi2:
            raise ValueError(
                "Cannot reshape attention_hi tokens to the same H_hi,W_hi with this aspect ratio."
            )

        H_lo_attn, W_lo_attn = find_hw_for_aspect_ratio(L_lo2, aspect_ratio)
        if H_lo_attn * W_lo_attn != L_lo2:
            raise ValueError(
                "Cannot reshape attention_lo tokens to the same H_lo,W_lo with this aspect ratio."
            )

        attn_hi_2d = attention_hi.reshape(B * n_heads, L_hi2, L_hi2)
        attn_lo_2d = attention_lo.reshape(B * n_heads, L_lo2, L_lo2)

        # We need a 4D shape for interpolation: [B*n_heads, 1, H_lo_attn, W_lo_attn].
        # But note that the attention is 2D in each dimension (query vs. key).
        # A simpler approach is to just do a 2D interpolation for each row, but that’s more complex.
        # We'll do a naive approach: treat each row as 1D and upsample.
        # For a full 2D approach, we'd interpret (query index -> y-axis, key index -> x-axis)
        # and reshape to [B*n_heads, 1, H_lo_attn, W_lo_attn], then upsample to [B*n_heads, 1, H_hi_attn, W_hi_attn].

        # Quick demonstration: we can just do a normal 2D interpolation:
        attn_hi_4d = attn_hi_2d.reshape(B * n_heads, 1, H_hi_attn, W_hi_attn)
        attn_lo_4d = attn_lo_2d.reshape(B * n_heads, 1, H_lo_attn, W_lo_attn)

        attn_lo_up = F.interpolate(
            attn_lo_4d, size=(H_hi_attn, W_hi_attn), mode=upsample_mode
        )
        # Now compare attn_hi_4d vs. attn_lo_up
        # For example, compute difference:
        attn_diff = attn_hi_4d - attn_lo_up  # [B*n_heads, 1, H_hi_attn, W_hi_attn]

        # Visualize for one example (and maybe one head):
        fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
        idx = 0  # pick the first from B*n_heads
        axes2[0].imshow(attn_hi_4d[idx, 0].cpu().detach().numpy(), cmap="viridis")
        axes2[0].set_title("Attention Hi")
        axes2[1].imshow(attn_lo_up[idx, 0].cpu().detach().numpy(), cmap="viridis")
        axes2[1].set_title("Attention Lo Up")
        diff_map = attn_diff[idx, 0].cpu().detach().numpy()
        im = axes2[2].imshow(diff_map, cmap="bwr")
        axes2[2].set_title("Attn Difference (Hi - LoUp)")
        plt.colorbar(im, ax=axes2[2], fraction=0.046, pad=0.04)
        for ax in axes2:
            ax.axis("off")
        plt.tight_layout()
        plt.show()

    return (k_dist, q_dist, v_dist)
