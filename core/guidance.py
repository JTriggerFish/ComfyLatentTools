import numpy as np
import random

from .latent_filters import *
from comfy.ldm.pixart.blocks import MultiHeadCrossAttention
from comfy.model_patcher import ModelPatcher, set_model_options_patch_replace
from comfy.samplers import calc_cond_batch
import kornia.geometry
from torch import Tensor
import torch
import math
import enum
from comfy.ldm.modules.attention import optimized_attention
from torch.nn import functional as F

# Seems like for ComfyUI / ComfyScript, only relative imports work properly
try:
    from .latent_filters import (
        gaussian_blur_2d,
        gaussian_kernel_size_for_img,
        moment_match,
        tokens_to_spatial,
        spatial_to_tokens,
        normalize_tensor,
    )
    from . import utils
except ImportError:
    from latent_filters import gaussian_blur_2d, gaussian_kernel_size_for_img
    import utils


class GuidanceType(str, enum.Enum):
    VALUE_RESCALE = "ValueRescale"
    SCRAMBLE = "Scramble"
    RANDOM_ROTATION = "RandomRotation"
    FUZZY = "Fuzzy"
    AAT = "AAT"
    SEG = "SEG"
    PAG = "PAG"
    PERMUTE = "Permute"
    RANDOM_DROP = "RandomDrop"
    RANDOM_SUBSPACE = "RandomSubspace"
    SVD = "SVD"
    # DOWNSAMPLED = "Downsampled" # NOT IMPLEMENTED YET
    # DISTANCE_WEIGHTED = "DistanceWeighted" #NOT IMPLEMENTED YET


class GuidanceScalingMethod(str, enum.Enum):
    NONE = "None"
    PRED_SPACE_RESCALE = "PredSpaceRescale"
    V_SPACE_RESCALE = "VSpaceRescale"
    SNF = "SNF"
    SOFTMAX = "Softmax"
    NORMALIZE = "Normalize"


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
    rescaling_fraction: float,
) -> Tensor:
    base = cond_pred
    cfg = max(cfg - 1, 0) * (cond_pred - uncond_pred)

    if alternate_pred is None:
        raise ValueError("Alternate prediction must be provided for SNF guidance.")
    else:
        if alternate_pred_ref is None:
            alternate_pred_ref = 0
        guidance_adj = alternate_guidance_weight * (alternate_pred_ref - alternate_pred)

    adjusted = base + utils.saliency_tensor_combination(cfg, guidance_adj)
    simple = base + cfg + guidance_adj
    return (1 - rescaling_fraction) * simple + rescaling_fraction * adjusted


def softmax_guidance_combine(
    cond_pred: Tensor,
    uncond_pred: Tensor,
    cfg: float,
    alternate_pred: Tensor | None,
    alternate_pred_ref: Tensor | None,
    alternate_guidance_weight: float,
    temperature: float = 1.0,
) -> Tensor:

    base = cond_pred
    cfg = max(cfg - 1, 0) * (cond_pred - uncond_pred)

    if alternate_pred is None:
        raise ValueError("Alternate prediction must be provided for SNF guidance.")
    else:
        if alternate_pred_ref is None:
            alternate_pred_ref = 0
        guidance_adj = alternate_guidance_weight * (alternate_pred_ref - alternate_pred)

    return base + utils.softmax_weighted_combination(cfg, guidance_adj, temperature)


def guidance_normalize(
    cond_pred: Tensor,
    uncond_pred: Tensor,
    cfg: float,
    alternate_pred: Tensor | None,
    alternate_pred_ref: Tensor | None,
    alternate_guidance_weight: float,
    apply_rescaling_to_guidance: bool,
    rescaling_fraction: float = 1.0,
    target_std: float = 1.0,
    per_channel: bool = True,
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
        guidance = (
            1 - rescaling_fraction
        ) * guidance + rescaling_fraction * normalize_tensor(
            guidance, target_std=target_std, per_channel=per_channel
        )
        return guidance
    else:
        base_guidance = (
            1 - rescaling_fraction
        ) * base_guidance + rescaling_fraction * normalize_tensor(
            base_guidance, target_std=target_std, per_channel=per_channel
        )
        return base_guidance + guidance_adj


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
            ref_tensor=cond_pred,
            z=guidance,
            rescaled_fraction=rescaling_fraction,
            match_mean=False,
        )
        return guidance
    else:
        base_guidance = utils.partial_rescaling(
            ref_tensor=cond_pred, z=base_guidance, rescaled_fraction=rescaling_fraction
        )
        return base_guidance + guidance_adj


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
    per_channel: bool = True,
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
            ref_tensor=cond,
            z=guidance,
            rescaled_fraction=rescaling_fraction,
            match_mean=False,
            per_channel=per_channel,
        )
        final_pred = utils.v_to_pred(x, sigma, [guidance])[0]
    else:
        base_guidance = utils.partial_rescaling(
            ref_tensor=cond,
            z=base_guidance,
            rescaled_fraction=rescaling_fraction,
            match_mean=False,
            per_channel=per_channel,
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
                rescaling_fraction,
            )
        case GuidanceScalingMethod.SOFTMAX:
            return softmax_guidance_combine(
                cond_pred,
                uncond_pred,
                cfg,
                alternate_pred,
                alternate_pred_ref,
                alternate_guidance_weight,
                temperature=rescaling_fraction,
            )
        case GuidanceScalingMethod.NORMALIZE:
            return guidance_normalize(
                cond_pred,
                uncond_pred,
                cfg,
                alternate_pred,
                alternate_pred_ref,
                alternate_guidance_weight,
                apply_rescaling_to_guidance,
                rescaling_fraction,
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

        q = tokens_to_spatial(q, aspect_ratio)

        blur_sigma = scaled_blur_sigma * math.sqrt(height_orig * width_orig * 64) / 10

        if blur_sigma >= 0:
            kernel_size = gaussian_kernel_size_for_img(
                blur_sigma, q, cap_at_half_smallest_dim=False
            )
            q = gaussian_blur_2d(q, kernel_size, blur_sigma)
        else:
            # Negative blur_sigma => set entire q to the mean
            q[:] = q.mean(dim=(-2, -1), keepdim=True)

        q = spatial_to_tokens(q)

        return optimized_attention(q, k, v, heads=heads)

    return seg_perturbed_attention


def scramble_attention_wrapper(
    set_size: float = 1, value_scaling: float = 1.0
) -> callable:

    def scrambled_perturbed_attention(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        extra_options,
        mask=None,
    ) -> Tensor:
        heads = extra_options["n_heads"]
        """
        Replaces attention with random-subset averaging of V for each query token.

        For each of the M query tokens, we randomly select r = max(1, floor(fraction * T))
        rows of V (T total), then average those r vectors to produce the output. This 
        ignores Q entirely (no dot-products) and thus perturb" attention severely,
        but in a structured way.

        Args:
            Q: (B, M, D)  -> B=batch_size, M=#image tokens (e.g. "areas"), D=inner_dim
            V: (B, T, D)  -> T=#text tokens (or values) 
            fraction: in [0,1]
                - 0 means pick exactly 1 random value per query token,
                - 1 means average all T values (uniform attention),
                - anything in between picks r ~ fraction*T.

        Returns:
            out: (B, M, D)
                The "perturbed attention" output where each query token's
                result is a random average of value vectors.
        """
        b, m, d = q.shape
        _, t, _ = v.shape

        # Number of values to pick per query token
        r = min(t, max(1, int(set_size)))

        # 1) Flatten (B,T,D) => (B*T, D) for simpler gathering along dim=0
        v_flat = v.view(b * t, d)

        # 2) We need an index in [0 .. B*T) for each (b, m, r_idx).
        #    We'll first pick random positions in [0..T) for each (b, m, r_idx),
        #    then add an offset = b*T to shift into the [b*T .. (b+1)*T) block.
        #
        # Shape: (B, M, r)
        idx_local = torch.randint(low=0, high=t, size=(b, m, r), device=v.device)

        # 3) Build the batch offsets => for b in [0..B), offset = b*T
        #    We want shape (B,1,1) so it can broadcast to (B,M,r)
        batch_offsets = (torch.arange(b, device=v.device) * t).view(b, 1, 1)
        # expand to (B, M, r), then add
        idx_global = idx_local + batch_offsets

        # 4) Flatten idx_global => shape (B*M*r,) so we can gather in one shot
        idx_flat = idx_global.view(-1)

        # 5) Gather from V_flat => shape (B*M*r, D)
        subset_vals = v_flat[idx_flat, :]

        # 6) Reshape => (B, M, r, D)
        subset_vals = subset_vals.view(b, m, r, d)

        # 7) Average over r => (B, M, D)
        out = subset_vals.mean(dim=2)

        # If we assume that the keys are independent,
        # to keep the same std as choosing a single value, we need to scale by sqrt(r)
        mean = out.mean()
        out = (out - mean) * np.sqrt(r) * value_scaling + mean

        return out

    return scrambled_perturbed_attention


def permute_attention_wrapper(
    permute_mix: float = 1.0, value_scaling: float = 1.0
) -> callable:

    def permuted_attention(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        extra_options,
        mask=None,
    ) -> Tensor:
        heads = extra_options["n_heads"]
        """
        Randomly permute the values before computing attention.
        """
        _, t, _ = v.shape
        perm = torch.randperm(t, device=v.device)
        permuted_v = v[:, perm, :]
        new_v = permute_mix * permuted_v + (1 - permute_mix) * v
        new_v *= value_scaling
        return optimized_attention(q, k, new_v, heads=heads)

    return permuted_attention


def value_rescale_attention_wrapper(
    scaling_factor: float = 1.0,
    global_random_scaling_std: float = 0.0,
    local_random_scaling_std: float = 0.0,
) -> callable:

    def scaled_values_attention(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        extra_options,
        mask=None,
    ) -> Tensor:
        heads = extra_options["n_heads"]
        """
        Randomly permute the values before computing attention.
        """
        orig_dtype = v.dtype
        global_scaling = scaling_factor * torch.exp(
            torch.randn(1, device=v.device) * global_random_scaling_std
            - 0.5 * global_random_scaling_std**2
        ).clamp(0.01, 100.0)
        local_scaling = torch.exp(
            torch.randn(v.shape, device=v.device) * local_random_scaling_std
            - 0.5 * local_random_scaling_std**2
        ).clamp(0.01, 100.0)
        new_v = global_scaling * local_scaling * v
        new_v = new_v.to(orig_dtype)

        return optimized_attention(q, k, new_v, heads=heads)

    return scaled_values_attention


def rank_k_svd_approx(M, k=2, niter=2):
    """
    Compute a rank-k approximation of M via truncated SVD (torch.svd_lowrank).
    M: (T, D) float tensor
    k:  integer <= min(T, D)
    niter: number of power-iteration refinements for the lowrank SVD
    Returns:
      U:  (T, k)
      S:  (k,)
      V:  (D, k)
    so that M_k = (U * S) @ V^T is the rank-k approximation.
    """
    # Safeguard rank
    k = min(k, min(M.shape[0], M.shape[1]))

    orig_dtype = M.dtype
    M = M.to(torch.float32)
    # Use PyTorch's truncated SVD
    U, S, V = torch.svd_lowrank(M, q=k, niter=niter)
    # shapes:
    #  U: (T, k)
    #  S: (k,)
    #  V: (D, k)
    U = U.to(orig_dtype)
    S = S.to(orig_dtype)
    V = V.to(orig_dtype)

    return U, S, V


def rank_k_svd_approx_full(M: torch.Tensor, k: int):
    """
    Uses the full SVD (torch.linalg.svd) to get a rank-k approximation.
    M: (T, D) float tensor
    k: number of singular components to keep (<= min(T, D))

    Returns:
      U_k: (T, k)
      S_k: (k,)
      V_k: (D, k)
    so M_approx = (U_k * S_k) @ V_k.T is the rank-k reconstruction.
    """
    orig_dtype = M.dtype
    M = M.to(torch.float32)
    T, D = M.shape
    # Clamp k to min(T, D)
    r = min(T, D)
    k = min(k, r)

    # Perform full SVD
    # U: (T, r),  S: (r,),  Vh: (r, D)
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)

    # Slice out the top-k
    U_k = U[:, :k]  # (T, k)
    S_k = S[:k]  # (k,)
    # Vh[:k, :] => (k, D), so we transpose to get V_k => (D, k)
    V_k = Vh[:k, :].transpose(-2, -1)

    U_k = U_k.to(orig_dtype)
    S_k = S_k.to(orig_dtype)
    V_k = V_k.to(orig_dtype)

    return U_k, S_k, V_k


def project_onto_cols(X, U):
    """
    Project X (T, D) onto the column-space of U (T, k), which has orthonormal columns.
    The result is in the same shape (T, D).
    i.e.  Proj_{U}(X) = U (U^T X).
    We'll do this with matmul.
    """
    # U^T X => shape (k, D)
    # then U * that => (T, D)
    return U @ (U.transpose(-2, -1) @ X)


def add_orthogonal_noise_rankk(M, rank=2, niter_svd=2, noise_scaling=1.0):
    """
    1) Get rank-k approximation of M via truncated SVD
    2) Compute residual R = M - M_k
    3) Match R's mean/std -> generate random noise
    4) Remove top-k subspace from that noise (so it's orth to the principal components)
    5) Combine M_k + noise_orth.

    M: (T, D)
    rank: how many singular components to keep
    niter_svd: passes for truncated SVD
    Returns:
      M_k (rank-k approx),
      noise_orth (final noise in orth subspace),
      M_noisy = M_k + noise_orth
    """
    T, D = M.shape

    # ---- (1) SVD for rank-k
    U, S, V = rank_k_svd_approx(M, k=rank, niter=niter_svd)
    # U, S, V = rank_k_svd_approx_full(M, k=rank)
    # Reconstruct rank-k portion: M_k = U * diag(S) * V^T
    # We'll do (U * S) => (T, k), then matmul with V^T => (T, D)
    M_k = (U * S) @ V.transpose(-2, -1)

    # ---- (2) Residual & stats
    R = M - M_k
    r_std = R.std(unbiased=False)

    if noise_scaling <= 0:
        return M_k, torch.zeros_like(M), M_k

    # ---- (3) Generate random noise that matches R's mean/std
    noise = torch.randn_like(M) * noise_scaling * r_std

    # ---- (4) Remove top-k subspace from noise
    # The top-k subspace on the left side is spanned by columns of U.
    # We'll do noise_orth = noise - Proj_{U}(noise).
    # Provided U has orthonormal columns, project_onto_cols(noise, U) yields
    #   the portion inside the top-k space, so subtract it out.
    noise_proj = project_onto_cols(noise, U)  # shape (T, D)
    noise_orth = noise - noise_proj

    # Optionally re-scale once again if you want exact mean/std in the orth space.
    # We'll do a single pass for demonstration:
    n_std = noise_orth.std(unbiased=False)
    if n_std > 1e-12:
        noise_orth = noise_orth * (r_std / n_std)

    # ---- (5) Combine
    M_noisy = M_k + noise_orth
    return M_k, noise_orth, M_noisy


def add_orthogonal_noise_rankk_batched(M, rank=2, niter_svd=2, noise_scaling=1.0):
    """
    Batched version: M: (B, T, D)
    For each b in [0..B-1], we:
      1. rank-k SVD
      2. get M_k
      3. measure residual stats
      4. create noise, remove top-k subspace
      5. M_noisy = M_k + noise_orth
    Returns: M_k, noise_orth, M_noisy => all shape (B, T, D).
    """
    B, T, D = M.shape
    M_k_out = torch.empty_like(M)
    noise_orth_out = torch.empty_like(M)
    M_noisy_out = torch.empty_like(M)

    for b_idx in range(B):
        Mb = M[b_idx]  # shape (T, D)
        # Single-matrix approach
        Mk_b, noise_b, Mnoisy_b = add_orthogonal_noise_rankk(
            Mb, rank=rank, niter_svd=niter_svd, noise_scaling=noise_scaling
        )
        M_k_out[b_idx] = Mk_b
        noise_orth_out[b_idx] = noise_b
        M_noisy_out[b_idx] = Mnoisy_b

    return M_k_out, noise_orth_out, M_noisy_out


def random_subspace_projection(values: torch.Tensor, rank: int = 2, add_noise_std=0.1):
    B, T, D = values.shape
    device, orig_dtype = values.device, values.dtype

    # 1) Random subspace Q
    R = torch.randn(D, rank, device=device, dtype=torch.float32)
    Q, _ = torch.linalg.qr(R, mode="reduced")  # (D, k)
    Q = Q.to(orig_dtype)

    # 2) Project onto subspace
    # vQ = (B,T,D) x (D,k) -> (B,T,k)
    vQ = torch.matmul(values, Q)  # "batch matmul"
    # v_proj = (B,T,k) x (k,D) -> (B,T,D)
    v_proj = torch.matmul(vQ, Q.transpose(0, 1))

    # 3) Orthogonal noise
    noise = torch.randn_like(values) * add_noise_std
    nQ = torch.matmul(noise, Q)  # (B, T, k)
    n_proj = torch.matmul(nQ, Q.transpose(0, 1))  # (B, T, D)
    noise_orth = noise - n_proj

    # 4) Combine
    values_new = v_proj + noise_orth

    return values_new


def value_svd_attention_wrapper(
    rank: int = 1,
    add_noise_std: float = 0.0,
) -> callable:

    def svd_attention(
        q: Tensor,
        k: Tensor,
        values: Tensor,
        extra_options,
        mask=None,
    ) -> Tensor:
        heads = extra_options["n_heads"]
        """
        Randomly permute the values before computing attention.
        """

        if rank < 1:
            return optimized_attention(q, k, torch.zeros_like(values), heads=heads)

        orig_dtype = values.dtype
        v_truncated, noise_orth, v_noisy = add_orthogonal_noise_rankk_batched(
            values, niter_svd=3, rank=rank, noise_scaling=add_noise_std
        )
        new_v = v_noisy

        return optimized_attention(q, k, new_v, heads=heads)

    return svd_attention


def random_subspace_projection_wrapper(
    rank: int = 10, add_noise_std: float = 0.1
) -> callable:
    """
    Wraps the random_subspace_projection function to be used as an attention function.
    on the values

    :param rank: Dimension of the random subspace.
    :param add_noise_std: Standard deviation of the noise to be added orthogonally.
    :return: Callable function for attention.
    """

    def random_subspace_attention(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        extra_options: dict,
        mask=None,
    ) -> Tensor:
        heads = extra_options["n_heads"]
        new_v = random_subspace_projection(v, rank=rank, add_noise_std=add_noise_std)
        return optimized_attention(q, k, new_v, heads=heads, mask=mask)

    return random_subspace_attention


def affine_attention_transform_wrapper(
    scaling_factor: float = 1.0,
    translation_factor: float = 0.0,
    mean_extrapolation_factor: float = 0.0,
) -> callable:
    """

    A perturbed attention function that applies an affine transformation to the queries before calling the attention

    Parameters
    ----------
    scaling_factor : float
        Multiplier applied to the query vectors (default: 1.0).
    translation_factor : float
        Constant added to every element of the query vectors (default: 0.0).
    mean_extrapolation_factor : float
        Weight of the global mean of Q that is added to each query vector
        (default: 0.0).

    Returns
    -------
    callable
        A function that takes (q, k, v, extra_options, mask) and returns the
        output of `optimized_attention` using the transformed queries.
    """

    def random_drop_dual_perturbed_attention(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        extra_options: dict,
        mask=None,
    ) -> Tensor:
        """
        Applies an affine transform to the queries Q and then calls an optimized
        attention function.

        The queries are transformed as follows:
          new_q = scaling_factor * q
                   + translation_factor
                   + mean_extrapolation_factor * mean(q)

        Where mean(q) is computed over the tokens dimension (area), yielding
        a per-sample global mean that is broadcast back into each token.

        Parameters
        ----------
        q : Tensor
            Query tensor of shape [batch_size, area, inner_dim].
        k : Tensor
            Key tensor of shape [batch_size, area, inner_dim].
        v : Tensor
            Value tensor of shape [batch_size, area, inner_dim].
        extra_options : dict
            Additional options for attention, must contain "n_heads" for multi-head.
        mask : Optional[Tensor]
            An optional attention mask to be applied in the attention step.

        Returns
        -------
        Tensor
            The output of the attention mechanism using the transformed queries.
        """
        heads = extra_options["n_heads"]
        bs, area, inner_dim = q.shape

        # 1) Scale q
        new_q = q * scaling_factor

        # 2) Add translation offset (broadcast over all elements if float)
        if translation_factor != 0.0:
            new_q = new_q + translation_factor

        # 3) Mean extrapolation: add the mean of q multiplied by factor
        if mean_extrapolation_factor != 0.0:
            # Mean across tokens (dim=1), keep dimension for broadcast
            q_mean = q.mean(dim=1, keepdim=True)
            new_q = new_q + mean_extrapolation_factor * q_mean

        return optimized_attention(new_q, k, v, heads=heads, mask=mask)

    return random_drop_dual_perturbed_attention


def fuzzy_attention_wrapper(
    noise_std: float = 1.0, scaling_factor: float = 1.0
) -> callable:
    """
    Add noise to the values

    :param noise_std:
    :param scaling_factor:
    :return:
    """

    def fuzzy_attention(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        extra_options: dict,
        mask=None,
    ) -> Tensor:
        heads = extra_options["n_heads"]

        # Scale queries
        new_v = v * scaling_factor

        # Add Gaussian noise
        if noise_std > 0.0:
            noise = torch.randn_like(new_v) * noise_std
            new_v += +noise

        return optimized_attention(q, k, new_v, heads=heads, mask=mask)

    return fuzzy_attention


def random_drop_wrapper(
    drop_percentage: float = 0.5,
    rescale: float = 1.0,
    replacement_std_mult: float = 0.0,
) -> callable:
    """
    Drop a random subset of the values before computing attention.
    """

    # def random_drop(
    #     q: Tensor,
    #     k: Tensor,
    #     v: Tensor,
    #     extra_options: dict,
    #     mask=None,
    # ) -> Tensor:
    #     heads = extra_options["n_heads"]
    #
    #     b, t, d = v.shape
    #     r = max(0, min(int(drop_percentage * t), t))
    #     idx = torch.randperm(t, device=v.device)[:r]
    #     # Generate random samples following the Gaussian distribution
    #     # mean = v.mean(dim=1, keepdim=True)
    #     # std = v.std(dim=1, keepdim=True)
    #     std = v.std()
    #     # mean = v.mean()
    #     random_samples = replacement_std_mult * std * torch.randn_like(v)
    #
    #     # Replace the dropped values with the random samples
    #     new_v = v.clone()
    #     new_v[:, idx, :] = random_samples[:, idx, :]
    #     new_v *= rescale
    #
    #     return optimized_attention(q, k, new_v, heads=heads, mask=mask)
    def random_drop_orth(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        extra_options: dict,
        mask=None,
    ) -> Tensor:
        """
        v: (B, T, D)  each item in the batch has T tokens, each token is D-dim.
        We drop a random set of tokens (size ~ drop_percentage * T) per batch item.
        Then we generate noise and remove the subspace spanned by the remain tokens.
        """
        heads = extra_options["n_heads"]
        B, T, D = v.shape

        # We'll build a new copy of v to modify
        new_v = v.clone()

        # For each batch item
        for b_idx in range(B):
            # 1) Decide which tokens to drop for this item
            r = max(0, min(int(drop_percentage * T), T))
            idx = torch.randperm(T, device=v.device)[:r]  # tokens to drop
            remain_idx = torch.tensor(
                list(set(range(T)) - set(idx.tolist())),
                device=v.device,
                dtype=torch.long,
            )
            remain_count = remain_idx.numel()

            if remain_count == 0:
                # Edge case: if everything got dropped, we can't build a subspace.
                # Just fill noise (optionally do it isotropically).
                noise_b = (
                    replacement_std_mult
                    * v.std()
                    * torch.randn((r, D), device=v.device, dtype=v.dtype)
                )
                new_v[b_idx, idx, :] = noise_b
                continue

            # 2) Gather the remain vectors => shape (remain_count, D)
            remain_v = new_v[b_idx, remain_idx, :]  # (remain_count, D)

            dropped = new_v[b_idx, idx, :]
            dropped_std = dropped.std()
            noise_scale = replacement_std_mult * dropped_std
            dropped_mean = dropped.mean()

            # 3) Build subspace via QR on remain_v^T => shape (D, remain_count)
            #    so that columns of Q are an orthonormal basis in R^D
            M = remain_v.transpose(0, 1)  # shape (D, remain_count)
            # If remain_count > D, we clamp remain_count to D to avoid errors
            # Usually remain_count <= T, but T can be bigger than D or smaller, depends.
            # We'll do 'reduced' mode
            # But handle edge cases:
            rcount = min(M.shape[1], M.shape[0])  # min(d, remain_count)
            if rcount < 1:
                # No subspace
                # i.e. remain_count < 1 => all dropped
                noise_b = noise_scale * torch.randn(
                    (r, D), device=v.device, dtype=v.dtype
                )
                new_v[b_idx, idx, :] = noise_b
                continue

            # Possibly slice M if remain_count > D
            M = M[:, :rcount]  # shape (D, rcount)
            orig_dtype = M.dtype
            Q, _ = torch.linalg.qr(
                M.to(dtype=torch.float32), mode="reduced"
            )  # Q: (D, rcount)
            Q = Q.to(orig_dtype)

            # 4) Generate random noise for the dropped tokens => shape (r, D)
            noise_b = torch.randn((r, D), device=v.device, dtype=v.dtype)

            # 5) Project out the subspace spanned by Q => noise_orth
            #    noise_b: (r, D)
            #    Q: (D, rcount)
            # => noise_b @ Q: (r, rcount)
            # => (noise_b @ Q) @ Q^T: (r, D) is the portion inside subspace
            proj = (noise_b @ Q) @ Q.transpose(-2, -1)  # shape (r, D)
            noise_orth = noise_b - proj
            noise_orth *= noise_scale / noise_orth.std()
            # noise_orth += dropped_mean

            # 6) Insert into new_v
            new_v[b_idx, idx, :] = noise_orth

        new_v *= rescale

        return optimized_attention(q, k, new_v, heads=heads, mask=mask)

    return random_drop_orth


def upscale_and_transfer_previous_attention_wrapper(
    downsample_factor: float = 2.0,
) -> callable:

    prev_k: None | Tensor = None
    prev_v: None | Tensor = None
    prev_q: None | Tensor = None

    def transfer_attention(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        extra_options: dict,
        mask=None,
    ) -> Tensor:
        nonlocal prev_k, prev_v, prev_q
        if prev_k is None:
            prev_k = k
            prev_v = v
            prev_q = q
            return optimized_attention(
                q, k, v, heads=extra_options["n_heads"], mask=mask
            )
        else:
            upsample_mode = "bicubic"
            height_orig, width_orig = extra_options["original_shape"][2:4]
            aspect_ratio = width_orig / height_orig
            # q_lo_2d = tokens_to_spatial(prev_q, aspect_ratio)
            # q_2d = tokens_to_spatial(q, aspect_ratio)
            # k_lo_2d = tokens_to_spatial(prev_k, aspect_ratio)
            # k_2d = tokens_to_spatial(k, aspect_ratio)
            # v_lo_2d = tokens_to_spatial(prev_v, aspect_ratio)
            # v_2d = tokens_to_spatial(v, aspect_ratio)
            q_lo, k_lo, v_lo = prev_q, prev_k, prev_v

            b, tokens, dim = q.shape
            b_lo, tokens_lo, dim_lo = q_lo.shape
            assert dim == dim_lo
            assert b == b_lo
            heads = extra_options["n_heads"]
            dim_head = dim // heads
            attn = torch.nn.MultiheadAttention(dim, heads, device=q.device)
            attn_lo, attn_map_lo = attn(q_lo, k_lo, v_lo, need_weights=True)
            attn_hi, attn_map_hi = attn(q, k, v, need_weights=True)

            # B, C, H_hi, W_hi = q_2d.shape
            # _, _, H_lo, W_lo = q_lo_2d.shape

            # # 2. Upsample lo-res -> hi-res
            # scale_factor_h = H_hi / H_lo
            # scale_factor_w = W_hi / W_lo

            # q_lo_up = F.interpolate(
            #     q_lo_2d, size=(H_hi, W_hi), mode=upsample_mode
            # )  # [B, C, H_hi, W_hi]
            # k_lo_up = F.interpolate(
            #     k_lo_2d, size=(H_hi, W_hi), mode=upsample_mode
            # )  # [B, C, H_hi, W_hi]
            # v_lo_up = F.interpolate(v_lo_2d, size=(H_hi, W_hi), mode=upsample_mode)

            # q = spatial_to_tokens(q_2d)
            # k = spatial_to_tokens(k_2d)

            # v = spatial_to_tokens(v_2d)
            # latent_filters.compare_kqv_resolutions(
            #     k, q, v, prev_k, prev_q, prev_v, aspect_ratio
            # )
            return optimized_attention(
                q, k, v, heads=extra_options["n_heads"], mask=mask
            )

    return transfer_attention


def batch_copy_attention_keys_queries_wrapper() -> callable:

    def copy_attention_keys_queries(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        extra_options,
        mask=None,
    ) -> Tensor:
        """
        Takes a batch of size 2 and copies the keys and queries of the first batch to the second batch.
        before calling the attention function
        :param q:
        :param k:
        :param v:
        :param extra_options:
        :param mask:
        :return:
        """
        heads = extra_options["n_heads"]
        bs, area, inner_dim = q.shape
        assert bs == 2, "This function requires a batch size of 2"

        q[1] = q[0]
        k[1] = k[0]

        # Copy the keys and queries to the value tensor
        return optimized_attention(v, k, q, heads=heads, mask=mask)

    return copy_attention_keys_queries


def random_rotation_wrapper(
    rotation_intensity: float = 0.0, rescale: float = 1.0
) -> callable:
    """
    Returns a function that, when called, applies a random orthonormal rotation
    in R^d to the query vectors. The rotation is parameterized by
    rotation_intensity in [0,1], where 0 means 'no rotation' and 1 means
    'full random rotation'.

    Internally:
      1. We generate a skew-symmetric matrix S (size d x d) once per forward pass.
      2. For a given rotation_intensity = alpha, we compute R(alpha) = exp(alpha * S).
      3. We apply R(alpha) to each query vector q[i].

    Parameters
    ----------
    rotation_intensity : float
        A scalar in [0,1]. 0 means identity transform, 1 means a random
        orthonormal rotation given by exp(S).
    rescale : float
        A multiplier applied after rotation (default=1.0).

    Returns
    -------
    callable
        A function that takes (q, k, v, extra_options, mask) and returns the
        output of `optimized_attention` with the queries rotated by R(alpha).
    """

    # We'll clamp alpha to [0,1] just to be safe (in case user inputs out of range).
    alpha = max(0.0, min(1.0, rotation_intensity))

    def create_random_skew_symmetric(d: int, device) -> Tensor:
        """
        Create a random d x d skew-symmetric matrix S by sampling a random
        matrix A and taking S = (A - A^T)/2, ensuring S^T = -S.
        """
        A = torch.randn(d, d, device=device, dtype=torch.float32)
        S = A - A.transpose(0, 1)  # Make S skew-symmetric
        S *= 0.5
        return S

    def random_orthonormal_attention(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        extra_options: dict,
        mask=None,
    ) -> Tensor:
        """
        Applies an orthonormal rotation R(alpha) = exp(alpha * S)
        to each query vector in q, then calls `optimized_attention`.
        """

        heads = extra_options["n_heads"]
        bs, area, d = q.shape

        # Generate a new random skew-symmetric matrix S each forward pass
        S = create_random_skew_symmetric(d, q.device)

        if alpha == 0.0:
            # No rotation
            rotated_q = q
        else:
            # Compute exp(alpha * S) via torch.matrix_exp
            transform_matrix = torch.matrix_exp(S * alpha)  # shape: [d, d]
            transform_matrix = transform_matrix.to(dtype=q.dtype, device=q.device)

            # Perform the rotation via a standard matrix multiplication
            # q is [bs, area, d]. We'll treat (bs*area) as a batch dimension:
            #   [bs*area, d] x [d, d] -> [bs*area, d]
            # and reshape back to [bs, area, d].
            rotated_q = q.reshape(-1, d) @ transform_matrix
            rotated_q = rotated_q.view(bs, area, d)

            if rescale != 1.0:
                rotated_q = rotated_q * rescale

        return optimized_attention(rotated_q, k, v, heads=heads, mask=mask)

    return random_orthonormal_attention


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
