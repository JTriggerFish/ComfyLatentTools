- [ComfyLatentTools](#comfylatenttools)
  - [Installation](#installation)
  - [Node Overview](#node-overview)
    - [Rescaled PAG ( Perturbed Attention Guidance )](#rescaled-pag--perturbed-attention-guidance-)
      - [Parameters](#parameters)
    - [Latent Normalized Lanczos Resize (LNLR)](#latent-normalized-lanczos-resize-lnlr)
      - [Internal Operation Order](#internal-operation-order)
      - [Parameters](#parameters-1)
- [Math notes](#math-notes)
  - [Why is the CFG and PAG rescaling done in v-pred space](#why-is-the-cfg-and-pag-rescaling-done-in-v-pred-space)
  - [TODO more rigorous derivation of this argument](#todo-more-rigorous-derivation-of-this-argument)
- [References](#references)

# ComfyLatentTools

A set of custom nodes for ComfyUI, providing a specialized **Latent Normalized Lanczos Resize** workflow.

## Installation

1. Download or clone this repository into your ComfyUI's `custom_nodes` folder.
2. Restart ComfyUI. The node(s) will appear in the **image/upscaling** category.

## Node Overview

### Rescaled PAG ( Perturbed Attention Guidance )

Implementation of Perturbed Attention Guidance that also takes into account the insight of RescaleCFG.

Compared to vanilla PAG this results in much better dynamic range and much less "deep frying" or oversaturation.

This is still a little bit experimental and some amount of experimentation with sampling parameters and PAG parameters is recommended.
pag_scale = 1/2 cfg is a good starting point and I recommend a post rescale close to 1.0.

The samplers seem to converge faster with this turned on so a decrease of number of steps might be possible with little loss of quality.

#### Parameters

### Latent Normalized Lanczos Resize (LNLR)

A specialized upscaling node designed to:
- Perform a Lanczos upscale in **image space**,
- Re-encode to latent space,
- Match the original latent's mean/variance,
- Optionally add correlated noise or blend with a pure latent-based upscale.

This aims to produce an upscaled latent that stays more faithful to the original diffusion pass, avoiding excessive blur, and optionally adding noise.
It can serve as a fast base for subsequent (re-)diffusion or refinement steps, in which case the noise addition can help to introduce additional details and variations at different scales.

#### Internal Operation Order

1. **Soft Outlier Clamp** (optional)  
   Uses a “huberize_quantile” method to softly clamp outliers.
2. **Decode**  
   Converts latent to image.
3. **Lanczos Upscale**  
   Upscales the image.
4. **Encode**  
   Converts upscaled image back to latent space.
5. **Weighted Latent Upscale** (optional)  
   Blends the new latent with a nearest exact upsampled version of the original latent, if the corresponding weight > 0
6. **Moment Matching**  
   Aligns mean and variance of the upscaled latent with the original.
7. **Add Correlated Gaussian Noise** (optional)  
   Injects correlated noise for additional variation.

#### Parameters

- **size_multiplier**  
  Multiplies original spatial dimensions (width/height).  
  *(Default: 2.0; Range: 0.1–4.0)*

- **soft_clamp_outliers** (enable/disable)  
  Toggles outlier soft-clamping before decoding.  
  *(Default: enable)*

- **outlier_quantile**  
  Quantile threshold for outlier soft-clamping.  
  *(Default: 0.01; Range: 0–1)*

- **outlier_clamp_slope**  
  Slope of the clamp outside the quantile range.  
  *(Default: 0.1; Range: 0–1)*

- **add_latent_noise** (enable/disable)  
  Adds correlated Gaussian noise to the final latent.  
  *(Default: disable)*

- **latent_noise_std**  
  Standard deviation for the generated noise.  
  *(Default: 1.0; Range: 0–10)*

- **latent_noise_scale**  
  Scaling factor applied to the correlated noise before adding.  
  *(Default: 0.1; Range: 0.01–10)*

- **add_latent_upscale_with_weight**  
  Blends the newly encoded latent with a direct “latent nearest” upscale.  
  *(Default: 0.0; Range: 0–1)*



# Math notes
## Why is the CFG and PAG rescaling done in v-pred space
A shown in [2](#ref2), the v-prediction velocity is essentially the velocity of the discretized ODE that needs to be solved to sample at inference.
A heuristic argument is that we can tweak the direction of this vector to change the trajectory of the denoising path without affecting the convergence of the scheme significantly.

However, chaging the magnitude of this vector will cause over or under shooting the trajectory and either fail to converge or diverge, causing the "burnt" saturated images where the latent vectors get clipped to their max values.

Therefore, by switching our prediction target to v-space instead of $x_0$ or $\epsilon$ space, which can all be done linearly, and ensuring that we renormalize the new vector, 
we perform interpolation in a "safe" space where we can weigh things as much as we want while avoiding these issues.

We can then convert the v-prediction back to $x_0$ prediction as expected by the rest of the diffusion code as implemented in ComfyUI.

## TODO more rigorous derivation of this argument

Thefreo

# References
<a id="ref1"></a>
- [1]Efficient Diffusion Training via Min-SNR Weighting Strategy [paper](https://arxiv.org/abs/2303.09556)
<a id="ref2"></a>
- [2]Progressive Distillation for Fast Sampling of Diffusion Models [paper](https://arxiv.org/abs/2202.00512)
- [3]Common Diffusion Noise Schedules and Sample Steps are Flawed [paper](https://arxiv.org/abs/2305.08891)
- [4]sd_perturbed_attention [github](https://github.com/pamparamm/sd-perturbed-attention/tree/master)
- [5]Perturbed-Attention Guidance from Self-Rectifying Diffusion Sampling with Perturbed-Attention Guidance [paper](https://cvlab-kaist.github.io/Perturbed-Attention-Guidance/)
- [6]Smoothed Energy Guidance: Guiding Diffusion Models with Reduced Energy Curvature of Attention [paper](https://arxiv.org/abs/2408.00760)
- [7]Sliding Window Guidance from The Unreasonable Effectiveness of Guidance for Diffusion Models [paper](https://arxiv.org/abs/2411.10257)