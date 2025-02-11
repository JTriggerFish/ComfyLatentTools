---
layout: default
title: "Scrambled Attention Guidance"
toc: true        # Enable automatic table of contents generation.
---

<!-- Draft Warning Banner -->
<div style="border: 2px solid red; background-color: #ffe6e6; padding: 15px; text-align: center; margin-bottom: 20px;">
  <h2 style="margin-top: 0;">Draft Document</h2>
  <p>This document is a draft. Content is incomplete and subject to change.</p>
</div>

# Scrambled Attention Guidance

<!-- Table of Contents marker (Kramdown will replace this with the TOC) -->
* Table of Contents
  {:toc}

# Background
## General diffusion models formulation

There are various approaches and formulations in the literature, here we follow the setup that best matches the implementation in ComfyUI to make it easier to match the formulas and the code.

### Model

We assume that our data, in this case images, follows some unknown and intractable probability distribution in some high dimensional space. \
Let $x$ be an arbitrary sample from this distribution $x \sim p(x)$ \
Since we can always rescale it we may WLOG assume that its standard deviation $\text{std}(x)=1$.

In diffusion models we seek to build progressively noisier latent variables indexed by time

{% raw %}
$$
\begin{align}
z_t &= \alpha_t\, x + \sigma_t\, \epsilon_t, \\
\epsilon_t &\sim \mathcal{N}(0,1)
\end{align}
$$
{% endraw %}

where the $\epsilon_t$ are standard normal noise samples that are independent of $x$, and $\alpha_t$ and $\sigma_t$ are functions of time,
with $\sigma_t$ strictly increasing in $t$.

This latent model is in fact an integrated form of a stochastic differential equation (SDE) with a time dependent diffusion coefficient.\
The time index $t$ is in an arbitrary range $[0,T]$ where we may take $T=1$ for simplicity - note that by the time change property of Brownian motions,
rescaling time is equivalent to rescaling the diffusion coefficient.

#### Variance preserving formulation
By independence, we have that $\text{var}(z_T) = \alpha_t^2 + \sigma_t^2$. \
At the last step we want the latent variable to be almost pure noise so we set $\alpha_T << 1$ and $\sigma_T \approx \Sigma^2$. \
In the variance preserving formulation we keep the variance of the latent variable constant, i.e.

{% raw %}
$$
\forall t, \quad \text{var}(z_t) = \Sigma^2 = \sigma_T^2.
$$
{% endraw %}

Depending on the formulation we may take $\alpha_T = 0$ exactly ( "Zero Signal to Noise Ratio" or ZSNR ) or not, this has important implications that we sidestep here.

In many papers it is assumed that $\Sigma^2 = 1$ for simplicity, but note that this is typically not the case in the ComfyUI code thus we will not make this assumption here. \
The $\alpha_t$ parameters is entirely redundant in the variance preserving formulation, since it can be rewritten as $\alpha_t = \sqrt{\Sigma^2 - \sigma_t^2}$.

Note that we can normalize any of the samples $z_t$ to have unit variance by dividing by $\Sigma$ ( this is typically done in the ComfyUI code before calling the decoder model )

### Forward and backward diffusions
So far we have built some variables $z_t$ that are progressively noisier versions of the original image $x$.\
$z_0$ is almost exactly the original image, and $z_T$ is almost pure noise, and it is easy to show that

{% raw %}
$$
\forall s>t, \quad z_s = \frac{\alpha_s}{\alpha_t}\, z_t + \sqrt{\sigma_s^2 - \left(\frac{\alpha_s}{\alpha_t}\right)^2 \sigma_t^2}\,\epsilon_{s\rightarrow t},
$$
{% endraw %}

were $\epsilon_{s\rightarrow t}$ is still a standard normal random variable independent of $z_t$. 

In other words, we can build the final pure noise image iteratively by a forward diffusion process starting from the original image and adding more and more noise.

Now if we were to try and run this process backwards, we would start with pure noise and estimate the conditional distribution of $z_t$ given $z_s$ for $s>t$ iteratively until we reach $z_0$,
at whcih point we have an image that would follow the original distribution $p(x)$. \
This is exactly what we do when sampling - the difficulty being in estimating the conditional distribution ( in practice we want just the mean ) of $\epsilon_t$,
or equivalently $z_t$, given $z_s$ for $s>t$.

Intuitively, by taking small steps the conditional distribution of $z_t$ given $z_s$ is close to a normal distribution and should thus be easier to approximate - this is why we run
the process iteratively with small steps.

Note that the sampling process is best formulated as an SDE or the equivalent Fokker-Planck probability flow ODE, and then discretized in time to obtain a numerical scheme, in which
case the fact we need to take small steps is more clear.

What's left to do now is to find a way to sample from the conditional distribution of $z_t$ given $z_s$ for $s>t$ - this is where the neural networks come in.
Specifically in the Stable Diffusion family of models, we parametrize a UNet architecture together with attention mechanisms to estimate these conditional distributions.

Note that in stable diffusion the sampling is actually done in the latent space of a VAE in order to greatly reduce the dimensionality of the problem compared
to doing it directly in the image space. The VAE can be thought of as in image compression scheme.

Furthermore, in all the above we have been working on unconditional diffusion, but in practice we want to condition the sampling on some text embedding or other information, so the model
can be used for text to image generation.

This is actually fairly simple to do, we just need to add the conditioning information to the latent variable $z_t$ and the sampling process is the same.

## Some key estimates, and model training.

From the above formulation it is immediate that

{% raw %}
$$
x = \frac{z_t - \sigma_t \epsilon_t}{\alpha_t},
$$
{% endraw %}

and thus

{% raw %}
$$
\mathbb{E}[x|z_t] = \frac{z_t - \sigma_t \mathbb{E}[\epsilon_t | z_t] }{\alpha_t}
$$
{% endraw %}

Here $\mathbb{E}[\epsilon_t | z_t]$ is the quantity that eps-prediction models try to estimate, with 

{% raw %}
$$
\hat{\epsilon}_{\theta}(z_t, \sigma_t) \approx \mathbb{E}[\epsilon_t | z_t]
$$
{% endraw %}

where 
$\hat{\epsilon}_{\theta}(z_t, \sigma_t)$ is the model's noise prediction for the current time step, using fitted neural network parameters $\theta$.

$\mathbb{E}\left[ x | z_t\right]$ is the denoised output or prediction of the diffusion model at time $t$.

Note that the divison by $\alpha_t$ is a problem at the last step, since $\alpha_T$ is very small, or even zero for ZSNR models. \
However this rescaling is not necessary and is not done in the ComfyUI implementation, there the "denoised output" is simply 
$z_t  - \sigma_t  \hat{\epsilon}_{\theta}(z_t, \sigma_t)$.

TODO CHECK WHETHER THE RESCALING HAPPENS SOMEWHERE ELSE IN THE SAMPLER OR BEFORE THE VAE DECODING

### Training the model
In typical model training we aim to minimise the weighted sum of the MSE between the denoised output and the original image at various time steps, where the noisy images are generated by the diffusion process.

### V-prediction and its relationship to the score function
Using Tweedie's formula we can compute another estimate of the conditional expectation of $x$ given $z_t$.
(See [here](https://www.weideng.org/posts/tweedie_formula/) for a detailed derivation):

{% raw %}
$$
\mathbb{E}[x|z_t] = \frac{1}{\alpha_t} \left( z_t + \sigma_t^2 \nabla_{z_t} \log p(z_t) \right)
$$
{% endraw %}

Let $\hat{v_t} = \nabla_{z_t} \log p(z_t)$ be the v-prediction vector.\
We can equivalently fit the model to predict this vector instead, this is equivalent to reweighting the loss function that we fit the model to minimize.

indeed, note that rearranging the above formula we get that

{% raw %}
$$
\begin{align}
\hat{v_t} &= \frac{1}{\sigma_t^2} \left( \alpha_t \mathbb{E}[x|z_t] - z_t \right)  \\
&= - \frac{\hat{\epsilon}}{\sigma_t}
\end{align}
$$
{% endraw %}

# Scrambled Attention Guidance
TODO


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
