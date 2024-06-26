"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import torch
import numpy as np
from scipy.stats import betaprime
#----------------------------------------------------------------------------
# Loss function corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

randn_like=torch.randn_like

SIGMA_MIN=0.002
SIGMA_MAX=80
rho=7
S_churn= 1
S_min=0
S_max=float('inf')
S_noise=1

def sample(net, num_samples, dim, num_steps = 50, device = 'cuda:0', disc_model=None):
    latents = torch.randn([num_samples, dim], device=device)

    step_indices = torch.arange(num_steps, dtype=torch.float32, device=latents.device)

    sigma_min = max(SIGMA_MIN, net.sigma_min)
    sigma_max = min(SIGMA_MAX, net.sigma_max)

    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
                sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    with torch.no_grad():
        t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    x_next = latents.to(torch.float32) * t_steps[0]

    # with torch.no_grad():
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_next = sample_step(net, num_steps, i, t_cur, t_next, x_next, disc_model)

    return x_next


def forward_guidance(inputs, disc_model):
    inputs_clone = torch.clone(inputs)
    inputs_clone.requires_grad_(True)
    loss = -torch.sum(torch.log(torch.sigmoid(disc_model(inputs_clone))))
    guidance = torch.autograd.grad(loss, inputs_clone)

    return guidance[0]

def sample_step(net, num_steps, i, t_cur, t_next, x_next, disc_model):
    guidance_weight = 100
    x_cur = x_next
    # Increase noise temporarily.
    gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
    with torch.no_grad():
        t_hat = net.round_sigma(t_cur + gamma * t_cur) 
    x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
    # Euler step.
    with torch.no_grad():
        denoised = net(x_hat, t_hat).to(torch.float32)
    d_cur = (x_hat - denoised) / t_hat
    if disc_model is not None:
        d_cur += forward_guidance(x_cur, disc_model) * guidance_weight
    x_next = x_hat + (t_next - t_hat) * d_cur

    # Apply 2nd order correction.
    if i < num_steps - 1:
        with torch.no_grad():
            denoised = net(x_next, t_next).to(torch.float32)
        d_prime = (x_next - denoised) / t_next
        if disc_model is not None:
            d_prime += forward_guidance(x_next, disc_model) * guidance_weight
        x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next

def guided_sample(
        net, num_samples, dim, num_steps=50, device='cuda:0',disc_model=None,
        joint_temperature=1.0, 
        uncond_temperature=0.5, 
        max_guidance=0.004, 
        add_factor=0.3,
):
    latents = torch.randn([num_samples, dim], device=device)
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(SIGMA_MIN, net.sigma_min)
    sigma_max = min(SIGMA_MAX, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
               sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])  # t_N = 0

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)

        # classifier guidance
        add_noise = S_noise * torch.randn_like(x_cur)
        if disc_model is not None:
            with torch.enable_grad():
                x_cur_norm = (((x_cur ** 2).mean(1)) ** 0.5)
                x0 = x_cur * 0
                for index in range(x_cur_norm.shape[0]):
                    if x_cur_norm[index] > 1:
                        x0[index] = x_cur[index] / x_cur_norm[index]
                    else:
                        x0[index] = x_cur[index]
                x0 = x0.detach().requires_grad_(True).float()
                logits = disc_model(x0)
                numerator = torch.exp(logits * joint_temperature)[:, 1].unsqueeze(1)
                denominator = torch.exp(logits * uncond_temperature).sum(1, keepdims=True) * 2
                selected = torch.log(numerator / denominator)
                # probs = torch.log(logits)

                current_time = i
                current_guidance = (max_guidance/len(t_steps)) * (len(t_steps) - current_time)
                current_guidance = max(current_guidance, 0.00001)

                interval = len(t_steps) - 1
                add_value = np.sin( current_time / interval * (1*np.pi) ) * max_guidance * add_factor
                current_guidance = current_guidance + add_value

                grads = torch.autograd.grad(selected.sum(), x0)[0]
                grads_norm = ( ((grads**2).mean((1)))**0.5 )
                for index in range(x_cur_norm.shape[0]):
                    grads[index] = (grads[index]/grads_norm[index]) * x_cur_norm[index] * current_guidance
        else:
            grads = 0

        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * add_noise + grads
        # Euler step.
        denoised = net(x_hat, t_hat).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, t_next).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next


class VPLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t

    def __call__(self, denosie_fn, data, labels, augment_pipe=None):
        rnd_uniform = torch.rand([data.shape[0], 1, 1, 1], device=data.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(data) if augment_pipe is not None else (data, None)
        n = torch.randn_like(y) * sigma
        D_yn = denosie_fn(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()

#----------------------------------------------------------------------------
# Loss function corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

class VELoss:
    def __init__(self, sigma_min=0.02, sigma_max=100, D=128, N=3072, opts=None):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.D = D
        self.N = N
        print(f"In VE loss: D:{self.D}, N:{self.N}")

    def __call__(self, denosie_fn, data, labels = None, augment_pipe=None, stf=False, pfgmpp=False, ref_data=None):
        if pfgmpp:

            # N, 
            rnd_uniform = torch.rand(data.shape[0], device=data.device)
            sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)

            r = sigma.double() * np.sqrt(self.D).astype(np.float64)
            # Sampling form inverse-beta distribution
            samples_norm = np.random.beta(a=self.N / 2., b=self.D / 2.,
                                          size=data.shape[0]).astype(np.double)

            samples_norm = np.clip(samples_norm, 1e-3, 1-1e-3)

            inverse_beta = samples_norm / (1 - samples_norm + 1e-8)
            inverse_beta = torch.from_numpy(inverse_beta).to(data.device).double()
            # Sampling from p_r(R) by change-of-variable
            samples_norm = r * torch.sqrt(inverse_beta + 1e-8)
            samples_norm = samples_norm.view(len(samples_norm), -1)
            # Uniformly sample the angle direction
            gaussian = torch.randn(data.shape[0], self.N).to(samples_norm.device)
            unit_gaussian = gaussian / torch.norm(gaussian, p=2, dim=1, keepdim=True)
            # Construct the perturbation for x
            perturbation_x = unit_gaussian * samples_norm
            perturbation_x = perturbation_x.float()

            sigma = sigma.reshape((len(sigma), 1, 1, 1))
            weight = 1 / sigma ** 2
            y, augment_labels = augment_pipe(data) if augment_pipe is not None else (data, None)
            n = perturbation_x.view_as(y)
            D_yn = denosie_fn(y + n, sigma, labels,  augment_labels=augment_labels)
        else:
            rnd_uniform = torch.rand([data.shape[0], 1, 1, 1], device=data.device)
            sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
            weight = 1 / sigma ** 2
            y, augment_labels = augment_pipe(data) if augment_pipe is not None else (data, None)
            n = torch.randn_like(y) * sigma
            D_yn = denosie_fn(y + n, sigma, labels, augment_labels=augment_labels)

        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).

class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, hid_dim = 100, gamma=5, opts=None):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.hid_dim = hid_dim
        self.gamma = gamma
        self.opts = opts


    def __call__(self, denoise_fn, data):

        rnd_normal = torch.randn(data.shape[0], device=data.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()

        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

        y = data # z0
        n = torch.randn_like(y) * sigma.unsqueeze(1) # zt = z0 + eps*sigma
        D_yn = denoise_fn(y + n, sigma) # eps_theta = f(zt, t)
    
        target = y
        loss = weight.unsqueeze(1) * ((D_yn - target) ** 2) # mse(eps_theta, z0) ? 

        return loss

