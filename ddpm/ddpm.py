import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

import sys
curPath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(curPath)



# def _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, warmup_frac):
#     betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
#     warmup_time = int(num_diffusion_timesteps * warmup_frac)
#     betas[:warmup_time] = np.linspace(beta_start, beta_end, warmup_time, dtype=np.float64)
#     return betas


# def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
#     """
#     This is the deprecated API for creating beta schedules.

#     See get_named_beta_schedule() for the new library of schedules.
#     """
#     if beta_schedule == "quad":
#         betas = (
#             np.linspace(
#                 beta_start ** 0.5,
#                 beta_end ** 0.5,
#                 num_diffusion_timesteps,
#                 dtype=np.float64,
#             )
#             ** 2
#         )
#     elif beta_schedule == "linear":
#         betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
#     elif beta_schedule == "warmup10":
#         betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.1)
#     elif beta_schedule == "warmup50":
#         betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.5)
#     elif beta_schedule == "const":
#         betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
#     elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
#         betas = 1.0 / np.linspace(
#             num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
#         )
#     else:
#         raise NotImplementedError(beta_schedule)
#     assert betas.shape == (num_diffusion_timesteps,)
#     return betas

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class DDPM(nn.Module):
    def __init__(self, 
                 input_dim=1,
                 hidden_dim=1024,
                 n_steps=1000,
                 device=torch.device('cuda:0')) -> None:
        super().__init__()
        self.device=device
        self.input_dim = input_dim
        self.diffuser = MLPDiffusion(d_in=input_dim, dim_t=hidden_dim).to(self.device)
        
        self.n_steps = n_steps
        betas = torch.linspace(-6, 6, self.n_steps)
        betas = torch.sigmoid(betas)
        betas = betas * (0.5e-2 - 1e-5) + 1e-5

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))
        
        self.to(device)
        
    def get_loss_weights(self, action_weight, discount, weights_dict):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        self.action_weight = action_weight
        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        ## set loss coefficients for dimensions of observation
        if weights_dict is None: weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[ind] *= w

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

        return loss_weights

    #------------------------------------------ sampling ------------------------------------------#
    
    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, eps=None):
        if eps is None:
            epsilon = self.diffuser(x, t)
        else:
            epsilon = eps

        t = t.detach().to(torch.int64)
        x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, eps=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, eps=eps)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape):
        batch_size = shape[0]
        x = torch.randn(shape, device=self.device)

        for i in reversed(range(0, self.n_steps)):
            timesteps = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            x = self.p_sample(x, timesteps)
            
        return x
    
    @torch.no_grad()
    def sample_wo_guidance(self, batch_size=400, n_samples=0, extra_step=0):            
        x = np.random.randn(n_samples, self.input_dim)

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            x[start:end] = self.p_sample_loop(x[start:end].shape).cpu().detach().numpy()
        
        return x

    def forward_guidance(self, disc_model, z0, zt):
        preds = disc_model(z0)
        loss = -torch.sum(torch.log(torch.sigmoid(preds)))
        guidance = torch.autograd.grad(loss, zt)
        return guidance[0]
    
    def universal_guided_sample_batch(self, batch_size=400, disc_model=None, forward_weight=1, backward_step=10, self_recurrent_step=10):
        x = torch.randn((batch_size, self.input_dim), dtype=torch.float32).to(self.device).requires_grad_(True)
        for i in reversed(range(0, self.n_steps)):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            eps_theta = self.diffuser(x, t)
            z0_hat = (x - extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * eps_theta) / extract(self.sqrt_alphas_cumprod, t, x.shape)
            eps_theta_hat = eps_theta + forward_weight * extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * self.forward_guidance(disc_model, z0_hat, x) # forward universal guidance
            x = self.p_sample(x, t, eps_theta_hat).requires_grad_(True)
        return x.detach().cpu().numpy()
    
    
    def universal_guided_sample(self, batch_size, n_samples, disc_model, forward_weight, backward_step, self_recurrent_step):
        if disc_model == None:
            print('No discriminator added, running without guidance')
            return self.sample_wo_guidance(batch_size, n_samples)
        
        res = []
        with tqdm(total=n_samples) as pbar:
            for _ in range(0, n_samples, batch_size):
                res.append(self.universal_guided_sample_batch(batch_size, disc_model, forward_weight, backward_step, self_recurrent_step))
                pbar.update(batch_size)
        return np.concatenate(res, 0)

    #------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, t):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon = self.diffuser(x_noisy, t)
        loss = ((x_recon - noise) ** 2).mean()
        return loss

    def forward(self, x):
        batch_size = len(x)
        t = torch.randint(0, self.n_steps, (batch_size,), device=x.device).long()
        return self.p_losses(x, t)
    
        # self.betas = cosine_beta_schedule(self.n_steps)
    #     betas = torch.linspace(-6, 6, self.n_steps)
    #     betas = torch.sigmoid(betas).to(self.device)
    #     self.betas = betas * (0.5e-2 - 1e-5) + 1e-5  
    #     self.input_dim=input_dim
        
    #     self.alphas = 1 - self.betas			
    #     self.alphas_prod = torch.cumprod(self.alphas, 0).to(self.device)
    #     self.alphas_prod_p = torch.cat([torch.tensor([1]).float().cuda(), self.alphas_prod[:-1]], 0)  
    #     self.alphas_bar_sqrt = torch.sqrt(self.alphas_prod)
    #     self.one_minus_alphas_bar_log = torch.log(1 - self.alphas_prod)
    #     self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - self.alphas_prod)
    #     self.name = 'ddpm'
    
    # def forward(self, data):
    #     batch_size = data.shape[0]

    #     t = torch.randint(0, self.n_steps, size=(batch_size // 2,)).to(self.device)
    #     t = torch.cat([t, self.n_steps - 1 - t], dim=0)
    #     if (batch_size % 2) != 0:
    #         t = torch.randint(0, self.n_steps, size=(batch_size,)).to(self.device)     					 
    #     t = t.unsqueeze(-1)

    #     a = self.alphas_bar_sqrt[t]
    #     aml = self.one_minus_alphas_bar_sqrt[t]
    #     epsilon = torch.randn_like(data).to(self.device)

    #     xt = data * a + epsilon * aml
    #     output = self.diffuser(xt, t.squeeze(-1))
    #     loss = (epsilon - output).square().mean()
        
    #     return loss
    
    # def sample_one_step(self, x, t, eps_theta=None):
    #     coeff = self.betas[t] / self.one_minus_alphas_bar_sqrt[t]
    #     diffuser_t = torch.LongTensor([t]).to(self.device)
    #     if eps_theta is None:
    #         eps_theta = self.diffuser(x, diffuser_t)
    #     mean = (1 / (1 - self.betas[t]).sqrt()) * (x - (coeff * eps_theta))
    #     z = torch.randn_like(x)
    #     sigma_t = self.betas[t].sqrt()
    #     sample = mean + sigma_t * z
    #     return sample

    # def sample_wo_guidance(self, batch_size=400, n_samples=0, extra_step=0):            
    #     x = np.random.randn(n_samples, self.input_dim)

    #     for start in range(0, n_samples, batch_size):
    #         end = min(start + batch_size, batch_size)
    #         batch_x = torch.tensor(x[start:end], dtype=torch.float32).to(self.device)
    #         for i in range(self.n_steps-1, -extra_step, -1):
    #             t = max(i, 0)
    #             batch_x = self.sample_one_step(batch_x, t)

    #         x[start:end] = batch_x.cpu().detach().numpy()
        
    #     return x

    
    # def set_logger(self, logger):
    #     self.logger = logger
        
    
# class MLPDiffusion(nn.Module):					
#     def __init__(self, input_dim=13, n_steps=100, num_layers=3, hidden_dim=128):
#         super(MLPDiffusion, self).__init__()

#         self.num_layers = num_layers

#         self.linears = nn.ModuleList([nn.Linear(input_dim, hidden_dim), nn.LeakyReLU()])
#         for _ in range(num_layers - 1):
#             self.linears.append(nn.Linear(hidden_dim, hidden_dim))
#             self.linears.append(nn.LeakyReLU())
#         self.linears.append(nn.Linear(hidden_dim, input_dim))

#         self.embeddings = nn.ModuleList([nn.Embedding(n_steps, hidden_dim) for _ in range(num_layers)])
    
#         self._init_net()

#     def _init_net(self):
#         for layer in self.modules():
#             if isinstance(layer, torch.nn.Linear):
#                 # nn.init.orthogonal_(layer.weight)
#                 nn.init.kaiming_uniform_(layer.weight) # no difference 

#     def forward(self, x, t):
#         for idx, embedding_layer in enumerate(self.embeddings):	
#             t_embedding = embedding_layer(t)
#             x = self.linears[2 * idx](x)
#             x += t_embedding
#             x = self.linears[2 * idx + 1](x)
#         x = self.linears[-1](x)	
        				
#         return x



class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=1000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class MLPDiffusion(nn.Module):
    def __init__(self, d_in, dim_t=1024):
        super().__init__()
        self.dim_t = dim_t

        self.proj = nn.Linear(d_in, dim_t)

        self.mlp = nn.Sequential(
            nn.Linear(dim_t, dim_t * 2),
            nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t * 2),
            nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, d_in),
        )

        self.map_noise = PositionalEmbedding(num_channels=dim_t)
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t)
        )
    
    def forward(self, x, t, class_labels=None):
        emb = self.map_noise(t)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape) # swap sin/cos
        emb = self.time_embed(emb)
    
        x = self.proj(x) + emb
        return self.mlp(x)

