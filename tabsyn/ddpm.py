import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

import sys
curPath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(curPath)

def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype)


class DDPM(nn.Module):
    def __init__(self, 
                 num_layers=5,
                 input_dim=1,
                 hidden_dim=1024,
                 n_steps=1000,
                 diff_lr=1e-3,
                 device=torch.device('cuda:0')) -> None:
        super().__init__()
        self.device=device
        self.diffuser = MLPDiffusion(input_dim=input_dim, n_steps=n_steps, num_layers=num_layers, hidden_dim=hidden_dim).to(self.device)
        self.diff_optimizer = torch.optim.Adam(self.diffuser.parameters(), lr=diff_lr)
        self.n_steps = n_steps
        # self.betas = cosine_beta_schedule(self.n_steps)
        betas = torch.linspace(-6, 6, self.n_steps)
        betas = torch.sigmoid(betas).to(self.device)
        self.betas = betas * (0.5e-2 - 1e-5) + 1e-5  
        self.input_dim=input_dim
        
        self.alphas = 1 - self.betas			
        self.alphas_prod = torch.cumprod(self.alphas, 0).to(self.device)
        self.alphas_prod_p = torch.cat([torch.tensor([1]).float().cuda(), self.alphas_prod[:-1]], 0)  
        self.alphas_bar_sqrt = torch.sqrt(self.alphas_prod)
        self.one_minus_alphas_bar_log = torch.log(1 - self.alphas_prod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - self.alphas_prod)
        self.name = 'ddpm'
    
    def update(self, data):
        batch_size = data.shape[0]

        t = torch.randint(0, self.n_steps, size=(batch_size // 2,)).to(self.device)
        t = torch.cat([t, self.n_steps - 1 - t], dim=0)
        if (batch_size % 2) != 0:
            t = torch.randint(0, self.n_steps, size=(batch_size,)).to(self.device)     					 
        t = t.unsqueeze(-1)

        a = self.alphas_bar_sqrt[t]
        aml = self.one_minus_alphas_bar_sqrt[t]
        epsilon = torch.randn_like(data).to(self.device)

        xt = data * a + epsilon * aml
        output = self.diffuser(xt, t.squeeze(-1))

        loss = (epsilon - output).square().mean()
        
        self.diff_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.diffuser.parameters(), 1.)		
        self.diff_optimizer.step()
        
        return loss.item()
    
    def sample_one_step(self, x, t, eps_theta=None):
        coeff = self.betas[t] / self.one_minus_alphas_bar_sqrt[t]
        diffuser_t = torch.LongTensor([t]).to(self.device)
        if eps_theta is None:
            eps_theta = self.diffuser(x, diffuser_t)
        mean = (1 / (1 - self.betas[t]).sqrt()) * (x - (coeff * eps_theta))
        z = torch.randn_like(x)
        sigma_t = self.betas[t].sqrt()
        sample = mean + sigma_t * z
        return sample

    def sample_wo_guidance(self, batch_size=400, n_samples=0, extra_step=100):            
        x = np.random.randn(n_samples, self.input_dim)

        for start in range(0, n_samples, batch_size):
            end = min(start+batch_size, batch_size)
            batch_x = torch.tensor(x[start:end], dtype=torch.float32).to(self.device)
            # for i in range(self.n_steps-1, -1):
            for i in range(self.n_steps-1, -extra_step, -1):
                t = max(i, 0)
                batch_x = self.sample_one_step(batch_x, t)

            x[start:end] = batch_x.cpu().detach().numpy()
        
        return x

    def forward_guidance(self, disc_model, z0, zt):
        preds = disc_model(z0)
        loss = -torch.sum(torch.log(torch.sigmoid(preds)))
        guidance = torch.autograd.grad(loss, zt)
        return torch.sqrt(torch.abs(guidance[0])) * torch.sign(guidance[0])
        # return guidance[0]
        
    def backward_guidance(self, disc_model, z0, m):
        delta = torch.zeros_like(z0, requires_grad=True)
        lr=1e-3
        for _ in range(m):
            output = disc_model(z0 + delta)
            loss = -torch.sum(torch.log(torch.sigmoid(output)))
            with torch.no_grad():
                delta -= lr * torch.autograd.grad(loss, delta)[0]
        
        return delta
    
    
    def universal_guided_sample_batch(self, batch_size=400, disc_model=None, forward_weight=1, backward_step=10, self_recurrent_step=10):
        zt = torch.randn((batch_size, self.input_dim), dtype=torch.float32).to(self.device).requires_grad_(True)
        for t in range(self.n_steps-1, 0, -1):
            for _ in range(self_recurrent_step):
                eps_theta = self.diffuser(zt, torch.LongTensor([t]).to(self.device))
                z0_hat = (zt - self.one_minus_alphas_bar_sqrt[t] * eps_theta) / self.alphas_bar_sqrt[t]
                eps_theta_hat = eps_theta + forward_weight * self.one_minus_alphas_bar_sqrt[t] * self.forward_guidance(disc_model, z0_hat, zt) # forward universal guidance
                if backward_step > 0:
                    eps_theta_hat = eps_theta_hat - self.alphas_bar_sqrt[t] / self.one_minus_alphas_bar_sqrt[t] * self.backward_guidance(disc_model, z0_hat, backward_step) # backward universal guidance
                zt_1 = self.sample_one_step(zt, t, eps_theta=eps_theta_hat)
                eps_hat = torch.randn_like(zt_1)
                w = self.alphas_bar_sqrt[t] / self.alphas_bar_sqrt[t-1]
                zt = w * zt_1 + torch.sqrt(1-torch.square(w)) * eps_hat
            zt = zt_1 if self_recurrent_step > 0 else self.sample_one_step(zt, t)
        return zt.detach().cpu().numpy()
    
    
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
    
    def set_logger(self, logger):
        self.logger = logger
        
    
class MLPDiffusion(nn.Module):					
    def __init__(self, input_dim=13, n_steps=100, num_layers=3, hidden_dim=128):
        super(MLPDiffusion, self).__init__()

        self.num_layers = num_layers

        self.linears = nn.ModuleList([nn.Linear(input_dim, hidden_dim), nn.LeakyReLU()])
        for _ in range(num_layers - 1):
            self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.LeakyReLU())
        self.linears.append(nn.Linear(hidden_dim, input_dim))

        self.embeddings = nn.ModuleList([nn.Embedding(n_steps, hidden_dim) for _ in range(num_layers)])
    
        self._init_net()

    def _init_net(self):
        for layer in self.modules():
            if isinstance(layer, torch.nn.Linear):
                nn.init.orthogonal_(layer.weight)
                # nn.init.kaiming_uniform_(layer.weight) # no difference 

    def forward(self, x, t):
        for idx, embedding_layer in enumerate(self.embeddings):	
            t_embedding = embedding_layer(t)
            x = self.linears[2 * idx](x)
            x += t_embedding
            x = self.linears[2 * idx + 1](x)
        x = self.linears[-1](x)	
        				
        return x

