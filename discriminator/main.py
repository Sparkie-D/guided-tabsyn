import torch
import torch.nn as nn
import pickle
import os
from tqdm import tqdm
from torch.utils.data.sampler import WeightedRandomSampler
from ddpm.ddpm import DDPM
from tabsyn.model import MLPDiffusion, Model
import argparse
from utils.latent_utils import get_input_generate_disc, get_input_generate
from utils.diffusion_utils import sample
from discriminator.model import discriminator
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class Trainer(object):
    def __init__(self, 
                 model,
                 pos_data,
                 neg_data, 
                 valid_pos,
                 valid_neg, 
                 logger,
                #  args,
                model_path,
                 device,
                 ) -> None:
        self.pos_data = pos_data
        self.neg_data = neg_data
        self.valid_pos = valid_pos
        self.valid_neg = valid_neg
        self.logger = logger
        self.device = device
        self.model = model
        # self.log_path = args.log_path
        # self.model_path = args.model_path
        self.model_path = model_path
        # self.save_model_epoch = args.save_model_epoch
        self.save_model_epoch = 10
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        
    def train_epoch(self, neg_loader, pos_loader):
        total_loss = 0
        n_batch = 0
        criterion = nn.CrossEntropyLoss()
        
        for i, (neg_data, pos_data) in enumerate(zip(neg_loader, pos_loader)):
            pos_data = pos_data.to(self.device).float()
            neg_data = neg_data.to(self.device).float()
            pos_pred = self.model(pos_data)
            neg_pred = self.model(neg_data)
            # outputs = self.model(torch.cat([pos_data, neg_data]))
            # labels = torch.cat([torch.ones_like(pos_data)[..., 0], torch.zeros_like(neg_data)[..., 0]]).to(pos_data).long()
            
            learner_loss = -torch.mean(torch.log(1- torch.sigmoid(neg_pred)))
            expert_loss = -torch.mean(torch.log(torch.sigmoid(pos_pred)))
            loss = learner_loss + expert_loss + self._gradient_penalty(neg_data, pos_data, LAMBDA=0)
            # loss = criterion(outputs, labels)
            
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            total_loss += loss.item()
            n_batch += 1
            
        return total_loss / n_batch
        
    def train(self, batch_size=32, num_epoch=1000):  
        save_path = os.path.join(self.model_path, f'discriminator.pickle')
        
        sampler = WeightedRandomSampler(weights=torch.ones(len(self.pos_data)), num_samples=batch_size, replacement=True)
        pos_loader = torch.utils.data.DataLoader(self.pos_data, batch_size=batch_size, sampler=sampler)
        neg_loader = torch.utils.data.DataLoader(self.neg_data, batch_size=batch_size, shuffle=True)             
        
        self.model.train()     
        for epoch in tqdm(range(num_epoch), desc='Training'):
            loss = self.train_epoch(neg_loader, pos_loader)
            
            self.logger.add_scalar('train/loss', loss, epoch)
            
            self.eval_epoch(batch_size, epoch)
            
            if self.save_model_epoch > 0 and (epoch + 1) % self.save_model_epoch == 0:
                with open(save_path, 'wb') as f:
                    pickle.dump(self.model.state_dict(), f)
                    # print(f'model saved at {save_path}')
        else:
            with open(save_path, 'wb') as f:
                pickle.dump(self.model.state_dict(), f)
                # print(f'model saved at {save_path}')

    def eval_epoch(self, batch_size, epoch):
        sampler = WeightedRandomSampler(weights=torch.ones(len(self.valid_pos)), num_samples=batch_size, replacement=True)
        pos_loader = torch.utils.data.DataLoader(self.valid_pos, batch_size=batch_size, sampler=sampler)
        neg_loader = torch.utils.data.DataLoader(self.valid_neg, batch_size=batch_size, shuffle=True) 
        pos_pred, neg_pred = [], []
        
        for i, (neg_data, pos_data) in enumerate(zip(neg_loader, pos_loader)):
            pos_data = pos_data.to(self.device).float()
            neg_data = neg_data.to(self.device).float()
            pos_pred.append(self.model(pos_data).squeeze())
            neg_pred.append(self.model(neg_data).squeeze())

        pos_pred = torch.sigmoid(torch.cat(pos_pred))
        neg_pred = torch.sigmoid(torch.cat(neg_pred))
        self.logger.add_histogram('eval/pos_prediction', pos_pred, epoch)
        self.logger.add_histogram('eval/neg_prediction', neg_pred, epoch)            

   
    def _gradient_penalty(self, real_data, generated_data, LAMBDA=10):
        batch_size = real_data.size()[0]

        # Calculate interpolationsubsampling_rate=20
        alpha = torch.rand(batch_size, 1).requires_grad_()
        alpha = alpha.expand_as(real_data).to(self.device)
        # print(alpha.shape, real_data.shape, generated_data.shape)
        interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data

        # Calculate probability of interpolated examples
        # print(self.device, self.energy_model.device)
        prob_interpolated = self.model(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size()).to(self.device),
                               create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)

        # Return gradient penalty
        return LAMBDA * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

def main(args):
    train_pos, valid_pos, info = get_input_generate_disc(args)
    x = get_input_generate(args)[0] # decoder in info
    mean = x.mean(0)
    in_dim = x.shape[1]
    
    print('Pre-sampling for discriminator training')
    if args.method == 'ddpm':
        model = DDPM(
            input_dim=in_dim,
            hidden_dim=args.ddpm_hidden_dim,
            n_steps=args.ddpm_steps,
            device=args.device
        )
        model.load_state_dict(torch.load(f'ddpm/ckpt/{args.dataname}/model.pt'))
        
        x_train = model.sample_wo_guidance(batch_size=args.sample_batch_size,
                                           n_samples=1000,)
        
        x_test = model.sample_wo_guidance(batch_size=args.sample_batch_size,
                                          n_samples=1000,)
    elif args.method == 'tabsyn':
        denoise_fn = MLPDiffusion(in_dim, 1024).to(args.device)
        model = Model(denoise_fn = denoise_fn, hid_dim = train_pos.shape[1]).to(args.device)
        model.load_state_dict(torch.load(f'tabsyn/ckpt/{args.dataname}/model.pt'))

        x_train = sample(model.denoise_fn_D, 20000, in_dim)
        # x_train = x_train * 2 + mean.to(args.device)
        x_train = x_train.float().detach().cpu().numpy()
        
        x_test = sample(model.denoise_fn_D, 1000, in_dim)
        # x_test = x_test * 2 + mean.to(args.device)
        x_test = x_test.float().detach().cpu().numpy()
    
    print('Samping finished.')    
    train_pos = (train_pos - mean.to(args.device).detach().cpu().numpy()) / 2.
    valid_pos = (valid_pos - mean.to(args.device).detach().cpu().numpy()) / 2.

    trainer = Trainer(
        model=discriminator(input_dim=in_dim, 
                            hidden_dim=args.disc_hidden_dim,
                            device=args.device),
        pos_data=train_pos,
        neg_data=x_train,
        valid_pos=valid_pos,
        valid_neg=x_test,
        device=args.device,
        logger=SummaryWriter(args.logdir),
        model_path=f'discriminator/ckpt/{args.method}/{args.dataname}'
    )
    trainer.train(batch_size=256, num_epoch=args.disc_epoch)

   