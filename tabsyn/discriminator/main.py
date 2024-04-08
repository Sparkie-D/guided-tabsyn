import torch
import torch.nn as nn
import pickle
import os
from tqdm import tqdm
from torch.utils.data.sampler import WeightedRandomSampler
from tabsyn.ddpm import DDPM
import argparse
from tabsyn.latent_utils import get_input_generate_disc as get_input_generate
from tabsyn.discriminator.model import discriminator
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class Trainer(object):
    def __init__(self, 
                 model,
                 pos_data,
                 neg_data, 
                 valid_data, 
                 logger,
                #  args,
                model_path,
                 device,
                 ) -> None:
        self.pos_data = pos_data
        self.neg_data = neg_data
        self.valid_data = valid_data
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
        for i, (neg_data, pos_data) in enumerate(zip(neg_loader, pos_loader)):
            pos_data = pos_data.to(self.device)
            neg_data = neg_data.to(self.device)
            pos_pred = self.model(pos_data)
            neg_pred = self.model(neg_data)
            
            learner_loss = -torch.mean(torch.log(1 - torch.sigmoid(neg_pred)))
            expert_loss = -torch.mean(torch.log(torch.sigmoid(pos_pred)))
            
            loss = learner_loss + expert_loss + self._gradient_penalty(neg_data, pos_data, LAMBDA=0)
            # loss = learner_loss + expert_loss
            
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            total_loss += loss.item()
            n_batch += 1
            
        return total_loss / n_batch
            
        

    def train(self, batch_size=32, num_epoch=1000):  
        sampler = WeightedRandomSampler(weights=torch.ones(len(self.pos_data)), num_samples=batch_size, replacement=True)
        pos_loader = torch.utils.data.DataLoader(self.pos_data, batch_size=batch_size, sampler=sampler)
        neg_loader = torch.utils.data.DataLoader(self.neg_data, batch_size=batch_size, shuffle=True)             
        for epoch in tqdm(range(num_epoch), desc='Training'):
            loss = self.train_epoch(neg_loader, pos_loader)
            
            self.logger.add_scalar('train/loss', loss, epoch)
            
            self.eval_epoch(batch_size, epoch)
            
            if self.save_model_epoch > 0 and (epoch + 1) % self.save_model_epoch == 0:
                with open(os.path.join(self.model_path, f'discriminator.pickle'), 'wb') as f:
                    pickle.dump(self.model.state_dict(), f)

    def eval_epoch(self, batch_size, epoch):
        sampler = WeightedRandomSampler(weights=torch.ones(len(self.pos_data)), num_samples=batch_size, replacement=True)
        pos_loader = torch.utils.data.DataLoader(self.valid_data, batch_size=batch_size, sampler=sampler)
        neg_loader = torch.utils.data.DataLoader(self.neg_data, batch_size=batch_size, shuffle=True) 
        pos_pred, neg_pred = [], []
        
        for i, (neg_data, pos_data) in enumerate(zip(neg_loader, pos_loader)):
            pos_data = pos_data.to(self.device)
            neg_data = neg_data.to(self.device)
            pos_pred.append(self.model(pos_data).squeeze())
            neg_pred.append(self.model(neg_data).squeeze())
        
        pos_pred = torch.sigmoid(torch.cat(pos_pred))
        neg_pred = torch.sigmoid(torch.cat(neg_pred))
        # print(pos_pred, neg_pred)
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
    pos_data, _, _, ckpt_path, disc_path, info, num_inverse, cat_inverse = get_input_generate(args)
    in_dim = pos_data.shape[1]
    
    model = DDPM(
        num_layers=3,
        input_dim=in_dim,
        hidden_dim=1024,
        n_steps=1000,
        diff_lr=1e-3,
        device=args.device
    )
    
    model.load_state_dict(torch.load(f'tabsyn/ckpt/{args.dataname}/model.pt'))

    train_z, _, _, ckpt_path, disc_path, info, num_inverse, cat_inverse = get_input_generate(args)
    mean = train_z.mean(0)
    x_next = model.sample_wo_guidance(batch_size=256,
                                      n_samples=1024,)

    neg_data = x_next.astype(np.float32) * 2 + mean.to(args.device).detach().cpu().numpy()

    trainer = Trainer(
        model=discriminator(input_dim=train_z.shape[1], 
                            hidden_dims=4,
                            device=args.device),
        # model=model,
        pos_data=pos_data,
        neg_data=neg_data,
        valid_data=train_z,
        device=args.device,
        logger=SummaryWriter(args.logdir),
        model_path=f'tabsyn/discriminator/ckpt/{args.dataname}_fewshot'
    )
    trainer.train(batch_size=256,
                  num_epoch=100)
    
# if __name__ == '__main__':
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--dataname', type=str, default='pksim')
    # parser.add_argument('--method', type=str, default='discriminator')
    
    # args = parser.parse_args()
    # assert os.path.exists(os.path.join('data', args.dataname+'_fewshot')), f"The dataset [{args.dataname}] have no fewshot dataset yet."
    
    # args.logdir=os.path.join('logs', f'{args.dataname}', f'{args.method}')
    # if not os.path.exists(args.logdir):
    #     os.makedirs(args.logdir)
   