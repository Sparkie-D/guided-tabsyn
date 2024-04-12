import os
import torch

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import warnings
import time

from tqdm import tqdm
# from tabsyn.model import MLPDiffusion, Model
from ddpm.ddpm import DDPM
from utils.latent_utils import get_input_train
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings('ignore')


def main(args): 
    device = args.device

    train_z, _, _, ckpt_path, _ = get_input_train(args)

    print(ckpt_path)
    logger = SummaryWriter(args.logdir)

    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    in_dim = train_z.shape[1] 

    mean, std = train_z.mean(0), train_z.std(0)

    train_z = (train_z - mean) / 2
    train_data = train_z


    batch_size = 4096
    train_loader = DataLoader(
        train_data,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 4,
    )

    num_epochs = args.ddpm_epoch

    # denoise_fn = MLPDiffusion(in_dim, 1024).to(device)
    # print(denoise_fn)
    
    model = DDPM(
        input_dim=in_dim,
        hidden_dim=args.ddpm_hidden_dim,
        n_steps=args.ddpm_steps,
        device=args.device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=20, verbose=True)

    model.train()

    best_loss = float('inf')
    patience = 0
    start_time = time.time()
    with tqdm(total=num_epochs) as pbar:
        pbar.set_description(f"Training {args.method}")

        for epoch in range(num_epochs):

            batch_loss = 0.0
            len_input = 0
            for batch in train_loader:
                inputs = batch.float().to(device)
                loss = model(inputs)
            
                loss = loss.mean()

                batch_loss += loss.item() * len(inputs)
                len_input += len(inputs)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_postfix({"Loss": loss.item()})

            curr_loss = batch_loss/len_input
            scheduler.step(curr_loss)

            if curr_loss < best_loss:
                best_loss = loss.item()
                patience = 0
                torch.save(model.state_dict(), f'{ckpt_path}/model.pt')
            else:
                patience += 1
                if patience == 2000:
                    print('Early stopping')
                    break

            if epoch % 1000 == 0:
                torch.save(model.state_dict(), f'{ckpt_path}/model_{epoch}.pt')

            pbar.update(1)

    end_time = time.time()
    print('Time: ', end_time - start_time)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training of TabSyn')

    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'