import torch

import argparse
import warnings
import time
import numpy as np

from ddpm.ddpm import DDPM
from utils.latent_utils import get_input_generate, recover_data, split_num_cat_target
from discriminator.model import discriminator

warnings.filterwarnings('ignore')


def main(args):
    dataname = args.dataname
    device = args.device
    steps = args.steps
    save_path = args.save_path

    train_z, _, _, ckpt_path, disc_path, info, num_inverse, cat_inverse = get_input_generate(args)
    in_dim = train_z.shape[1] 

    mean = train_z.mean(0)

    model = DDPM(
        num_layers=args.ddpm_num_layers,
        input_dim=in_dim,
        hidden_dim=args.ddpm_hidden_dim,
        n_steps=args.ddpm_steps,
        diff_lr=args.ddpm_lr,
        device=args.device
    )

    model.load_state_dict(torch.load(f'{ckpt_path}/model.pt'))
    

    '''
        Generating samples    
    '''
    start_time = time.time()

    num_samples = train_z.shape[0]

    # x_next = model.universal_guided_sample()
    x_next = model.sample_wo_guidance(batch_size=args.sample_batch_size,
                                      n_samples=num_samples,)

    syn_data = x_next.astype(np.float32) * 2 + mean.to(device).detach().cpu().numpy()
    
    # syn_data = x_next.float().cpu().numpy()
    syn_num, syn_cat, syn_target = split_num_cat_target(syn_data, info, num_inverse, cat_inverse, args.device) 

    syn_df = recover_data(syn_num, syn_cat, syn_target, info)

    idx_name_mapping = info['idx_name_mapping']
    idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}

    syn_df.rename(columns = idx_name_mapping, inplace=True)
    syn_df.to_csv(save_path, index = False)
    
    end_time = time.time()
    print('Time:', end_time - start_time)

    print('Saving sampled data to {}'.format(save_path))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generation')

    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
    parser.add_argument('--epoch', type=int, default=None, help='Epoch.')
    parser.add_argument('--steps', type=int, default=None, help='Number of function evaluations.')

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'