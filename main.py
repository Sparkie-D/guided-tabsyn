import torch
import os
from utils import execute_function, get_args

if __name__ == '__main__':
    args = get_args()
    if torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'

    if not args.save_path:
        args.save_path = f'synthetic/{args.dataname}/{args.method}.csv'
    
    args.logdir=os.path.join('logs', f'{args.dataname}', f'{args.method}')
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    
    main_fn = execute_function(args.method, args.mode)

    main_fn(args)