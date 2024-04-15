import torch
import os
from utils.utils import execute_function, get_args

if __name__ == '__main__':
    args = get_args()
    if torch.cuda.is_available() and args.gpu >= 0:
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'
    print(f"Using device : {args.device}")
    
    if not args.save_path:
        if not args.enable_guidance:
            args.save_path = f'synthetic/{args.dataname}/{args.method}.csv'
        else:
            args.save_path = f'synthetic/{args.dataname}/{args.method}_guided.csv'
        
    args.logdir=os.path.join('logs', f'{args.dataname}', f'{args.method}' if not args.enable_guidance else f'{args.method}_discriminator')
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    
    main_fn = execute_function(args.method, args.mode, args.enable_guidance)

    main_fn(args)