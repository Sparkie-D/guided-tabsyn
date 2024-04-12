import matplotlib.pyplot as plt
import pandas as pd
import argparse
import seaborn as sns
import os
import numpy as np

def visual_correlation(data:dict, savefig=True, name='correlations'):
    # plot correlations in pair
    fig, axes = plt.subplots(nrows=2, ncols=len(data.keys()), figsize=(10*len(data.keys()), 10))
    fig.suptitle("Correlations")
    for i, key in enumerate(data.keys()):
        sns.heatmap(data=data[key][0].corr(min_periods=1), ax=axes[0, i], annot=False, vmin=0, vmax=1, xticklabels=False, yticklabels=False)
        axes[0, i].set_title(f'{key} Pretrain Data')
        sns.heatmap(data=data[key][1].corr(min_periods=1), ax=axes[1, i], annot=False, vmin=0, vmax=1, xticklabels=False, yticklabels=False)
        axes[1, i].set_title(f'{key} Finetune Data')
    if savefig:
        plt.savefig(f'{name}.png')
        plt.close()
    
def visual_correlation_row(data:dict, savefig=True, path='correlations', enable_labels=False):
    # plot given correlations in one row
    fig, axes = plt.subplots(nrows=1, ncols=len(data.keys()), figsize=(10*len(data.keys()), 10))
    fig.suptitle("Correlations")
    for i, key in enumerate(data.keys()):
        sns.heatmap(data=data[key].corr(min_periods=1), ax=axes[i], annot=False, vmin=0, vmax=1, xticklabels=enable_labels, yticklabels=enable_labels)
        axes[i].set_title(f'{key}')
    if savefig:
        plt.savefig(f'{path}')
        plt.close()
    return fig, axes

def get_range(data):
    mins, maxs = [[] for _ in range(len(data.keys()))], [[] for _ in range(len(data.keys()))]
    for i, key in enumerate(data.keys()):
        cur_data = data[key]
        for name in cur_data.columns:
            mins[i].append(cur_data[name].min())
            maxs[i].append(cur_data[name].max())
    mins, maxs = np.array(mins), np.array(maxs)
    return mins.min(axis=0), maxs.max(axis=0)

def visual_distribution(data:dict, savefig=True, path='distributions', cat_cols=[], enable_labels=False):
    n_cols = len(data.keys())
    n_rows = data[list(data.keys())[0]].shape[1]
    num_cols = [col for col in data[list(data.keys())[0]].columns if col not in cat_cols]
    num_data = {key:data[key][num_cols] for key in data.keys()}
    xmin, xmax = get_range(num_data)
    
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(10*n_cols, 10*n_rows))
    fig.suptitle('Distributions')
    for i, key in enumerate(num_data.keys()):
        cur_data = num_data[key]
        for j, name in enumerate(cur_data.columns):
            axes[j, i].hist(cur_data[name])
            axes[j, i].set_title(f'{key}-{name}')
            axes[j, i].set_xlim(xmin[j], xmax[j])
            
    if len(cat_cols) > 0:
        cat_data = {key:data[key][cat_cols].astype(str) for key in data.keys()}
        unique_values = {col:list(set(sum([cat_data[key][col].unique().tolist() for key in cat_data.keys()], []))) for col in cat_cols}
        for i, key in enumerate(cat_data.keys()):
            cur_data = cat_data[key]
            for j, name in enumerate(cur_data.columns):
                axes[j+len(num_cols), i].hist(sorted(cur_data[name].values.tolist()))
                axes[j+len(num_cols), i].set_title(f'{key}-{name}')        
    
    plt.subplots_adjust(top=1)
    plt.tight_layout()
    if savefig:
        plt.savefig(f'{path}')
        plt.close()
    return fig, axes
 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', type=str, default='pksim')
    parser.add_argument('--method', type=str, default='ddpm')
    parser.add_argument('--enable_guidance', action='store_true')
    
    args = parser.parse_args()
    args.method = args.method if not args.enable_guidance else args.method+"_guided"
    args.save_dir = os.path.join('eval', 'distribution', args.dataname, args.method)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    raw = pd.read_csv(os.path.join('synthetic', args.dataname, f'real.csv'))
    syn = pd.read_csv(os.path.join('synthetic', args.dataname, f'{args.method}.csv'))
    
    visual_dict = {
        'Original':raw[raw.columns],
        'synthetic':syn[raw.columns],
    }
    if 'guided' in args.method:
        fewshot = pd.read_csv(os.path.join('data', f'{args.dataname}_fewshot', f'{args.dataname}_fewshot.csv'))
        fewshot_all = pd.read_csv(os.path.join('data', f'{args.dataname}_fewshot', f'{args.dataname}_fewshot_all.csv'))
        visual_dict['Fewshot'] = fewshot[raw.columns]
        visual_dict['Fewshot All'] = fewshot_all[raw.columns]
    
    cat_cols = [col for col in raw.columns if raw[col].dtype == 'object']

    visual_distribution(visual_dict, 
                        path=os.path.join(args.save_dir, 'distributions.png'), 
                        cat_cols=cat_cols, 
                        enable_labels=True)