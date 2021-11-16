from pathlib import Path

import torch
import matplotlib.pyplot as plt

from lib.utils.evaluation import ecdf

def resultsTable(transErr, rotErr, transNorm, overlaps, times):
    overlap_ratio_ranges = [0.6, 0.5, 0.4, 0] # translation ranges (meters)
    rot_err_threshold = 5 # degrees
    trans_err_threshold = 0.6 # meters

    for min_overlap in overlap_ratio_ranges:
        msk = overlaps > min_overlap
        sTE, mTE = torch.std_mean(transErr[msk])
        sRE, mRE = torch.std_mean(rotErr[msk])
        recall = torch.sum((transErr[msk]<trans_err_threshold)*(rotErr[msk]<rot_err_threshold))/torch.sum(msk)
        print(f'For overlap_ratio > {min_overlap:.1f}. MTE: {mTE.item():.3f} +- {sTE.item():.2f} m. MRE: {mRE.item():.3f} +- {sRE.item():.2f} deg. Recall {recall.item():.3f}')
    print(f'Inference time: {times.mean():.2f} +- {times.std():.2f} ms')

def plot(dataset, results_path):
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, sharey=True, figsize=(7.0,2.0))
    for path in results_path:
        label = path.split('/')[-1].split('-')[0]
        stat = torch.load(path)
        ax1.plot(*ecdf(stat['transErr']), label=label)
        ax2.plot(*ecdf(stat['rotErr']), label=label)
        ax3.plot(*ecdf(stat['times']/1000), label=label)
        print(f'Model {label}:')
        resultsTable(stat['transErr'], stat['rotErr'], stat['tNorms'], stat['overlaps'], stat['times'])

    #axis
    ax1.set_xlim([0,1.0])
    ax1.set_xticks(torch.arange(0,1.25,0.25).numpy())
    ax2.set_xlim([0,3])
    ax2.set_xticks(torch.arange(0,4,1).numpy())
    ax3.set_xscale('log')

    #labels
    ax1.set_ylabel('Cumulative Density')
    ax1.set_xlabel('Translation Error (m)')
    ax2.set_xlabel('Rotation Error (deg)')
    ax3.set_xlabel('Sample Execution Time (s)')

    #grids
    ax1.grid()
    ax2.grid()
    ax3.grid()

    #legend
    plt.title('{dataset} dataset')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.37, top=0.99)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower left', ncol=3, borderaxespad=0, frameon=False, bbox_to_anchor=(0.07, -0.03, 0.9, 0.1), mode='expand')
    plt.savefig(f'results/plots-{dataset}.png')

if __name__ == '__main__':
    # load all results in folder
    res_path = Path('results/')
    res_kitti = map(str, res_path.glob('*-kitti.pth'))
    res_codd = map(str, res_path.glob('*-codd.pth'))

    # plot results for each dataset 
    print('Results KITTI')
    print('=======================================')
    plot('kitti', res_kitti)

    print('Results CODD')
    print('=======================================')
    plot('codd', res_codd)
