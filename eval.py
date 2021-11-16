import argparse
from pathlib import Path
from tqdm import tqdm
import torch

import config
from lib.dataset.codd import CODDAggDataset
from lib.dataset.kitti import KITTIOdometryDataset
from lib.dataset.datatransforms import VoxelSampling

from lib.models.fastreg import FastReg
from lib.models.baseline.icp import ICPModel, ICPPostModel
from lib.models.baseline.ransac_fpfh import FPFH_RANSAC
from lib.models.baseline.teaser_fpfh import FPFH_TEASER
from lib.utils.evaluation import overlap_ratio, registrationMetrics 

def eval_model(model, loader, save_path):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    tes, res, tnorms, overlaps, times = [], [], [], [], []
    for (pb,RtGT) in tqdm(loader, total=len(loader)):
        pb = pb.cuda()
        RtGT = RtGT.cuda()

        with torch.no_grad():
            #forward pass with timing
            start.record()
            R,t = model(pb)[:2]
            end.record()
            torch.cuda.synchronize()
            time = start.elapsed_time(end) / loader.batch_size # time in milisecs

            #compute registration metrics
            rotErr, transErr = registrationMetrics(RtGT, R, t)

            #compute overlap ratio
            overlap = torch.FloatTensor([overlap_ratio(pb[i], RtGT[i], config.VOXEL_SAMPLING_SIZE) for i in range(pb.shape[0])])

            #save stats
            res.append(rotErr)
            tes.append(transErr)
            tnorms.append(torch.linalg.norm(RtGT[:, :3, -1].view(-1, 3), dim=-1).view(-1))
            overlaps.append(overlap)
            times.append(time)

    print('Finished evaluation')
    rotErr = torch.cat(res)
    transErr = torch.cat(tes)
    tnorms = torch.cat(tnorms)
    overlaps = torch.cat(overlaps)
    times = torch.FloatTensor(times[2:]) #discards first 2 values due to noise
    print(f'Mean Translation Error {transErr.mean().item():.2f}m +- {transErr.std().item():.2f}')
    print(f'Mean Rotation Error {rotErr.mean().item():.2f}deg +- {rotErr.std().item():.2f}')
    print(f'Mean Execution Time per Sample {times.mean().item():.2f}ms +- {times.std().item():.2f}')
    evalData = {'rotErr': rotErr, 'transErr': transErr, 'tNorms': tnorms, 'overlaps': overlaps, 'times': times}
    torch.save(evalData, save_path)

def main(args):
    # create dataset and dataloader
    transform = VoxelSampling(config.VOXEL_SAMPLING_SIZE)
    if args.dataset == 'codd':
        dataset = CODDAggDataset(config.CODD_PATH, mode='test', transform=transform)
    elif args.dataset == 'kitti':
        dataset = KITTIOdometryDataset(config.KITTI_PATH, mode='test', minDist=10, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, pin_memory=True, drop_last=True, num_workers=args.batch_size, shuffle=False)

    # create model
    if args.model in ('fastreg', 'fastregicp'):
        model = FastReg(T=config.T)
    elif args.model == 'icp':
        model = ICPModel(30, 1e-8, 5)
    elif args.model == 'fpfh_ransac':
        model = FPFH_RANSAC(0.3, 1., 1000000, 10000)
    elif args.model == 'fpfh_teaser':
        model = FPFH_TEASER(voxel_size=0.3, max_iter=10000)

    # load checkpoint, if available
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint))

    model.eval()
    model = model.cuda()

    # adds ICP post-processing
    if args.model == 'fastregicp':
        model = ICPPostModel(model, 10, 1e-4, 0.5)

    # run evaluation
    Path('results/').mkdir(parents=True, exist_ok=True)
    eval_model(model, loader, f'results/{args.model}-{args.dataset}.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluates a registration model and saves the results into a file')
    parser.add_argument('model', choices=('fastreg','fastregicp','icp','fpfh_ransac','fpfh_teaser'), help='model to be evaluated')
    parser.add_argument('dataset', choices=('codd','kitti'), help='dataset used for evaluation')
    parser.add_argument('--checkpoint', type=str, help='path to model checkpoint')
    parser.add_argument('-bs', '--batch_size', default=10)
    args = parser.parse_args()
    main(args)
