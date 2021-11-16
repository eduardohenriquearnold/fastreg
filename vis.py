import argparse
from pathlib import Path

from tqdm import tqdm
import torch
from mayavi import mlab

import config
from lib.dataset.codd import CODDAggDataset
from lib.dataset.kitti import KITTIOdometryDataset
from lib.dataset.datatransforms import VoxelSampling

from lib.models.fastreg import FastReg
from lib.models.baseline.icp import ICPModel, ICPPostModel
from lib.models.baseline.ransac_fpfh import FPFH_RANSAC
from lib.models.baseline.teaser_fpfh import FPFH_TEASER
from lib.utils.evaluation import registrationMetrics 
from lib.utils.transforms import transformPoints
from lib.utils.visualisation import plotPCL

def change_viewport(scale=1):
    engine = mlab.get_engine()
    scene = engine.scenes[0]
    scene.scene.camera.position = [20.79690843496835/scale, 129.1599054406037/scale, 164.5706258484855/scale]
    scene.scene.camera.focal_point = [12.726333618164062, -1.1529998779296875, 7.12680721282959]
    scene.scene.camera.view_angle = 30.0
    scene.scene.camera.view_up = [-0.00867483185325536, -0.7701118480340653, 0.6378498952025272]
    scene.scene.camera.clipping_range = [53.46434895139427, 395.1983662760175]
    scene.scene.camera.compute_view_plane_normal()
    scene.scene.render()

def vis_model(model, dataset, frame):

    pb, RtGT = dataset[frame]
    pb = torch.from_numpy(pb).unsqueeze(0).cuda()
    RtGT = torch.from_numpy(RtGT).unsqueeze(0).cuda()
    tnorm = torch.linalg.norm(RtGT[0,:3,-1]).item()

    #forward pass
    with torch.no_grad():
        R,t = model(pb)[:2]
        Rt = torch.eye(4).unsqueeze(0).cuda()
        Rt[:,:3,:3] = R
        Rt[:,:3,-1:] = t.transpose(1,2)

    #compute registration metrics
    rotErr, transErr = registrationMetrics(RtGT, R, t)

    #align pcls
    pbt = pb.clone()
    pbt[0,0] = transformPoints(Rt[0], pbt[0,0])
    pbt_gt = pb.clone()
    pbt_gt[0,0] = transformPoints(RtGT[0], pbt_gt[0,0])

    #plot pcls (blu)
    fig, _ = plotPCL(pbt[0,0,:,:3], color=(1,0,0))  # red -> source transformed by model prediction
    fig, _ = plotPCL(pbt[0,1,:,:3], color=(0,0,1), fig=fig) # blue -> target
    fig, _ = plotPCL(pbt_gt[0,0,:,:3], color=(0,1,0), fig=fig) # green -> source transformed by GT

    change_viewport()
    mlab.show()

    print(f'Frame {frame}. ||t_gt|| = {tnorm:.2f}')
    print(f'Translation Error {transErr.item():.2f}m')
    print(f'Rotation Error {rotErr.item():.2f}deg')

def main(args):
    # create dataset and dataloader
    transform = VoxelSampling(config.VOXEL_SAMPLING_SIZE)
    if args.dataset == 'codd':
        dataset = CODDAggDataset(config.CODD_PATH, mode='test', transform=transform)
    elif args.dataset == 'kitti':
        dataset = KITTIOdometryDataset(config.KITTI_PATH, mode='test', minDist=10, transform=transform)

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

    # run visualisation 
    for frame in args.frames:
        vis_model(model, dataset, frame)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualise registration model results')
    parser.add_argument('model', choices=('fastreg','fastregicp','icp','fpfh_ransac','fpfh_teaser'), help='model to be evaluated')
    parser.add_argument('dataset', choices=('codd','kitti'), help='dataset used for evaluation')
    parser.add_argument('--checkpoint', type=str, help='path to model checkpoint')
    parser.add_argument('--frames', type=str, default='43,522,901')
    args = parser.parse_args()
    args.frames = map(int, args.frames.split(','))
    main(args)
