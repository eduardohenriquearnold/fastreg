import argparse
import torch
from torch.utils.tensorboard import SummaryWriter

import config
from lib.dataset.codd import CODDAggDataset
from lib.dataset.kitti import KITTIOdometryDataset
from lib.dataset.datatransforms import Compose, VoxelSampling, RandomRotationTransform

from lib.models.fastreg import FastReg
from lib.utils.evaluation import registrationMetrics

def executeEpoch(model, loader, opt, sched, e, sw, mode='train'):
    assert mode == 'train' or mode =='val', 'mode should be train or val'

    if mode == 'train':
        model.train()
    else:
        model.eval()

    lE, lpE, lnE = 0, 0, 0
    rotE, transE = 0, 0
    maxInliersE, actualInliersE = 0, 0

    for b, (pb,RtGT) in enumerate(loader):
        pb = pb.cuda()
        RtGT = RtGT.cuda()

        if mode == 'train':
            R, t, loss_pos, loss_neg, loss, maxInliers, actualInliers = model(pb, RtGT)
        else:
            with torch.no_grad():
                R, t, loss_pos, loss_neg, loss, maxInliers, actualInliers = model(pb, RtGT)

        lE += loss.detach().item()
        lpE += loss_pos.detach().item()
        lnE += loss_neg.detach().item()

        rotErr, transErr = registrationMetrics(RtGT, R, t)
        rotErr = rotErr.detach().median().item()
        transErr = transErr.detach().median().item()
        rotE += rotErr
        transE += transErr

        maxInliersE += maxInliers.item()
        actualInliersE += actualInliers.item()

        if mode == 'train':
            #optimise model
            loss.backward()
            opt.step()
            opt.zero_grad()
            print(f'E {e}/B {b}. Loss {loss.detach().item():.4f}. PLoss {loss_pos.detach().item():.4f}. NLoss {loss_neg.detach().item():.4f}. MRE {rotErr:.2f}, MTE {transErr:.2f}. Inliers {actualInliers.item()}/{maxInliers.item()}')

    #stats
    batches = len(loader)
    lE /= batches
    lpE /= batches
    lnE /= batches
    rotE /= batches
    transE /= batches
    maxInliersE /= batches
    actualInliersE /= batches
    print(f'{mode} {e}. Loss {lE:.4f}. PLoss {lpE:.4f}. NLoss {lnE:.4f}. MRE {rotE:.2f}. MTE {transE:.2f}. MInliers {actualInliersE:.1f}/{maxInliersE:.1f}')

    #update tensorboard
    sw.add_scalar(f'{mode}/loss', lE, e)
    sw.add_scalar(f'{mode}/loss_pos', lpE, e)
    sw.add_scalar(f'{mode}/loss_neg', lnE, e)
    sw.add_scalar(f'{mode}/rot_err', rotE, e)
    sw.add_scalar(f'{mode}/trans_err', transE, e)
    sw.add_scalar(f'{mode}/trans_err', transE, e)
    sw.add_scalar(f'{mode}/maxInliers', maxInliersE, e)
    sw.add_scalar(f'{mode}/actualInliers', actualInliersE, e)

    #update scheduler
    if mode == 'train':
        sched.step()

def train(args):
    if args.dataset == 'codd':
        trainDataset = CODDAggDataset(config.CODD_PATH, mode='train', transform=Compose([VoxelSampling(0.3), RandomRotationTransform(rsig=40)]))
        valDataset = CODDAggDataset(config.CODD_PATH, mode='val', transform=VoxelSampling(0.3))
    elif args.dataset == 'kitti':
        trainDataset = KITTIOdometryDataset(config.KITTI_PATH, mode='train', transform=Compose([VoxelSampling(0.3), RandomRotationTransform(rsig=40)]))
        valDataset = KITTIOdometryDataset(config.KITTI_PATH, mode='val', transform=VoxelSampling(0.3))

    trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=config.batch_size, pin_memory=True, drop_last=True, num_workers=config.batch_size, shuffle=True)
    valLoader= torch.utils.data.DataLoader(valDataset, batch_size=config.batch_size, pin_memory=True, drop_last=True, num_workers=config.batch_size)

    model = FastReg(config.T).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=1e-1, eps=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)

    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint))
        model = model.cuda()

    expPath = 'runs/'
    writer = SummaryWriter(expPath)

    for e in range(config.epochs):
        executeEpoch(model, trainLoader, opt, sched, e, writer, mode='train')

        if (e + 1) % config.val_period == 0:
            #run validation
            executeEpoch(model, valLoader, opt, sched, e, writer, mode='val')

            #saves model
            torch.save(model.state_dict(), f'{expPath}/model{e}.pth')

    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains FastReg registration model')
    parser.add_argument('dataset', choices=('codd','kitti'), help='dataset used for evaluation')
    parser.add_argument('--checkpoint', type=str, help='path to model checkpoint (continue training)')
    args = parser.parse_args()
    train(args)