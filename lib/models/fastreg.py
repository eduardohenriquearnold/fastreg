import torch
import torch.nn.functional as F

from lib.pointnet2.pointnet2_utils import three_nn
from lib.pointnet2.pytorch_utils import SharedMLP

from lib.models.attention import GNNAttention
from lib.models.encoder import Encoder
from lib.utils.transforms import transformPoints,transformPointsBatch

import open3d as o3d
import numpy as np

class _FastRegbase(torch.nn.Module):
    '''Base-class for registration network. Requires a point-wise encoder and a GNN to propagate point features. '''

    def __init__(self, T):
        super().__init__()

        self.debug = False
        self.T = T

    def encoder(self, pts):
        '''Encoder must accept pts [batch*2, npts, 4] and return [batch*2, npts, D] feature vectors.'''
        raise NotImplementedError()

    def graphNet(self, xyz0, xyz1, f0, f1):
        '''Computes graph attention. Must accept xyz[batch,npts,3] and f[batch,npts,D] and return new features f0a and f1a with same dimensions'''
        raise NotImplementedError()

    def forward(self, pb, RtGT=None):
        '''Takes pb [batch, 2, npts', 4] on CUDA device and outputs R [batch,3,3], t [batch,1,3].
        If RtGT [batch, 4,4] is given, also outputs the loss and stats
        Where R,t are estimated rotation matrices and translation vectors and RtGT are the ground-truth extended transformation matrix that transforms xyz0 into xyz1.'''
        assert len(pb.shape) == 4, 'pb should have 4 dimensions: [batch, 2, npts, 4]'
        assert pb.size(1) == 2, 'pb dim 1 should have size 2'
        assert pb.size(3) == 4, 'pb dim 1 should have size 4'

        batchSize = pb.size(0)
        pb = pb.view(2*batchSize, pb.size(2), pb.size(3)).cuda() # [2*B, npts, 4]

        #Get point-wise features through encoder
        xyz, f = self.encoder(pb)
        xyz0, xyz1 = xyz[0::2], xyz[1::2] # [B,npts,3] note npts < npts' (original pcl size)
        f0, f1 = f[0::2], f[1::2]         # [B,npts,D]
        npts = xyz0.size(1)

        #runs graphNet
        f0, f1 = self.graphNet(xyz0, xyz1, f0, f1)

        #normalise features in unit sphere R^D
        f0 = f0/(torch.linalg.norm(f0, dim=-1).unsqueeze(-1) + 1e-6)
        f1 = f1/(torch.linalg.norm(f1, dim=-1).unsqueeze(-1) + 1e-6)

        #obtain soft-mapping from p0 in p1 through feature space
        smap = F.softmax(torch.bmm(f0, f1.transpose(1,2))/self.T, dim=-1)  # [B, npts, npts]

        # get points in xyz1 with highest score for each point in xyz0
        idxmax = torch.argmax(smap, dim=2)
        sxyz1 = xyz1[torch.arange(batchSize).repeat_interleave(npts).to(pb.device), idxmax.view(-1)].view(batchSize, npts, 3)

        # get transformation parameters using RANSAC
        R, t, inliers = [], [], []
        corrIdx = o3d.utility.Vector2iVector(torch.arange(xyz0.size(1)).reshape(-1,1).repeat(1,2).int().numpy())
        ransacCriteria = o3d.pipelines.registration.RANSACConvergenceCriteria(max_iteration=100000, confidence=0.999)
        checker = o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(0.5)
        for i in range(batchSize):
            pc0 = o3d.geometry.PointCloud()
            pc0.points = o3d.utility.Vector3dVector(xyz0[i].detach().cpu().numpy())
            pc1 = o3d.geometry.PointCloud()
            pc1.points = o3d.utility.Vector3dVector(sxyz1[i].detach().cpu().numpy())
            regResult = o3d.pipelines.registration.registration_ransac_based_on_correspondence(pc0, pc1, corrIdx, 0.5, criteria=ransacCriteria, ransac_n=3, checkers=[checker])
            R.append(regResult.transformation[:3,:3].copy())
            t.append(regResult.transformation[:3,-1].copy())
            inliers.append(np.asarray(regResult.correspondence_set)[:, 0])
        R = torch.cat([torch.from_numpy(r).reshape(1,3,3) for r in R], dim=0).float().cuda()
        t = torch.cat([torch.from_numpy(t).reshape(1,1,3) for t in t], dim=0).float().cuda()

        if self.debug:
            self.debugVis(pb, xyz0, sxyz1, RtGT, inliers)

        # compute loss
        if RtGT is not None:
            maxCorrDistance = 1.6 # meters
            xyz0t = transformPointsBatch(RtGT, xyz0)
            dist, idx = three_nn(xyz0t.contiguous(), xyz1.contiguous()) 
            dist = dist[..., 0]  # [B, npts]
            idx = idx[..., 0]

            hasCorr = dist < maxCorrDistance
            idxCorr = idx[hasCorr].long()
            bidx, pidx1 = torch.nonzero(hasCorr).T
            mask_pos = torch.zeros_like(smap)
            mask_neg = torch.zeros_like(smap)
            mask_neg[hasCorr] = 1.
            mask_neg[bidx.long(), pidx1.long(), idxCorr] = 0.
            mask_pos[bidx.long(), pidx1.long(), idxCorr] = 1.

            loss_pos = -(smap * mask_pos).sum() / mask_pos.sum() # we want to maximise smap at mask_pos
            loss_neg = (smap * mask_neg).sum() / mask_neg.sum()
            loss = loss_pos + 10 * loss_neg

            maxInliers = hasCorr.sum() # maximum number of inliers in batch (independent of how good matches are, based on GT)
            actualInliers = (idxmax == idx)[hasCorr].sum() # number of correct inliers (according to the model)

            return R, t, loss_pos, loss_neg, loss, maxInliers, actualInliers

        return R, t

    def debugVis(self, pb, Bxyz0, Bxyz1, BRtGT, Binliers):
        '''Show point clouds and lines between top-K correspondences, with line colour indicating the correspondence weight'''

        from lib.utils.visualisation import plotPCL, plotLinesWithWeights
        from mayavi import mlab

        for b in range(Bxyz0.size(0)):
            #transform all pts to pcl1 CS
            pcl0 = transformPoints(BRtGT[b], pb[2*b, :, :3])
            xyz0 = transformPoints(BRtGT[b], Bxyz0[b])
            pcl1 = pb[2*b + 1, :, :3]
            xyz1 = Bxyz1[b]
            inliers = Binliers[b]

            # whole point clouds in red and blue
            fig,_ = plotPCL(pcl0, (1,0,0))
            plotPCL(pcl1, (0,0,1), fig=fig)

            # sampled coordinates
            plotPCL(xyz0, (0.5,0.5,0), fig=fig)
            plotPCL(xyz1, (0,0.5,0.5), fig=fig)

            # plot lines between corr inliers
            xyz0in = xyz0[inliers]
            xyz1in = xyz1[inliers]
            print(torch.linalg.norm(xyz0in - xyz1in, dim=-1))
            color = torch.ones(xyz0in.shape[0], 1)
            plotLinesWithWeights(xyz0in, xyz1in, color)
            mlab.show()
            
class FastReg(_FastRegbase):
    def __init__(self, T):
        super().__init__(T)

        self.enc = Encoder() #output feature dim: 128
        self.gnn = GNNAttention(dim=128, k=32)

    def encoder(self, pts):
        return self.enc(pts)

    def graphNet(self, xyz0, xyz1, f0, f1):
        return self.gnn(xyz0, xyz1, f0, f1)
