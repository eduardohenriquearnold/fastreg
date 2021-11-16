import torch
import numpy as np

import open3d as o3d

class FPFH_RANSAC(torch.nn.Module):
    def __init__(self, voxel_size, max_dist, max_iter, max_val):
        super().__init__()

        self.voxel_size = voxel_size
        self.max_iter = max_iter
        self.max_val = max_val
        self.max_dist = max_dist

    def preprocess_point_cloud(self, p):
        '''p: numpy array [npts, 3]. returns downsampled pointcloud and its fpfh features'''

        voxel_size = self.voxel_size
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(p[:,:3])
        pcd_down = pcd.voxel_down_sample(voxel_size)
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0,
                                                 max_nn=30))
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0,
                                                 max_nn=100))
        return (pcd_down, pcd_fpfh)

    def forward(self, pb):
        '''pb [batch, 2, npts, 4]. returns the registration result from FPFH RANSAC model'''

        assert pb.size(0) == 1, 'batch size must be 1'

        #compute fpfh features
        p0, fp0 = self.preprocess_point_cloud(pb[0,0].cpu().numpy())
        p1, fp1 = self.preprocess_point_cloud(pb[0,1].cpu().numpy())

        #compute registration
        res = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(p0, p1, fp0, fp1, self.max_dist, ransac_n=3, criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(self.max_iter, self.max_val))
        Rt = res.transformation.astype(np.float32)
        R, t = Rt[:3,:3], Rt[:3, -1]
        R = torch.from_numpy(R).unsqueeze(0).cuda()
        t = torch.from_numpy(t).view(1,1,3).cuda()
        return R,t

