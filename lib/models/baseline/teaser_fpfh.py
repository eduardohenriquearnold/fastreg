import torch
import numpy as np

import open3d as o3d

try:
    import teaserpp_python
except:
    pass
from scipy.spatial import cKDTree


class FPFH_TEASER(torch.nn.Module):
    def __init__(self, voxel_size, max_iter):
        super().__init__()

        self.voxel_size = voxel_size
        self.max_iter = max_iter

        solver_params = teaserpp_python.RobustRegistrationSolver.Params()
        solver_params.cbar2 = 1.0
        solver_params.noise_bound = voxel_size
        solver_params.estimate_scaling = False
        solver_params.inlier_selection_mode = teaserpp_python.RobustRegistrationSolver.INLIER_SELECTION_MODE.PMC_EXACT
        solver_params.rotation_tim_graph = teaserpp_python.RobustRegistrationSolver.INLIER_GRAPH_FORMULATION.CHAIN
        solver_params.rotation_estimation_algorithm = (
            teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
        )
        solver_params.rotation_gnc_factor = 1.4
        solver_params.rotation_max_iterations = max_iter
        solver_params.rotation_cost_threshold = 1e-16
        self.solver_params = solver_params

    def pcd2xyz(self, pcd):
        return np.asarray(pcd.points).T

    def extract_fpfh(self, pcd):
        radius_normal = self.voxel_size * 2
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        radius_feature = self.voxel_size * 5
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
        )
        return np.array(fpfh.data).T

    def find_knn_cpu(self, feat0, feat1, knn=1, return_distance=False):
        feat1tree = cKDTree(feat1)
        dists, nn_inds = feat1tree.query(feat0, k=knn, n_jobs=-1)
        if return_distance:
            return nn_inds, dists
        else:
            return nn_inds

    def find_correspondences(self, feats0, feats1, mutual_filter=True):
        nns01 = self.find_knn_cpu(feats0, feats1, knn=1, return_distance=False)
        corres01_idx0 = np.arange(len(nns01))
        corres01_idx1 = nns01

        if not mutual_filter:
            return corres01_idx0, corres01_idx1

        nns10 = self.find_knn_cpu(feats1, feats0, knn=1, return_distance=False)
        corres10_idx1 = np.arange(len(nns10))
        corres10_idx0 = nns10

        mutual_filter = corres10_idx0[corres01_idx1] == corres01_idx0
        corres_idx0 = corres01_idx0[mutual_filter]
        corres_idx1 = corres01_idx1[mutual_filter]

        return corres_idx0, corres_idx1

    def forward(self, pb):
        """pb [batch, 2, npts, 4]. returns the registration result from FPFH RANSAC model"""

        assert pb.size(0) == 1, "batch size must be 1"

        # load point clouds into o3d format
        pb = pb[0].cpu().numpy()
        A_pcd_raw = o3d.geometry.PointCloud()
        A_pcd_raw.points = o3d.utility.Vector3dVector(pb[0, :, :3])
        B_pcd_raw = o3d.geometry.PointCloud()
        B_pcd_raw.points = o3d.utility.Vector3dVector(pb[1, :, :3])

        # voxel downsample both clouds
        A_pcd = A_pcd_raw.voxel_down_sample(voxel_size=self.voxel_size)
        B_pcd = B_pcd_raw.voxel_down_sample(voxel_size=self.voxel_size)
        A_xyz = self.pcd2xyz(A_pcd)  # np array of size 3 by N
        B_xyz = self.pcd2xyz(B_pcd)  # np array of size 3 by M

        # extract FPFH features
        A_feats = self.extract_fpfh(A_pcd)
        B_feats = self.extract_fpfh(B_pcd)

        # establish correspondences by nearest neighbour search in feature space
        corrs_A, corrs_B = self.find_correspondences(A_feats, B_feats, mutual_filter=True)
        A_corr = A_xyz[:, corrs_A]  # np array of size 3 by num_corrs
        B_corr = B_xyz[:, corrs_B]  # np array of size 3 by num_corrs

        # robust global registration using TEASER++
        solver = teaserpp_python.RobustRegistrationSolver(self.solver_params)
        solver.solve(A_corr, B_corr)
        solution = solver.getSolution()
        R, t = np.copy(solution.rotation), np.copy(solution.translation)
        R = torch.from_numpy(R).unsqueeze(0).float().cuda()
        t = torch.from_numpy(t).view(1, 1, 3).float().cuda()
        return R, t
