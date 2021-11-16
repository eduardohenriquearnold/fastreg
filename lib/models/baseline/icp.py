import torch
import open3d as o3d

from lib.utils.transforms import transformPointsBatch

class ICPModel(torch.nn.Module):
    '''ICP model from Open3D'''

    def __init__(self, maxiter, tolerance, maxDist, mode='po2po'):
        '''maxiter: max number of interations. tolerance: relative change in distance to stop criteria. maxDist: maximum distance allowed between correspondences'''
        super().__init__()
        self.maxDist = maxDist
        self.criteria = o3d.pipelines.registration.ICPConvergenceCriteria(relative_rmse=tolerance, max_iteration=maxiter)
        self.mode = mode

        if mode == 'po2po':
            self.estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        elif mode == 'po2pl':
            self.estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPlane()
        else:
            raise NotImplementedError

    def preprocess_point_cloud(self, p):
        '''p: numpy array [npts, 3]. returns downsampled pointcloud and its fpfh features'''

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(p[:,:3])

        #if point to plane we need to estimate normals
        if self.mode == 'po2pl':
            pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.6, max_nn=10))

        return pcd 

    def forward(self, pb):
        '''pb [batch, 2, npts, 4]. returns the registration result from FPFH RANSAC model'''

        assert pb.size(0) == 1, 'batch size must be 1'

        src, tgt = pb[0,0,:,:3], pb[0,1,:,:3]

        #remove empty points
        valid = (src[...,0] != 0) * (src[...,1] != 0) * (src[...,2] != 0)
        src = src[valid]
        valid = (tgt[...,0] != 0) * (tgt[...,1] != 0) * (tgt[...,2] != 0)
        tgt = tgt[valid]

        #create open3d point cloud objects
        p0 = self.preprocess_point_cloud(src.cpu().numpy())
        p1 = self.preprocess_point_cloud(tgt.cpu().numpy())

        #call registration pipeline
        res = o3d.pipelines.registration.registration_icp(p0, p1, self.maxDist, estimation_method=self.estimation_method, criteria=self.criteria)

        #return result
        Rt = res.transformation.astype('float32')
        R, t = Rt[:3,:3], Rt[:3, -1]
        R = torch.from_numpy(R).unsqueeze(0).cuda()
        t = torch.from_numpy(t).view(1,1,3).cuda()
        return R,t

class ICPPostModel(ICPModel):
    '''Uses ICP as a post-processing step of another model, given as input'''

    def __init__(self, preModel, maxiter, tolerance, maxDist):
        super().__init__(maxiter, tolerance, maxDist)
        self.preModel = preModel

    def forward(self, pb):
        '''Runs pb through original model, transform points and then runs ICP. Run batched original model and loops for ICP'''

        b, _, npts, _ = pb.shape

        #Compute R,t using Pre Model
        R,t = self.preModel(pb)[:2]
        Rt = torch.eye(4).unsqueeze(0).repeat(b,1,1).to(pb.device)
        Rt[:,:3,:3] = R
        Rt[:,:3,-1:] = t.transpose(1,2)

        #Apply transform to src points
        pb[:,0] = transformPointsBatch(Rt, pb[:,0])

        #Refine using ICP, loop for each sample in batch
        RtICP = torch.eye(4).unsqueeze(0).repeat(b,1,1).to(pb.device)
        for sidx in range(b):
            Ricp,ticp = super().forward(pb[sidx].unsqueeze(0))[:2]
            RtICP[sidx,:3,:3] = Ricp
            RtICP[sidx,:3,-1:] = ticp.transpose(1,2)

        #Obtain final matrix
        Rt = RtICP @ Rt
        return Rt[:,:3,:3], Rt[:,:3,-1:].view(-1,1,3)
