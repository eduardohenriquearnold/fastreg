import numpy as np
import open3d as o3d

from lib.utils.transforms import getTransform, getRandomTransform, transformPoints, transformPointsBatch

class RandomRotationTransform(object):
    '''To be used with CODDAgg dataset. Apply random pose transform to both point clouds and adjust the RtGT to provide the right GT pose between them'''

    def __init__(self, rmean=0, rsig=10):
        '''Random transformation parameters'''
        self.rmean = rmean
        self.rsig = rsig

    def __call__(self, pb, RtGT):
        '''Transform both point clouds in pb [2, npts, 4] with a random transform (rotation only) and adjust the RtGT[4,4] accordingly'''

        #get random transform for each of the point clouds
        randomRt = getRandomTransform(tmean=0, tsig=0, rmean=self.rmean, rsig=self.rsig, size=2)

        #transform them
        pb = transformPointsBatch(randomRt, pb)

        #update RtGT
        RtGT = randomRt[1] @ RtGT @ np.linalg.inv(randomRt[0])
        RtGT = RtGT.astype(np.float32)

        return pb, RtGT

class VoxelSampling(object):
    '''Voxelise and sample mean voxel point, which creates more uniform density of points in the point clouds'''
    
    def __init__(self, voxel_size):
        self.voxel_size = voxel_size

    def __call__(self, pb, RtGT):
        '''Voxelise both poin clouds in pb [2, npts, 4], padding the points'''

        for i in range(2):
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(pb[i,:,:3])
            pc.colors = o3d.utility.Vector3dVector(np.concatenate([pb[i,:,-1:], np.zeros((pb.shape[1],2))], axis=1))
            pcs = pc.voxel_down_sample(self.voxel_size)
            npts = len(pcs.points)
            pb[i,:npts,:3] = np.asarray(pcs.points)
            pb[i,:npts,3] = np.asarray(pcs.colors)[:,0]
            pb[i,npts:] = 0

        return pb, RtGT

class Compose(object):
    '''Compose transforms'''

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, pb, RtGT):
        for t in self.transforms:
            pb, RtGT = t(pb, RtGT)
        return pb, RtGT

class FullOverlapTransform(object):
    '''Makes the second point cloud in pb [2, npts,4] as the direct transformation of the first one such that all points have a match'''

    def __call__(self, pb, RtGT):

        #remove central points
        r = 16
        idx = np.linalg.norm(pb[0,:,:3], axis=-1) < r
        pb[0,idx] = 0 

        pb[1] = transformPoints(RtGT, pb[0])

        #shuffle points
        pb[0] = np.random.permutation(pb[0]) 
        pb[1] = np.random.permutation(pb[1]) 

        #add random noise to random points in first point cloud 
        n = 15000
        idx = np.random.permutation(np.arange(pb.shape[1]))[:n]
        pb[0,idx,:3] = 0#np.random.normal(loc=30, scale=0.2, size=(n,3))
        return pb, RtGT

class DropoutTransform(object):
    '''Drops a percentage of valid points from both point clouds with uniform probability. The dropped points are replaced with all zeros to keep input dimensionality.'''

    def __init__(self, ratio):
        '''Ratio of valid points to drop'''

        self.r = ratio

    def __call__(self, pts, RtGT):
        valid = (pts[...,0] != 0) * (pts[...,1] != 0) * (pts[...,2] != 0)
        numvalid = valid.sum(axis=1)
        numdrop = self.r * numvalid

        valididx = np.nonzero(valid[0])[0]
        np.random.shuffle(valididx)
        pts[0, valididx[:int(numdrop[0])]] = 0

        valididx = np.nonzero(valid[1])[0]
        np.random.shuffle(valididx)
        pts[1, valididx[:int(numdrop[1])]] = 0

        return pts, RtGT

class RemoveGroundTransform(object):
    '''Removes ground points based on normal vector'''
    def __init__(self, angleThresh):
        '''angleThresh (degrees) indicates the minimum angle of difference between normal vector and +Z vector not to be considered a ground point'''

        self.thresh = np.cos(np.radians(angleThresh))

    def __call__(self, pts, RtGT):
        '''pts [2, npts, 4] '''

        for i in [0, 1]:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts[i,:,:3])
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=10))
            projNormalZ = np.asarray(pcd.normals)[:,2]
            msk = np.abs(projNormalZ) > self.thresh
            pts[i, msk] = 0

        return pts, RtGT

class RangeImageTransform(object):
    '''Transforms point cloud into a range image with 5 channels [x,y,z,intensity,range]'''

    def __init__(self):
        '''h represents image height and w image width'''
        
        self.h = 64
        self.w = 800
        self.vfovup = np.radians(10) 
        self.vfovdown = np.radians(25) 

    def __call__(self, pts, RtGT):
        '''Transforms pair of input point clouds into pair of range image representations with shape [2,5,64,780]'''

        r = np.linalg.norm(pts[:,:,:3], axis=-1) + 1e-6
        u = 0.5 * (self.w - 1) * (1 - np.arctan2(pts[:,:,1], pts[:,:,0])/np.pi)
        v = (self.h - 1) * (1 - (np.arcsin(pts[:,:,2] / r) + self.vfovdown)/(self.vfovup + self.vfovdown))

        #rounds and clips
        u = np.clip(np.rint(u), 0, self.w - 1).astype(np.int32)
        v = np.clip(np.rint(v), 0, self.h - 1).astype(np.int32)

        #creates range images [2,5,h,w]. 5 channels are [x,y,z,intensity,r]
        img = np.zeros((2, 5, self.h, self.w))
        img[0, :4, v[0], u[0]] = pts[0,:,:4]
        img[1, :4, v[1], u[1]] = pts[1,:,:4]
        img[0, 4, v[0], u[0]] = r[0]
        img[1, 4, v[1], u[1]] = r[1]

        return img, RtGT
