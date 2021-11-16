# Adapted from the code of Chris Choy and Wei Dong in https://github.com/chrischoy/DeepGlobalRegistration/blob/master/dataloader/kitti_loader.py

import os
import glob

import torch
import numpy as np
import open3d as o3d

kitti_cache = {}
kitti_icp_cache = {}

class KITTIOdometryDataset(torch.utils.data.Dataset):
    def __init__(self, path, mode, minDist=10, transform=None):
        super().__init__()

        self.root = path
        self.transform = transform

        max_time_diff = 3
        drive_ids = {'train': [0,1,2,3,4,5], 'val': [6,7], 'test' : [8,9,10]}
        assert mode in drive_ids.keys(), 'Invalid dataset mode, please choose from train, val, test'

        #creates list of samples 
        self.samples = []

        if minDist:
            #samples with translation norm above minDist
            for drive_id in drive_ids[mode]:
                inames = sorted(self.get_all_scan_ids(drive_id))
                all_odo = self.get_video_odometry(drive_id, return_all=True)
                all_pos = np.array([self.odometry_to_positions(odo) for odo in all_odo])
                Ts = all_pos[:, :3, 3]
                pdist = (Ts.reshape(1, -1, 3) - Ts.reshape(-1, 1, 3))**2
                pdist = np.sqrt(pdist.sum(-1))
                more_than_minDist= pdist > minDist
                curr_time = inames[0]
                while curr_time in inames:
                    # Find the min index
                    next_time = np.where(more_than_minDist[curr_time][curr_time:curr_time + 100])[0]
                    if len(next_time) == 0:
                        curr_time += 1
                    else:
                        # Follow https://github.com/yewzijian/3DFeatNet/blob/master/scripts_data_processing/kitti/process_kitti_data.m#L44
                        next_time = next_time[0] + curr_time - 1

                    if next_time in inames:
                        self.samples.append((drive_id, curr_time, next_time))
                        #  print(pdist[curr_time,next_time])
                        curr_time = next_time + 1

        else:
            #include all consecutive samples (within max_time_diff of 3)
            for drive_id in drive_ids[mode]:
                inames = self.get_all_scan_ids(drive_id)
                for start_time in inames:
                    for time_diff in range(2, max_time_diff):
                        pair_time = time_diff + start_time
                        if pair_time in inames:
                            self.samples.append((drive_id, start_time, pair_time))

        # Remove problematic sequence
        for item in [(8, 15, 58),]:
            if item in self.samples:
                self.samples.pop(self.samples.index(item))

        #set velo2cam transformation
        R = np.array([
        7.533745e-03, -9.999714e-01, -6.166020e-04, 1.480249e-02, 7.280733e-04,
        -9.998902e-01, 9.998621e-01, 7.523790e-03, 1.480755e-02
        ]).reshape(3, 3)
        T = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01]).reshape(3, 1)
        velo2cam = np.hstack([R, T])
        self.velo2cam = np.vstack((velo2cam, [0, 0, 0, 1])).T
 
    def __len__(self):
        return len(self.samples)

    def get_all_scan_ids(self, drive_id):
        fnames = glob.glob(self.root + '/sequences/%02d/velodyne/*.bin' % drive_id)
        assert len(fnames) > 0, f"Make sure that the path {self.root} has drive id: {drive_id}"
        inames = [int(os.path.split(fname)[-1][:-4]) for fname in fnames]
        return inames

    def get_video_odometry(self, drive, indices=None, ext='.txt', return_all=False):
        data_path = self.root + '/poses/%02d.txt' % drive
        if data_path not in kitti_cache:
            kitti_cache[data_path] = np.genfromtxt(data_path)
        if return_all:
            return kitti_cache[data_path]
        else:
            return kitti_cache[data_path][indices]

    def odometry_to_positions(self, odometry):
        T_w_cam0 = odometry.reshape(3, 4)
        T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
        return T_w_cam0

    def rot3d(self, axis, angle):
        ei = np.ones(3, dtype='bool')
        ei[axis] = 0
        i = np.nonzero(ei)[0]
        m = np.eye(3)
        c, s = np.cos(angle), np.sin(angle)
        m[i[0], i[0]] = c
        m[i[0], i[1]] = -s
        m[i[1], i[0]] = s
        m[i[1], i[1]] = c
        return m

    def pos_transform(self, pos):
        x, y, z, rx, ry, rz, _ = pos[0]
        RT = np.eye(4)
        RT[:3, :3] = np.dot(np.dot(self.rot3d(0, rx), self.rot3d(1, ry)), self.rot3d(2, rz))
        RT[:3, 3] = [x, y, z]
        return RT

    def get_position_transform(self, pos0, pos1, invert=False):
        T0 = self.pos_transform(pos0)
        T1 = self.pos_transform(pos1)
        return (np.dot(T1, np.linalg.inv(T0)).T if not invert else np.dot(
            np.linalg.inv(T1), T0).T)

    def _get_velodyne_fn(self, drive, t):
        fname = self.root + '/sequences/%02d/velodyne/%06d.bin' % (drive, t)
        return fname    

    def apply_transform(self, pts, trans):
        R = trans[:3, :3]
        T = trans[:3, 3]
        pts = pts @ R.T + T
        return pts 

    def _icp(self, xyz0, xyz1, M):
        #apply estimated transform from GPS
        xyz0t = self.apply_transform(xyz0, M)

        #down sample both point clouds
        pc0 = o3d.geometry.PointCloud()
        pc0.points = o3d.utility.Vector3dVector(xyz0t)
        pc0 = pc0.voxel_down_sample(0.05) #down sample to voxels of 0.05m == 5cm

        pc1 = o3d.geometry.PointCloud()
        pc1.points = o3d.utility.Vector3dVector(xyz1[:, :3])
        pc1 = pc1.voxel_down_sample(0.05) #down sample to voxels of 0.05m == 5cm

        #perform ICP
        reg = o3d.pipelines.registration.registration_icp(pc0, pc1, 0.2, np.eye(4),
                                   o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                   o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))

        M2 = M @ reg.transformation
        return M2

    def __getitem__(self, idx):
        drive, t0, t1 = self.samples[idx]
        all_odometry = self.get_video_odometry(drive, [t0, t1])
        positions = [self.odometry_to_positions(odometry) for odometry in all_odometry]
        fname0 = self._get_velodyne_fn(drive, t0)
        fname1 = self._get_velodyne_fn(drive, t1)

        # XYZ and reflectance
        xyzr0 = np.fromfile(fname0, dtype=np.float32).reshape(-1, 4)
        xyzr1 = np.fromfile(fname1, dtype=np.float32).reshape(-1, 4)

        xyz0 = xyzr0[:, :3]
        xyz1 = xyzr1[:, :3]

        #load refined transformation matrix, if doesnt exist perform icp and save results
        key = '%d_%d_%d' % (drive, t0, t1)
        filename = self.root + '/icp/' + key + '.npy'
        if key not in kitti_icp_cache:
            if not os.path.exists(filename):
                M = (self.velo2cam @ positions[0].T @ np.linalg.inv(positions[1].T) @ np.linalg.inv(self.velo2cam)).T

                #get ICP correction
                M = self._icp(xyz0, xyz1, M)

                # write to a file
                np.save(filename, M)
            else:
                M = np.load(filename)
                kitti_icp_cache[key] = M
        else:
            M = kitti_icp_cache[key]

        #concatenate points in single tensor filled with 0's
        pts = np.zeros((2, 129300, 4), dtype=np.float32)
        pts[0, :xyzr0.shape[0]] = xyzr0
        pts[1, :xyzr1.shape[0]] = xyzr1

        #transform
        Rt = M.astype(np.float32)
        if self.transform:
            pts, Rt = self.transform(pts, Rt)

        return pts, Rt
