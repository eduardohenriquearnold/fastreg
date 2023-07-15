# Dataset paths
CODD_PATH = "/data/CODD/"
KITTI_PATH = "/data/KITTIodometry/"

# Fastreg model parameters
T = 1e-2
VOXEL_SAMPLING_SIZE = 0.3

# Training parameters
lr = 1e-2
batch_size = 6
val_period = 1  # Run validation evaluation every val_period epochs
epochs = 50
