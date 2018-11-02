import numpy as np

im_size = (480, 640)
sigma_p = 5  # Some white noise variance thing
index_matrix = np.reshape(
    np.dstack(
        np.meshgrid(
            np.arange(480),
            np.arange(640),
            indexing='ij')),
    (480 * 640,
     2))

# 3x3 Intrinsic camera matrix - converts 3x3 point in camera frame to
# homogeneous repreentation of an image coordiante
cam_matrix = np.eye(3, 3)
cam_matrix_inv = np.linalg.inv(cam_matrix)