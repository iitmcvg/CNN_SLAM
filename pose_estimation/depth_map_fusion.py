# Check actual_fuse

import numpy as np
import cv2
import tensorflow as tf
import sys
import time

from pose_estimation.keyframe_utils import fix_u
from params import im_size,camera_matrix as cam_matrix,camera_matrix_inv as cam_matrix_inv
# 3x3 Intrinsic camera matrix - converts 3x3 point in camera frame to
# homogeneous repreentation of an image coordiante

def actual_fuse(u, frame, prev_keyframe):
    '''
    Does the actual fusion of depth and uncertainty map

    Arguments:
            u: Pixel location
            frame: current keyframe of Keyframe class
            prev_keyframe: Previous keyframe of Keyframe class

    Returns:
            Depth and Uncertainty map values at u
    '''
    u = np.append(u, np.ones(1))
    u = u.astype(np.int32)
    v_temp = frame.D[u[0]][u[1]] * \
        np.matmul(cam_matrix_inv, u)  # Returns 3x1 point
    v_temp = np.append(v_temp, np.ones(1))
    v_temp = np.matmul(frame.T, v_temp)  # Return 3x1
    v_temp = np.matmul(cam_matrix, v_temp)
    v = (v_temp / v_temp[2])[:2]
    v = v.astype(np.int32)
    v = fix_u(v)
    u_p = (prev_keyframe.D[v[0]][v[1]] * prev_keyframe.U[v[0]]
           [v[1]] / frame.D[u[0]][u[1]]) + sigma_p**2
    D = (u_p * frame.D[u[0]][u[1]] + frame.U[u[0]][u[1]] * prev_keyframe.D[v[0]]
         [v[1]]) / (u_p + frame.U[u[0]][u[1]])  # Kalman filter update step 1
    # Kalman filter update step 2
    U = u_p * frame.U[u[0]][u[1]] / (u_p + frame.U[u[0]][u[1]])
    return D, U


def fuse_depth_map(frame, prev_keyframe):
    '''
    Fuses depth map for new keyframe

    Arguments:
            frame: New keyframe of Keyframe class
            prev_keyframe: Previous keyframe of Keyframe class

    Returns:
            The new keyframe as Keyframe object
    '''
    actual_fuse_v = np.vectorize(
        actual_fuse,
        signature='(1)->(),()',
        excluded=[
            1,
            2])
    D, U = actual_fuse_v(index_matrix, frame, prev_keyframe)
    frame.D = np.reshape(D, (im_size))
    frame.U = np.reshape(U, (im_size))
    return frame.D, frame.U

def temp():
    return 1

if __name__ == "__main__":
    temp()
