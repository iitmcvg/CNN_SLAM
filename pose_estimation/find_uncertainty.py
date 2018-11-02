import numpy as np
import cv2
import tensorflow as tf
import sys
import time
import argparse
import math

from pose_estimation.camera_pose_estimation import fix_u
from params import im_size,camera_matrix as cam_matrix,camera_matrix_inv as cam_matrix_inv,sigma_p,index_matrix

def find_uncertainty(u, D, D_prev, T):
    '''
    Finds uncertainty for one element of new keyframe

    Arguments:
            u: Pixel location
            D: New keyframe's depth map
            D_prev: Previous keyframe's depth map
            T: Pose of new keyframe

    Returns: Uncertainty at position u
    '''
    u = np.append(u, np.ones(1))  # Convert to homogeneous
    u = u.astype(np.int32)

    V = D[u[0], u[1]] * np.matmul(cam_matrix_inv, u)  # World point
    V = np.append(V, np.ones(1))

    u_prop = np.matmul(cam_matrix, T)
    u_prop = np.matmul(u_prop, V)
    u_prop = u_prop / u_prop[2]
    u_prop = u_prop[:-1]
    u_prop = u_prop.astype(np.int32)
    u_prop = fix_u(u_prop)
    U = D[u[0]][u[1]] - D_prev[u_prop[0]][u_prop[1]]
    return U**2


def get_uncertainty(T, D, prev_keyframe):
    '''
    Finds the uncertainty map for a new keyframe

    Arguments:
            T: Pose of new keyframe wrt prev keyframe
            D: Depth map of new keyframe
            prev_keyframe: Previous keyframe of Keyframe class

    Returns:
            U: Uncertainty map
    '''
    # Write vectorize properly
    # T = np.matmul(np.linalg.inv(T),prev_keyframe.T) #Check if this is right
    a = find_uncertainty((5,5),D,prev_keyframe.D,T)
    print("\n\nadwadwa\n\n")
    find_uncertainty_v = np.vectorize(
        find_uncertainty,
        signature='(1)->()',
        excluded=[
            1,
            2,
            3])
    U = np.zeros(im_size)
    U = find_uncertainty_v(index_matrix, D, prev_keyframe.D, T)  # Check
    U = np.reshape(U, im_size)
    return U


def get_initial_uncertainty():
    '''
    To get uncertainty map for the first frame
    '''
    return np.ones(im_size) * sigma_p
