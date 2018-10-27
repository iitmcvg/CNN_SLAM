# Library imports
import numpy as np
import cv2
import sys
import time
import argparse
import math

from pose_estimation.keyframe_utils import *
from pose_estimation.config import *
import pose_estimation.optimiser as optimiser

def calc_photo_residual(i, frame, cur_keyframe, T):
    '''
    Calculates the photometric residual for one point

    Arguments:
            i: Pixel location
            frame: Current frame as numpy array
            cur_keyframe: Previous keyframe as Keyframe object
            T: Estimated pose

    Returns:
            r: Photometric residual
    '''
    # Make i homogeneous
    i = np.append(i, np.ones(1))
    i = i.astype(int)
    # 3D point 3*1

    V = cur_keyframe.D[i[0]][i[1]] * np.matmul(cam_matrix_inv, i)

    # Make V homogeneous 4*1
    V = np.append(V, 1)

    # 3D point in the real world shifted (3*4 x 4*1 = 3*1)
    u_prop = np.matmul(T, V)[:3]

    # 3D point in camera frame (3*3 * 3*1)
    u_prop = np.matmul(cam_matrix, u_prop)

    # Projection onto image plane
    u_prop = (u_prop / u_prop[2])[:2]
    u_prop = u_prop.astype(int)

    u_prop = fix_u(u_prop)

    r = (int(cur_keyframe.I[i[0], i[1]]) - int(frame[u_prop[0], u_prop[1]]))
    return r


def calc_r_for_delr(u, D, frame, cur_keyframe, T):
    '''
    Finds photometric residual given one point

    Argumemnts:
            u: numpy array oof x and y location
            D: Depth map value at u
            frame: current frame
            cur_keyframe: previous keyframe of keyframe class
            T: current estimated pose

    Returns:
            r: photometric residual
    '''
    u = np.append(u, [1])
    v = D * np.matmul(cam_matrix_inv, u)
    v = np.append(v, [1])
    u_prop = np.matmul(T, v)[:3]
    u_prop = np.matmul(cam_matrix, u_prop)
    u_prop = ((u_prop / u_prop[2])[:2]).astype(int)

    u_prop = fix_u(u_prop)

    r = int(cur_keyframe.I[u[0], u[1]]) - int(frame[u_prop[0], u_prop[1]])
    return r

def delr_delD(u, frame, cur_keyframe, T):
    '''
    Finds the derivative of the photometric residual wrt depth (r wrt d)
    delr/delD  = (delr/delu)*(delu/delD)
    delr/delu = delr/delx + delr/dely - finding root of sum of squares now
    delD/delu = delD/delx + delD/dely - finding root of sum of squares now
    r = cur_keyframe.I[u[0]][u[1]] - frame[u_prop[0]][u_prop[1]] - How r is defined normally

    Arguments:
            u: High gradient pixel location
            frame: Current frame as numpy array
            cur_keyframe: Previous keyframe as a Keyframe object
            T: Estimated pose

    Returns:
            delr: The derivative
    '''

    # Convert u to int
    u = u.astype(int)

    # For finding right and left sides for x and y
    ulx = np.array([u[0] - 1, u[1]])
    urx = np.array([u[0] + 1, u[1]])
    uly = np.array([u[0], u[1] - 1])
    ury = np.array([u[0], u[1] - 1])

    ulx = fix_u(ulx)
    uly = fix_u(uly)
    urx = fix_u(urx)
    ury = fix_u(ury)

    # Depth map values
    Dlx = cur_keyframe.D[ulx[0]][ulx[1]]
    Drx = cur_keyframe.D[urx[0]][urx[1]]
    Dly = cur_keyframe.D[uly[0]][uly[1]]
    Dry = cur_keyframe.D[ury[0]][ury[1]]

    # Finding delD/delu
    delDdelu = ((Drx - Dlx)**2 + (Dry - Dly)**2)**0.5
    deludelD = 1.0 / delDdelu

    r_list = [0, 0, 0, 0]  # Just random

	# u = np.append(u,[1])
	# v = D*np.matmul(cam_matrix_inv,u)
	# v = np.append(v,[1])
	# u_prop = np.matmul(T,v)[:3]
	# u_prop = np.matmul(cam_matrix,u_prop)
	# u_prop = ((u_prop/u_prop[2])[:2]).astype(int)
	# r_list[0] = cur_keyframe.I[u[0],u[1]] - frame[u_prop[0],u_prop[1]]

    # Calculate r_list
    calc_r_for_delr_v = np.vectorize(
        calc_r_for_delr, excluded=[
            2, 3, 4], signature='(1),()->()')
    u_list = [ulx, urx, uly, ury]
    D_list = [Dlx, Drx, Dly, Dry]
    r_list = calc_r_for_delr_v(u_list, D_list, frame, cur_keyframe, T)

    delrdelu = ((r_list[0] - r_list[1])**2 + (r_list[2] - r_list[3])**2)**0.5

    delr = delrdelu * deludelD
    return delr

def calc_photo_residual_uncertainty(u, frame, cur_keyframe, T):
    '''
    Calculates the photometric residual uncertainty

    Arguments:
            u: High gradient pixel location
            frame: Current frame as a numpy array
            cur_keyframe: Previous keyframe as a Keyframe object
            T: Estimated pose

    Returns:
            sigma: Residual uncertainty
    '''
    deriv = delr_delD(u, frame, cur_keyframe, T)
    sigma = (sigma_p**2 + (deriv**2) * cur_keyframe.U[u[0]][u[1]])**0.5
    return sigma

def ratio_residual_uncertainty(u, frame, cur_keyframe, T):
    return huber_norm(calc_photo_residual(u, frame, cur_keyframe, T) /
                      calc_photo_residual_uncertainty(u, frame, cur_keyframe, T))

# WHAT IS FLAG?
def calc_cost(uu, frame, cur_keyframe, T):
    '''
    Calculates the residual error as a stack.

    Arguments:
            uu: An array containing the high gradient elements (X,2)
            frame: Numpy array o the current frame
            cur_keyframe: Previous keyframe as a Keyframe class
            pose: Current estimated Pose

    Returns:
            r: Residual error as an array
    '''
    
    return ratio_residual_uncertainty_v(uu, frame, cur_keyframe, T)


def loss_fn(uu, frame, cur_keyframe, T_s):
    T = get_back_T(T_s)
    cost = calc_cost(uu, frame, cur_keyframe, T)
    cost = np.sum(cost)
    return cost 

def grad_fn(u, frame, cur_keyframe, T_s, frac = 0.01):
    '''
    Calculate gradients
    '''    
    costa = loss_fn(u, frame, cur_keyframe, T_s * (1 - frac))
    costb = loss_fn(u, frame, cur_keyframe, T_s * (1 + frac))
    grad = (costb - costa) / (2 * T_s * frac)

    return grad

# Vectorised implementations
ratio_residual_uncertainty_v = np.vectorize(
        ratio_residual_uncertainty, excluded=[
            1, 2, 3], signature='(1)->()')


def minimize_cost(u, frame, cur_keyframe, 
    variance = 0.01,
    mean = 5.0,
    learning_rate = 0.05,
    max_iter= 100,
    loss_bound = 0.1):

    dof = len(u)
    T_s = np.random.random((6)) * variance + mean

    optim = optimiser.Adam(lr = learning_rate)
    i = 0

    while True:  # Change later
        loss = loss_fn(u, frame, cur_keyframe, T_s)
        grads = grad_fn(u, frame, cur_keyframe, T_s)

        print("loss ", loss)

        T_s = optim.get_update([T_s],[grads])[0]
        i = i + 1

        print("grad: ", np.max(grads))
        print("T_s", T_s)

        # Stopping condtions
        if (loss < loss_bound) or (i == max_iter) or (abs(np.max(grads)) > 100):
            break

    return get_back_T(T_s), loss_fn(u, frame, cur_keyframe, T_s)

def test_min_cost_func():
    '''
    Test minimum cost function:

    * Take current frame, keyframe
    '''
    # Image Size
    im_x, im_y = im_size

    # Random high grad points
    u_test = np.array([[5, 4], [34, 56], [231, 67], [100, 100], [340, 237]])

    # Random frame
    frame_test = np.uint8(np.random.random((im_x, im_y)) * 256)

    # Current key frame, depth, pose, uncertainuty
    cur_key_test_im = np.uint8(np.random.random((im_x, im_y, 3)) * 256)
    cur_key_test_im_grey = np.uint8(
        (cur_key_test_im[:, :, 0] + cur_key_test_im[:, :, 1] + cur_key_test_im[:, :, 2]) / 3)
    cur_key_depth = np.random.random((im_x, im_y))
    dummy_pose = np.eye(4)[:3]
    cur_key_unc = np.ones((im_x, im_y))

    cur_key = Keyframe(
        dummy_pose,
        cur_key_depth,
        cur_key_unc,
        cur_key_test_im_grey)

    print(
        "Testing minimize cost func",
        minimize_cost(
            u_test,
            frame_test,
            cur_key))

if __name__ == '__main__':
    test_min_cost_func()
