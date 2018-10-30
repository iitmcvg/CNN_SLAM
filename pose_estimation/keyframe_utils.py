'''
Key frame Utils 

Imported by camera pose estimation
'''

import numpy as np
import math

from pose_estimation.config import *

def isRotationMatrix(R):
    '''
    Checks if a matrix is a valid rotation matrix.
    '''
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def extract_angles(R):
    '''
    Extract rotation angles

    Returns: aplha, beta, gamma (as np array)
    '''

    assert(isRotationMatrix(R))  # Throws error if false

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])

def get_min_rep(T):
    '''
    Convert 3*4 matrix into 6*1 vector

    [x y z alpha beta gamma]

    '''
    t = T[:, 3]
    x, y, z = t

    angles = extract_angles(T[:, :3])

    T_vect = np.zeros(6)
    T_vect[:3] = t
    T_vect[3:6] = angles
    return T_vect



def eulerAnglesToRotationMatrix(theta):
    '''
    Converts rotation angles about x,y and z axis to a rotation matrix
    '''
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])

    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])

    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])
    R = np.dot(R_z, np.dot(R_y, R_x))

    return R


def fix_u(u_prop):
    '''
    Fixes a pixel location if it is negative or out of bounds

    Arguments;
            u_prop: pixel location

    Returns:
            u_prop: fixed pixel location
    '''
    if u_prop[0] >= im_size[0]:
        u_prop[0] = im_size[0] - 1
    elif u_prop[0] < 0:
        u_prop[0] = 0
    if u_prop[1] >= im_size[1]:
        u_prop[1] = im_size[1] - 1
    elif u_prop[1] < 0:
        u_prop[1] = 0
    return u_prop


def get_back_T(T_fl):
    '''
    Converts the minimal representation of the pose into the normal 3x4 transformation matrix
    '''
    # print "The flattened pose input is ",T_fl,'\n\n\n'
    T = np.ones((3, 4))
    T[:, 3] = T_fl[:3]  # 4th column of T = first 3 elements of T_fl
    R = eulerAnglesToRotationMatrix(T_fl[3:6])
    T[:, :3] = R
    return T

def get_delD(D):
    return 0.01  # Change later to calculate based on input depth map


def huber_norm(x):
    '''
    Calculates and Returns the huber norm

    Arguments:
            x: Input

    Returns:
            Huber norm of x
    '''
    delta = 1  # Change later
    if abs(x) < delta:
        return 0.5 * (x**2)
    else:
        return delta * (abs(x) - (delta / 2))