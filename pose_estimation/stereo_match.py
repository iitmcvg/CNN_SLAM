# Error with fnding epipoles
# See https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
# Need camera matrix for depth from disparity
# Depth from disparity

# Find standard deviation for whole image?
# Interpolation?
# Post processing on disparity map?
# Check do_transform

# Try increasing search range
# See SVD and minimizing least squares in Zisserman
# Do normalized corelation
# See graph cuts or DP formulations

# Always index images as x and then y
# when trying to display a numpy array as an image, noralize to (0,1)
'''
Small Baseline Stereo Matching
'''
import cv2
import numpy as np
import time
from multiprocessing import Pool
#from keyframe_utils import fix_u
from matplotlib import pyplot as plt
#from params import im_size,camera_matrix,camera_matrix_inv
camera_matrix = np.eye(3,3) # Camera matrix
camera_matrix_inv = np.linalg.inv(camera_matrix) 
im_size = (480,640) # S

def get_essential_matrix(T):
    '''
    Returns the essential matrix E given the pose T
    '''
    t = T[:3, 3]
    R = T[:3, :3]  # 3x3
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    E = np.matmul(tx, R)
    return E


"""
def find_epipolar_lines(u,E):
    '''
    Finds epipolar lines Line1 and Line2 in cur_keyframe and frame respectively

    Arguments:
        u: Point in cur_keyframe
        E: Essential matrix to go from cur_keyframe to frame

    Returns:
        line1,line2: The epipolar lines
    '''
    # Find line 2 in frame
    u = np.append(u,(1)) # Make u homogeneous
    v = np.matmul(camera_matrix_inv,u) # Point in world coordinates
    l = np.matmul(E,v) # Line in world coords (or something like that)
    l = np.matmul(camera_matrix_inv,l) # Epipolar line
    line2 = l[:2]/l[2] # Line as (a,b). So ax' + by' + 1 = 0

    #Finding line 1 in cur_keyframe
    E_inv = np.linalg.inv(E)
    u2 = (0,(-1/line2[1]))
    v = np.matmul(camera_matrix_inv,u) # Point in world coordinates
    l = np.matmul(E_inv,v) # Line in world coords (or something like that)
    l = np.matmul(camera_matrix_inv,l) # Epipolar line
    line1 = l[:2]/l[2]

    return line1,line2
"""


def find_epipoles(F):
    '''
    F.e1 = 0
    F.transpose.e2 = 0
    '''
    e1 = np.cross(F[0] + F[1], F[1] + F[2])
    if e1[2] != 0:
        e1 = e1 / e1[2]
    e2 = np.cross(F[:, 0] + F[:, 1], F[:, 1] + F[:, 2])
    if e2[2] != 0:
        e2 = e2 / e2[2]

    if(np.dot(F[2], e1) > 1e-8 or np.dot(F[:, 2], e2) > 1e-8):  # Change later
        print("Error with finding epipoles")
        # Add something here for error handling

    return e1, e2


def get_H2(frame, e, F):
    '''
    H2 = GRT
    '''
    # Move center of image to (0,0)
    T = np.array([[1, 0, -im_size[1] / 2], [0, 1, im_size[0] / 2], [0, 0, 1]])
    e_trans = np.matmul(T, e)

    # Rotate epipole so that its on the x axis
    e_new = np.array([(e_trans[0]**2 + e_trans[1]**2)**0.5, 0, 1])
    cos = np.dot(e_new, e_trans) / (e_trans[0]**2 + e_trans[1]**2)
    sin = (1 - cos**2)**0.5
    R = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])

    # Move epipole to infinity
    G = np.eye(3)
    G[2][0] = -1 / (e_new[0])

    H2 = np.matmul(G, np.matmul(R, T))
    return H2

# Check


def do_transform(frame, H):
    '''
    Transforms frame according to projective transform H
    '''
    dst = cv2.warpPerspective(frame, H, im_size)
    return dst


def get_rect_pose(T):
    return T  # Change later


def rectify_frames(frame1, frame2, F, rel_T):
    '''
    Makes the image planes parallel

    Arguments:
            frame1: First image
            frame2: Second image
            E: Fundamental matrix (frame1 to frame2). frame2 is x'. x'Fx = 0

    Returns:
            frame_rect_1: Rectified image1
            frame_rect_2: Rectified image2
            rect_rel_T: Rectified relative pose
    '''
    e1, e2 = find_epipoles(
        F)  # F.e1 = 0 and (F.T).e2 = 0. e1 and e2 in homogeneous form
    H2 = get_H2(frame2, e2, F)  # H2 = GRT

    # Compute H1 = H2.M.Ha
    R = rel_T[:3, :3]
    M = np.matmul(camera_matrix, np.matmul(R, camera_matrix))

    # Need to find Ha
    a, b, c = 1, 0, 0  # Initialise randomly later
    H0 = np.matmul(H2, M)
    frame2_rect = do_transform(frame2, H2)
    frame1_temp = do_transform(frame1, H0)

    # Minimise and find a,b,c (or just take as 1,0,0?)

    Ha = np.array([[a, b, c], [0, 1, 0], [0, 0, 1]])
    H1 = np.matmul(Ha, H0)
    frame1_rect = do_transform(frame1, H1)

    # Use old baseline only? Its small enough
    rect_rel_T = get_rect_pose(rel_T)
    return frame1_rect, frame2_rect, rect_rel_T


def actual_match(vec1, vec2):
    std_dev = int((np.var(vec2))**0.5)
    D = np.ones(im_size[1]) * 0.05
    for j in range(im_size[1] - 8):
        five_points = np.zeros(5)
        for k in range(5):
            five_points[k] = vec1[j + 2 * k]
        min_cost = -1
        min_pos = -1
        a = time.time()
        for k in range(j - 3 * std_dev, j + 3 * std_dev + 1):  # Change to 2?
            if(k < 0 or k + 10 > im_size[1]):
                continue
            cost = 0
            for l in range(5):
                cost = cost + (five_points[l] - vec2[k + 2 * l])**2
            if min_cost == -1:
                min_cost = cost
                min_pos = k + 4
            if cost < min_cost:
                # print cost,min_cost,j,k
                min_cost = cost
                min_pos = k + 4
        b = time.time()
        # print b-a
        """
        l = 0
        u = 8
        if j-2*std_dev>0:
            l = j-2*std_dev
        else:
            l = 0
        if j+2*std_dev+1+5>im_size[1]: #Check
            u = im_size[1]
        else:
            u = j+2*std_dev+1
        vec = np.flip(vec2,axis = 0) # change search range
        corr = np.correlate(five_points,vec)
        min_pos = np.argmax(corr)+2"""
        D[j + 2] = np.abs(min_pos - j)  # Add im_size(0) also? (and take abs)?
    return D


def five_pixel_match(img1, img2):
    '''
    Computes the disparity map for two parallel plane images img1 and img2
    '''
    D = np.zeros(im_size)
    for i in range(im_size[0]):
    	D[i] = actual_match(img1[i],img2[i])
  # Initilize with some white noise variance?
    return D

def depth_from_disparity(disparity_map, T):
    '''
    Computes depth map from disparity map

    Arguments:
            disparity_map
            T: Pose

    Returns:
            depth_map
    '''
    return 1.0 / \
        ((disparity_map / (T[0, 3]**2 + T[1, 3]**2 + T[2, 3]**2)**0.5) + 0.001)

def stereo_match(frame1, frame2, T1, T2):
    '''
    Function to do stereo matching and return the depth map

    Arguments:
            frame1: 1st frame
            frame2: 2nd frame
            T1: Pose of frame1 wrt previous keyframe
            T2: Pose of frame2 wrt previous keyframe

    Returns:
            D: Depth map
    '''
    T1 = np.append(T1, np.array([[0, 0, 0, 1]]), 0)
    T2 = np.append(T2, np.array([[0, 0, 0, 1]]), 0)
    # Go from frame1 to prev keyframe and then to frame2
    rel_T = np.matmul(np.linalg.inv(T1), T2)
    rel_T = rel_T[:3]
    E = get_essential_matrix(rel_T)
    F = np.matmul(camera_matrix_inv.T, np.matmul(E, camera_matrix_inv))  # Fundamental Matrix
    #frame_rect_1, frame_rect_2, rect_rel_T = rectify_frames(frame1, frame2, F, rel_T)
    frame_rect_1,frame_rect_2 = cv2.StereoRectify(camera_matrix,camera_matrix,imageSize = im_size,R = rel_T[:3,:3],T = rel_T[:3,3])
    cv2.imshow("adwadaw",frame_rect_1)
    frame_rect_1 = np.transpose(frame_rect_1.astype(np.uint8))
    frame_rect_2 = np.transpose(frame_rect_2.astype(np.uint8))
    #disparity_map = five_pixel_match(frame1, frame2)  # Disparity map
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=7)
    disparity_map = stereo.compute(frame_rect_1,frame_rect_2)
    depth_map = depth_from_disparity(disparity_map, rect_rel_T)
    plt.imshow(disparity_map,cmap = 'gray')
    plt.show()
    return depth_map


def test_5_match():
    img1 = cv2.resize(
        cv2.imread(
            "stereo.jpeg",
            0),
        (im_size[1],
         im_size[0]),
        interpolation=cv2.INTER_CUBIC)
    img2 = cv2.resize(
        cv2.imread(
            "stereo(1).jpeg",
            0),
        (im_size[1],
         im_size[0]),
        interpolation=cv2.INTER_CUBIC)
    im1 = np.array(img1)
    im2 = np.array(img2)
    D = np.zeros(im_size)
    D_1 = five_pixel_match(img1, img2)
    """stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(img1,img2)
    cv2.imshow('dawwd',disparity)
    cv2.waitKey(0)"""

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


"""
def test_stereo_match():
    img1 = cv2.resize(cv2.imread("pose_estimation/stereo.jpeg",0),(im_size[1], im_size[0]),interpolation=cv2.INTER_CUBIC)
    img2 = cv2.resize(cv2.imread("pose_estimation/stereo(1).jpeg",0),(im_size[1],im_size[0]),interpolation=cv2.INTER_CUBIC)
    img1_rect = img2_rect = img1
    T1 = np.array([[1, 0, 0, 9.5], [0, 1, 0, 0], [0, 0, 1, 0]])
    T2 = np.array([[1, 0, 0, 10], [0, 1, 0, 0], [0, 0, 1, 0]])
    T_rel = np.matmul(np.linalg.inv(np.append(T1,[[0,0,0,1]],0)),np.append(T2,[[0,0,0,1]],0))
    R1 = R2 = np.zeros((3,3))
    P1 = P2 = np.zeros((3,4))
    cv2.stereoRectify(cameraMatrix1 = camera_matrix,cameraMatrix2 = camera_matrix,distCoeffs1 = np.zeros(5),distCoeffs2 = np.zeros(5),imageSize = im_size,R = T_rel[:3,:3],T = -T_rel[:3,3],R1 = R1,R2 = R2,P1 = P1,P2 = P2,newImageSize = im_size)
    print(T_rel[:3,:3],P1)
	map1,map2 = cv2.initUndistortRectifyMap(camera_matrix,np.zeros(5),R1,size = im_size,m1type = cv2.CV_16SC2,newCameraMatrix = P1)
	img1_rect = cv2.remap(img1,map1 = map1,map2 = map2,interpolation = cv2.INTER_CUBIC)
	plt.imshow(map1,cmap = 'gray')
	plt.show()
    stereo = cv2.StereoBM_create(numDisparities=64, blockSize=11)
    disparity_map = stereo.compute(img1,img2)
    #plt.imshow(disparity_map,cmap = 'gray')
    #plt.show()
    return 1
"""

def for_just_refine(img1,img2):
	T1 = np.array([[1, 0, 0, 3], [0, 1, 0, 0], [0, 0, 1, 0]])
	T2 = np.array([[1, 0, 0, 5], [0, 1, 0, 0], [0, 0, 1, 0]])
    # Go from frame1 to prev keyframe and then to frame2
	T_rel = np.matmul(np.linalg.inv(np.append(T1,[[0,0,0,1]],0)),np.append(T2,[[0,0,0,1]],0))
	rel_T = T_rel[:3]
	E = get_essential_matrix(rel_T)
	F = np.matmul(camera_matrix_inv.T, np.matmul(E, camera_matrix_inv))  # Fundamental Matrix
	disparity_map = five_pixel_match(img1, img2)  # Disparity map
	#stereo = cv2.StereoBM_create(numDisparities=16, blockSize=7)
	#disparity_map = stereo.compute(frame_rect_1,frame_rect_2)
	depth_map = depth_from_disparity(disparity_map, rel_T)
	plt.imshow(disparity_map,cmap = 'gray')
	plt.show()
	return disparity_map,depth_map	



if __name__ == '__main__':
    #test_stereo_match()
    """img1 = cv2.resize(
        cv2.imread(
            "pose_estimation/stereo.jpeg",
            0),
        (im_size[1],
         im_size[0]),
        interpolation=cv2.INTER_CUBIC)
    img2 = cv2.resize(
        cv2.imread(
            "pose_estimation/stereo(1).jpeg",
            0),
        (im_size[1],
         im_size[0]),
        interpolation=cv2.INTER_CUBIC) 
    print(time.time())""
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=7)
    disparity = stereo.compute(img1,img2)
    disparity = cv2.GaussianBlur(disparity,(5,5),0)
    print(time.time())
    plt.imshow(disparity,'gray')
    plt.show()
    #cv2.imshow('eesfse',disparity)
    #cv2.waitKey(0)"""
