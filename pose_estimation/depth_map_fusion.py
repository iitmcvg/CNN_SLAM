import numpy as np 
import cv2
import tensorflow as tf
import sys
import time

sigma_p = 0 # Some white noise variance thing

def actual_fuse(u,frame,prev_keyframe):
	'''
	Does the actual fusion of depth and uncertainty map

	Arguments:
		u: Pixel location
		frame: current keyframe of Keyframe class
		prev_keyframe: Previous keyframe of Keyframe class

	Returns:
		Depth and Uncertainty map values at u
	'''
	u = np.append(u,np.ones(1))
	v_temp = frame.D[u[0]][u[1]]*np.matmul(cam_matrix_inv,u) #Returns 3x1 point
	v_temp.appenf(1)
	v_temp = np.matmul(np.matmul(np.linalg.inv(frame.T),prev_keyframe.T),v_temp) #Return 3x1
	v_temp = np.matmul(cam_matrix,v_temp)
	v = (v_temp/v_temp[2])[:2]
	u_p = (prev_keyframe.D[v[0]][v[1]]*prev_keyframe.U[v[0]][v[1]]/frame.D[u[0]][u[1]]) + sigma_p**2
	frame.D[u[0]][u[1]] = (u_p*frame.D[u[0]][u[1]] + frame.U[u[0]][u[1]]*prev_keyframe.D[v[0]][v[1]])/(u_p + frame.U[u[0]][u[1]]) #Kalman filter update step 1
	frame.U[u[0]][u[1]] = u_p*frame.U[u[0]][u[1]]/(u_p + frame.U[u[0]][u[1]]) #Kalman filter update step 2
	return frame.D[u[0]][u[1]],frame.U[u[0]][u[1]]

def fuse_depth_map(frame,prev_keyframe):
	'''
	Fuses depth map for new keyframe

	Arguments:
		frame: New keyframe of Keyframe class
		prev_keyframe: Previous keyframe of Keyframe class

	Returns:
		The new keyframe as Keyframe object
	'''
	actual_fuse_v = np.vectorize(actual_fuse)
	frame.D,frame.U = actual_fuse_v(index_matrix,frame,prev_keyframe)
	return frame.D,frame.U
