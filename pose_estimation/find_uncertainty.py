import numpy as np 
import cv2
import tensorflow as tf
import sys
import time
import argparse
import math

sigma_p = 0 # White noise variance

def find_uncertainty(u,D,D_prev,T):
	'''
	Finds uncertainty for one element of new keyframe

	Arguments:
		u: Pixel location
		D: New keyframe's depth map
		D_prev: Previous keyframe's depth map
		T: Pose of new keyframe

	Returns: Uncertainty at position u
	'''
	u = np.append(u,np.ones(1)) # Convert to homogeneous

	V = D * np.matmul(cam_matrix_inv,u) # World point
	V = np.append(V,np.ones(1))

	u_prop = np.matmul(cam_matrix,T)
	u_prop = np.matmul(u_prop,V)
	u_prop = u_prop/u_prop[2]
	u_prop = u_prop[:-1]

	U = D[u[0]][u[1]] - D_prev[u_prop[0]][u_prop[1]]
	return U**2

def get_uncertainty(T,D,prev_keyframe):
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
	find_uncertainty_v = np.vectorize(find_uncertainty)
	U = find_uncertainty_v(index_matrix,D,prev_keyframe.D,T) #Check
	return U

def get_initial_uncertainty(): 
	'''
	To get uncertainty map for the first frame
	'''
	return np.ones(im_size)*sigma_p