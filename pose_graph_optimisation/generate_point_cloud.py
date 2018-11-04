# Use point only if uncertainty is below threshold

import numpy as np
from params import index_matrix_2,camera_matrix_inv

def generate_point_cloud(keyframes,world_poses):
	points = []
	for i in keyframes:
		point_in_cam_frame = np.transpose(i.D*index_matrix_2) # 3x480*640
		points_in_world = np.matmul(camera_matrix_inv,point_in_cam_frame) # 3x480*640
		points_in_world = np.transpose(points_in_world) # 480*640x3
		points_colours = np.reshape(i.I,(480*640,3)) # RGB values for each point
		points.append(points_in_world,points_colours)
	return points