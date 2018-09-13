# See semi-dense visual odometry
# Do we need prev_cost

# Do normalized corelation
# Try out window sizes
# How to get deoth from disparity for verged cameras
# Can do some stuff later like blurring or adge enhancement and all?
# See graph cuts or DP formulations

'''
Small Baseline Stereo Matching
'''

import numpy as np

# Put in some doc later
im_size = (480,640)

camera_matrix = np.eye(3,3)
camera_matrix_inv = np.linalg.inv(camera_matrix)

def get_essential_matrix(T):
	'''
	Returns the essential matrix given the pose T
	'''

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

def calc_cost_stereo(five_points,line2,j,frame,cur_keyframe):
	five_points_2 = np.zeros((5,2))
	cost = 0
	for i in range(5):
		five_points_2[i] = [(-1-line2[0]*(i+j))/line2[1],i+j]
		cost = cost + (cur_keyframe.I[five_points[i][0],five_points[i][1]] - frame[five_points_2[i][0],five_points_2[i][1]])**2
	return cost

def calc_cost_stereo_change(prev_cost,five_points,j,line2,frame,cur_keyframe):
	prev_cost = prev_cost - (cur_keyframe.I[five_points[0][0],five_points[0][1]] - frame[(-1-line2[0]*(i+j-1))/line2[1],i+j-1])**2
	prev_cost = prev_cost + (cur_keyframe.I[five_points[0][0],five_points[0][1]] - frame[(-1-line2[0]*(i+j-1))/line2[1],i+j-1])**2

def calc_disp():


def disp_to_depth():

def five_pixel_match(D,frame,cur_keyframe,line1,line2):
	var = np.var(frame) # Variance of frame
	five_points = np.zeros((5,2)) # Points in cur_keyframe
	for i in range(im_size(1)-4): # i is the first pixel

		# Get the five points in cur_keyframe
		for j in range(5): 
			five_points[j] = [i+j,(-1-line1[0]*(i+j))/line1[1]]
		start = 0

		# For points in frame
		for j in range(i - int(3*var*line2[1]/line2[0]),i+int(3*var*line2[1]/line2[0])): # j goes from i - 3*sigma*cos(theta) to i + 3*sigma*cos(theta) - x values
			if j<0:
				continue
			if j>im_size(1):
				break
			if start == 0:
				min_cost = calc_cost_stereo(five_points,line2,j,frame,cur_keyframe)
				min_pos = j
				start = 1
				prev_cost = min_cost
				continue
			cost = calc_cost_stereo(five_points,line2,j,frame,cur_keyframe)
			prev_cost = cost
			if (cost<min_cost):
				min_cost = cost
				min_pos = j
		# Pixels being matched
		min_pos = min_pos + 2
		i = i+2
		disp = calc_disp(min_pos,i,line1,line2)
		D[i,(-1-(a*i))/line1[1]] = disp_to_depth(disp)
		return D

def stereo_match(cur_keyframe,frame,T):
	'''
	Function to do stereo matching

	Arguments:
		cur_keyframe: previous keyframe as a keyframe object
		frame: current frame as numpy image
		T: Estimated pose of frame wrt cur_keyframe

	Returns:
		D: Depth map
	'''
	E = get_essential_matrix(T)
	D = np.zeros(im_size) # Initilize with some white noise variance?

	for i in range(im_size[0]):
		line1,line2 = find_epipolar_lines((0,i),E) # line 1 is in cur_keyframe and line2 in frame. Each line is of the form (a,b) so that ax + by +1 = 0
		D = five_pixel_match(D,frame,cur_keyframe,line1,line2)

		#Check which parts of keyframe havent been matched yet
	
	return D