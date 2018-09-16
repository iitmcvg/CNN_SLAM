# Interpolation?
# Post processing on disparity map?
# Match windows not pixels

# Do normalized corelation
# Try out window sizes
# How to get deoth from disparity for verged cameras
# Can do some stuff later like blurring or adge enhancement and all?
# See graph cuts or DP formulations

#Always index images as x and then y

'''
Small Baseline Stereo Matching
'''
import cv2
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


def five_pixel_match(D,img1,img2):
	'''
	Computes the disparity map for two parallel plane images img1 and img2
	'''
	std_dev = int((np.var(img2))**0.5) # Standard Deviation

	for i in range(im_size[0]):
		for j in range(im_size[1] - 4):
			five_points = np.zeros(5)
			for k in range(5):
				five_points[k] = img1[j+k][i]
			min_cost = -1
			min_pos = -1
			for k in range(j-2*std_dev,j+2*std_dev+1):
				if(k<0 or k+5>640):
					continue
				cost = 0
				for l in range(5):
					cost = cost + (five_points[l] - img2[k+l][i])**2
				if cost>min_cost:
					min_cost = cost
					min_pos = k
			D[i][j+2] = (min_pos + im_size[1] - j-2)*(255/850.0)
			print i,D[i][j+2],'\n'
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
		D = five_pixel_match(D,frame,cur_keyframe.I,line1,line2)

		#Check which parts of keyframe havent been matched yet
	
	return D

def test_stereo_match():
	img1 = cv2.resize(cv2.imread("stereo.jpeg",0),im_size,interpolation = cv2.INTER_CUBIC)
	img2 = cv2.resize(cv2.imread("stereo(1).jpeg",0),im_size,interpolation = cv2.INTER_CUBIC)
	im1 = np.array(img1)
	im2 = np.array(img2)
	D = np.zeros(im_size)
	D = five_pixel_match(D,img1,img2)
	cv2.imshow('dawwd',D)
	cv2.waitKey(0)

if __name__=='__main__':
	test_stereo_match()