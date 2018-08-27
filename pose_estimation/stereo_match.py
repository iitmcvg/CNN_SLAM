import numpy as np

# Put in some doc later
im_size = (480,640)
camera_matrix_inv = np.eye(3,3)
camera_matrix = np.eye(3,3)

def get_essential_matrix(T):
	'''
	Returns the essential matrix given the transformation matrix

	Arguments:
		T: 3x4 tranformation matrix

	Returns:
		E: Essential matrix
	'''
	R = T[:3,:3] # Rotation matrix
	t = T[:,3]
	tx = np.array([[0,-t[2],t[1]],[t[2],0,-t[0]],[-t[1],t[0],0]])
	E = np.matmul(tx,R) # 3x3
	return E 

def find_epipolar_line(i,j,T):
	'''
	Finds the epipolar line in img 1 for a point in img 2

	Arguments:
		i: y coord of point
		j: x coord of point
		T: pose of img2 wrt img 1

	Returns:
		v_prop: stores the a,b and c parameters of the line
	'''
	E = get_essential_matrix(T)
	u = np.array([i,j,1]) # Homogeneous point in img 2
	v = np.matmul(cam_matrix_inv,u) # We dont know depth so this 3D point could lie anywhere along a ray.
	v = (v/v[2])
	v_prop = np.matmul(E,v)
	v_prop = np.matmul(cam_matrix_inv,v_prop)

	# Given (x,y,1) x*v_prop[0] + y*v_prop[1] + v_prop[2] = 0 -> line
	return v_prop

def stereo_match(img1,img2,T):
	'''
	Returns an estimated disparity map given two stereo images and their relative pose

	Arguments:
		img1: image 1 as numpy array - previous keyframe
		img2: image 2 as numpy array - current frame
		T: 3x4 transformation matrix for img2 wrt im1
		im_size: shape of image

	Returns:
		D: Disparity map
	'''
	for i in im_size[0]:
		for j in im_size[1]:
			line = find_epipolar_line(i,j,T)
			# Points on line are given by (x,(-line[2] - x*line[0])/line(1)) in img 1
			# Find other line also
			# Do 5 pixel matching with gaussian prior


