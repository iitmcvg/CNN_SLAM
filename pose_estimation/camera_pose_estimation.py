# Singular matrix encountered sometimes
# See Gauss Newton in Zisserman

# Check later
# Newton gauss is for least squares; Here we are using for huber norm (also should huber norm be included in the calculation of individual costs)
# delr/delu = delr/delx + delr/dely - should we divide by 2 or something or find the root of sum of squares
# initial uncertainty and pose

# To do
# Dont just cast to int. Interpolate instead. Make sure you are doing inverse warping and not forward - do everywhere
# Pixel from keyframe should always be propagated
# Put hyperparamaters in some doc
# Change exit criteria
# Make sure values are in float before finding inverse(1.0/x)
# See if the entire keyframe needs to be passed everytime. Sometimes only the image needs to be passed. (See in stereo match also)

'''
Camera Pose Estimation
'''

# Libraries
import numpy as np 
import cv2
import tensorflow as tf
import sys
import time
import argparse
import math

# Modules
#import depth_map_fusion as depth_map_fusion
from pose_estimation.stereo_match import *
#import monodepth

'''
Variable nomenclature:

* u : high grad elements
* uu: array of high elements
* T: pose (3*4)
* T_s: pose compressed (6x1)
* D: depth map
* U: uncertainity
* dof: len(uu)

'''

im_size = (480,640)
sigma_p = 5 # Some white noise variance thing
index_matrix = np.reshape(np.dstack(np.meshgrid(np.arange(480),np.arange(640),indexing = 'ij')),(480*640,2))
cam_matrix = np.eye(3,3) # 3x3 Intrinsic camera matrix - converts 3x3 point in camera frame to homogeneous repreentation of an image coordiante
cam_matrix_inv = np.linalg.inv(cam_matrix)

def fix_u(u_prop):
	'''
	Fixes a pixel location if it is negative or out of bounds
	
	Arguments;
		u_prop: pixel location

	Returns:
		u_prop: fixed pixel location
	'''
	if u_prop[0]>=im_size[0]:
		u_prop[0] = im_size[0] - 1
	elif u_prop[0]<0:
		u_prop[0] = 0
	if u_prop[1]>=im_size[1]:
		u_prop[1] = im_size[1] - 1
	elif u_prop[1]<0:
		u_prop[1] = 0
	return u_prop

def isRotationMatrix(R) :
	'''
	Checks if a matrix is a valid rotation matrix.
	'''
	Rt = np.transpose(R)
	shouldBeIdentity = np.dot(Rt, R)
	I = np.identity(3, dtype = R.dtype)
	n = np.linalg.norm(I - shouldBeIdentity)
	return n < 1e-6
	

def extract_angles(R):
	'''
	Extract rotation angles

	Returns: aplha, beta, gamma (as np array)
	'''

	assert(isRotationMatrix(R)) #Throws error if false
     
	sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
		
	singular = sy < 1e-6
	
	if  not singular :
		x = math.atan2(R[2,1] , R[2,2])
		y = math.atan2(-R[2,0], sy)
		z = math.atan2(R[1,0], R[0,0])
	else :
		x = math.atan2(-R[1,2], R[1,1])
		y = math.atan2(-R[2,0], sy)
		z = 0
	
	return np.array([x, y, z])


def get_min_rep(T):
	'''
	Convert 3*4 matrix into 6*1 vector

	[x y z alpha beta gamma]
	
	'''
	t=T[:,3]
	x,y,z=t

	angles=extract_angles(T[:,:3])

	T_vect=np.zeros(6)
	T_vect[:3]=t
	T_vect[3:6]=angles
	return T_vect

def eulerAnglesToRotationMatrix(theta) :
	'''
	Converts rotation angles about x,y and z axis to a rotation matrix
	'''
	R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
         
	R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
                 
	R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])                     
	R = np.dot(R_z, np.dot( R_y, R_x ))
 
	return R

def _get_back_T(T_fl):
	'''
	Converts the minimal representation of the pose into the normal 3x4 transformation matrix
	'''
	#print "The flattened pose input is ",T_fl,'\n\n\n'
	T = np.ones((3,4))
	T[:,3] = T_fl[:3]
	R = eulerAnglesToRotationMatrix(T_fl[3:6])
	T[:,:3] = R
	return T

def get_initial_pose(): 
	'''
	Pose for the first frame
	'''
	return np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])


def calc_photo_residual(i,frame,cur_keyframe,T):
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
	i = np.append(i,np.ones(1)) 
	i = i.astype(int)
	#3D point 3*1

	V = cur_keyframe.D[i[0]][i[1]] * np.matmul(cam_matrix_inv,i) 

	#Make V homogeneous 4*1
	V=np.append(V,1)

	#3D point in the real world shifted (3*4 x 4*1 = 3*1)
	u_prop = np.matmul(T,V)[:3]

	#3D point in camera frame (3*3 * 3*1)
	u_prop = np.matmul(cam_matrix,u_prop) 

	# Projection onto image plane
	u_prop = (u_prop/u_prop[2])[:2] 
	u_prop = u_prop.astype(int)

	u_prop = fix_u(u_prop)

	#print i,'\n',u_prop

	r = (int(cur_keyframe.I[i[0],i[1]]) - int(frame[u_prop[0],u_prop[1]]))
	# print r,'\n',cur_keyframe.I[i[0],i[1]],frame[u_prop[0],u_prop[1]],'\n\n'
	return r

#Not needed?
"""
def calc_photo_residual_d(u,D,T,frame,cur_keyframe): #For finding the derivative only
	'''
	Calculates photometric residual but only for finding the derivative

	Arguments:
		u: High gradient pixel location
		D: Depth value in previous keyframe at u
		T: Estimated pose
		frame: current frame as numpy array
		cur_keyframe: Previous keyframe as a Keyframe object

	Returns:
		r: Photometric residual
	'''
	u = np.append(u,np.ones(1))
	u = u.astype(int)
	Vp = D*np.matmul(cam_matrix_inv,u)
	Vp = tf.reshape(tf.concat([Vp,tf.constant(np.array([1],np.float64))],0),[4,1]) # 4x1
	T_t = tf.constant(T) # 3x4
	
	u_prop = tf.matmul(T_t,Vp)[:3] #3x1
	
	u_prop = tf.matmul(tf.constant(cam_matrix),u_prop)
	u_prop = (u_prop/u_prop[2])[:2]
	u_prop = tf.cast(u_prop,tf.int32)
	r = cur_keyframe.I[u[0]][u[1]] - frame[u_prop[0]][u_prop[1]]
	return r 
"""

def get_delD(D):
	return 0.01 #Change later to calculate based on input depth map


def calc_r_for_delr(u,D,frame,cur_keyframe,T):
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
	u = np.append(u,[1])
	v = D*np.matmul(cam_matrix_inv,u)
	v = np.append(v,[1])
	u_prop = np.matmul(T,v)[:3]
	u_prop = np.matmul(cam_matrix,u_prop)
	u_prop = ((u_prop/u_prop[2])[:2]).astype(int)

	u_prop = fix_u(u_prop)

	r = int(cur_keyframe.I[u[0],u[1]]) - int(frame[u_prop[0],u_prop[1]])
	return r

def delr_delD(u,frame,cur_keyframe,T):
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
	ulx = np.array([u[0] - 1,u[1]])
	urx = np.array([u[0] + 1,u[1]])
	uly = np.array([u[0],u[1] - 1])
	ury = np.array([u[0],u[1] - 1])

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
	deludelD = 1.0/delDdelu

	r_list = [0,0,0,0] # Just random

	"""
	u = np.append(u,[1])
	v = D*np.matmul(cam_matrix_inv,u)
	v = np.append(v,[1])
	u_prop = np.matmul(T,v)[:3]
	u_prop = np.matmul(cam_matrix,u_prop)
	u_prop = ((u_prop/u_prop[2])[:2]).astype(int)
	r_list[0] = cur_keyframe.I[u[0],u[1]] - frame[u_prop[0],u_prop[1]]"""

	# Calculate r_list
	calc_r_for_delr_v = np.vectorize(calc_r_for_delr,excluded = [2,3,4],signature = '(1),()->()')
	u_list = [ulx,urx,uly,ury]
	D_list = [Dlx,Drx,Dly,Dry]
	r_list = calc_r_for_delr_v(u_list,D_list,frame,cur_keyframe,T)

	delrdelu = ((r_list[0] - r_list[1])**2 + (r_list[2] - r_list[3])**2)**0.5

	delr = delrdelu*deludelD
	return delr

def calc_photo_residual_uncertainty(u,frame,cur_keyframe,T):
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
	deriv = delr_delD(u,frame,cur_keyframe,T)
	sigma = (sigma_p**2 + (deriv**2)*cur_keyframe.U[u[0]][u[1]])**0.5
	return sigma

def huber_norm(x):
	'''
	Calculates and Returns the huber norm

	Arguments:
		x: Input

	Returns:
		Huber norm of x
	'''
	delta = 1 #Change later
	if abs(x)<delta:
		return 0.5*(x**2)
	else:
		return delta*(abs(x) - (delta/2))

def ratio_residual_uncertainty(u,frame,cur_keyframe,T):
	return huber_norm(calc_photo_residual(u,frame,cur_keyframe,T)/calc_photo_residual_uncertainty(u,frame,cur_keyframe,T))

def calc_cost(uu,frame,cur_keyframe,T,flag = 1):
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
	# Should we include huber norm also here
	if flag==0:
		T = _get_back_T(T)
	ratio_residual_uncertainty_v = np.vectorize(ratio_residual_uncertainty,excluded = [1,2,3],signature = '(1)->()')
	return ratio_residual_uncertainty_v(uu,frame,cur_keyframe,T)

def get_jacobian(dof,u,frame,cur_keyframe,T_s):
	'''
	Returns the Jacobian of the Residual Error wrt the Pose (r wrt T)

	r == (dof,1)
	T == (6,1)

	delr/delT == (dof,6)
	delr/delu == (dof,1)
	delT/delu == (6,1)
	delu/delT == (1,6) 
	delr/delT = (delr/delu)*(delu/delT)

	Arguments:
		dof: Number of high gradient elements we are using
		u: An array containing the high gradient elements
		frame: Numpy array o the current frame
		cur_keyframe: Previous keyframe as a Keyframe class
		T_s: Current estimated Pose in minimal representation

	Returns:
		J: The required Jacobian (dofx6)
	'''
	ratio = 5 # Change later
	T_list1 = np.array([])
	T_list2 = np.array([])
	for i in range(0,6):
		temp1 = np.array(T_s) # So it actually creates a copy and does not refer to the same array
		temp2 = np.array(T_s)  
		temp1[i] = T_s[i]+(ratio*T_s[i])
		temp2[i] = T_s[i]-(ratio*T_s[i])
		#print '\n\na == \n',T_s,'\n',temp1,'\n',temp2,'\n\n'
		if i==0:
			T_list1 = np.array([temp1])
			T_list2 = np.array([temp2])
			continue
		T_list1 = np.append(T_list1,[temp1],0)
		T_list2 = np.append(T_list2,[temp2],0)
	calc_cost_v = np.vectorize(calc_cost,excluded = [0,1,2,4],signature = '(1)->(dof)') # Dont tell number of rows. Just tell shape of each row(for input and output)
	r1 = calc_cost_v(u,frame,cur_keyframe,T_list1,0) # 6xdof
	r2 = calc_cost_v(u,frame,cur_keyframe,T_list2,0) # 6xdof
	J = np.array(r1 - r2)
	#print T_list1,'\n\n'
	#print T_list2
	return J.T 

def get_W(dof,stack_r):
	'''
	Returns the weight matrix for weighted Gauss-Newton Optimization

	Arguments:
		dof: Number of high gradient elements we are using
		stack_r: The stacked residual error as a numpy array (of length dof)

	Returns:
		W: Weight Matrix
	'''
	W = np.zeros((dof,dof))
	for i in range(dof): 
		W[i][i] = (dof + 1)/(dof + stack_r[i]**2)
	return W

def exit_crit(delT):
	'''
	Checks for when to exit the loop while doing Gauss - Newton Optimization
	
	Arguments: 
		delT: The right multiplied increment of the pose

	Returns:
		1(to exit) or 0(not to exit)
	'''
	#Change later
	# TO DO
	return 1 

def minimize_cost_func(u,frame, cur_keyframe):
	'''
	Does Weighted Gauss-Newton Optimization

	Arguments:
		u: array of points in high gradient areas of the current frame
		frame: Current frame(as a Numpy array)
		cur_keyframe: The previous keyframe of the Keyframe Class

	Returns:
		T: The camera Pose
	'''
	dof = len(u)
	# Random pose
	T_s = np.random.random((6))
	T = _get_back_T(T_s)

	while True:
		stack_r = np.array(calc_cost(u,frame,cur_keyframe,T)) # dofx1
		J = get_jacobian(dof,u,frame,cur_keyframe,T_s) # dofx6
		Jt = J.transpose() # 6xdof
		W = get_W(dof,stack_r) # dofxdof - diagonal matrix
		temp = np.matmul(np.matmul(Jt,W),J) # 6x6
		if np.linalg.det(temp) == 0:
			print "Singular matrix encountered"
			print J
		hess = np.linalg.inv(temp) # 6x6
		delT = np.matmul(hess,Jt) # 6xdof
		delT = np.matmul(delT,W) # 6xdof
		delT = -np.matmul(delT,stack_r) # 6x1

		if exit_crit(delT):
			break

		delT = _get_back_T(delT) #3x4
		delT = np.append(delT,[[0,0,0,1]],0) # 4x4
		T_4 = np.append(T,[[0,0,0,1]],0) # 4x4
		T = np.matmul(T_4,delT)[:3] # 3x4
		T_s = get_min_rep(T) # 6x1
	return T

def check_keyframe(T):
	'''
	Checks the Pose of a new frame to see if it is a keyframe(if the camera has moved too far from the previous keyframe)

	Arguments: 
		T: Pose of new frame wrt to prev keyframe

	Returns:
		Either 1(is a keyframe) or 0(not a keyframe)
	'''
	W = np.ones((6,6)) #Weight Matrix - change later
	threshold = 0
	T_s = get_min_rep(T)	
	temp = matmul(W,T_s) # 6x1
	temp = matmul(T_s.transpose(),temp)
	return temp>=threshold

def _delay():
	'''
	Adds a time delay
	'''
	time.sleep(60) #Change later

def _exit_program():
	'''
	Exits the program
	'''
	sys.exit(0)

def test_highgrad():
	'''
	Test thresholding based extraction of high gradient element.

	Laplace filter used
	'''
	im_x,im_y=im_size
	dummy_image=np.uint8(np.random.random((im_x,im_y,3))*256)
	dummy_image_grey=np.uint8((dummy_image[:,:,0]+dummy_image[:,:,1]+dummy_image[:,:,2])/3)

	# Test high  grad
	result=get_highgrad_element(dummy_image_grey)
	print("Testing high grad {} ".format(result))

	assert(result.shape[1]==2)

def test_min_cost_func():
	'''
	Test minimum cost function:

	* Take current frame, keyframe
	'''
	# Image Size
	im_x,im_y = im_size

	# Random high grad points
	u_test = np.array([[5,4],[34,56],[231,67],[100,100],[340,237]])

	# Random frame
	frame_test = np.uint8(np.random.random((im_x,im_y))*256)

	# Current key frame, depth, pose, uncertainuty
	cur_key_test_im = np.uint8(np.random.random((im_x,im_y,3))*256)
	cur_key_test_im_grey = np.uint8((cur_key_test_im[:,:,0]+cur_key_test_im[:,:,1]+cur_key_test_im[:,:,2])/3)
	cur_key_depth = np.random.random((im_x,im_y))
	dummy_pose=np.eye(4)[:3]
	cur_key_unc = np.ones((im_x,im_y))

	cur_key = Keyframe(dummy_pose,cur_key_depth,cur_key_unc,cur_key_test_im_grey)

	print("Testing minimize cost func",minimize_cost_func(u_test,frame_test,cur_key))

def test_get_min_rep():
	T = np.array([[0.36,0.48,-0.8,5],[-0.8,0.6,0,3],[0.48,0.64,0.60,8]])
	#T = np.array([[1,0,0,5],[0,math.sqrt(3)/2,0.5,3],[0,-0.5,math.sqrt(3)/2,8]]) # 30 degree rotation about x axis - works
	print (T,'\n')
	print ("Testing get_min_rep",get_min_rep(T))
	return get_min_rep(T)

def test_get_back_T():
	T_s = test_get_min_rep()
	print (T_s,'\n')
	print("Testing get_back_T",_get_back_T(T_s))

def test_find_epipoles():
	t_s = np.random.random((6))
	T = _get_back_T(t_s)
	E = stereo_match.get_essential_matrix(T)
	F = np.matmul(camera_matrix_inv.T,np.matmul(E,camera_matrix_inv))
	e1,e2 = stereo_match.find_epipoles(F)
	print F
	print e1

	
if __name__=='__main__':
	test_find_epipoles()