import numpy as np 
import cv2
import tensorflow as tf
import sys
import time
import argparse
import math

im_size = (480,640)
sigma_p = 5 # Some white noise variance thing
index_matrix = np.reshape(np.dstack(np.meshgrid(np.arange(480),np.arange(640),indexing = 'ij')),(480*640,2))
cam_matrix = np.eye(3,3) # 3x3 Intrinsic camera matrix - converts 3x3 point in camera frame to homogeneous repreentation of an image coordiante
cam_matrix_inv = np.linalg.inv(cam_matrix)

class Keyframe:
	def __init__(self, pose, depth, uncertainty, image):
		self.T = pose # 4x4 transformation matrix # 6 vector
		self.D = depth
		self.U = uncertainty
		self.I = image


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

def loss_tf(u,frame,cur_keyframe,T_s):
	T = _get_back_T(T_s.numpy())
	cost = calc_cost(u,frame,cur_keyframe,T)
	cost = tf.reduce_sum(cost)
	return cost#+T_s[0]-T_s[0]

def grad_tf(u,frame,cur_keyframe,T_s):
	T1 = T_s.numpy()
	grad = np.array([1.0,1.0,1.0,1.0,1.0,1.0])
	for i in range(6):
		Ta = T1.copy()
		x = Ta[i]/10.0
		Ta[i] = Ta[i] - x
		Tb = T1.copy()
		Tb[i] = Tb[i] + x
		Ta = tf.contrib.eager.Variable(Ta)
		Tb = tf.contrib.eager.Variable(Tb)
		costa = loss_tf(u,frame,cur_keyframe,Ta)
		costb = loss_tf(u,frame,cur_keyframe,Tb)
		grad[i] = (costb - costa)/(2*x)
	"""with tf.GradientTape() as tape:
		lossa = loss_tf(u,frame,cur_keyframe,T_s)
	#print(lossa)
	#print(T_s)
	grad = tape.gradient(lossa,T_s)
	#print(grad)"""
	grads = tf.contrib.eager.Variable(grad,dtype = tf.float32)
	#print(grad)
	return grads

def minimize_cost_with_tf(u,frame,cur_keyframe):
	tf.enable_eager_execution()
	dof = len(u)
	T_s = tf.contrib.eager.Variable(np.random.random((6)),dtype = tf.float32)
	optimizer = tf.train.AdamOptimizer(learning_rate = 0.1)
	i = 0
	while(loss_tf(u,frame,cur_keyframe,T_s)>0.1): # Change later
		grads = grad_tf(u,frame,cur_keyframe,T_s)
		#print("grad = ",grads)
		optimizer.apply_gradients(zip([grads],[T_s]),global_step = tf.train.get_or_create_global_step())
		i = i+1
		print(grads)
	return _get_back_T(T_s.numpy()),loss_tf(u,frame,cur_keyframe,T_s),i

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

	print("Testing minimize cost func",minimize_cost_with_tf(u_test,frame_test,cur_key))


if __name__=='__main__':
	test_min_cost_func()