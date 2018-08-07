#Change u to be a numpy array

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

# Modules
#import Pose_Estimation.depth_map_fusion as depth_map_fusion
#import pose_estimation.monodepth as monodepth

im_size = (480,640)
sigma_p = 0 # Some white noise variance thing
index_matrix = np.dstack(np.meshgrid(np.arange(480),np.arange(640),indexing = 'ij'))
cam_matrix = np.eye(3,3) #Change later
cam_matrix_inv = np.eye(3,3) #Change later

class Keyframe:
	def __init__(self, pose, depth, uncertainty, image):
		self.T = pose # 4x4 transformation matrix # 6 vector
		self.D = depth
		self.U = uncertainty
		self.I = image

	def _isRotationMatrix(self,R) :
		'''
		Checks if a matrix is a valid rotation matrix.
		'''
		Rt = np.transpose(R)
		shouldBeIdentity = np.dot(Rt, R)
		I = np.identity(3, dtype = R.dtype)
		n = np.linalg.norm(I - shouldBeIdentity)
		return n < 1e-6
	

	def _extract_angles(self):
		'''
		Extract rotation angles

		Returns: aplha, beta, gamma (as np array)
		'''

		assert(self._isRotationMatrix(R))
     
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

	@property
	def T_vec(self):
		'''
		Convert 4*4 matrix into 6*1 vector

		[x y z alpha beta gamma]
	
		'''

		t=self.T[:3,3].T
		x,y,z=t

		angles=self._extract_angles()

		self.T_vec=np.zeros(6)
		self.T_vec[:3]=t
		self.T_vec[:3]=angles

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
	u=np.append(u,np.ones(1)) #Convert to homogeneous

	V = D * np.matmul(cam_matrix_inv,u) #World point
	V=np.append(V,np.ones(1))

	u_prop = np.matmul(cam_matrix,T)
	u_prop = np.matmul(u_prop,V)
	u_prop = u_prop/u_prop[2]
	u_prop=u_prop[:-1]

	U = D[u[0]][u[1]] - D_prev[u_prop[0]][u_prop[1]]
	return U**2

def get_uncertainty(T,D,prev_keyframe):
	'''
	Finds the uncertainty map for a new keyframe

	Arguments:
		T: Pose of new keyframe
		D: Depth map of new keyframe
		prev_keyframe: Previous keyframe of Keyframe class

	Returns:
		U: Uncertainty map
	'''
	T = np.matmul(np.linalg.inv(T),prev_keyframe.T) #Check if this is right
	find_uncertainty_v = np.vectorize(find_uncertainty)
	U = find_uncertainty_v(index_matrix,D,prev_keyframe.D,T) #Check
	return U

def get_initial_uncertainty(): 
	'''
	To get uncertainty map for the first frame
	'''
	raise NotImplementedError

def get_initial_pose(): 
	'''
	Pose for the first frame
	'''
	raise NotImplementedError

def get_highgrad_element(img,threshold=100):
	'''
	Finds high gradient areas in the image

	Arguments:
		img: Input image

	Returns:
		u: List of pixel locations
	'''
	
	laplacian = cv2.Laplacian(img,cv2.CV_8U)
	ret,thresh = cv2.threshold(laplacian,threshold,255,cv2.THRESH_BINARY)
	u = cv2.findNonZero(thresh)
	return u

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
	V = cur_keyframe.D[i[0]][[1]] * np.matmul(cam_matrix_inv,i) 

	#Make V homogeneous 4*1
	V=np.append(V,1)

	#3D point in the real world shifted (3*4 x 4*1 = 3*1)
	u_prop = np.matmul(T,V)[:3]

	#3D point in camera frame (3*3 * 3*1)
	u_prop = np.matmul(cam_matrix,u_prop) 

	# Projection onto image plane
	u_prop = (u_prop/u_prop[2])[:2] 
	u_prop = u_prop.astype(int)
	# Residual width*height
	r = (cur_keyframe.I[i[0]][i[1]] - frame[u_prop[0]][u_prop[1]])

	return r

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
	V = D*np.matmul(cam_matrix_inv,i)
	V.append(1)
	u_prop = np.matmul(T,V)
	u_prop = np.matmul(cam_matrix,u_prop)
	u_prop = u_prop/u_prop[2]
	u_prop.pop()
	r = cur_keyframe.I[u[0]][u[1]] - frame.I[u_prop[0]][u_prop[1]]
	return r 

def delr_delD(u,frame,cur_keyframe,T):
	'''
	Finds the derivative of the photometric residual wrt depth

	Arguments:
		u: High gradient pixel location
		frame: Current frame as numpy array
		cur_keyframe: Previous keyframe as a Keyframe object
		T: Estimated pose

	Returns:
		delr: The derivative
	'''
	D = tf.constant(cur_keyframe.D[u[0]][u[1]])
	r = calc_photo_residual_d(u,D,T,frame,cur_keyframe)
	delr = 0
	with tf.Session() as sess:
		_,delr = tf.test.compute_gradient(r,(1),D,(1))
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
		return delta*(abs(a) - (delta/2))

def calc_cost(u,frame,cur_keyframe,T):
	'''
	Calculates the residual error.

	Arguments:
		u: A list containing the high gradient elements
		frame: Numpy array o the current frame
		cur_keyframe: Previous keyframe as a Keyframe class
		T: Current estimated Pose

	Returns:
		r: Residual error as a list
	'''
	r = []
	for i in u:
		r.append(huber_norm(calc_photo_residual(i,frame,cur_keyframe,T)/calc_photo_residual_uncertainty(i,frame,cur_keyframe,T))) #Is it uncertainty or something else?
	return r

def calc_cost_jacobian(u,frame,cur_keyframe,T_s): 
	'''
	Calculates the residual error for the Jacobian

	Arguments:
		u: A list containing the high gradient elements
		frame: Numpy array o the current frame
		cur_keyframe: Previous keyframe as a Keyframe class
		T_s: Current estimated Pose as a flattened numpy array

	Returns:
		r: Residual error as a list
	'''
	T = np.reshape(T_s,(3,4))
	r = []
	for i in u:
		r.append(huber_norm(calc_photo_residual(i,frame,cur_keyframe,T)/calc_photo_residual_uncertainty(i,frame,cur_keyframe,T)))
	return r

def get_min_rep(T):
	'''
	Converts the pose to its minimal representation (epsilon)

	Arguments:
		T: Input pose

	Returns: 
		T_s: Minimal representation of pose
	'''
	
	
def get_jacobian(dof,u,frame,cur_keyframe,T):
	'''
	Returns the Jacobian of the Residual Error wrt the Pose

	Arguments:
		dof: Number of high gradient elements we are using
		u: A list containing the high gradient elements
		frame: Numpy array o the current frame
		cur_keyframe: Previous keyframe as a Keyframe class
		T: Current estimated Pose

	Returns:
		J: The required Jacobian
	'''
	T_s = get_min_rep(T)
	T_c = tf.constant(T_s) #Flattened pose in tf
	r_s = calc_cost_jacobian(u,frame,keyframe,T_c)
	with tf.Session() as sess:
		_,J = tf.run(tf.test.compute_gradient(r_s,(dof,1),T_c,(12,1))) #Returns two jacobians... (Other two parameters are the shapes)
	return J

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
	Checks for the when to exit the loop while doing Gauss - Newton Optimization
	
	Arguments: 
		delT: The right multiplied increment of the pose

	Returns:
		1(to exit) or 0(not to exit)
	'''

def minimize_cost_func(u,frame, cur_keyframe):
	'''
	Does Weighted Gauss-Newton Optimization

	Arguments:
		u: List of points in high gradient areas of the current frame
		frame: Current frame(as a Numpy array)
		cur_keyframe: The previous keyframe of the Keyframe Class

	Returns:
		T: The camera Pose
	'''
	dof = len(u)
	T = np.zeros((3,4)) #Do random initialization later
	while(1):
		stack_r = calc_cost(u,frame,cur_keyframe,T)
		J = get_jacobian(dof,u,frame,cur_keyframe,T)
		Jt = J.transpose()
		W = get_W(dof,stack_r) #dof x dof - diagonal matrix
		hess = np.linalg.inv(np.matmul(np.matmul(Jt,W),J)) # 12x12
		delT = np.matmul(hess,Jt)
		delT = np.matmul(delT,W)
		delT = -np.matmul(delT,stack_r) 
		T = np.dot(delT,T.flatten()) #Or do subtraction?
		T = np.reshape(T,(3,4))
		if exit_crit(delT):
			break
	return T

def check_keyframe(T):
	'''
	Checks the Pose of a new frame to see if it is a keyframe(if the camera has moved too far from the previous keyframe)

	Arguments: 
		T: Pose of new frame wrt to prev keyframe

	Returns:
		Either 1(is a keyframe) or 0(not a keyframe)
	'''
	W = np.zeros((12,12)) #Weight Matrix
	threshold = 0
	R = T[:3][:3]
	t = T[3][:3]
	R = R.flatten()
	E = np.concatenate(R,t) # 12 dimensional 	
	temp = matmul(W,E)
	temp = matmul(E.transpose(),temp)
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
	im_x,im_y=im_size
	dummy_image=np.uint8(np.random.random((im_x,im_y,3))*256)

	dummy_image_grey=(dummy_image[:,:,0]+dummy_image[:,:,1]+dummy_image[:,:,2])/3
	dummy_image_grey=np.uint8(dummy_image_grey)

	dummy_depth=np.random.random((im_x,im_y))

	dummy_pose=np.eye(4)[:3]
	dummy_uncertainity=np.ones((im_x,im_y))

	# Dummy keyframe
	dummy_frame= Keyframe(dummy_pose,dummy_depth, dummy_uncertainity, dummy_image_grey)
	
	# Test high  grad
	print("Testing high grad",get_highgrad_element(dummy_image_grey))

def test_min_cost_func():
	im_x,im_y = im_size
	u_test = np.array([[5,4],[34,56],[231,67],[100,100],[340,237]])
	frame_test = np.uint8(np.random.random((im_x,im_y))*256)
	cur_key_test_im = np.uint8(np.random.random((im_x,im_y,3))*256)
	cur_key_test_im_grey = (cur_key_test_im[:,:,0]+cur_key_test_im[:,:,1]+cur_key_test_im[:,:,2])/3
	cur_key_test_im_grey = np.uint8(cur_key_test_im_grey)

	cur_key_depth = np.random.random((im_x,im_y))
	dummy_pose=np.eye(4)[:3]
	cur_key_unc = np.ones((im_x,im_y))

	cur_key = Keyframe(dummy_pose,cur_key_depth,cur_key_unc,cur_key_test_im_grey)

	print("Testing minimize cost func",minimize_cost_func(u_test,frame_test,cur_key))


if __name__=='__main__':
	test_min_cost_func()