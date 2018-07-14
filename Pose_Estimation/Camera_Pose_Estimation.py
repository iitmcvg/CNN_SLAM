import numpy as np 
import cv2
import tensorflow as tf
import sys
import time
import pose_estimation.refine_depth_map 
import pose_estimation.depth_map_fusion 

im_size = (480,640)
sigma_p = 0 # Some white noise variance thing
index_matrix = np.dstack(np.meshgrid(np.arange(480),np.arange(640),indexing = 'ij'))

class Keyframe:
	def __init__(self, pose, depth, uncertainty, image):
		self.T = pose # 3x4 transformation matrix
		self.D = depth
		self.U = uncertainty
		self.I = image

def get_camera_image():
	cam = cv2.VideoCapture(0)
	ret,frame = cam.read()
	frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #Using single channel image
	frame_ar = np.array(frame)
	return ret,frame_ar,frame

def get_camera_matrix(): #Change to read from camera calib file
	return np.zeros((3,3))

cam_matrix = get_camera_matrix()
cam_matrix_inv = np.linalg.inv(cam_matrix)

def get_cnn_depth(): #To get CNN predicted depth from an image


def find_uncertainty(u,D,D_prev,T):
	u.append(1) #Convert to homogeneous
	V = D * np.matmul(cam_matrix_inv,u) #World point
	V.append(1)
	u_prop = np.matmul(cam_matrix,T)
	u_prop = np.matmul(u_prop,V)
	u_prop = u_prop/u_prop[2]
	u_prop.pop()
	U = D[u[0]][u[1]] - D_prev[u_prop[0]][u_prop[1]]
	return U**2

def get_uncertainty(T,D,prev_keyframe):
	T = np.matmul(np.linalg.inv(T),prev_keyframe.T) #Check if this is right
	find_uncertainty_v = np.vectorize(find_uncertainty)
	U = find_uncertainty_v(index_matrix,D,prev_keyframe.D,T) #Check
	return U

def get_initial_uncertainty(): #To get uncertainty map for the first frame


def get_initial_pose(): #Pose for the first frame


def get_highgrad_element(img): #Test this out separately
	threshold_grad = 100 #Change later
	laplacian = cv2.Laplacian(img,cv2.CV_8U)
	ret,thresh = cv2.threshold(laplacian,threshold_grad,255,cv2.THRESH_BINARY)
	u = cv2.findNonZero(thresh)
	return u

def calc_photo_residual(i,frame,cur_keyframe,T):
	i.append(1) #Make i homogeneous
	V = cur_keyframe.D[i[0]][i[1]] * np.matmul(cam_matrix_inv,i) #3D point
	V.append(1) #Make V homogeneous
	u_prop = np.matmul(T,V) #3D point in the real world shifted
	u_prop = np.matmul(cam_matrix,u_prop) #3D point in camera frame
	u_prop = u_prop/u_prop[2] #Projection onto image plane
	u_prop.pop()
	r = (cur_keyframe.I[i[0]][i[1]] - frame.I[u_prop[0]][u_prop[1]])
	return r

def calc_photo_residual_d(u,D,T,frame,cur_keyframe): #For finding the derivative only
	u.append(1)
	V = D*np.matmul(cam_matrix_inv,i)
	V.append(1)
	u_prop = np.matmul(T,V)
	u_prop = np.matmul(cam_matrix,u_prop)
	u_prop = u_prop/u_prop[2]
	u_prop.pop()
	r = cur_keyframe.I[u[0]][u[1]] - frame.I[u_prop[0]][u_prop[1]]
	return r 

def delr_delD(u,frame,cur_keyframe,T):
	D = tf.constant(cur_keyframe.D[u[0]][u[1]])
	r = calc_photo_residual_d(u,D,T,frame,cur_keyframe)
	delr = 0
	with tf.Session() as sess:
		delr = tf.gradients(r,D)
	return delr

def calc_photo_residual_uncertainty(u,frame,cur_keyframe,T):
	deriv = delr_delD(u,frame,cur_keyframe,T)
	sigma = (sigma_p**2 + (deriv**2)*cur_keyframe.U[u[0]][u[1]])**0.5
	return sigma

def huber_norm(x):
	delta = 1 #Change later
	if abs(x)<delta:
		return 0.5*(x**2)
	else 
		return delta*(abs(a) - (delta/2))

"""
def calc_cost_func(u,frame,cur_keyframe,T): #Calculates the aggregate cost (not as a list)
	sum = 0
	for i in u:
		sum = sum + huber_norm(w[i]*calc_photo_residual(i,frame,cur_keyframe,T)/calc_photo_residual_uncertainty(i,frame,cur_keyframe,T))
	return sum
"""

def calc_cost(u,frame,cur_keyframe,T):
	r = []
	for i in u:
		r.append(huber_norm(calc_photo_residual(i,frame,cur_keyframe,T)/calc_photo_residual_uncertainty(i,frame,cur_keyframe,T))) #Is it uncertainty or something else?
	return r

def calc_cost_jacobian(u,frame,cur_keyframe,T_s): #Use just for calculating the Jacobian
	T = np.reshape(T_s,(3,4))
	r = []
	for i in u:
		r.append(huber_norm(calc_photo_residual(i,frame,cur_keyframe,T)/calc_photo_residual_uncertainty(i,frame,cur_keyframe,T)))
	return r

def get_jacobian(dof,u,frame,cur_keyframe,T):
	T_s = T.flatten()
	T_c = tf.constant(T_s) #Flattened pose in tf
	r_s = calc_cost_jacobian(u,frame,keyframe,T_c)
	with tf.Session() as sess:
		_,J = tf.run(tf.test.compute_gradient(r_s,(dof,1),T_c,(12,1))) #Returns two jacobians... (Other two parameters are the shapes)
	return J

def get_W(dof,stack_r):
	W = np.zeros((dof,dof))
	for i in range(dof):
		W[i][i] = (dof + 1)/(dof + stack_r[i]**2)
	return W

def exit_crit(delT):


def minimize_cost_func(u,frame, cur_keyframe): #Does Weighted Gauss-Newton Optimization
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
	W = np.zeros((12,12)) #Weight Matrix
	threshold = 0
	R = T[:3][:3]
	t = T[3][:3]
	R = R.flatten()
	E = np.concatenate(R,t) # 12 dimensional 	
	temp = matmul(W,E)
	temp = matmul(E.transpose(),temp)
	if temp>=threshold:
		return 1
	else
		return 0

def put_delay():
	time.sleep(60) #Change later

def exit_program():
	sys.exit(0)

def main():
	ret,frame,image = get_camera_image() #frame is a numpy array
	K = [] #Will be a list of keyframe objects
	ini_depth = get_cnn_depth(frame)
	ini_uncertainty = get_initial_uncertainty()
	ini_pose = get_initial_pose()
	K.append(Keyframe(ini_pose,ini_depth,ini_uncertainty,frame)) #First Keyframe appended
	cur_keyframe = K[0]
	cur_index = 0
	while(True): #Loop for keyframes
		while(True): #Loop for normal frames
			ret,frame,image = get_camera_image() #frame is the numpy array
			if not ret:
				exit_program()
			u = get_highgrad_element(image) #consists of a list of points. Where a point is a list of length 2.
			T = minimize_cost_func(u,frame,cur_keyframe) 
			if check_keyframe(T):                    
				depth = get_cnn_depth(frame)	
				cur_index += 1
				uncertainty = get_uncertainty(T,D,K[cur_index - 1])
				K.append(Keyframe(T,depth,uncertainty,frame))
				K[cur_index].D,K[cur_index].U = fuse_depth_map(K[cur_index],K[cur_index - 1])
				cur_keyframe = K[cur_index]
				put_delay()
				break
			else:
				cur_keyframe.D,cur_keyframe.U = refine_depth_map(frame,T,cur_keyframe)
				put_delay()

if__name__ == "__main__":
	main()