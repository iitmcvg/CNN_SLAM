# Check all incomplete functions - go through all and see
# Parallelize all for loops
# Normalize depth map****
# Pack better while parallelizing

# Libraries
import numpy as np 
import cv2
import tensorflow as tf
import sys
import time
import argparse
from matplotlib import pyplot as plt
from PIL import Image

# Modules
import pose_estimation.depth_map_fusion as depth_map_fusion
import pose_estimation.stereo_match as stereo_match
from params import *
import pose_estimation.camera_pose_estimation as camera_pose_estimation
import pose_estimation.find_uncertainty as find_uncertainty
from keyframe_utils import Keyframe as Keyframe
import monodepth_infer.monodepth_single as monodepth_single
#from pose_graph_optimisation.pose_graph_optimisation import cloud_for_vis

parser = argparse.ArgumentParser(description='Monodepth TensorFlow implementation.')
parser.add_argument('--mono_checkpoint_path', default = "checkpoints/model_kitti_resnet/model_kitti_resnet.data" ,type=str,   help='path to a specific checkpoint to load')
parser.add_argument('--input_height', type=int,   help='input height', default=480)
parser.add_argument('--input_width', type=int,   help='input width', default=640)
args = parser.parse_args()


# Video cam
cam = cv2.VideoCapture(0)

def get_camera_image():
	'''
	Returns:

	* ret: Whether camera captured or not 
	* frame: 3 channel image
	* frame_grey greyscale
	'''
	ret,frame = cam.read()
	frame_grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) # Using single channel image
	return ret,frame,frame_grey

def get_camera_matrix(path=None): 
	'''
	Read intrinsic matrix from npy file.

	Change to read from camera calib file.

	Use identity matrix for testing.
	'''
	if path:
		return np.load(path)
	else:
		return np.eye(3)

def get_highgrad_element(img):
	'''
	Finds high gradient areas in the image

	Arguments:
		img: Input image

	Returns:
		u: Array of pixel locations
		Shape (X,2)
		X: number of high grad elements
	'''
	laplacian = cv2.Laplacian(img,cv2.CV_8U)
	ret,thresh = cv2.threshold(laplacian,threshold_for_high_grad,255,cv2.THRESH_BINARY)
	u = cv2.findNonZero(thresh) # Returns in (x,y) format. Need to exchange
	u = np.squeeze(np.array(u))
	temp = np.copy(u[:,0])
	u[:,0] = u[:,1]
	u[:,1] = temp
	return np.squeeze(np.array(u))

def check_keyframe(T):
	return 0 # Change later

def _exit_program():
	raise NotImplementedError

def main():
	# INIT monodepth_single session
	sess=monodepth_single.init_monodepth(args.mono_checkpoint_path)
	keyf1 = cv2.resize(cv2.imread("pose_estimation/stereo.jpeg"),(im_size[1],im_size[0]),interpolation = cv2.INTER_CUBIC)

	# INIT camera matrix
	cam_matrix = get_camera_matrix()

	try: 
		cam_matrix_inv = np.linalg.inv(cam_matrix)
	except:
		raise (Error, "Verify camera matrix")

	# Image is 3 channel, frame is grayscale
	#ret,image,frame = get_camera_image()

	# List of keyframe objects
	K = []

	# Predict depth
	image = cv2.imread("pose_estimation/stereo.jpeg")
	ini_depth = monodepth_single.get_depth_map(sess,keyf1)
	plt.imshow(ini_depth)
	plt.show()

    # Initalisation
	ini_uncertainty = find_uncertainty.get_initial_uncertainty()
	ini_pose,ini_covariance = camera_pose_estimation.get_initial_pose()
	ini_covariance = camera_pose_estimation.get_initial_covariance()
	K.append(Keyframe(ini_pose,ini_depth,ini_uncertainty,frame,image,ini_covariance)) 
	cur_keyframe = K[0]
	cur_index = 0
	prev_frame = cur_keyframe.I
	prev_pose = cur_keyframe.T
	while(True):

		ret,image,frame = get_camera_image() # frame is the numpy array

		if not ret:
			_exit_program()

        # Finds the high gradient pixel locations in the current frame
		u = get_highgrad_element(frame) 

        # Finds pose of current frame by minimizing photometric residual (wrt prev keyframe)
		T,C,_ = camera_pose_estimation.minimize_cost_func(u,frame,cur_keyframe) 
            
		if check_keyframe(T):			
			# If it is a keyframe, add it to K after finding depth and uncertainty map                    
			depth = monodepth_single.get_depth_map(sess,image)	
			cur_index += 1
			uncertainty = find_uncertainty.get_uncertainty(T,D,K[cur_index - 1])
			# T = np.append(T,np.array([[0,0,0,1]]),0)
			# cur_keyframe.T = np.append(cur_keyframe.T,np.array([[0,0,0,1]]),0)
			# T_abs = np.matmul(T,cur_keyframe.T) # absolute pose of the new keyframe
			# T = T[:3]
			# cur_keyframe.T = cur_keyframe.T[:3]
			K.append(Keyframe(T,depth,uncertainty,frame,image,C))
			K[cur_index].D,K[cur_index].U = depth_map_fusion.fuse_depth_map(K[cur_index],K[cur_index - 1])
			cur_keyframe = K[cur_index]

			point_cloud = graph_optimization.pose_graph_optimization(K)
			# Visualise point cloud

		else: # Refine and fuse depth map. Stereo matching consecutive frame
			D_frame = stereo_match.stereo_match(prev_frame,frame,prev_pose,T)
			U_frame = find_uncertainty.get_uncertainty(T,D,cur_keyframe)
			frame_obj = Keyframe(T,D_frame,U_frame,frame) # frame as a keyframe object
			cur_keyframe.D,cur_keyframe.U = depth_map_fusion.fuse_depth_map(frame_obj,cur_keyframe)
		
			# Generate and visualise point cloud?

		_delay()
		prev_frame = frame
		prev_pose = T
		continue

def test_without_cnn():
	keyf1 = cv2.resize(cv2.imread("pose_estimation/stereo.jpeg"),(im_size[1],im_size[0]),interpolation = cv2.INTER_CUBIC)
	f = cv2.resize(cv2.imread("pose_estimation/stereo(1).jpeg"),(im_size[1],im_size[0]),interpolation = cv2.INTER_CUBIC)
	gray_keyf1 = cv2.cvtColor(keyf1,cv2.COLOR_BGR2GRAY)
	# INIT camera matrix
	cam_matrix = get_camera_matrix()
	try: 
		cam_matrix_inv = np.linalg.inv(cam_matrix)
	except:
		raise (Error, "Verify camera matrix")

	# Image is 3 channel, frame is grayscale
	ret,image,frame = 1,keyf1,gray_keyf1

	# List of keyframe objects
	K = []

	# Predict depth
	ini_depth = np.random.random(im_size)*255

    # Initalisation
	ini_uncertainty = find_uncertainty.get_initial_uncertainty()
	ini_pose = camera_pose_estimation.get_initial_pose()
	ini_C = camera_pose_estimation.get_initial_covariance()
	K.append(Keyframe(ini_pose,ini_depth,ini_uncertainty,frame,image,ini_C)) 
	cur_keyframe = K[0]
	cur_index = 0
	prev_frame = cur_keyframe.F
	prev_pose = cur_keyframe.T

	# ret,image,frame = get_camera_image() # frame is the numpy array

	print ("*****************************")
	print ("Initialised first keyframe")
	print ("*****************************\n")

	frame = cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)
    # Finds the high gradient pixel locations in the current frame
	u = get_highgrad_element(frame) 
 
	print ("**********************************************")
	print ("Got high grad elements. Going to estimate pose")
	print ("**********************************************\n")

    # Finds pose of current frame by minimizing photometric residual (wrt prev keyframe)
	T,C,_ = camera_pose_estimation.minimize_cost_func(u,frame,cur_keyframe) #ini_pose,ini_C,3#
            
	print ("*****************************")
	print ("Estimated Pose")
	print ("*****************************\n")
	print ("T = ", T,'\n')
	if check_keyframe(T):	
		print ("Error: second frame cant be keyframe\n")	
		# If it is a keyframe, add it to K after finding depth and uncertainty map                    
		depth = monodepth_single.get_cnn_depth(sess,image)	
		cur_index += 1
		uncertainty = find_uncertainty.get_uncertainty(T,D,K[cur_index - 1])
		T = np.append(T,np.array([[0,0,0,1]]),0)
		cur_keyframe.T = np.append(cur_keyframe.T,np.array([[0,0,0,1]]),0)
		T_abs = np.matmul(T,cur_keyframe.T) # absolute pose of the new keyframe
		T = T[:3]
		cur_keyframe.T = cur_keyframe.T[:3]
		K.append(Keyframe(T_abs,depth,uncertainty,frame))
		K[cur_index].D,K[cur_index].U = depth_map_fusion.fuse_depth_map(K[cur_index],K[cur_index - 1])
		cur_keyframe = K[cur_index]

		update_pose_graph.update_pose_graph()
		update_pose_graph.graph_optimization()

	else: # Refine and fuse depth map. Stereo matching consecutive frame
		print ("*****************************")
		print ("Going to do stereo matching")
		print ("*****************************\n")
		D_frame = stereo_match.stereo_match(prev_frame,frame,prev_pose,T)
		print ("*****************************")
		print ("Stereo Matching Done")
		print ("*****************************\n")
		#plt.imshow(D_frame)
		#plt.show()
		print ("**********************************")
		print ("Going to find uncertainty")
		print ("**********************************")
		U_frame = find_uncertainty.get_uncertainty(T,D_frame,cur_keyframe)
		frame_obj = Keyframe(T,D_frame,U_frame,frame,image,C) # frame as a keyframe object
		cur_keyframe.D,cur_keyframe.U = depth_map_fusion.fuse_depth_map(frame_obj,cur_keyframe)
		print ("**********************************")
		print ("Found uncertainty and fused it")
		print ("**********************************")

	cv2.imshow("twrer",K[cur_index].D)
	cv2.waitKey(0)


def test_cam_pose_est():
	img1 = cv2.resize(cv2.imread("stereo.jpeg",0),(im_size[1],im_size[0]),interpolation = cv2.INTER_CUBIC)
	img2 = cv2.resize(cv2.imread("stereo(1).jpeg",0),(im_size[1],im_size[0]),interpolation = cv2.INTER_CUBIC)
	u = get_highgrad_element(frame)

def just_refine():
	# INIT monodepth_single session
	sess=monodepth_single.init_monodepth(args.mono_checkpoint_path)
	img1 = cv2.resize(cv2.imread("pose_estimation/stereo_kitti1.png"),(im_size[1],im_size[0]),interpolation = cv2.INTER_CUBIC)
	img2 = cv2.resize(cv2.imread("pose_estimation/stereo_kitti2.png"),(im_size[1],im_size[0]),interpolation = cv2.INTER_CUBIC)

	gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
	gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
	# INIT camera matrix
	cam_matrix = get_camera_matrix()

	try: 
		cam_matrix_inv = np.linalg.inv(cam_matrix)
	except:
		raise (Error, "Verify camera matrix")

	# Image is 3 channel, frame is grayscale
	#ret,image,frame = get_camera_image()

	# Predict depth
	ini_depth = monodepth_single.get_depth_map(sess,img1)
	#plt.imshow(ini_depth)
	#pslt.savefig('cnn.png')

    # Initalisation
	ini_uncertainty = find_uncertainty.get_initial_uncertainty()
	cv2.imwrite("cnn2.jpg",ini_depth)
	disparity,depth = stereo_match.for_just_refine(gray1,gray2)
	cv2.imwrite("disp2.jpg",disparity)

	"""
	fused_disparity = 0.33*disparity + 0.66*ini_depth
	plt.subplot(2,2,1),plt.imshow(img1,cmap = 'gray')
	plt.title('Original'), plt.xticks([]), plt.yticks([])
	plt.subplot(2,2,2),plt.imshow(ini_depth,cmap = 'gray')
	plt.title('CNN - Predicted depth'), plt.xticks([]), plt.yticks([])
	plt.subplot(2,2,3),plt.imshow(disparity,cmap = 'gray')
	plt.title('Stereo matching results'), plt.xticks([]), plt.yticks([])
	plt.subplot(2,2,4),plt.imshow(fused_disparity,cmap = 'gray')
	plt.title('Fused Results'), plt.xticks([]), plt.yticks([])
	plt.show()
	plt.savefig('output1.png')
"""
	plt.imshow(ini_depth)
	plt.savefig('kitti_original.png')
	plt.imshow(disparity)
	plt.savefig('stereo_matched.png')
	fused = 0.33*ini_depth + 0.66*disparity
	plt.imshow(fused)
	plt.show()
	plt.savefig('fused.png')
	#cloud_for_vis(img1,disparity)

if __name__ == "__main__":
	just_refine()