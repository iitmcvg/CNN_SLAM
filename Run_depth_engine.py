import numpy as np 
import cv2
import tensorflow as tf
import sys
import time
import argparse
from matplotlib import pyplot as plt
from PIL import Image

# Modules
#import pose_estimation.depth_map_fusion as depth_map_fusion
#import pose_estimation.stereo_match as stereo_match
from params import *
#import pose_estimation.camera_pose_estimation as camera_pose_estimation
#import pose_estimation.find_uncertainty as find_uncertainty
#from keyframe_utils import Keyframe as Keyframe
import monodepth_infer.monodepth_single as monodepth_single

parser = argparse.ArgumentParser(description='Monodepth TensorFlow implementation.')
parser.add_argument('--mono_checkpoint_path', default = "checkpoints/model_kitti_resnet/model_kitti_resnet.data" ,type=str,   help='path to a specific checkpoint to load')
parser.add_argument('--input_height', type=int,   help='input height', default=480)
parser.add_argument('--input_width', type=int,   help='input width', default=640)
args = parser.parse_args()

sess=monodepth_single.init_monodepth(args.mono_checkpoint_path)
img1 = cv2.resize(cv2.imread("pose_estimation/stereo.jpeg"),(im_size[1],im_size[0]),interpolation = cv2.INTER_CUBIC)

img2 = cv2.resize(cv2.imread("disp_main.jpg",0),(im_size[1],im_size[0]),interpolation = cv2.INTER_CUBIC).astype(np.float64)
# Predict depth
ini_depth = monodepth_single.get_depth_map(sess,img1)
ini_depth = cv2.resize(ini_depth,(im_size[1],im_size[0]),interpolation = cv2.INTER_CUBIC).astype(np.float64)
weight = 0.99
plt.imshow(0.001*ini_depth)
plt.show()
fused_disparity = cv2.addWeighted(ini_depth,weight,img2,1.0-weight,0)

plt.subplot(2,2,1),plt.imshow(img1,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(ini_depth,cmap = 'gray')
plt.title('CNN - Predicted depth'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(img2,cmap = 'gray')
plt.title('Stereo matching results'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(fused_disparity,cmap = 'gray')
plt.title('Fused Results'), plt.xticks([]), plt.yticks([])
plt.show()
plt.savefig('output1.png')


plt.imshow(img1)
plt.show()
plt.savefig('prop_depth1.png')