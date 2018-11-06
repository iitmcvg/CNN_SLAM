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
#from params import *
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
img1 = cv2.resize(cv2.imread("pose_estimation/cityscapes2.jpg"),(im_size[1],im_size[0]),interpolation = cv2.INTER_CUBIC)

# Predict depth
ini_depth = monodepth_single.get_depth_map(sess,img1)
plt.imshow(ini_depth)
plt.show()