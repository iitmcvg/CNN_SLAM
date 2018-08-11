from __future__ import absolute_import, division, print_function

# only keep warnings and errors
import os
import cv2     

os.environ['TF_CPP_MIN_LOG_LEVEL']='0'

import numpy as np
import argparse
import re
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.misc
from monodepth_model import *
from monodepth_dataloader import *
from average_gradients import *

parser = argparse.ArgumentParser(description='Monodepth TensorFlow implementation.')

parser.add_argument('--encoder',          type=str,   help='type of encoder, vgg or resnet50', default='vgg')
parser.add_argument('--video_path',       type=str,   help='path to the video', required=True)
parser.add_argument('--checkpoint_path',  type=str,   help='path to a specific checkpoint to load', required=True)
parser.add_argument('--input_height',     type=int,   help='input height', default=256)
parser.add_argument('--input_width',      type=int,   help='input width', default=512)

args = parser.parse_args()
feature_params = dict( maxCorners = 1000,
                       qualityLevel = 0.01,
                       minDistance = 7,
                       blockSize = 7 )
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
color = np.random.randint(0,255,(1000,3))
# Here we have created 100 different colors keeping in mind that we want only 100 Corners at the MAX 
def post_process_disparity(disp):
    _, h, w = disp.shape
    l_disp = disp[0,:,:]
    r_disp = np.fliplr(disp[1,:,:])
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

def test_simple(params):
    """Test function."""

    left  = tf.placeholder(tf.float32, [2, args.input_height, args.input_width, 3])
    model = MonodepthModel(params, "test", left, None)
    cap = cv2.VideoCapture(args.video_path)
    restore_path = args.checkpoint_path.split(".")[0]

    # SESSION
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)

    # SAVE
    train_saver = tf.train.Saver()

    # INIT
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

    # RESTORE
        
    train_saver.restore(sess, restore_path)
    for i in range(100000):
        End = False
        i+=1
        check,frame = cap.read()
        if (check == False):
            cap.release()
            cv2.destroyAllWindows()
            break
        image_map = np.zeros([frame.shape[0],frame.shape[1],1],dtype = np.uint8)
        old_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(frame)
        p0 = cv2.goodFeaturesToTrack(old_frame , mask = None , **feature_params)
        j = 0
        while(j<=20):
            ret,frame_new = cap.read()
            if(ret == False):
                break
            frame_new_gray = cv2.cvtColor(frame_new , cv2.COLOR_BGR2GRAY)
            p1 , st, err = cv2.calcOpticalFlowPyrLK(old_frame, frame_new_gray, p0, None, **lk_params)
             # Select good points
            good_new = p1[st==1]
            good_old = p0[st==1]
            # draw the tracks
            for i,(new,old) in enumerate(zip(good_new,good_old)):
                a,b = new.ravel()
                c,d = old.ravel()
                mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 1)
                if (a<old_frame.shape[1]):
                    a = int(a)
                else:
                    a = old_frame.shape[1]-1
                if (b<old_frame.shape[0]):
                    b = int(b)
                else:
                    b = old_frame.shape[0] -1
                x = int(10*(np.sqrt((a-c)**2 + (b-d)**2)))
                image_map[b,a,0] = x  
        
         
            image_map = image_map.astype(dtype = np.uint8)

            image_map_cmap = cv2.applyColorMap(image_map,cv2.COLORMAP_HOT)

            img = cv2.add(frame_new,mask)

            input_image = frame_new
            original_height, original_width, num_channels = input_image.shape
            input_image = scipy.misc.imresize(input_image, [args.input_height, args.input_width], interp='lanczos')
            input_image = input_image.astype(np.float32) / 255
            input_images = np.stack((input_image, np.fliplr(input_image)), 0)
        
            disp = sess.run(model.disp_left_est[0], feed_dict={left: input_images})
            disp_pp = post_process_disparity(disp.squeeze()).astype(np.float32)
            disp_pp = 255*3*np.reshape(disp_pp , (256,512,1))
            

            disp_pp = disp_pp.astype(np.uint8)

            disp_cmap = cv2.applyColorMap(disp_pp, cv2.COLORMAP_HOT)   
            disp_cmap = cv2.resize(disp_cmap , (image_map_cmap.shape[1],image_map_cmap.shape[0])) 

            alpha=0.3      
            img_final = cv2.addWeighted(disp_cmap,0.7,image_map_cmap,1-alpha,0)     
            cv2.namedWindow("Depth_only",cv2.WINDOW_NORMAL)
            cv2.namedWindow("Original",cv2.WINDOW_NORMAL)
            cv2.namedWindow("Optical_flow_only",cv2.WINDOW_NORMAL)
            cv2.namedWindow("Final",cv2.WINDOW_NORMAL)
            cv2.imshow('Final',img_final)
            cv2.imshow("Original",frame_new)
            cv2.imshow("Depth_only",disp_cmap)       
            cv2.imshow('Optical_flow_only',img)
            #print(np.max(image_map) , np.max(disp_cmap))
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                End = True
                break
            # Now update the previous frame and previous points
            old_frame = frame_new_gray.copy()
            p0 = good_new.reshape(-1,1,2)
            j+=1
        if (i == 16):
            i = 0
        if (End == True):
            break
    cap.release()
    cv2.destroyAllWindows()
    print('done!')

def main(_):

    params = monodepth_parameters(
        encoder=args.encoder,
        height=args.input_height,
        width=args.input_width,
        batch_size=2,
        num_threads=1,
        num_epochs=1,
        do_stereo=False,
        wrap_mode="border",
        use_deconv=False,
        alpha_image_loss=0,
        disp_gradient_loss_weight=0,
        lr_loss_weight=0,
        full_summary=False)

    test_simple(params)

if __name__ == '__main__':
    tf.app.run()
