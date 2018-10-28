# Library imports
import scipy.misc
import tensorflow.contrib.slim as slim
import tensorflow as tf
import argparse
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# Module imports
import monodepth_infer.monodepth_single as monodepth_single
import monodepth.monodepth_model as monodepth_model


parser = argparse.ArgumentParser(description='Monodepth Batch Inference.')

# Parser args
parser.add_argument("--path", default = "dummy_data", help="path to files. Needs to have a rgb folder containing images.")
parser.add_argument("--checkpoint_path", default = "checkpoints/model_kitti_resnet/model_kitti_resnet.data", help="path to checkpoint")
parser.add_argument("--image_format", default= ".png", help="path to files. Needs to have a rgb folder containing images.")

args=parser.parse_args()

# Monodepth Params
input_height, input_width = 256, 512
batch_size = 1
encoder = 'resnet50'

params = monodepth_model.monodepth_parameters(
    encoder=encoder,
    height=input_height,
    width=input_width,
    batch_size= batch_size,
    num_threads=1,
    num_epochs=1,
    do_stereo=False,
    wrap_mode="border",
    use_deconv=False,
    alpha_image_loss=0,
    disp_gradient_loss_weight=0,
    lr_loss_weight=0,
    full_summary=False)

# Parser Fn for Tf datasets

def _parse_fn(example):
    '''
    Example is a file name, absolute path
    '''
    img = tf.image.decode_png(tf.read_file(example))
    img = tf.image.resize_images(img, tf.constant([input_height, input_width]))
    img = img [: , : , ::-1]
    img = tf.reshape(img, [input_height, input_width, 3])
    img= img/255
    imgl = img
    imgr = tf.image.flip_left_right(img)/ 255
    
    return example, tf.stack((imgl, imgr), 0)

def save(d, e):
    '''
    Resize and save disparity maps
    '''
    path_depth = os.path.join(args.path, "depth")
    path_rgb = os.path.join(args.path, "rgb")

    if not os.path.exists(path_depth):
        os.mkdir(path_depth)

    e = e[0]
    e = e.decode('ascii')
    w, h, _ = cv2.imread(e).shape
    disp = cv2.resize(d, (h, w)) 
    
    name = e.split("/")[-1]
    name_path = os.path.join(path_depth, name)
    plt.imsave(name_path, disp, cmap='plasma')

# Get rgb images
dataset = tf.data.Dataset.list_files(args.path+"/rgb/*"+ args.image_format)
dataset = dataset.map(_parse_fn)
dataset = dataset.batch(batch_size)
dataset = dataset.make_one_shot_iterator()
example, image = dataset.get_next()

# Instantiate monodepth model
model = monodepth_model.MonodepthModel(params, "test", image[0], None)
disp_pp = model.disp_left_est[0]

with tf.Session() as sess:
    monodepth_single.init_monodepth(args.checkpoint_path,sess = sess)

    i = 1
    # Run dataset iterator
    while True:
        try:
            print ("Count ", i)
            _example, _image, _disp = sess.run([example,
                image, 
                model.disp_left_est[0]])

            _disp_pp = monodepth_single.post_process_disparity(_disp.squeeze().astype(np.float32))

            save(_disp_pp, _example)
            
            i+=1

        except tf.errors.OutOfRangeError:
            print("End of dataset")
            break