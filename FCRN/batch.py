# Library imports
import scipy.misc
import tensorflow.contrib.slim as slim
import tensorflow as tf 
import argparse
import numpy as np
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt
import utils.numpngw as numpngw

# Module imports
import FCRN.models as models

parser = argparse.ArgumentParser(description='Monodepth Batch Inference.')

# Parser args
parser.add_argument("--path", default = "dummy_data", help="path to files. Needs to have a rgb folder containing images.")
parser.add_argument("--checkpoint_path", default = "checkpoints/FCRN/NYU_FCRN.ckpt", help="path to checkpoint")
parser.add_argument("--image_format", default= ".jpg", help="path to files. Needs to have a rgb folder containing images.")
parser.add_argument("--batch", default= 1 , type = int, help="Batch size")

args=parser.parse_args()

# Monodepth Params
input_height, input_width = 228, 304
batch_size = args.batch
MAX = 256**2 -1 

# Parser Fn for Tf datasets
def _parse_fn(example):
    '''
    Example is a file name, absolute path
    '''
    img = tf.image.decode_png(tf.read_file(example))
    img = tf.image.resize_images(img, tf.constant([input_height, input_width]))
    img = img [: , : , ::-1]
    img = tf.reshape(img, [input_height, input_width, 3])
    # img= img/255
    
    return example, img

def save(d, e):
    '''
    Resize and save disparity maps
    '''
    path_depth = os.path.join(args.path, "depth")
    path_rgb = os.path.join(args.path, "rgb")

    if not os.path.exists(path_depth):
        os.mkdir(path_depth)

    e = e.decode('ascii')
    w, h, _ = cv2.imread(e).shape
    disp = cv2.resize(d, (h, w)) 
    
    name = e.split("/")[-1]
    name = ".".join(name.split(".")[:-1])
    name_path = os.path.join(path_depth, name )

    # Matplotlib stuff 
    # plt.imsave(name_path + ".png", disp, cmap='plasma')
    
    # Rescale to 0-255 and convert to uint8
    rescaled = (MAX / disp.ptp() * (disp - disp.min())).astype(np.uint16)

    # im = Image.fromarray(disp)
    # if im.mode != 'RGB':
    #     im = im.convert('RGB')
    # im.save(name_path + ".png")
    numpngw.write_png(name_path+".png",rescaled)

    # np.save(name_path,disp)

# Get rgb images
dataset = tf.data.Dataset.list_files(args.path+"/rgb/*")
dataset = dataset.map(_parse_fn)
dataset = dataset.batch(batch_size)
dataset.prefetch(2*batch_size)
dataset = dataset.make_one_shot_iterator()
example, image = dataset.get_next()

# Instantiate depth FCRN model
net = models.ResNet50UpProj({'data': image}, batch_size, 1, False)

# GPU options
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config = config) as sess:
    # Load the converted parameters
    print('Loading the model')

    # Use to load from ckpt file
    saver = tf.train.Saver()     
    saver.restore(sess, args.checkpoint_path)

    i = 1
    # Run dataset iterator
    while True:
        try:
            print ("Count ", i)
            _example, _image, _disp = sess.run([example,
                image, 
                net.get_output()])

            print(_disp.shape)
            _disp = np.squeeze(_disp)
            print(_disp.shape)

            if len(_disp.shape) ==3:
                for _d,_e in zip(_disp,_example):
                    print(_d.shape)
                    save(_d, _e)
                    i+=1
            else:
                save(_disp, _example[0])
                i+=1

        except tf.errors.OutOfRangeError:
            print("End of dataset")
            break
