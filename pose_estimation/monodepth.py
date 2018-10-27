import monodepth.average_gradients as average_gradients
import monodepth.monodepth_dataloader as monodepth_dataloader
import monodepth.monodepth_model as monodepth_model
import scipy.misc
import tensorflow.contrib.slim as slim
import tensorflow as tf
import time
import re
import argparse
import numpy as np
'''
Extract monodepth prediction for a given frame

'''

from __future__ import absolute_import, division, print_function

# only keep warnings and errors
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'


def _post_process_disparity(disp):
    '''
    Post processing.

    not intended to be used outside module.
    '''
    _, h, w = disp.shape
    l_disp = disp[0, :, :]
    r_disp = np.fliplr(disp[1, :, :])
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def init_monodepth(checkpoint_path):
    '''
    Intialises a monodepth session.

    Returns session.
    '''
    # SESSION
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)

    # SAVER
    train_saver = tf.train.Saver()

    # INIT
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

    # RESTORE
    restore_path = checkpoint_path.split(".")[0]
    train_saver.restore(sess, restore_path)

    return sess


def get_depth_map(sess, image_array, encoder="resnet50"):
    '''
    Returns monocular depth map

    Args:
    * image_array: input image array
    * checkpoint path: Path to restore from
    * encoder:


    TODO:
    * Use frozen graphs instead
    * Switch to tf.data.Datasets
    '''
    input_height, input_width = image_array.shape()[:2]
    params = monodepth_parameters(
        encoder=encoder,
        height=input_height,
        width=input_width,
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

    left = tf.placeholder(tf.float32, [2, input_height, input_width, 3])
    model = MonodepthModel(params, "test", left, None)

    input_image = image_array
    original_height, original_width, num_channels = input_image.shape
    input_image = input_image.astype(np.float32) / 255
    input_images = np.stack((input_image, np.fliplr(input_image)), 0)

    disp = sess.run(model.disp_left_est[0], feed_dict={left: input_images})
    disp_pp = _post_process_disparity(disp.squeeze()).astype(np.float32)

    disp_to_img = scipy.misc.imresize(
        disp_pp.squeeze(), [
            original_height, original_width])

    return disp_to_img
