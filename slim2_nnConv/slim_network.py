#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
import create_impulse_img
import numpy as np
import rawpy
import tensorflow.contrib.slim as slim
import sys
sys.path.append("/home/zhu/PycharmProjects/SeeInTheDark_Threading/visualize")
from tensorflow.python import pywrap_tensorflow


# sess = tf.InteractiveSession()
global img_conv1
ratio = 28
# TensorBoard情報出力ディレクトリ
log_dir = './logs'
checkpoint_dir = '../checkpoint/Sony/'
input_path = '/home/zhu/PycharmProjects/SeeInTheDark_Threading/dataset/short/00001_00_0.1s.ARW'


def lrelu(x):
    return tf.maximum(x * 0.2, x)

def network(op_layer_number, input):
    list_out = []
    conv1_1 = slim.conv2d(input, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_1')
    list_out.append(conv1_1)
    conv1_2 = slim.conv2d(conv1_1, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_2')
    pool1 = slim.max_pool2d(conv1_2, [2, 2], padding='SAME')
    list_out.append(pool1)

    conv2_1 = slim.conv2d(pool1, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_1')
    list_out.append(conv2_1)
    conv2_2 = slim.conv2d(conv2_1, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_2')
    pool2 = slim.max_pool2d(conv2_2, [2, 2], padding='SAME')
    list_out.append(pool2)

    conv3_1 = slim.conv2d(pool2, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_1')
    list_out.append(conv3_1)
    conv3_2 = slim.conv2d(conv3_1, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_2')
    pool3 = slim.max_pool2d(conv3_2, [2, 2], padding='SAME')
    list_out.append(pool3)

    conv4_1 = slim.conv2d(pool3, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_1')
    list_out.append(conv4_1)
    conv4_2 = slim.conv2d(conv4_1, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_2')
    pool4 = slim.max_pool2d(conv4_2, [2, 2], padding='SAME')
    list_out.append(pool4)

    conv5_1 = slim.conv2d(pool4, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_1')
    list_out.append(conv5_1)
    conv5_2 = slim.conv2d(conv5_1, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_2')
    list_out.append(conv5_2)

    return list_out[op_layer_number-1]

def load_image(input_path):
    """ 将图像处理成numpy格式
    :param input_path:  图像路径
    :return:
    """

    def pack_raw(raw):
        # pack Bayer image to 4 channels
        im = raw.raw_image_visible.astype(np.float32)
        im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

        im = np.expand_dims(im, axis=2)
        img_shape = im.shape
        H = img_shape[0]
        W = img_shape[1]

        out = np.concatenate((im[0:H:2, 0:W:2, :],
                              im[0:H:2, 1:W:2, :],
                              im[1:H:2, 1:W:2, :],
                              im[1:H:2, 0:W:2, :]), axis=2)

        return out

    raw = rawpy.imread(input_path)
    input_full = np.expand_dims(pack_raw(raw), axis=0) * ratio
    input_full = np.minimum(input_full, 1.0)

    return input_full

def pack_raw(raw):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out

def init(op_layer_number, input_full):
    #初始化
    tf.compat.v1.reset_default_graph()
    sess = tf.compat.v1.Session()
    init_sess = tf.compat.v1.global_variables_initializer()
    sess.run(init_sess)
    op_layer_number = op_layer_number -1

    # raw = rawpy.imread(input_path)
    # input_full = np.expand_dims(pack_raw(raw), axis=0) * ratio
    # input_full = np.minimum(input_full, 1.0)
    h = input_full.shape[0]
    w = input_full.shape[1]
    input_full = tf.reshape(input_full,[1,h,w,4])
    in_image = input_full

    conv_out = network(op_layer_number,in_image)
    #
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt:
        print('loaded ' + ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)

    conv_out = sess.run(conv_out)

    return conv_out

if __name__ == '__main__':

    # 获取自定义input_img
    img_height = 15
    img_width = 15
    impulse_type = 0
    impulse_size = 1
    input_img, pattern_name = create_impulse_img.CreateImpulseImg(impulse_type, impulse_size,
                                                 img_height, img_width).create()
    print ("===================== Start to calculate the result of {} =======================".format(pattern_name))

    op_layer_number = 4
    conv_out = init(op_layer_number, input_img)
    print (conv_out.shape)
    print (conv_out)
    for i in range(len(conv_out[:,:,:,])):
        print(conv_out[:,:,:,i][0])





