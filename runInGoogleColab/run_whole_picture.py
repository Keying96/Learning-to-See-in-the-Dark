#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
import os, scipy.io
import sys
import numpy as np
import tensorflow.contrib.slim as slim
import divide_into_block
import matplotlib.pyplot as plt
import time
import rawpy


ratio = 30
input_path = '../dataset/short/00001_00_0.1s.ARW'
checkpoint_dir = '../checkpoint/Sony/'
result_dir = './result_divide/'

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

def lrelu(x):
    return tf.maximum(x * 0.2, x)

def upsample_and_concat(x1, x2, output_channels, in_channels):
    pool_size = 2
    deconv_filter = tf.Variable(tf.truncated_normal([pool_size, pool_size, output_channels, in_channels], stddev=0.02))
    deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2), strides=[1, pool_size, pool_size, 1])

    deconv_output = tf.concat([deconv, x2], 3)
    deconv_output.set_shape([None, None, None, output_channels * 2])

    return deconv_output

def network(input):

    conv1_1 = slim.conv2d(input, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_1')
    conv1 = slim.conv2d(conv1_1, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_2')
    pool1 = slim.max_pool2d(conv1, [2, 2], padding='SAME')

    conv2 = slim.conv2d(pool1, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_1')
    conv2 = slim.conv2d(conv2, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_2')
    pool2 = slim.max_pool2d(conv2, [2, 2], padding='SAME')

    conv3 = slim.conv2d(pool2, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_1')
    conv3 = slim.conv2d(conv3, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_2')
    pool3 = slim.max_pool2d(conv3, [2, 2], padding='SAME')

    conv4 = slim.conv2d(pool3, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_1')
    conv4 = slim.conv2d(conv4, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_2')
    pool4 = slim.max_pool2d(conv4, [2, 2], padding='SAME')

    conv5 = slim.conv2d(pool4, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_1')
    conv5 = slim.conv2d(conv5, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_2')

    up6 = upsample_and_concat(conv5, conv4, 256, 512)
    conv6 = slim.conv2d(up6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_1')
    conv6 = slim.conv2d(conv6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_2')

    up7 = upsample_and_concat(conv6, conv3, 128, 256)
    conv7 = slim.conv2d(up7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_1')
    conv7 = slim.conv2d(conv7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_2')

    up8 = upsample_and_concat(conv7, conv2, 64, 128)
    conv8 = slim.conv2d(up8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_1')
    conv8 = slim.conv2d(conv8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_2')

    up9 = upsample_and_concat(conv8, conv1, 32, 64)
    conv9 = slim.conv2d(up9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_1')
    conv9 = slim.conv2d(conv9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_2')

    conv10 = slim.conv2d(conv9, 12, [1, 1], rate=1, activation_fn=None, scope='g_conv10')
    out = tf.depth_to_space(conv10, 2)
    return out


sess = tf.Session()
in_image = tf.placeholder(tf.float32, [None, None, None, 4])
out_image = network(in_image)

saver = tf.train.Saver()
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt:
    print('loaded ' + ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)

#读入图像
raw = rawpy.imread(input_path)
input_full = np.expand_dims(pack_raw(raw), axis=0) * ratio
input_full = np.minimum(input_full, 1.0)

start_time = time.time()
output = sess.run(out_image, feed_dict={in_image: input_full})
# output = sess.run(out_image)
output = np.minimum(np.maximum(output, 0), 1)
output = output[0, :, :, :]


end_time = time.time()
print ("the runtime is {}".format(end_time-start_time))


plt.axis("off")
plt.imshow(output)
savename = os.path.join(result_dir, 'whole_image.png') #the runtime is 5.30000114441

if not os.path.isdir(result_dir):
    os.makedirs(result_dir)
plt.savefig(savename, dpi=600)
print (savename)
plt.close()

# #读入图像
# input_full = divide_into_block.laod_input(input_path)
# for i in range(len(input_full)):
#     sess = tf.Session()
#     in_image = tf.placeholder(tf.float32, [None, None, None, 4])
#
#     # tf.initialize_all_variables() no long valid from
#     # 2017-03-02 if using tensorflow >= 0.12
#     if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
#         init = tf.initialize_all_variables()
#     else:
#         init = tf.global_variables_initializer()
#     sess.run(init)
#
#     input = np.expand_dims(input_full[i], axis=0) * ratio
#     input = np.minimum(input, 1.0)
#     in_image = input
#
#     out_image = network(in_image)
#     #
#     saver = tf.train.Saver()
#     ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
#     if ckpt:
#         print('loaded ' + ckpt.model_checkpoint_path)
#         saver.restore(sess, ckpt.model_checkpoint_path)
#
#     output = sess.run(out_image)
#     output = np.minimum(np.maximum(output, 0), 1)
#     output = output[0, :, :, :]
#
#     plt.imshow(output)
#     savename = os.path.join(result_dir, 'result_{}.png'.format(i))
#     if not os.path.isdir(result_dir ):
#         os.makedirs(result_dir)
#     plt.savefig(savename, dpi=600)
#     print (savename)
#     plt.close()