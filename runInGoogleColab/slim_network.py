# -*- coding:utf-8 -*-
import tensorflow as tf
import os, scipy.io
import sys
import numpy as np
import rawpy
import tensorflow.contrib.slim as slim
import sys
sys.path.append("/home/zhu/PycharmProjects/SeeInTheDark_Threading/visualize")
import pack_image0703

# sess = tf.InteractiveSession()
global img_conv1
ratio = 28
# TensorBoard情報出力ディレクトリ
log_dir = './logs'
checkpoint_dir = '../checkpoint/Sony/'
input_path = '/home/zhu/PycharmProjects/SeeInTheDark_Threading/dataset/short/00001_00_0.1s.ARW'


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

    conv1 = slim.conv2d(input, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_1')
    conv1 = slim.conv2d(conv1, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_2')
    pool1 = slim.max_pool2d(conv1, [2, 2], padding='SAME')
    # print("Within session, tf.shape(conv1)： ", sess.run(tf.shape(pool1)))
    # tf.summary.image('conv1', tf.reshape(tf.transpose(conv1,perm=[0,3,1,2]),[-1,1510,2014,1]), 32)
    # tf.summary.image('conv1', tf.reshape(conv1, [-1, 1512, 2016, 1]), 32)
    # tf.summary.image('pool1', tf.reshape(pool1, [-1, 756, 1008, 1]), 32)
    # conv1_image = conv1[0:2, :, :, 0:32]
    # conv1_image = tf.transpose(conv1_image, perm=[3,1,2,0])

    # conv2 = slim.conv2d(pool1, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_1')
    # conv2 = slim.conv2d(conv2, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_2')
    # pool2 = slim.max_pool2d(conv2, [2, 2], padding='SAME')
    # # tf.summary.image('conv2', tf.reshape(conv2, [-1, 756, 1008, 1]), 64)
    # conv2_image = conv2[0:1, :, :, 0:64]
    # conv2_image = tf.transpose(conv2_image, perm=[3,1,2,0])
    #
    # conv3 = slim.conv2d(pool2, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_1')
    # conv3 = slim.conv2d(conv3, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_2')
    # pool3 = slim.max_pool2d(conv3, [2, 2], padding='SAME')
    # # tf.summary.image('conv3', tf.reshape(conv3, [-1, 378, 504, 1]), 128)
    # conv3_image = conv3[0:1, :, :, 0:128]
    # conv3_image = tf.transpose(conv3_image, perm=[3,1,2,0])
    #
    # conv4 = slim.conv2d(pool3, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_1')
    # conv4 = slim.conv2d(conv4, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_2')
    # pool4 = slim.max_pool2d(conv4, [2, 2], padding='SAME')
    # # tf.summary.image('conv4', tf.reshape(conv4, [-1, 189, 252, 1]), 256)
    # conv4_image = conv4[0:1, :, :, 0:256]
    # conv4_image = tf.transpose(conv4_image, perm=[3,1,2,0])
    #
    # conv5 = slim.conv2d(pool4, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_1')
    # conv5 = slim.conv2d(conv5, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_2')
    # # tf.summary.image('conv5', tf.reshape(conv5, [-1, 95, 126, 1]), 512)
    # conv5_image = conv5[0:1, :, :, 0:512]
    # conv5_image = tf.transpose(conv5_image, perm=[3,1,2,0])
    #
    # up6 = upsample_and_concat(conv5, conv4, 256, 512)
    # conv6 = slim.conv2d(up6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_1')
    # conv6 = slim.conv2d(conv6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_2')
    # # tf.summary.image('conv6', tf.reshape(conv6, [-1, 189, 252, 1]), 256)
    # conv6_image = conv6[0:1, :, :, 0:256]
    # conv6_image = tf.transpose(conv6_image, perm=[3,1,2,0])
    #
    # up7 = upsample_and_concat(conv6, conv3, 128, 256)
    # conv7 = slim.conv2d(up7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_1')
    # conv7 = slim.conv2d(conv7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_2')
    # # tf.summary.image('conv7', tf.reshape(conv7, [-1, 378, 504, 1]), 128)
    # conv7_image = conv7[0:1, :, :, 0:128]
    # conv7_image = tf.transpose(conv7_image, perm=[3,1,2,0])
    #
    # up8 = upsample_and_concat(conv7, conv2, 64, 128)
    # conv8 = slim.conv2d(up8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_1')
    # conv8 = slim.conv2d(conv8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_2')
    # # tf.summary.image('conv8', tf.reshape(conv8, [-1, 756, 1008, 1]), 64)
    # conv8_image = conv8[0:1, :, :, 0:64]
    # conv8_image = tf.transpose(conv8_image, perm=[3,1,2,0])
    #
    # up9 = upsample_and_concat(conv8, conv1, 32, 64)
    # conv9 = slim.conv2d(up9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_1')
    # conv9 = slim.conv2d(conv9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_2')
    # # tf.summary.image('conv9', tf.reshape(conv8, [-1, 1512, 2016, 1]), 32)
    # conv9_image = conv9[0:1, :, :, 0:32]
    # conv9_image = tf.transpose(conv9_image, perm=[3,1,2,0])
    #
    # conv10 = slim.conv2d(conv9, 12, [1, 1], rate=1, activation_fn=None, scope='g_conv10')
    # out = tf.depth_to_space(conv10, 2)
    # # tf.summary.image('conv10', tf.reshape(conv10, [-1, 1512, 2016, 1]), 12)
    # # tf.summary.image('out', tf.reshape(out, [-1, 3024, 4032, 3]), 1)
    # return out
    # return


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
    # out = np.concatenate((im[0:H:2, 0:W:2, :],
    #                       im[0:H:2, 0:W:2, :],
    #                       im[0:H:2, 0:W:2, :],
    #                       im[0:H:2, 0:W:2, :]), axis=2)

    return out

sess = tf.Session()
in_image = tf.placeholder(tf.float32, [None, None, None, 4])

# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)

#读入图像
raw = rawpy.imread(input_path)
input_full = np.expand_dims(pack_raw(raw), axis=0) * ratio
input_full = np.minimum(input_full, 1.0)
in_image = input_full
# input_img = pack_image0703.loadImage(input_path)
print (in_image[:, :, :, 0])


conv1 = slim.conv2d(in_image, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_1')
conv1_1_feature = conv1[:,:,:,0:32]
conv1 = slim.conv2d(conv1, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_2')
conv1_2_feature = conv1[:,:,:,0:32]

#
saver = tf.train.Saver()
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt:
    print('loaded ' + ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)

with sess.as_default():
    # output = sess.run(out_image, feed_dict={in_image: input_full})
    output = sess.run(conv1)
    print (output.shape)
    print (conv1_1_feature.eval()[:,:,:,0])
    print (conv1_2_feature.eval()[:,:,:,0])
    # output = np.minimum(np.maximum(output, 0), 1)
    # output = output[0, :, :, :]
    # sess.close()