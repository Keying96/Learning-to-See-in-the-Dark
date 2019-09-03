# -*- coding:utf-8 -*-
import tensorflow as tf
import os, scipy.io
import sys
import numpy as np
import rawpy
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import math
import errno
from keras_preprocessing import image

# sess = tf.InteractiveSession()
global img_conv1
ratio = 28
# TensorBoard情報出力ディレクトリ
log_dir = './logs_filters'
CONV_NAME = "g_conv9_2"
PLOT_DIR = './plot_conv_weights/'
layer_list = []

if not tf.gfile.Exists(PLOT_DIR):
    tf.gfile.MakeDirs(log_dir)

# 指定したディレクトリがあれば削除し、再作成
if tf.gfile.Exists(log_dir):
    tf.gfile.DeleteRecursively(log_dir)
tf.gfile.MakeDirs(log_dir)

# ファイル場所("C:\\"部分はエスケープ処理のため、"\"が2つあることに注意)
input_path = './dataset/Iphone/00034.dng'
checkpoint_dir = './checkpoint/Sony/'
# checkpoint_dir = './visualize_mnist_test/'

result_dir = './result_Iphone/'

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
    # filter1 =
    # print("Within session, tf.shape(conv1)： ", sess.run(tf.shape(pool1)))
    # tf.summary.image('conv1', tf.reshape(tf.transpose(conv1,perm=[0,3,1,2]),[-1,1510,2014,1]), 32)
    # tf.summary.image('conv1', tf.reshape(conv1, [-1, 1512, 2016, 1]), 32)
    # tf.summary.image('pool1', tf.reshape(pool1, [-1, 756, 1008, 1]), 32)
    conv1_image = conv1[0:1, :, :, 0:32]
    conv1_image = tf.transpose(conv1_image, perm=[3,1,2,0])
    # tf.summary.image("filtered_images_layer1", conv1_image, max_outputs=32)

    conv2 = slim.conv2d(pool1, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_1')
    conv2 = slim.conv2d(conv2, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_2')
    pool2 = slim.max_pool2d(conv2, [2, 2], padding='SAME')
    # tf.summary.image('conv2', tf.reshape(conv2, [-1, 756, 1008, 1]), 64)
    conv2_image = conv2[0:1, :, :, 0:64]
    conv2_image = tf.transpose(conv2_image, perm=[3,1,2,0])
    # tf.summary.image("filtered_images_layer2", conv2_image, max_outputs=64)

    conv3 = slim.conv2d(pool2, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_1')
    conv3 = slim.conv2d(conv3, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_2')
    pool3 = slim.max_pool2d(conv3, [2, 2], padding='SAME')
    # tf.summary.image('conv3', tf.reshape(conv3, [-1, 378, 504, 1]), 128)
    conv3_image = conv3[0:1, :, :, 0:128]
    conv3_image = tf.transpose(conv3_image, perm=[3,1,2,0])
    # tf.summary.image("filtered_images_layer3", conv3_image, max_outputs=128)

    conv4 = slim.conv2d(pool3, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_1')
    conv4 = slim.conv2d(conv4, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_2')
    pool4 = slim.max_pool2d(conv4, [2, 2], padding='SAME')
    # tf.summary.image('conv4', tf.reshape(conv4, [-1, 189, 252, 1]), 256)
    conv4_image = conv4[0:1, :, :, 0:256]
    conv4_image = tf.transpose(conv4_image, perm=[3,1,2,0])
    # tf.summary.image("filtered_images_layer4", conv4_image, max_outputs=256)

    conv5 = slim.conv2d(pool4, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_1')
    conv5 = slim.conv2d(conv5, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_2')
    # tf.summary.image('conv5', tf.reshape(conv5, [-1, 95, 126, 1]), 512)
    conv5_image = conv5[0:1, :, :, 0:512]
    conv5_image = tf.transpose(conv5_image, perm=[3,1,2,0])
    # tf.summary.image("filtered_images_layer5", conv5_image, max_outputs=512)

    up6 = upsample_and_concat(conv5, conv4, 256, 512)
    conv6 = slim.conv2d(up6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_1')
    conv6 = slim.conv2d(conv6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_2')
    # tf.summary.image('conv6', tf.reshape(conv6, [-1, 189, 252, 1]), 256)
    conv6_image = conv6[0:1, :, :, 0:256]
    conv6_image = tf.transpose(conv6_image, perm=[3,1,2,0])
    # tf.summary.image("filtered_images_layer6", conv6_image, max_outputs=256)

    up7 = upsample_and_concat(conv6, conv3, 128, 256)
    conv7 = slim.conv2d(up7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_1')
    conv7 = slim.conv2d(conv7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_2')
    # tf.summary.image('conv7', tf.reshape(conv7, [-1, 378, 504, 1]), 128)
    conv7_image = conv7[0:1, :, :, 0:128]
    conv7_image = tf.transpose(conv7_image, perm=[3,1,2,0])
    # tf.summary.image("filtered_images_layer7", conv7_image, max_outputs=128)

    up8 = upsample_and_concat(conv7, conv2, 64, 128)
    conv8 = slim.conv2d(up8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_1')
    conv8 = slim.conv2d(conv8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_2')
    # tf.summary.image('conv8', tf.reshape(conv8, [-1, 756, 1008, 1]), 64)
    conv8_image = conv8[0:1, :, :, 0:64]
    conv8_image = tf.transpose(conv8_image, perm=[3,1,2,0])
    # tf.summary.image("filtered_images_layer8", conv8_image, max_outputs=64)

    up9 = upsample_and_concat(conv8, conv1, 32, 64)
    conv9 = slim.conv2d(up9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_1')
    conv9 = slim.conv2d(conv9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_2')
    # tf.summary.image('conv9', tf.reshape(conv8, [-1, 1512, 2016, 1]), 32)
    conv9_image = conv9[0:1, :, :, 0:32]
    conv9_image = tf.transpose(conv9_image, perm=[3,1,2,0])
    # tf.summary.image("filtered_images_layer9", conv9_image, max_outputs=32)


    conv10 = slim.conv2d(conv9, 12, [1, 1], rate=1, activation_fn=None, scope='g_conv10')
    out = tf.depth_to_space(conv10, 2)
    # tf.summary.image('conv10', tf.reshape(conv10, [-1, 1512, 2016, 1]), 12)
    # tf.summary.image('out', tf.reshape(out, [-1, 3024, 4032, 3]), 1)
    return out


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

def get_grid_dim(x):
    """
    Transforms x into product of two integers
    :param x: int
    :return: two ints
    """
    factors = prime_powers(x)
    if len(factors) % 2 == 0:
        i = int(len(factors) / 2)
        return factors[i], factors[i - 1]

    i = len(factors) // 2
    return factors[i], factors[i]

def prime_powers(n):
    """
    Compute the factors of a positive integer
    Algorithm from https://rosettacode.org/wiki/Factors_of_an_integer#Python
    :param n: int
    :return: set
    """
    factors = set()

    n = float(n)
    for x in range(1, int(math.sqrt(n)) + 1):
        if n % x == 0:
            factors.add(int(x))
            factors.add(int(n // x))

    return sorted(factors)

def create_dir(path):
    """
    Creates a directory
    :param path: string
    :return: nothing
    """
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise


def prepare_dir(path, empty=False):
    """
    Creates a directory if it soes not exist
    :param path: string, path to desired directory
    :param empty: boolean, delete all directory content if it exists
    :return: nothing
    """
    if not os.path.exists(path):
        create_dir(path)

def plot_conv_weights(weights, name, channels_all=True):
    """
    Plots convolutional filters
    :param weights: numpy array of rank 4
    :param name: string, name of convolutional layer
    :param channels_all: boolean, optional
    :return: nothing, plots are saved on the disk
    """
    # make path to output folder
    plot_dir = os.path.join(PLOT_DIR, name)

    # create directory if does not exist, otherwise empty it
    prepare_dir(plot_dir, empty=True)

    w_min = np.min(weights)
    print ("weights: " + str(w_min))
    w_max = np.max(weights)
    print ("weights: " + str(w_max))

    channels = [0]
    # make a list of channels if all are plotted
    if channels_all:
        channels = range(weights.shape[2])
        print("channels: " + str(channels))

    # get number of convolutional filters
    num_filters = weights.shape[3]
    num_filters = str(num_filters)
    print ("filters: " + str(num_filters))

    # get number of grid rows and columns
    grid_r, grid_c = get_grid_dim(num_filters)

    # create figure and axes
    fig, axes = plt.subplots(min([grid_r, grid_c]),
                             max([grid_r, grid_c]))

    # iterate channels
    for channel in channels:
        # iterate filters inside every channel
        for l, ax in enumerate(axes.flat):
            # get a single filter
            img = weights[:, :, channel, l]
            # img = weights[0, :, :, l]
            # put it on the grid
            # ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='nearest', cmap='seismic')
            # ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='bicubic', cmap='Greys')
            ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='nearest', cmap = plt.get_cmap('gray'))

            # remove any labels from the axes
            ax.set_xticks([])
            ax.set_yticks([])
        # save figure
        # print ("plot_dir: " + str(plot_dir))
        plt.savefig(os.path.join(plot_dir, '{}-{}.png'.format(name, channel)), bbox_inches='tight')


with tf.Session() as sess:
    in_image = tf.placeholder(tf.float32, [None, None, None, 4])
    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)


    # 画像をTensorboardに出力
    out_image = network(in_image)

    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt:
        print('loaded ' + ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)


    def _get_layerList():
        # print ([v.name for v in tf.global_variables()])
        for v in tf.global_variables():
            if "weights" in v.name:
                layer_list.append(v.name)
        return (layer_list)
        # if "weights" in [v.name for v in tf.global_variables()]:
        #     layer_list.append(v.name)
        #     print (v.name)
        # print(layer_list)

    layer_list = _get_layerList()
    print (layer_list)
    #输出网络所有层的filters
    # for i in range(len(layer_list)):
    for i in  range(0, len(layer_list)):
        conv_name = layer_list[i].split('/')[0]
        print(conv_name)
        with tf.variable_scope(conv_name) as scope:
            tf.get_variable_scope().reuse_variables()
            weights = tf.get_variable("weights").eval()

            # print (weights)
            #
            # print ("filters: " + str(weights.shape[3]))
            # print (("channels: " + str(weights.shape[2])))
            # print (weights.shape[1])
            # print (weights.shape[0])

            plot_conv_weights(weights, conv_name)
    # 输出指定层的所有filters
    # with tf.variable_scope("g_conv1_2") as scope:
    #     tf.get_variable_scope().reuse_variables()
    #     weights = tf.get_variable("weights").eval()
    #     for i in range(3):
    #         # plt.matshow(weights[:,:,0,i], cmap = plt.get_cmap('gray'))
    #         img = weights[:,:,0,i]
    #         # plt.imshow(img, interpolation='nearest', cmap='seismic')
    #         plt.imshow(img, interpolation='bicubic', cmap='Greys')
    #         plt.show()
        # print (weights)
        #
        # print ("filters: " + str(weights.shape[3]))
        # print (("channels: " + str(weights.shape[2])))
        # print (weights.shape[1])
        # print (weights.shape[0])
        #
        # plot_conv_weights(weights, conv_name)



# direct to the local dir and run this in terminal:
# $ tensorboard --logdir=logs_filters



# print [v.name for v in tf.global_variables()]
