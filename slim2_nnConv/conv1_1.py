#!/usr/bin/env python
# -*- coding: utf-8 -*-


import load_model_checkpoint
import create_impulse_img
import tensorflow as tf
import  os, errno
import matplotlib.pyplot as plt

checkpoint_dir = '../checkpoint/Sony/'
result_dir = "./decompose_results/"


def prepare_dir(path, empty=False):
    """
    Creates a directory if it soes not exist
    :param path: string, path to desired directory
    :param empty: boolean, delete all directory content if it exists
    :return: nothing
    """

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

    if not os.path.exists(path):
        create_dir(path)

def lrelu(x):

    return tf.maximum(x * 0.2, x)

def concat_feature(feature_conv):
    """ 减少feature_list占据CPU的内存，每存储5次进行一次加法计算

    :return: 进行加法后的feature_list
    """
    feature_conv_list = feature_conv
    len_list = len(feature_conv_list)

    if len_list >= 5:
        concat_feature = tf.concat(feature_conv_list, axis=3)
        concat_feature = (tf.reduce_sum(concat_feature, axis=3, keepdims=True))
        feature_conv_list = concat_feature
    else :
        feature_conv_list = feature_conv

    return feature_conv_list
    # op_biases_1_1 = tf.reshape(op_biases_list[0].eval()[0], [1])
    # bias = tf.nn.bias_add(concat_feature, op_biases_1_1)
    # conv1_1 = lrelu(bias)
    # sess.run(conv1_1)
    # print ("conv1_1 {}".format(conv1_1.eval()))

def plot_channel_feature_map(channel_feature, decompose_conv_dir, image_channel):
    # the shape of "channel_feature_map": (1, 1424, 2128, 1)
    with sess.as_default():

        op_feature_channel = channel_feature.eval()
        h = op_feature_channel.shape[1]
        w = op_feature_channel.shape[2]

        image_array = op_feature_channel.reshape((h,w))

        plt.imshow(image_array,cmap="gray")
        savename = os.path.join(decompose_conv_dir, '{}.png'.format(image_channel))
        plt.savefig(savename,dpi = 600)
        print (savename)

def nn_conv1_1(input_img,
               op_kernel_list,
               op_biases_list,
               pattern_name,
               op_kernelshape_list,
               op_biasesshape_list,
               decompose_conv_name):

    decompose_conv_dir = os.path.join(result_dir, decompose_conv_name)
    prepare_dir(decompose_conv_dir)

    with sess.as_default():
        feature_channel_list = []
        curr_layer = 0
        input_img = tf.reshape(input_img, [1, 1424, 2128, 4])

        for image_channel in range(input_img.shape[3]):
            feature_conv1_1 = []
            op_img = tf.reshape(input_img[:, :, :, image_channel], [1, 1424, 2128, 1])
            num_output = op_kernelshape_list[curr_layer][3]

            for num_kernel1_1 in range(num_output):
                op_kernel_1_1 = tf.reshape(op_kernel_list[curr_layer].eval()[:, :, image_channel, num_kernel1_1],
                                           [3, 3, 1, 1])
                conv = tf.nn.conv2d(op_img, op_kernel_1_1, [1, 1, 1, 1], padding='SAME')
                sess.run(conv)
                feature_conv1_1.append(conv.eval())

                feature_conv1_1 = concat_feature(feature_conv1_1)

            feature_channel = feature_conv1_1
            feature_channel_list.append(feature_channel)
            plot_channel_feature_map(feature_channel, decompose_conv_dir, image_channel)


if __name__ == '__main__':
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.5)
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    sess = tf.compat.v1.Session(config=config)
    init_sess = tf.compat.v1.global_variables_initializer()

    # 载入卷积计算相关参数
    op_layer_number = 1
    load = load_model_checkpoint.LoadModelCheckpoint(sess, checkpoint_dir, op_layer_number)
    layer_name_list, op_kernel_list, op_biases_list, \
    op_kernelshape_list, op_biasesshape_list, decompose_conv_name = load.load_model_checkpoint()

    # 获取input_img
    img_height = 1424
    img_width = 2128
    impulse_type = 0
    impulse_size = 1
    input_img, pattern_name = create_impulse_img.CreateImpulseImg(impulse_type, impulse_size,
                                                 img_height, img_width).create()

    nn_conv1_1(input_img, op_kernel_list, op_biases_list,pattern_name,
                         op_kernelshape_list,op_biasesshape_list,decompose_conv_name)   #进行conv1_1卷积计算


    sess.close()