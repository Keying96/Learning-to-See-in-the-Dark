#!/usr/bin/env python
# -*- coding: utf-8 -*-
import  os, errno
import tensorflow as tf
import matplotlib.pyplot as plt

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

def concat_feature(sess,
                   feature_conv,
                   num_max_list):
    """ 当len(feature_conv)的值 >= num_max_list 进行一次加法计算
    来减少feature_list占据CPU的内存

    :return: 进行加法后的feature_list,类型是np
    """
    feature_conv_list = feature_conv
    len_list = len(feature_conv_list)
    # print ("len(feature_conv_list): {}".format(len_list))
    # print (feature_conv_list)

    with sess.as_default():
        if len_list >= num_max_list:
            concat_feature = tf.concat(feature_conv_list, axis=3)
            concat_feature = (tf.reduce_sum(concat_feature, axis=3, keepdims=True))
            feature_conv_list = []
            feature_conv_list.append(concat_feature.eval())

    return feature_conv_list

def plot_channel_feature_map(sess,
                             channel_feature,
                             decompose_conv_dir,
                             pattern_name,
                             image_channel):
    # the shape of "channel_feature_map": (1, 1424, 2128, 1)
    """
    :param sess:
    :param channel_feature: 需要被输出的channel_feature，类型是numpy的[1, height, weight, :]
    :param decompose_conv_dir:
    :param pattern_name:
    :param image_channel: 如果是channel_feature图像保存到的路径是
                                                "./decompose_conv_dir/pattern_name_image_channel.png"
                            如果是sub_channel_feature图像保存到的路径是
                                                "./decompose_conv_dir/pattern_name_image_channel/num.png"
    :return:
    """
    with sess.as_default():
    # channel_feature 图像
        num_channel = len(channel_feature)
        for curr_channel in range(num_channel):
            op_feature_channel = channel_feature[curr_channel]
            h = op_feature_channel.shape[1]
            w = op_feature_channel.shape[2]

            image_array = op_feature_channel.reshape((h,w))

            plt.imshow(image_array,cmap="gray")
            savename = os.path.join(decompose_conv_dir, '{}_{}.png'.format(pattern_name,image_channel))
            plt.savefig(savename,dpi = 600)
            print (savename)
            plt.close()

def plot_conv_feature_map(sess,
                             cov_feature,
                             decompose_conv_dir,
                             num_flag):
    # the shape of "channel_feature_map": (1, 1424, 2128, 1)
    """
    :param sess:
    :param channel_feature: 需要被输出的channel_feature，类型是numpy的[1, height, weight, :]
    :param decompose_conv_dir:
    :param pattern_name:
    :param image_channel: 如果是channel_feature图像保存到的路径是
                                                "./decompose_conv_dir/pattern_name_image_channel.png"
                            如果是sub_channel_feature图像保存到的路径是
                                                "./decompose_conv_dir/pattern_name_image_channel/num.png"
    :return:
    """
    with sess.as_default():
    # channel_feature 图像

        op_feature_channel = cov_feature
        h = op_feature_channel.shape[1]
        w = op_feature_channel.shape[2]

        image_array = op_feature_channel.reshape((h,w))

        plt.imshow(image_array,cmap="gray")
        savename = os.path.join(decompose_conv_dir, '{}.png'.format(num_flag))
        plt.savefig(savename,dpi = 600)
        print (savename)
        plt.close()

if __name__ == '__main__':
    pass