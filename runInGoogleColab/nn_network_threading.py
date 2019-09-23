# -*- coding:utf-8 -*-
"""
将input_image中的四个channel分多线程执行任务
"""
import numpy as np
import rawpy
import tensorflow as tf
import tensorflow.contrib.slim as slim
import make_impluse_patterns
import scipy.io, os,errno
from tensorflow.python import pywrap_tensorflow
import re
import matplotlib.pyplot as plt
import time
import sys
sys.path.append(r"/home/zhu/PycharmProjects/SeeInTheDark_Threading/visualize/")
import pack_image0703
input_path = '/home/zhu/PycharmProjects/SeeInTheDark_Threading/dataset/short/00001_00_0.1s.ARW'
# print("input_path: " + str(os.path.exists(input_path)))

ratio = 28

checkpoint_dir = '../checkpoint/Sony/'
result_dir = "./decompose_results/"


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

def lrelu(x):

    return tf.maximum(x * 0.2, x)

def upsample_and_concat(x1, x2, output_channels, in_channels):
    pool_size = 2
    deconv_filter = tf.Variable(tf.truncated_normal([pool_size, pool_size, output_channels, in_channels], stddev=0.02))
    deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2), strides=[1, pool_size, pool_size, 1])

    deconv_output = tf.concat([deconv, x2], 3)
    deconv_output.set_shape([None, None, None, output_channels * 2])

    return deconv_output


def plot_channel_feature_map(channel_feature_map,decompose_conv_dir,conv_number):
    # the shape of "channel_feature_map": (1, 1424, 2128, 1)
    with sess.as_default():

        print (len(channel_feature_map))
        for num_channel in range(len(channel_feature_map)):
            op_feature_channel = channel_feature_map[num_channel].eval()
            h = op_feature_channel.shape[1]
            w = op_feature_channel.shape[2]

            image_array = op_feature_channel.reshape((h,w))

            plt.imshow(image_array,cmap="gray")
            savename = os.path.join(decompose_conv_dir, '{}_{}.png'.format(conv_number,num_channel))
            plt.savefig(savename,dpi = 600)
            print (savename)


def slim2nn_conv2d(image, op_kernel_list, op_biases_list,op_kernelshape_list,op_biasesshape_list):
    start_conv = time.time()
    feature_channel_list = []

    with sess.as_default():
        print ("op_biases_list: " + str(op_biases_list[0].eval()[0]))

        num_layer = len(op_kernelshape_list)
        print ("num_layer: " + str(num_layer))

        for image_channel in range(image.shape[3]):
            curr_layer = 0
            op_img = tf.reshape(image[:, :, :, image_channel], [1, 1424, 2128, 1])
            # num_output = op_kernelshape_list[curr_layer][3]
            op_num_output = 1

            feature_conv1_1_list = []
            for num_kernel1_1 in range(op_num_output):
                curr_layer = 0

                op_kernel_1_1 = tf.reshape(op_kernel_list[curr_layer].eval()[:, :, image_channel, num_kernel1_1], [3, 3, 1, 1])
                op_biases_1_1 = tf.reshape(op_biases_list[curr_layer].eval()[num_kernel1_1], [1])
                conv = tf.nn.conv2d(op_img, op_kernel_1_1, [1, 1, 1, 1], padding='SAME')
                bias = tf.nn.bias_add(conv, op_biases_1_1)
                conv1_1 = lrelu(bias)

                curr_layer = curr_layer+1
                # num_output = op_kernelshape_list[curr_layer][3]
                op_num_output = 1
                num_input = op_kernelshape_list[curr_layer][2]
                feature_conv1_2_list = []

                sess.run(conv1_1)
                feature_channel_list.append(conv1_1)

            #     for num_kernel1_2 in range(op_num_output):
            #         op_img = conv1_1
            #         op_kernel_1_2 = tf.reshape(op_kernel_list[curr_layer].eval()[:, :, num_kernel1_1, num_kernel1_2], [3, 3, 1, 1])
            #         op_biases_1_2 = tf.reshape(op_biases_list[curr_layer].eval()[num_kernel1_2], [1])
            #         conv = tf.nn.conv2d(op_img, op_kernel_1_2, [1, 1, 1, 1], padding='SAME')
            #         bias = tf.nn.bias_add(conv, op_biases_1_2)
            #         conv1_2 = lrelu(bias)
            #
            #         sess.run(conv1_2)
            #         feature_conv1_2_list.append(conv1_2)
            #         # print ("conv1_2:{}".format (conv1_2.eval()[:,:,:,0]))
            #
            #         if num_kernel1_2 % 5  == 0:
            #             total = num_input * num_output
            #             curr = num_kernel1_1 * num_output + num_kernel1_2 + 1
            #             print ("Current Progress of image channel {} num_input: {}  num_output: {}:  "
            #                    "{:.0f}/{:.0f}".format(image_channel, num_kernel1_1,num_kernel1_2, curr, total))
            #             concat_conv1_2 = tf.concat(feature_conv1_2_list, axis=3)
            #             concat_conv1_2 = (tf.reduce_sum(concat_conv1_2, axis=3, keepdims=True))
            #             feature_conv1_2_list = []
            #             feature_conv1_2_list.append(concat_conv1_2)
            #
            #
            #     concat_conv1_1 = tf.concat(feature_conv1_2_list, axis=3)
            #     concat_conv1_1 = (tf.reduce_sum(concat_conv1_1, axis=3, keepdims=True))
            #     feature_conv1_1_list.append(concat_conv1_1)
            #
            # concat_feature = tf.concat(feature_conv1_1_list, axis=3)
            # concat_feature= (tf.reduce_sum(concat_feature, axis=3, keepdims=True))
            # feature_channel_list.append(concat_feature)

        print ("-------------------------------------------------------")
        print (feature_channel_list[0].eval().shape)
        print (feature_channel_list[0].eval()[:,:,:,0])
        print ("-------------------------------------------------------")
        print (feature_channel_list[1].eval().shape)
        print (feature_channel_list[1].eval()[:,:,:,0])
        print ("-------------------------------------------------------")
        print (feature_channel_list[2].eval().shape)
        print (feature_channel_list[2].eval()[:,:,:,0])
        print ("-------------------------------------------------------")
        print (feature_channel_list[3].eval().shape)
        print (feature_channel_list[3].eval()[:,:,:,0])
        print ("-------------------------------------------------------")

    end_conv = time.time()
    print (end_conv - start_conv)
    return feature_channel_list



def conv2d_mulit_channel(input_img,op_kernel_list, op_biases_list,pattern_name,
                         op_kernelshape_list,op_biasesshape_list,decompose_conv_name):
    """
    :param input_img: 输入的图像
    :param op_kernel_list: 原网络所有层的kernel的值
    :param op_biases_list: 原网络所有层的biase的值
    :param pattern_name: 输入图像的形状
    :param op_kernelshape_list: 本次计算中会用到的所有网络层的kernel的值
    :param op_biasesshape_list: 本次计算中
    :param decompose_conv_name: 最后一层网络层的层名
    :return:
    """

    # 命名文件夹result_dir + 输入的impluse名 + 最后一层layer名
    decompose_conv_name = decompose_conv_name
    str_name = str(time.time())[-8:-3]
    name = pattern_name + "_" + decompose_conv_name + "_" + str_name
    decompose_conv_dir = os.path.join(result_dir, name)
    prepare_dir(decompose_conv_dir)

    op_img = input_img
    features_channel = slim2nn_conv2d(op_img, op_kernel_list, op_biases_list,
                                  op_kernelshape_list,op_biasesshape_list)


    # # channel_feature_map的结果是一个颜色图层的卷积值
    # plot_channel_feature_map(features_channel, decompose_conv_dir, decompose_conv_name)

re_digits = re.compile(r'(\d+)')

def emb_numbers(s):
    pieces=re_digits.split(s)
    pieces[1::2]=map(int,pieces[1::2])
    return pieces

def sort_strings_with_emb_numbers(alist):
    aux = [(emb_numbers(s),s) for s in alist]
    aux.sort()
    return [s for __,s in aux]

def load_model_checkpoint(sess,layer_number):
    """
    :param layer_number: 本次计算过程中网络层数
    :return: 一系列和卷积操作相关的值
    所有层名：
    ['g_conv1_1', 'g_conv1_2', 'g_conv2_1', 'g_conv2_2', 'g_conv3_1', 'g_conv3_2',
     'g_conv4_1', 'g_conv4_2', 'g_conv5_1', 'g_conv5_2', 'g_conv6_1', 'g_conv6_2',
    'g_conv7_1', 'g_conv7_2', 'g_conv8_1', 'g_conv8_2', 'g_conv9_1', 'g_conv9_2', 'g_conv10']
    """

    layer_name_list = []
    op_kernelshape_list = []
    op_biasesshape_list = []
    op_kernel_list = []
    op_biases_list = []

    checkpoint_path = os.path.join(checkpoint_dir, "model.ckpt")
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)  # tf.train.NewCheckpointReader
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in var_to_shape_map:
        if "/weights/Adam_1" in key:
            layer_name_list.append(key.split("/weights/Adam_1")[0])

    layer_name_list = sort_strings_with_emb_numbers(layer_name_list)

    for i in range(layer_number):
        decompose_conv_name = layer_name_list[i]
        print (decompose_conv_name)
        key_kernel = decompose_conv_name + "/weights"
        key_biases = decompose_conv_name + "/biases"

        op_kernel_shape = var_to_shape_map[key_kernel]
        op_biases_shape = var_to_shape_map[key_biases]

        print (op_kernel_shape)
        print (op_biases_shape)

        op_kernelshape_list.append(op_kernel_shape)
        op_biasesshape_list.append(op_biases_shape)

    for i in range(layer_number):
        decompose_conv_name = layer_name_list[i]
        with tf.name_scope(decompose_conv_name) as scope:
            kernel = tf.Variable(tf.truncated_normal(op_kernelshape_list[i], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            biases = tf.Variable(tf.constant(0.0, shape=op_biasesshape_list[i], dtype=tf.float32),
                                 trainable=True, name='biases')
            op_kernel_list.append(kernel)
            op_biases_list.append(biases)



    # init = tf.global_variables_initializer()
    with sess.as_default():
    # with tf.compat.v1.Session() as sess:
    #     sess.run(init)

        saver = tf.train.Saver()
        # saver = tf.compat.v1.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt:
            print('loaded ' + ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
    #
    # print ("op_kernel_list: " + str(op_kernel_list[0].eval()))
    #     print ("op_biases_list: " + str(op_biases_list[0].eval()[0]))
    return layer_name_list,op_kernel_list,op_biases_list,op_kernelshape_list,op_biasesshape_list


def init(pattern_name,img_height,img_width):
    # 得到pattern矩阵
    pattern =make_impluse_patterns. create_pattern(pattern_name, img_height, img_width)
    pattern.create()
    pattern = pattern.getPattern()

    # 验证pattern矩阵是否正确, 并且对得到的pattern进行绘制和保存
    make_impluse_patterns.test(pattern,pattern_name)
    # convert2Tensor类型数据,并代入程序中进行测试
    input_img = make_impluse_patterns.conver2Tensor(pattern)

    # 创造对应的impulse图像
    return input_img

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


if __name__ == '__main__':
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.5)
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    sess = tf.compat.v1.Session(config=config)
    init_sess = tf.compat.v1.global_variables_initializer()

    # 方法一 获取input_img数据 以及 weight和biases
    op_layer_number = 2
    pattern_name = "raw"

    # 读入图像
    raw = rawpy.imread(input_path)
    input_full = np.expand_dims(pack_raw(raw), axis=0) * ratio
    input_full = np.minimum(input_full, 1.0)
    input_img = input_full

    # input_img = pack_image0703.loadImage(input_path)
    print (input_img[:,:,:,0])

    # # 方法三进行指定卷积层计算
    # # 网络第二层
    # op_layer_number = 2
    # pattern_name_list = ["center_square", "center_line"]
    # pattern_name = pattern_name_list[0]
    # img_height = 1424
    # img_width = 2128
    # input_img =  init(pattern_name,img_height,img_width)

    # layer_name_list: ['g_conv1_1', 'g_conv1_2', 'g_conv2_1', 'g_conv2_2', 'g_conv3_1', 'g_conv3_2', 'g_conv4_1', 'g_conv4_2', 'g_conv5_1', 'g_conv5_2', 'g_conv6_1', 'g_conv6_2',
    # 'g_conv7_1', 'g_conv7_2', 'g_conv8_1', 'g_conv8_2', 'g_conv9_1', 'g_conv9_2', 'g_conv10']
    # op_kernel_list: [<tf.Variable 'g_conv1_1/weights:0' shape=(3, 3, 4, 32) dtype=float32_ref>, <tf.Variable 'g_conv1_2/weights:0' shape=(3, 3, 32, 32) dtype=float32_ref>]
    # op_biases_list: [<tf.Variable 'g_conv1_1/biases:0' shape=(32,) dtype=float32_ref>, <tf.Variable 'g_conv1_2/biases:0' shape=(32,) dtype=float32_ref>]
    #op_kernelshape_list: [[3, 3, 4, 32], [3, 3, 32, 32]]
    #op_biasesshape_list: [[32], [32]]
    layer_name_list, op_kernel_list, op_biases_list,\
    op_kernelshape_list,op_biasesshape_list= load_model_checkpoint(sess,op_layer_number)
    decompose_conv_name = layer_name_list[op_layer_number-1]


    conv2d_mulit_channel(input_img,op_kernel_list, op_biases_list,pattern_name,
                         op_kernelshape_list,op_biasesshape_list,decompose_conv_name)

    sess.close()