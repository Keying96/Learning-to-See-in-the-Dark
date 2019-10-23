#!/usr/bin/env python
# -*- coding: utf-8 -*-


import load_model_checkpoint
import create_impulse_img
import tool
import tensorflow as tf
import  os, errno
import matplotlib.pyplot as plt

checkpoint_dir = '../checkpoint/Sony/'
result_dir = "./decompose_results/"


def lrelu(x):

    return tf.maximum(x * 0.2, x)

def nn_conv1_1(input_img,
               pattern_name,
               op_kernel_list,
               op_biases_list,
               op_kernelshape_list,
               op_biasesshape_list,
               decompose_conv_name):

    decompose_conv_dir = os.path.join(result_dir, decompose_conv_name)
    tool.prepare_dir(decompose_conv_dir)

    h = input_img.shape[0]
    w= input_img.shape[1]
    with sess.as_default():
        # feature_channel_list = []
        # curr_layer = 0
        # input_img = tf.reshape(input_img, [1, h, w, 4])
        # num_op_channel = input_img.shape[3]
        # print ("num_op_channel: {}".format(num_op_channel))
        #
        # for image_channel in range(num_op_channel):
        #     feature_conv1_1 = []
        #     feature_test = []
        #
        #     op_img = tf.reshape(input_img[:, :, :, image_channel], [1, h, w, 1])
        #
        #     #同一个channel图层下卷积结果的dir
        #     num_output = op_kernelshape_list[curr_layer][3]
        #     print ("num_output: {}".format(num_output))
        #     sub_decompose_conv_dir = os.path.join(decompose_conv_dir,
        #                                           "{}_{}".format(pattern_name, image_channel))
        #     tool.prepare_dir(sub_decompose_conv_dir)
        #
        #     for num_kernel1_1 in range(num_output):
        #         print ("image_channel: {} num_kernel1_1: {}".format(image_channel, num_kernel1_1))
        #         op_kernel_1_1 = tf.reshape(op_kernel_list[curr_layer].eval()[:, :, image_channel, num_kernel1_1],
        #                                    [3, 3, 1, 1])
        #         conv = tf.nn.conv2d(op_img, op_kernel_1_1, [1, 1, 1, 1], padding='SAME')
        #         conv_result = sess.run(conv)
        #         feature_conv1_1.append(conv.eval())
        #         feature_test.append(conv_result[:,0,0,0])
        #         tool.plot_conv_feature_map(sess,conv_result, sub_decompose_conv_dir, num_kernel1_1)
        #
        #         feature_conv1_1 = tool.concat_feature(sess, feature_conv1_1,5)
        #
        #     feature_channel = tool.concat_feature(sess,feature_conv1_1,1)
        #     tool.plot_channel_feature_map(sess,feature_channel, decompose_conv_dir,
        #                              pattern_name,image_channel)

        # 创建结果的对应保存地址
        sub_decompose_conv_dir0 = os.path.join(decompose_conv_dir,
                                              "{}_{}".format(pattern_name, 0)) #还需要num_kernel
        sub_decompose_conv_dir1 = os.path.join(decompose_conv_dir,
                                              "{}_{}".format(pattern_name, 1)) #还需要num_kernel
        sub_decompose_conv_dir2 = os.path.join(decompose_conv_dir,
                                              "{}_{}".format(pattern_name, 2)) #还需要num_kernel
        sub_decompose_conv_dir3 = os.path.join(decompose_conv_dir,
                                              "{}_{}".format(pattern_name, 3)) #还需要num_kernel
        merge_sub_decompose_conv_dir = os.path.join(decompose_conv_dir,
                                              "{}_{}".format(pattern_name, "merge")) #还需要num_kernel
        lrelu_sub_decompose_conv_dir = os.path.join(decompose_conv_dir,
                                              "{}_{}".format(pattern_name, "lrelu")) #还需要num_kernel
        sub_decompose_conv_dir = []
        sub_decompose_conv_dir.append(sub_decompose_conv_dir0)
        sub_decompose_conv_dir.append(sub_decompose_conv_dir1)
        sub_decompose_conv_dir.append(sub_decompose_conv_dir2)
        sub_decompose_conv_dir.append(sub_decompose_conv_dir3)
        sub_decompose_conv_dir.append(merge_sub_decompose_conv_dir)
        sub_decompose_conv_dir.append(lrelu_sub_decompose_conv_dir)
        for i in range(len(sub_decompose_conv_dir)):
            tool.prepare_dir(sub_decompose_conv_dir[i])

        # 获取每channel的输入图像
        input_img = tf.reshape(input_img, [1, h, w, 4])
        op_img_0 = tf.reshape(input_img[:, :, :, 0], [1, h, w, 1])
        op_img_1 = tf.reshape(input_img[:, :, :, 1], [1, h, w, 1])
        op_img_2 = tf.reshape(input_img[:, :, :, 2], [1, h, w, 1])
        op_img_3 = tf.reshape(input_img[:, :, :, 3], [1, h, w, 1])
        curr_layer = 0
        num_output = op_kernelshape_list[curr_layer][3]
        # for num_kernel1_1 in range(num_output):
        for num_kernel1_1 in range(1):
            merge_list = []
            # 获取对应kernel值
            op_kernel_1_1_0 = tf.reshape(op_kernel_list[curr_layer].eval()[:, :, 0, num_kernel1_1],
                                       [3, 3, 1, 1])
            op_kernel_1_1_1 = tf.reshape(op_kernel_list[curr_layer].eval()[:, :, 1, num_kernel1_1],
                                       [3, 3, 1, 1])
            op_kernel_1_1_2 = tf.reshape(op_kernel_list[curr_layer].eval()[:, :, 2, num_kernel1_1],
                                       [3, 3, 1, 1])
            op_kernel_1_1_3 = tf.reshape(op_kernel_list[curr_layer].eval()[:, :, 3, num_kernel1_1],
                                       [3, 3, 1, 1])
            # 进行卷积计算
            conv_0 = tf.nn.conv2d(op_img_0, op_kernel_1_1_0, [1, 1, 1, 1], padding='SAME')
            conv_1 = tf.nn.conv2d(op_img_1, op_kernel_1_1_1, [1, 1, 1, 1], padding='SAME')
            conv_2 = tf.nn.conv2d(op_img_2, op_kernel_1_1_2, [1, 1, 1, 1], padding='SAME')
            conv_3 = tf.nn.conv2d(op_img_3, op_kernel_1_1_3, [1, 1, 1, 1], padding='SAME')
            # 输出卷积结果
            merge_list.append(conv_0.eval())
            merge_list.append(conv_1.eval())
            merge_list.append(conv_2.eval())
            merge_list.append(conv_3.eval())
            for i in range(len(merge_list)):
                tool.plot_conv_feature_map(sess, merge_list[i], sub_decompose_conv_dir[i],num_kernel1_1)
            merge_feature = tool.concat_feature(sess, merge_list,4)[0]
            merge_list = []
            tool.plot_conv_feature_map(sess,merge_feature,merge_sub_decompose_conv_dir,num_kernel1_1)
            # 进行biase和lrelu计算
            op_biases_1_1 = tf.reshape(op_biases_list[curr_layer].eval()[num_kernel1_1], [1])
            bias = tf.nn.bias_add(merge_feature,op_biases_1_1)
            conv1_1_lrelu = lrelu(bias)
            # print ("conv1_1_lrelu[:,:,0,0]: {}".format(conv1_1_lrelu[:,:,0,0].eval()))
            tool.plot_conv_feature_map(sess,conv1_1_lrelu.eval(),lrelu_sub_decompose_conv_dir,num_kernel1_1)

    # for image_channel in range(num_op_channel):
        #     feature_conv1_1 = []
        #     feature_test = []
        #
        #     op_img = tf.reshape(input_img[:, :, :, image_channel], [1, h, w, 1])
        #
        #     #同一个channel图层下卷积结果的dir
        #     num_output = op_kernelshape_list[curr_layer][3]
        #     print ("num_output: {}".format(num_output))
        #     sub_decompose_conv_dir = os.path.join(decompose_conv_dir,
        #                                           "{}_{}".format(pattern_name, image_channel))
        #     tool.prepare_dir(sub_decompose_conv_dir)
        #
        #     for num_kernel1_1 in range(num_output):
        #         print ("image_channel: {} num_kernel1_1: {}".format(image_channel, num_kernel1_1))
        #         op_kernel_1_1 = tf.reshape(op_kernel_list[curr_layer].eval()[:, :, image_channel, num_kernel1_1],
        #                                    [3, 3, 1, 1])
        #         conv = tf.nn.conv2d(op_img, op_kernel_1_1, [1, 1, 1, 1], padding='SAME')
        #         conv_result = sess.run(conv)
        #         feature_conv1_1.append(conv.eval())
        #         feature_test.append(conv_result[:,0,0,0])
        #         tool.plot_conv_feature_map(sess,conv_result, sub_decompose_conv_dir, num_kernel1_1)
        #
        #         feature_conv1_1 = tool.concat_feature(sess, feature_conv1_1,5)
        #
        #     feature_channel = tool.concat_feature(sess,feature_conv1_1,1)
        #     tool.plot_channel_feature_map(sess,feature_channel, decompose_conv_dir,
        #                              pattern_name,image_channel)


if __name__ == '__main__':
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9)
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    sess = tf.compat.v1.Session(config=config)
    init_sess = tf.compat.v1.global_variables_initializer()

    # 载入卷积计算相关参数
    op_layer_number = 1
    load = load_model_checkpoint.LoadModelCheckpoint(sess, checkpoint_dir, op_layer_number)
    layer_name_list, op_kernel_list, op_biases_list, \
    op_kernelshape_list, op_biasesshape_list, decompose_conv_name = load.load_model_checkpoint()

    # 获取自定义input_img
    # img_height = 1424
    # img_width = 2128
    # impulse_type = 2
    img_height = 15
    img_width = 15
    impulse_type = 0
    impulse_size = 1
    input_img, pattern_name = create_impulse_img.CreateImpulseImg(impulse_type, impulse_size,
                                                 img_height, img_width).create()

    nn_conv1_1(input_img, pattern_name,op_kernel_list, op_biases_list,
                         op_kernelshape_list,op_biasesshape_list,decompose_conv_name)   #进行conv1_1卷积计算

    sess.close()