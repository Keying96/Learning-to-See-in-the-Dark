#!/usr/bin/env python
# -*- coding: utf-8 -*-

import load_model_checkpoint
import create_impulse_img
import tensorflow as tf
import os
import tensorflow.contrib.slim as slim
import tool
import slim_network

checkpoint_dir = '../checkpoint/Sony/'
result_dir = "./decompose_results/"

def lrelu(x):

    return tf.maximum(x * 0.2, x)

def creat_conv_dir(decompose_conv_dir,pattern_name, input_num):
    """ 创建保存当层卷积结果保存值的路径
    :param decompose_conv_dir:
    :param pattern_name:
    :param input_num:
    :return:
    """
    list_sub_decompose_conv_dir  = []

    for num in range(input_num):
        sub_decompose_conv_dir = os.path.join(decompose_conv_dir,
                                              "{}_{}".format(pattern_name, num)) #还需要num_kernel
        list_sub_decompose_conv_dir.append(sub_decompose_conv_dir)
    merge_sub_decompose_conv_dir = os.path.join(decompose_conv_dir,
                                                "{}_{}".format(pattern_name, "merge"))  # 还需要num_kernel
    lrelu_sub_decompose_conv_dir = os.path.join(decompose_conv_dir,
                                                "{}_{}".format(pattern_name, "lrelu"))  # 还需要num_kernel
    list_sub_decompose_conv_dir.append(merge_sub_decompose_conv_dir)
    list_sub_decompose_conv_dir.append(lrelu_sub_decompose_conv_dir)

    for i in range(len(list_sub_decompose_conv_dir)):
        tool.prepare_dir(list_sub_decompose_conv_dir[i])

    return list_sub_decompose_conv_dir

def nn_conv1_2(sess,
               input_img,
               pattern_name,
               op_kernel_list,
               op_kernelshape_list,
               decompose_conv_name):

    g = tf.Graph()
    with g.as_default():
        # 修改input_img的格式
        h = input_img.shape[1]
        w = input_img.shape[2]
        c = input_img.shape[3]
        input_img = tf.reshape(input_img,[1,h,w,c])

        #获取计算值
        curr_layer = 1
        input_num1_2 = op_kernelshape_list[curr_layer][2] #输入个数
        output_num1_2 = op_kernelshape_list[curr_layer][3] #输出个数

        # 准备保存的地址
        decompose_conv_dir = os.path.join(result_dir, decompose_conv_name)
        tool.prepare_dir(decompose_conv_dir)
        list_sub_decompose_conv_dir = creat_conv_dir(decompose_conv_dir,pattern_name, input_num1_2)

        #
        for kernel_num in range(output_num1_2):
            merge_list = []
            for con_num in range(input_num1_2):
                op_img_1_2 = tf.reshape(input_img[:,:,:,con_num],[1,h,w,1])

                # 获取conv1_2计算数值
                print ("con_num:{} kernel_num: {}".format(con_num, kernel_num))
                op_kernel_1_2 = tf.reshape(op_kernel_list[curr_layer].eval(session=sess)[:, :, con_num, kernel_num],
                                             [3, 3, 1, 1])
                conv1_2 = tf.nn.conv2d(op_img_1_2, op_kernel_1_2, [1, 1, 1, 1], padding='SAME')
                merge_list.append(conv1_2)

            s = tf.compat.v1.Session()
            with s.as_default():
                eval_merge_list = []
                for i in range(len(merge_list)):
                    print ("merge_list[i]: {} ".format(i))
                    conv_feature = merge_list[i].eval()
                    eval_merge_list.append(conv_feature)
                    tool.plot_conv_feature_map(sess, conv_feature, list_sub_decompose_conv_dir[i], kernel_num) #绘图

                concat_feature = tf.concat(eval_merge_list, axis=3)
                concat_feature = (tf.reduce_sum(concat_feature, axis=3, keepdims=True))
                eval_merge_list = []
                merge_list = []
                eval_merge_list.append(concat_feature.eval())
                merge_feature = eval_merge_list[0]
                tool.plot_conv_feature_map(s, merge_feature, list_sub_decompose_conv_dir[-2], kernel_num) #绘图

                op_biases_1_2 = tf.reshape(op_biases_list[curr_layer].eval(session=sess)[0], [1])
                bias = tf.nn.bias_add(merge_feature, op_biases_1_2)
                conv1_2_lrelu = lrelu(bias)
                tool.plot_conv_feature_map(s, conv1_2_lrelu.eval(), list_sub_decompose_conv_dir[-1], kernel_num) #绘图

                print (conv1_2_lrelu[:,:,0,0].eval())

if __name__ == '__main__':
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9)
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    sess = tf.compat.v1.Session(config=config)
    init_sess = tf.compat.v1.global_variables_initializer()
    sess.run(init_sess)

    # 载入卷积计算相关参数
    op_layer_number = 2
    load = load_model_checkpoint.LoadModelCheckpoint(sess,checkpoint_dir, op_layer_number)
    # load = load_model_checkpoint.LoadModelCheckpoint(checkpoint_dir, op_layer_number)
    layer_name_list, op_kernel_list, op_biases_list, \
    op_kernelshape_list, op_biasesshape_list, decompose_conv_name = load.load_model_checkpoint()
    print ("===================== Start to calculate the result of {} =======================".format(decompose_conv_name))

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
    print ("===================== Start to calculate the result of {} =======================".format(pattern_name))

    conv1_1,conv1_2 = slim_network.init(input_img)
    # print (conv1_1.shape)
    # print (conv1_2[:,:,0,0])
    # print(type(conv1_1))  #<type 'numpy.ndarray'>
    # print (conv1_1.shape) # (15, 15, 32)
    # print (op_kernel_list[0].eval(session= sess))

    # # 计算conv1_2卷积(15, 15, 32)
    nn_conv1_2(sess, conv1_1, pattern_name,op_kernel_list,
                         op_kernelshape_list,decompose_conv_name)


    sess.close()