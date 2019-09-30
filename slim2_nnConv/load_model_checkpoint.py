#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Functions for loading network's checkpoint
"""

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import re, os


class LoadModelCheckpoint(object):

    def __init__(self, sess, checkpoint_dir,op_layer_number):
        self._sess = sess
        self._checkpoint_dir = checkpoint_dir
        self._op_layer_number = op_layer_number

    def load_model_checkpoint(self):
        """
        :param layer_number: 本次计算过程中网络层数
        :return: 一系列和卷积操作相关的值
        所有层名：
        ['g_conv1_1', 'g_conv1_2', 'g_conv2_1', 'g_conv2_2', 'g_conv3_1', 'g_conv3_2',
         'g_conv4_1', 'g_conv4_2', 'g_conv5_1', 'g_conv5_2', 'g_conv6_1', 'g_conv6_2',
        'g_conv7_1', 'g_conv7_2', 'g_conv8_1', 'g_conv8_2', 'g_conv9_1', 'g_conv9_2', 'g_conv10']
        """
        print ("checkpoint_dir: {}  op_layer_number: {}".format(self._checkpoint_dir, self._op_layer_number))

        layer_name_list = []
        op_kernelshape_list = []
        op_biasesshape_list = []
        op_kernel_list = []
        op_biases_list = []

        __re_digits = re.compile(r'(\d+)')

        def __emb_numbers(s):
            pieces = __re_digits.split(s)
            pieces[1::2] = map(int, pieces[1::2])
            return pieces

        def __sort_strings_with_emb_numbers(alist):
            aux = [(__emb_numbers(s), s) for s in alist]
            aux.sort()
            return [s for __, s in aux]

        checkpoint_path = os.path.join(self._checkpoint_dir, "model.ckpt")
        reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)  # tf.train.NewCheckpointReader
        var_to_shape_map = reader.get_variable_to_shape_map()
        for key in var_to_shape_map:
            if "/weights/Adam_1" in key:
                layer_name_list.append(key.split("/weights/Adam_1")[0])

        layer_name_list = __sort_strings_with_emb_numbers(layer_name_list)
        decompose_conv_name = layer_name_list[op_layer_number]

        for i in range(self._op_layer_number):
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

        for i in range(self._op_layer_number):
            decompose_conv_name = layer_name_list[i]
            with tf.name_scope(decompose_conv_name) as scope:
                kernel = tf.Variable(tf.random.truncated_normal(op_kernelshape_list[i], dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
                biases = tf.Variable(tf.constant(0.0, shape=op_biasesshape_list[i], dtype=tf.float32),
                                     trainable=True, name='biases')
                op_kernel_list.append(kernel)
                op_biases_list.append(biases)

        # init = tf.global_variables_initializer()
        with self._sess.as_default():
            # with tf.compat.v1.Session() as sess:
            #     sess.run(init)

            saver = tf.compat.v1.train.Saver()
            ckpt = tf.train.get_checkpoint_state(self._checkpoint_dir)
            if ckpt:
                print('loaded ' + ckpt.model_checkpoint_path)
                saver.restore(self._sess, ckpt.model_checkpoint_path)
        #
        # print ("op_kernel_list: " + str(op_kernel_list[0].eval()))
        #     print ("op_biases_list: " + str(op_biases_list[0].eval()[0]))
        return layer_name_list, op_kernel_list, op_biases_list, \
               op_kernelshape_list, op_biasesshape_list, decompose_conv_name


if __name__ == '__main__':
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.5)
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    sess = tf.compat.v1.Session(config=config)
    init_sess = tf.compat.v1.global_variables_initializer()

    checkpoint_dir = '../checkpoint/Sony/'
    op_layer_number = 2

    load = LoadModelCheckpoint(sess, checkpoint_dir, op_layer_number)

    layer_name_list, op_kernel_list, op_biases_list,\
    op_kernelshape_list,op_biasesshape_list= load.load_model_checkpoint()

    print (layer_name_list)

