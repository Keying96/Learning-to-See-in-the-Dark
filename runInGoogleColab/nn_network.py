# -*- coding:utf-8 -*-
import tensorflow as tf
import tensorflow.contrib.slim as slim
import make_impluse_patterns
import scipy.io, os,errno
from tensorflow.python import pywrap_tensorflow
import re
import matplotlib.pyplot as plt

# checkpoint_dir = './checkpoint/Sony/'
checkpoint_dir = r'./Learning-to-See-in-the-Dark/checkpoint/Sony/'
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


def network(input):
    conv1 = slim.conv2d(input, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_1')
    conv1 = slim.conv2d(conv1, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_2')
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

def plot_channel_feature_map(channel_feature_map,decompose_conv_dir,conv_number,channel):
    # the shape of "channel_feature_map": (1, 1424, 2128, 1)
    h = channel_feature_map.shape[1]
    w = channel_feature_map.shape[2]

    image_array = channel_feature_map.reshape((h,w))

    plt.imshow(image_array,cmap="gray")
    savename = os.path.join(decompose_conv_dir, '{}_{}.png'.format(conv_number,channel))
    plt.savefig(savename,dpi = 600)
    print (savename)


def slim2nn_conv2d(op_img, op_channel_number,
                   op_kernel_list, op_biases_list,op_kernelshape_list,op_biasesshape_list):
    channel_list = []
    print ("op_kernel_list: " + str(op_kernel_list))
    print ("op_biases_list: " + str(op_biases_list))
    print ("op_kernel_list[0].shape: " + str(op_kernel_list[0].shape))
    print (type(op_kernel_list[0].shape))
    print ("op_biases_list[0].shape: " + str(op_biases_list[0].shape))
    print ("conv1_1/kernel: " +  str(op_kernelshape_list[0][3]))
    print ("conv1_2/kernel: " +  str(op_kernelshape_list[1][3]))
    print ("conv1_1/kernel/shape: " + str(op_kernelshape_list[0]))
    print ("conv1_2/kernel/shape: " + str(op_kernelshape_list[1]))
    print ("conv1_1/kernel/bias: " + str(op_biasesshape_list[0]))
    print ("conv1_2/kernel/bias: " + str(op_biasesshape_list[1]))

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        kernelshape = op_kernelshape_list[0][3]
        # tempo_op_kernelshape = 2
        for i in range(kernelshape):
            # op_kernel_1_1 = tf.reshape(op_kernel_list[0].eval()[:, :, op_channel_number, i], op_kernelshape_list[0])
            op_kernel_1_1 = tf.reshape(op_kernel_list[0].eval()[:, :, op_channel_number, i], [3,3,1,1])
            op_biases_1_1 = tf.reshape(op_biases_list[0].eval()[i], [1])
            conv = tf.nn.conv2d(op_img, op_kernel_1_1, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, op_biases_1_1)
            conv1 = lrelu(bias)

            out_num = op_kernelshape_list[1][3]
            # tempo_out_num = 2
            for j in range(out_num):
                print ("num_input: {}  num_output: {}".format(i,j))
                op_img = conv1
                op_kernel_1_2 = tf.reshape(op_kernel_list[1].eval()[:, :, i, j], [3,3,1,1])
                op_biases_1_2 = tf.reshape(op_biases_list[1].eval()[i], [1])
                conv = tf.nn.conv2d(op_img, op_kernel_1_2, [1, 1, 1, 1], padding='SAME')
                bias = tf.nn.bias_add(conv, op_biases_1_2)
                conv1 = lrelu(bias)

                sess.run(conv1)

                channel_list.append(conv1.eval())

    return channel_list


def conv2d_mulit_channel(input_img,op_kernel_list, op_biases_list,pattern_name,
                         op_kernelshape_list,op_biasesshape_list,decompose_conv_name):

    channel_number = input_img.shape[3]
    decompose_conv_name = decompose_conv_name
    name = pattern_name + "_"+decompose_conv_name
    decompose_conv_dir = os.path.join(result_dir, name)
    prepare_dir(decompose_conv_dir)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        for i in range(channel_number):
            op_img = input_img[:, :, :, i]
            op_img = tf.reshape(op_img, [1, 1424, 2128, 1])
            feature_maps = slim2nn_conv2d(op_img, i, op_kernel_list, op_biases_list,
                                          op_kernelshape_list,op_biasesshape_list)
            # first_list.append(feature)
            channel_feature_list = tf.concat(feature_maps, axis=3)
            channel_feature_map = tf.reduce_sum(channel_feature_list, axis=3, keepdims=True)

            # print (channel_feature_list)
            print (channel_feature_list.shape)
            print (channel_feature_map.shape)
            plot_channel_feature_map(channel_feature_map.eval(), decompose_conv_dir, decompose_conv_name, i)

re_digits = re.compile(r'(\d+)')

def emb_numbers(s):
    pieces=re_digits.split(s)
    pieces[1::2]=map(int,pieces[1::2])
    return pieces

def sort_strings_with_emb_numbers(alist):
    aux = [(emb_numbers(s),s) for s in alist]
    aux.sort()
    return [s for __,s in aux]



def load_model_checkpoint(layer_number):
    # ['g_conv1_1', 'g_conv1_2', 'g_conv2_1', 'g_conv2_2', 'g_conv3_1', 'g_conv3_2',
    #  'g_conv4_1', 'g_conv4_2', 'g_conv5_1', 'g_conv5_2', 'g_conv6_1', 'g_conv6_2',
    # 'g_conv7_1', 'g_conv7_2', 'g_conv8_1', 'g_conv8_2', 'g_conv9_1', 'g_conv9_2', 'g_conv10']
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

    # op_kernel_list = tf.concat(op_kernel_list, axis=3)
    # op_biases_list = tf.concat(op_biases_list, axis=3)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt:
            print('loaded ' + ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)

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


if __name__ == '__main__':
    # 方法三进行指定卷积层计算
    op_layer_number = 2
    pattern_name_list = ["center_square", "center_line"]
    pattern_name = pattern_name_list[0]
    img_height = 1424
    img_width = 2128
    input_img =  init(pattern_name,img_height,img_width)

    layer_name_list, op_kernel_list, op_biases_list,\
    op_kernelshape_list,op_biasesshape_list= load_model_checkpoint(op_layer_number)
    decompose_conv_name = layer_name_list[op_layer_number-1]
    conv2d_mulit_channel(input_img,op_kernel_list, op_biases_list,pattern_name,
                         op_kernelshape_list,op_biasesshape_list,decompose_conv_name)
    # slim2nn_conv2d_original(input_img)
    # conv2d_single_channel(input_img,pattern_name)
