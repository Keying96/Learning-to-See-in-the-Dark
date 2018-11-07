# -*- coding:utf-8 -*-
from __future__ import division
import os, scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import rawpy
import glob
import  threading,time

import cv2 as cv
import os.path


input_dir = "./dataset/Iphone/"
checkpoint_dir = './checkpoint/Sony/'
result_dir = './result_Iphone/'

ratiosList = {28, 87, 189, 366}
ratioA = 33


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

def RAW2PNG(test_name,raw_name):
    # print("gt_path: {gt_path}".format(gt_path = gt_path))

    gt_raw = rawpy.imread(test_name)
    im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
    gt_full = np.expand_dims(np.float32(im / 65535.0), axis=0)

    gt_full = gt_full[0, :, :, :]

    # scipy.misc.toimage(gt_full * 255, high=255, low=0, cmin=0, cmax=255).save(
    #     result_dir + 'final/%5d_00_%d_gt.png' % (test_id, ratio))

    scipy.misc.toimage(gt_full * 255, high=255, low=0, cmin=0, cmax=255).save(raw_name)

def clearLabel(_ax):
  _ax.tick_params(labelbottom="off",bottom="off")
  _ax.tick_params(labelleft="off",left="off")
  _ax.set_xticklabels([])
  _ax.axis('off')
  return _ax


def readImage(_filename):
  if os.path.exists(_filename):
    img = cv.imread(_filename)
    return img

#test the image
def toSeeInTheDark(ratio, in_path):
    sess = tf.Session()
    in_image = tf.placeholder(tf.float32, [None, None, None, 4])
    out_image = network(in_image)


    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

    if ckpt:
        print('loaded ' + ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)

    if not os.path.isdir(result_dir + 'final/'):
        os.makedirs(result_dir + 'final/')

    raw = rawpy.imread(in_path)
    input_full = np.expand_dims(pack_raw(raw), axis=0) * ratio

    im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)

    input_full = np.minimum(input_full, 1.0)

    output = sess.run(out_image, feed_dict={in_image: input_full})
    output = np.minimum(np.maximum(output, 0), 1)

    output = output[0, :, :, :]

    tmp_test_name = os.path.basename(in_path).split('.')[0]

    result_path = os.path.join(result_dir, r'final_%d/'%(ratio) + tmp_test_name+'_%d.png'% ratio)
    scipy.misc.toimage(output * 255, high=255, low=0, cmin=0, cmax=255).save(result_path)


def imageOps(ratio,tmp_test_path, result_dir):
    toSeeInTheDark(ratio,tmp_test_path)

    # tmp_test_name = os.path.basename(tmp_test_path).split('.')[0]
    # tmp_result_name = os.path.join(result_dir, r'final_%d/'%(ratio) + tmp_test_name+'.png')
    # new_imge.save( tmp_result_name )

def threadOPS(input_dir,result_dir):
    if os.path.isdir(input_dir):
        # get test IDs
        test_fns = glob.glob(input_dir + '/*.dng')
        test_ids = [os.path.basename(test_fn).split('.') for test_fn in test_fns]
    else:
        print('input dir is wrong')
        return -1

    threadImage = [0] * 20
    _index = 0
    for test_fn in test_fns:
        print(test_fn)
        tmp_test_path = test_fn

        if tmp_test_path.split('.')[1] != "DS_Store":
            threadImage[_index] = threading.Thread(target= imageOps,
                                                       args=(ratioA, tmp_test_path, result_dir, ))
            threadImage[_index].start()
            _index += 1
            time.sleep(1)

            # for ratio in ratiosList:
            #     threadImage[_index] = threading.Thread(target= imageOps,
            #                                            args=(ratio, tmp_test_path, result_dir))
            #     threadImage[_index].start()
            #     _index += 1



if __name__=="__main__":


    for ratio in ratiosList:
        if not os.path.isdir(result_dir + r'final_%d/'%(ratio)):
            os.makedirs(result_dir + r'final_%d/'%(ratio))

    if not os.path.isdir(result_dir + r'final_%d/' % (ratioA)):
         os.makedirs(result_dir + r'final_%d/' %(ratioA))

    threadOPS(input_dir,result_dir)


