# -*- coding:utf-8 -*-
import visualize_filters_0620
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from PIL import Image
import os
import errno

sess = tf.Session()


impulse_root = "./impulse_file"

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


    # out = np.concatenate((im[0:H:2, 0:W:2, :],
    #                       im[0:H:2, 1:W:2, :],
    #                       im[1:H:2, 1:W:2, :],
    #                       im[1:H:2, 0:W:2, :]), axis=2)
    #0 is black, 255 is whilt.

def create_random_img(img_height,img_width,black_number):
    pass

def create_appoint_img(img_height,img_width,black_number):
    pass

def create_center_img(img_height,img_width,black_number):
    center_width =  int(round(img_width/2))
    center_height =  int(round(img_height/2))
    impulse_size = 100

    # img = np.zeros([img_width,img_height,1],np.float32)
    # # img[:,:,0] = np.ones([img_width,img_height]) * 0
    # # for i in range(100):
    # #     img[center_width+i,center_height+i] = 1
    # for i  in range(img_width):
    #     for j in range(img_height):
    #         if (i > (center_width - impulse_size) and i < (center_width + impulse_size)) and (j > (center_height - impulse_size) and j < (center_height + impulse_size)):
    #             img [i,j,0] = 0.014
    #         else:
    #             img[i,j,0] = 0

    img = np.zeros([img_height,img_width,1],np.float32)
    # img[:,:,0] = np.ones([img_width,img_height]) * 0
    # for i in range(100):
    #     img[center_width+i,center_height+i] = 1
    for j in range(img_width):
        for i in range(img_height):
            if (j > (center_width - impulse_size) and j < (center_width + impulse_size)) \
                    and (i > (center_height - impulse_size) and i < (center_height + impulse_size)):
                img[i, j, :] = 0
            else:
                img[i, j, :] = 1

    print img

    return  img

def create_png(img_height,img_width):
    center_width =  int(round(img_width/2))
    center_height =  int(round(img_height/2))
    impulse_size = 100

    img = np.zeros([img_height,img_width,3],np.uint8)
    # img[:,:,0] = np.ones([img_width,img_height]) * 0
    # for i in range(100):
    #     img[center_width+i,center_height+i] = 1
    for j  in range(img_width):
        for i in range(img_height):
                if (j > (center_width - impulse_size) and j < (center_width + impulse_size)) \
                        and (i > (center_height - impulse_size) and i < (center_height + impulse_size)):
                    img [i,j,:] = 0
                else:
                    img[i,j,:] = 255

    # img.shape: [1424, 2128, 3]
    im = Image.fromarray(img)
    im_name = "impulse.jpeg"
    im.save(os.path.join(impulse_root,im_name))

def decompose(img,weight,baises,layer_name,channel,kernel):

    name = layer_name+"_"+"C" +  str(kernel)
    save_name = os.path.join(impulse_root,name)
    prepare_dir(save_name)

    img = tf.reshape(img,[1, 1424, 2128, 1])
    weight = tf.reshape(weight, [3, 3, 1, 1])
    baises = tf.reshape(baises,[1])

    op = tf.nn.conv2d(img, weight, strides=[1, 1, 1, 1], padding='SAME')
    op_baises = tf.nn.bias_add(op, baises)
    decompose_featuremap = tf.nn.relu(op_baises)
    
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        # sess.run(op)
        sess.run(decompose_featuremap)
    # print ("result: " + str(result.shape))
    # print (result)
    plot_image(decompose_featuremap,save_name,channel)

    return decompose_featuremap


def plot_image(image_array,save_name,channel):
    # h,w = image_array.shape
    h = image_array.shape[1]
    w = image_array.shape[2]

    # out_img = image_array.copy()
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        out_img =  image_array.eval()

    # out_img.shape: [1424, 2128]
    # matplotlib.pyplot方法
    # out_img = out_img.reshape((h, w))
    # print ("out_img: " + str(out_img))
    # im = Image.fromarray(np.uint8(out_img),"1")
    # im_name = '{}.JPEG'.format(channel)
    # im.save(os.path.join(save_name,im_name))

    # plt方法
    out_img = out_img.reshape((h,w))
    #subtract the black level
    # out_img = np.maximum(out_img - 512, 0) / (16383 - 512)
    # plt.axis("off")
    # plt.imshow(out_img, cmap="gray", aspect='auto')
    plt.imshow(out_img, cmap="gray")
    plt.savefig(os.path.join(save_name, '{}.png'.format( channel)),dpi = 300)
    print (os.path.join(save_name, '{}.png'.format( channel)))
    # plt.show()


def plot_feature_map(feature_map, image_name):
    h = feature_map.shape[1]
    w = feature_map.shape[2]

    out_img = feature_map.copy()
    out_img = out_img.reshape((h, w))
    plt.imshow(out_img,cmap="gray")
    image_name = image_name + ".png"
    plt.savefig(os.path.join(impulse_root, image_name),dpi = 300)
    # print (os.path.join(save_name, '{}.png'.format(channel)))


if __name__ == '__main__':
    prepare_dir(impulse_root)
    add_channel_imgs = []

    img_height = 1424
    img_width = 2128
    black_number = 1

    weights = visualize_filters_0620.weights
    baises = visualize_filters_0620.biases
    layer_name = visualize_filters_0620.visualize_layer_name

    # 输出制作的图片
    # create_png(img_height,img_width)

    # creat_random_img(img_height,img_width,black_number)
    # creat_appoint_img(img_height,img_width,black_number)

    # 进行运算
    img = create_center_img(img_height,img_width,black_number)
    in_channels = int(weights.shape[2])
    for i in range(in_channels):
        add_feature = decompose(img,weights[:,:,i,0],baises[0], layer_name, i,0)
        add_feature = tf.convert_to_tensor(add_feature,tf.float32)
        add_channel_imgs.append(add_feature)

    # init_op = tf.global_variables_initializer()
    # with sess.as_default():
    #     sess.run(init_op)
    #     print("===============================================================")
    #     print (add_channel_imgs[0].eval())
    #     print("===============================================================")
    #     print (add_channel_imgs[1].eval())
    #     print("===============================================================")
    #     print (add_channel_imgs[2].eval())
    #     print("===============================================================")
    #     print (add_channel_imgs[3].eval())


    add_channel_img = tf.concat(add_channel_imgs,axis=3)
    feature_map = tf.reduce_sum(add_channel_img, axis=3, keepdims=True)
    init_op = tf.global_variables_initializer()
    with sess.as_default():
        sess.run(init_op)
        # print("===============================================================")
        # print (add_channel_img.eval())
        # print("===============================================================")
        feature_map = (feature_map.eval())
        print (feature_map)
    plot_feature_map(feature_map,"test")

