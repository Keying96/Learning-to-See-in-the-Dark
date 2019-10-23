# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os,errno

# 在本程序中0:表示白色, 1:表示黑色
pattern_name_list = ["center_square","center_line"]
result_dir = "./decompose_results/impluse_patterns/"

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

# 在图像的正中心画一个impulse_size*impulse_size的正方形
class CenterSquare:
    def __init__(self,img_height,img_width):
        self.__img_height = img_height
        self.__img_width = img_width
        self.__pattern = 0

    def setPattern(self,pattern):
        self.__pattern = pattern
    def getPattern(self):
        return self.__pattern

    def create(self):
        center_width = int(round(self.__img_width / 2))
        center_height = int(round(self.__img_height / 2))
        impulse_size = 20

        img = np.zeros([self.__img_height, self.__img_width, 4], np.float32)
        # img[:,:,0] = np.ones([img_width,img_height]) * 0
        # for i in range(100):
        #     img[center_width+i,center_height+i] = 1
        for j in range(self.__img_width):
            for i in range(self.__img_height):
                if (j > (center_width - impulse_size) and j < (center_width + impulse_size)) \
                        and (i > (center_height - impulse_size) and i < (center_height + impulse_size)):
                    img[i, j, :] = 0
                else:
                    img[i, j, :] = 1

        self.__pattern = img

# 在图像的正中心画一个impulse_size_height*impulse_size_width的长方形
class CenterRectangle:
    def __init__(self, img_height, img_width):
        self.__img_height = img_height
        self.__img_width = img_width

    def setPattern(self, pattern):
        self.__pattern = pattern

    def getPattern(self):
        return self.__pattern

    def create(self):
        center_width = int(round(self.__img_width / 2))
        center_height = int(round(self.__img_height / 2))
        impulse_size_height = 10
        impulse_size_width = 300

        img = np.zeros([self.__img_height, self.__img_width, 4], np.float32)
        # img[:,:,0] = np.ones([img_width,img_height]) * 0
        # for i in range(100):
        #     img[center_width+i,center_height+i] = 1
        for j in range(self.__img_width):
            for i in range(self.__img_height):
                if (j > (center_width - impulse_size_width) and j < (center_width + impulse_size_width)) \
                        and (i > (center_height - impulse_size_height) and i < (center_height + impulse_size_height)):
                    img[i, j, :] = 0
                else:
                    img[i, j, :] = 1

        self.__pattern = img

def plot_pattern_img(pattern, patter_name):
    prepare_dir(result_dir)
    #     得到的pattern.shape(1,height,width,channel)
    h = pattern.shape[0]
    w = pattern.shape[1]

    # plt方法
    out_img = pattern.reshape((h, w))

    plt.imshow(out_img, cmap="gray")
    save_name = os.path.join(result_dir,'{}.png'.format(patter_name))
    plt.savefig(save_name, dpi=600)
    print (os.path.join(save_name))

def create_pattern(pattern_name,img_height,img_width):

    if pattern_name == pattern_name_list[0]:
        pattern = CenterSquare(img_height,img_width)
    elif pattern_name == pattern_name_list[1]:
        pattern = CenterRectangle(img_height,img_width)
    else:
        print ("sorry, we cannot create.")

    return pattern

def test(pattern,pattern_name):
    test_img = pattern[:,:,0]
    plot_pattern_img(test_img, pattern_name)

def conver2Tensor(pattern):
    input_img = tf.convert_to_tensor(pattern,tf.float32)
    input_img = tf.expand_dims(input_img, 0)
    return input_img

if __name__ == '__main__':
    # img_height = 1424
    # img_width = 2128

    img_height = 15
    img_width = 15

    pattern_name = pattern_name_list[0]

    # 得到pattern矩阵
    pattern = create_pattern(pattern_name,img_height,img_width)
    pattern.create()
    pattern = pattern.getPattern()

    # 验证pattern矩阵是否正确, 并且对得到的pattern进行绘制和保存
    test(pattern,pattern_name)
    # convert2Tensor类型数据,并代入程序中进行测试
    input_img = conver2Tensor(pattern)
    # print (input_img.shape)
