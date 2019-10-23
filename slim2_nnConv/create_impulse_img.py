#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, errno
import np
import matplotlib.pyplot as plt
import time
import rawpy

_Impulses = ["center_square","center_line","Sony"]
result_dir = "./decompose_results/impluse_patterns/"


def prepare_dir(path, empty=False):
    """
    Creates a directory if it soes not exist
    :param path: string, path to desired directory
    :param empty: boolean, delete all directory content if it exists
    :return: nothing
    """

    def _create_dir(path):
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
        _create_dir(path)

class CreateImpulseImg(object):

    def __init__(self,impulse_type,impulse_size,img_height,img_width):
        self._impulse_type = impulse_type
        self._impulse_size = impulse_size
        self._img_height = img_height
        self._img_width = img_width

    def create(self):
        center_width = int(round(self._img_width / 2))
        center_height = int(round(self._img_height / 2))

        img = np.zeros([self._img_height, self._img_width, 4], np.float32)

        print (self._impulse_type)
        if self._impulse_type == 0:
            pattern_name = _Impulses [0]
            pattern =  CreateCenterSquare(center_width, center_height,
                                          img, self._impulse_size).create()
        elif self._impulse_type == 1:
            pattern_name = _Impulses [1]
            pattern =  CreateCenterLine(center_width, center_height,
                                        img, self._impulse_size).create()

        elif self._impulse_type == 2:
            pattern_name = _Impulses [2]
            # pattern =  CreateSony(center_width, center_height,
            #                             img, self._impulse_size).create()
            pattern = CreateSony(center_width, center_height,
                                        img, self._impulse_size).create()
            pattern = pattern[0]

        # 输出图像plt
        h = pattern.shape[0]
        w = pattern.shape[1]
        out_img = pattern[:,:,0].reshape((h,w))
        plt.imshow(out_img, cmap="gray")

        # 设置图像文件名save_name
        prepare_dir(result_dir)
        time_now = str(time.time())[-8:-3]
        pattern_name = '{}_{}'.format(pattern_name, time_now)
        name = "{}.png".format(pattern_name)
        save_name = os.path.join(result_dir, name )
        plt.savefig(save_name, dpi=600)
        print (os.path.join(save_name))

        print("len(pattern): {}".format(len(pattern)))

        return pattern, pattern_name #返回图像图像所有channel数据,和图像名

class CreateSony(CreateImpulseImg):
    def __int__(self):
        print("Sony was created!")

        # CreateImpulseImg.__init__(self,impulse_type,
        #                           impulse_size,
        #                           img_height,
        #                           img_width)


    def create(self):
        self._ratio = 28
        self._input_path =  '../dataset/short/00001_00_0.1s.ARW'
        self._pattern = 0

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

        # 读入图像

        raw = rawpy.imread(self._input_path)
        input_full = np.expand_dims(pack_raw(raw), axis=0) * self._ratio
        input_full = np.minimum(input_full, 1.0)
        self._pattern = input_full

        return  self._pattern

class CreateCenterSquare(CreateImpulseImg):
    def __init__(self,center_width, center_height,img,impulse_size):
        self._center_width = center_width
        self._center_height = center_height
        self._impulse_size = impulse_size
        self._img = img
        self._pattern = 0

    def create(self):
        for j in range(self._center_width):
            for i in range(self._center_height):
                if (j >= (self._center_width - self._impulse_size) and j <= (self._center_width + self._impulse_size)) \
                        and (i >= (self._center_height - self._impulse_size) and i <= (self._center_height + self._impulse_size)):
                    print ("impulse image (i,j) : {},{}".format(i,j))
                    self._img[i, j, :] = 1
                else:
                    self._img[i, j, :] = 0

        self._pattern = self._img

        return  self._pattern


class CreateCenterLine(CreateImpulseImg):
    def __init__(self,center_width, center_height,img,impulse_size):
        self._center_width = center_width
        self._center_height = center_height
        self._impulse_size = impulse_size
        self._img = img
        self._pattern = 0

    def create(self):
        for j in range(self._center_width):
            for i in range(self._center_height):
                if (j >= (self._center_width - self._impulse_size) and j <= (self._center_width + self._impulse_size)) \
                        and (i >= (self._center_height - self._impulse_size * 3) and i <= (self._center_height + self._impulse_size * 3)):
                    self._img[i, j, :] = 1
                else:
                    self._img[i, j, :] = 0

        self._pattern = self._img

        return  self._pattern

if __name__ == '__main__':
    # img_height = 1424
    # img_width = 2128
    img_height = 15
    img_width = 15
    impulse_type = 0
    impulse_size = 1

    impulse_img, pattern_name = CreateImpulseImg(impulse_type, impulse_size,
                                   img_height,img_width).create()
    print (impulse_img.shape)
    # print (impulse_img[:,:,0])
    # print (pattern_name)