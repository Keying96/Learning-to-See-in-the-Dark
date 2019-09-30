#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, errno
import np
import matplotlib.pyplot as plt
import time
_Impulses = ["center_square","center_line"]
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

        img = np.ones([self._img_height, self._img_width, 4], np.float32)

        print (self._impulse_type)
        if self._impulse_type == 0:
            pattern_name = _Impulses [0]
            pattern =  CreateCenterSquare(center_width, center_height,
                                          img, self._impulse_size).create()
        elif self._impulse_type == 1:
            pattern_name = _Impulses [1]
            pattern =  CreateCenterLine(center_width, center_height,
                                        img, self._impulse_size).create()

        prepare_dir(result_dir)
        h = pattern.shape[0]
        w = pattern.shape[1]
        out_img = pattern[:,:,0].reshape((h,w))
        plt.imshow(out_img, cmap="gray")
        time_now = str(time.time())[-8:-3]
        pattern_name = '{}_{}.png'.format(pattern_name, time_now)
        save_name = os.path.join(result_dir, pattern_name )
        plt.savefig(save_name, dpi=600)
        print (os.path.join(save_name))

        return pattern, pattern_name


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
                    self._img[i, j, :] = 0
                else:
                    self._img[i, j, :] = 1

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
                    self._img[i, j, :] = 0
                else:
                    self._img[i, j, :] = 1

        self._pattern = self._img

        return  self._pattern

if __name__ == '__main__':
    img_height = 1424
    img_width = 2128
    impulse_type = 1
    impulse_size = 1

    impulse_img, pattern_name = CreateImpulseImg(impulse_type, impulse_size,
                                   img_height,img_width).create()
    print (impulse_img.shape)
    print (pattern_name)