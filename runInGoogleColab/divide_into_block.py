#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import rawpy

input_path = "../dataset/short/00001_00_0.1s.ARW"
ratio = 30

divide_list = []
#将AWR图像转化为我RGBG图像
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

#对图像数组进行分割
def divide_into_block(input_full, flag_block):
    vsplit = np.vsplit(input_full, flag_block)
    for i in range(len(vsplit)):
        hsplit = np.hsplit(vsplit[i], flag_block)
        for j in range(len(hsplit)):
            divide_list.append(hsplit[j])
    return divide_list

def laod_input(input_path, flag_block):
    # 通过图像路径读取图像
    raw = rawpy.imread(input_path)
    input_full = pack_raw(raw) * ratio
    print (input_full.shape)

    divide_list = divide_into_block(input_full, flag_block)
    print (len(divide_list))
    # input_div = np.expand_dims(input_divide[0],axis=0)

    # print (input_full[700, 700,:])
    # print (input_divide[0][700,700,:])
    return  divide_list

if __name__ == '__main__':
    flag_block = 4
    input_div = laod_input(input_path, flag_block)
    for i in range(len(input_div)):
        print (input_div[i])