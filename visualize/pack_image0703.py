# -*- coding:utf-8 -*-
import numpy as np
import os, scipy.io
import rawpy
import matplotlib.pyplot as plt
from scipy import signal

# input_image_dir = './dataset/Iphone/00003.dng'
input_image_dir = './dataset/Sony/short/00001_00_0.1s.ARW'
gt_image_dir = './dataset/Sony/long/00001_00_10s.ARW'

ratio = 100
input_images = []
color_channels = ["Red", "Green", "Blue","Green"]


def plot_img(array_img):
    h,w = array_img.shape

    out_img = array_img.copy()
    out_img = out_img.reshape((h,w))
    #subtract the black level
    out_img = np.maximum(out_img - 512, 0) / (16383 - 512)

    plt.imshow(out_img,cmap="gray")
    plt.show()


def plot_image(image_array):

    # create figure and axes
    fig, axes = plt.subplots(2,2)

    print ("channels: " + str(image_array.shape[2]))
    # for channel in range(image_array.shape[2]):
    for channel, ax in enumerate(axes.flat):
        print("channel: " + str(channel) )
        ax.imshow(image_array[0, :, :, channel])
        # ax.imshow(image_array[:, :, :])

        # remove any labels from the axes
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()



def pack_raw(green_raw):
    # pack Bayer image to 4 channels
    # im = raw.raw_image_visible.astype(np.float32)
    # print("im: " + str(im.shape))

    # plot_original_image(im)

    # subtract the black level
    # im = np.maximum(im - 512, 0) / (16383 - 512)
    im = np.expand_dims(green_raw, axis=2)
    img_shape = im.shape
    print ("img_shape: " + str(im.shape))
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out

def mirror(x, min, max):
    if x < min:
        return min - x
    elif x >= max:
        return 2 * max - x - 2
    else:
        return x

def moveG1ToR1G2ToB1(pack_img):

    h,w = pack_img.shape
    moveG1ToR1G2ToB1 = np.zeros((h,w))
    print (str(h) +" " +  str(w) +" " )

    for y in range(0,h,2):
        for x in range(0, w, 2):
            # 第一层:Red:
            # green_img[y+0, x+0] = pack_img[y+0, x+0]
            moveG1ToR1G2ToB1[y+0, x+0] = 0.0
            # moveG1ToR1G2ToB1[y + 0, x + 0] =  pack_img[y + 0, x + 1]
            # 第二层:Green
            # green_img[y + 0, x + 1] = pack_img[y + 0, x + 1]
            moveG1ToR1G2ToB1[y + 0, x + 1] = 0.0
            # 第三层:Green
            # green_img[y + 1, x + 0] = pack_img[y + 1, x + 0]
            moveG1ToR1G2ToB1[y + 1, x + 0] = 0.0
            # 第四层:Blue
            moveG1ToR1G2ToB1[y + 1, x + 1] = pack_img[y + 1, x + 0]
            # moveG1ToR1G2ToB1[y + 1, x + 1] = pack_img[y + 1, x + 1]
            # moveG1ToR1G2ToB1[y + 1, x + 1] = 0.0

    return moveG1ToR1G2ToB1

def interpolation_green_img(pack_img):
    h,w = pack_img.shape
    interpolation_img_green = np.zeros((h,w))

    for y in range(0,h,2):
        for x in range(0,w,2):
            # 第一层:Red:
            # green_img[y+0, x+0] = pack_img[y+0, x+0]
            interpolation_img_green[y + 0, x + 0] = 0.0
            # 第二层:Green
            interpolation_img_green[y + 0, x + 1] = pack_img[y + 0, x + 1]
            # interpolation_img[y + 0, x + 1] = 0.0
            # 第三层:Green
            interpolation_img_green[y + 1, x + 0] = pack_img[y + 1, x + 0]
            # interpolation_img[y + 1, x + 0] = 0.0
            # 第四层:Blue
            # interpolation_img[y + 1, x + 1] = pack_img[y + 1, x + 1]
            interpolation_img_green[y + 1, x + 1] = 0.0

    # print (interpolation_img_green[0:h:2 , 0:w:2])
    # print (interpolation_img_green[0:h:2 , 1:w:2])

    for y in range(0, h, 2):
        for x in range(0, w, 2):
            if interpolation_img_green[y,x] == 0:
                y0 = mirror(y - 1, 0, h)
                y1 = mirror(y + 1, 0, h)
                x0 = mirror(x - 1, 0, w)
                x1 = mirror(x + 1, 0, w)

                interpolation_img_green[y,x] = (interpolation_img_green[y0,x] + interpolation_img_green[y,x0] +
                                                interpolation_img_green[y,x1] + interpolation_img_green[y1,x]) / 4


    for y in range(1, h, 2):
        for x in range(1, w, 2):
            if interpolation_img_green[y,x] == 0:
                y0 = mirror(y - 1, 0, h)
                y1 = mirror(y + 1, 0, h)
                x0 = mirror(x - 1, 0, w)
                x1 = mirror(x + 1, 0, w)

                interpolation_img_green[y,x] = (interpolation_img_green[y0,x0] + interpolation_img_green[y0,x1] +
                                                interpolation_img_green[y1,x0] + interpolation_img_green[y1,x1]) / 4

    green_filter = np.array([[0, 1/4, 0], [1/4, 1, 1/4], [0, 1/4, 0]])
    green = signal.convolve2d(interpolation_img_green, green_filter, boundary='symm', mode='same')
    # print (green[0:h:2 , 0:w:2])
    # print (green[0:h:2 , 1:w:2])


    return green

def get_green_pixel(pack_img):

    h,w = pack_img.shape
    green_img = np.zeros((h,w))
    print (str(h) +" " +  str(w) +" " )

    # green_img[0:h:2, 0:w:2, :] = 0
    # # green_img[0:h:2, 1:w:2, :] = pack_img[0:h:2, 1:w:2, :]
    # green_img[1:h:2, 1:w:2, :] = 0
    # # green_img[1:h:2, 0:w:2, :] = pack_img[1:h:2, 0:w:2, :]
    #
    # green_img = np.concatenate((green_img[0:h:2, 0:w:2, :] , pack_img[0:h:2, 1:w:2, :],
    #                             green_img[1:h:2, 1:w:2, :],pack_img[1:h:2, 0:w:2, :]), axis=2)

    for y in range(0,h,2):
        for x in range(0, w, 2):
            # 第一层:Red:
            # green_img[y+0, x+0] = pack_img[y+0, x+0]
            green_img[y+0, x+0] = 0.0
            # 第二层:Green
            # green_img[y + 0, x + 1] = pack_img[y + 0, x + 1]
            green_img[y + 0, x + 1] = 0.0
            # 第三层:Green
            # green_img[y + 1, x + 0] = pack_img[y + 1, x + 0]
            # green_img[y + 1, x + 0] = pack_img[y + 1, x + 0]
            green_img[y + 1, x + 0] = 0.0
            # 第四层:Blue
            green_img[y + 1, x + 1] = pack_img[y + 1, x + 1]
            # green_img[y + 1, x + 1] = 0.0


    return green_img


def raw_to_numpy(raw):

    # raw_array = np.array(raw_img).reshape((h, w)).astype("float")
    raw_array = raw.raw_image_visible.astype(np.float32)
    # subtract_balck_level(raw,raw_array)
    # subtract the black level
    raw_array = np.maximum(raw_array - 512, 0) / (16383 - 512)

    return raw_array

def loadImage(input_image_dir):
    #获取图片名
    in_fn = os.path.basename(input_image_dir)
    print ("in_fn: {0}" .format(in_fn))

    #RAW读入图像
    raw = rawpy.imread(input_image_dir)

    #将RAW图像转化为numpy的array数据
    numpy_img = raw_to_numpy(raw)
    print ("numpy_img: " + str(numpy_img.shape))

    #将原始raw图像转化为想要的raw图像
    # green_images_raw = get_green_pixel(numpy_img)
    # green_images_raw = interpolation_green_img(numpy_img)
    G1ToR1G2ToB1 = moveG1ToR1G2ToB1(numpy_img)
    print ("green_images_raw: " + str(G1ToR1G2ToB1.shape))

    input_images_raw = pack_raw(G1ToR1G2ToB1)
    print ("input_images_raw: " + str(input_images_raw.shape))

    # scale the data by the amplication ratio
    input_images = np.expand_dims(input_images_raw, axis=0) * ratio

    # #crop
    H = input_images.shape[1]
    W = input_images.shape[2]
    print ("input_images: " + str(input_images.shape))

    input_full = np.minimum(input_images, 1.0)
    # print ("input_full: " + str(input_full.shape))

    plot_image(input_full)

    # plot_img(green_images_raw)
    # plot_img(input_images)
    return  input_full

if __name__ == '__main__':
    loadImage(input_image_dir)
