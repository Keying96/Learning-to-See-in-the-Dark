#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
import os, scipy.io
import sys
import numpy as np
import tensorflow.contrib.slim as slim
import divide_into_block
import matplotlib.pyplot as plt
import time
import SID

ratio = 30
input_path = '../dataset/short/00001_00_0.1s.ARW'
checkpoint_dir = '../checkpoint/Sony/'
result_dir = './result_divide/'

result_list = []
vstack_list = []
hstack_list = []

def reslut_stack(result_list,flag_block):
    for i in range(flag_block):
        flag = i * 8
        opreate_list = result_list[flag:flag+(flag_block)]
        print ("flag:{} flag+8:{}".format(flag, flag+(flag_block)))
        print (len(opreate_list))
        vstak = opreate_list[0]
        for j in range(len(opreate_list)-1):
            vstak = np.hstack((vstak, opreate_list[j+1]))
        vstack_list.append(vstak)

    hstack = vstack_list[0]
    for i in range(len(vstack_list)-1):
        hstack = np.vstack((hstack, vstack_list[i+1]))
    # vstak1 = np.hstack((result_list[0], result_list[1],result_list[2], result_list[3]))
    # vstak2 = np.hstack((result_list[4], result_list[5],result_list[6], result_list[7]))
    # vstak3 = np.hstack((result_list[8], result_list[9],result_list[10], result_list[11]))
    # vstak4 = np.hstack((result_list[12], result_list[13],result_list[14], result_list[15]))
    # result = np.vstack((vstak1,vstak2,vstak3,vstak4))
    return hstack


sess = tf.Session()
in_image = tf.placeholder(tf.float32, [None, None, None, 4])
out_image = SID.network(in_image)

saver = tf.train.Saver()
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt:
    print('loaded ' + ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)

#读入图像
flag_block = 8
input_full = divide_into_block.laod_input(input_path, flag_block)
start_time = time.time()
for i in range(len(input_full)):
    input = np.expand_dims(input_full[i],axis=0)
    input = np.minimum(input, 1.0)
    # in_image = input

    output = sess.run(out_image, feed_dict={in_image: input})
    # output = sess.run(out_image)
    output = np.minimum(np.maximum(output, 0), 1)
    output = output[0, :, :, :]
    result_list.append(output)



end_time = time.time()
print ("the runtime is {}".format(end_time-start_time)) #the runtime is 5.81841087341


print (len(result_list))
result_image = reslut_stack(result_list,flag_block)
plt.axis("off")
plt.imshow(result_image)
savename = os.path.join(result_dir, 'result_image64.png')
if not os.path.isdir(result_dir):
    os.makedirs(result_dir)
plt.savefig(savename, dpi=600)
print (savename)
plt.close()
