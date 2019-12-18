#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import shutil

def remove_file(src_path, dst_path):
    """ 将src_path路径下的文件移动到dst_path路径下
    :param src_path:
    :param dst_path:
    :return:
    """
    #读取src_path路径下的文件名dict
    list_layer_filenames = os.listdir(src_path)
    print (list_layer_filenames)

    #通过循环移动每个文件名下的文件
    #并重命名保存在新地址dst_path中
    for i in range(len(list_layer_filenames)):
        layer_file_path = os.path.join(src_path, list_layer_filenames[i])
        list_filter_names = os.listdir(layer_file_path)
        flag_layer = list_layer_filenames[i].split("_")[-1]
        print (flag_layer)
        # print (list_filter_names)

        #获取“list_layer_filenames[i]”中最后_后的字符
        #获取“list_filter_names[j]”中.png前的字符
        for j in range(len(list_filter_names)):
            flag_string = list_filter_names[j].split(".")[0]  #新的文件夹名
            f_dst = os.path.join(dst_path,flag_string)
            if not os.path.exists(f_dst):
                os.mkdir(f_dst)
            new_name = flag_string+"_"+flag_layer+".png"
            dst = os.path.join(f_dst,new_name)
            src = os.path.join(layer_file_path,list_filter_names[j])
            print ("dst:{}".format(dst))
            print ("src:{}".format(src))
            shutil.copy(src,dst)

if __name__ == '__main__':

    src_path = "/home/zhu/PycharmProjects/Learning-to-See-in-the-Dark/slim2_nnConv" \
               "/decompose_results/g_conv1_2/center_square_23602/" #移动前文件路径
    dst_path = "/home/zhu/PycharmProjects/Learning-to-See-in-the-Dark/slim2_nnConv" \
               "/decompose_results/g_conv1_2/center_square_23602_remove/" #移动后文件路径
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
    remove_file(src_path, dst_path)