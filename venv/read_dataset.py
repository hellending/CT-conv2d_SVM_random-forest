import os
import numpy as np
import tensorflow.compat.v1 as tf
from PIL import Image as Img
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
def creat_x_database(rootdir,resize_row,resize_col):
    #列出文件夹下所有的，目录和文件
    list = os.listdir(rootdir)
    #创建一个随机矩阵，作为多个图片转换为矩阵后传入其中
    database=np.arange(len(list)*resize_row*resize_col*3).reshape(len(list)
    ,resize_row,resize_col,3)
    for i in range(0,len(list)):
        path = os.path.join(rootdir,list[i])    #把目录和文件名合成一个路径
        if os.path.isfile(path):                ##判断路径是否为文件
            #一次性代码
            # img = Img.open(path).convert('RGB')
            # img.save(path)
            image_raw_data = tf.io.gfile.GFile(path,'rb').read()#读取图片
            with tf.Session() as sess:
                img_data = tf.image.decode_jpeg(image_raw_data)#图片解码
                #压缩图片矩阵为指定大小
                resized=tf.image.resize_images(img_data,[resize_row,resize_col],method=0)
                database[i]=resized.eval()
    return database