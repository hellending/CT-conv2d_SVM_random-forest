import tensorflow.compat.v1 as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import grey_picture as rd
import os,sys
from sklearn.model_selection import train_test_split
import time
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
tf.disable_eager_execution()
def creat_label(length,classfication_value,one_hot_value):
    #创建一个适当大小的矩阵来接收
    array=np.arange(length*classfication_value).reshape(length,classfication_value)
    for i in range(0,length):
        array[i]=one_hot_value #这里采用one hot值来区别合格与不合格
    return array

x = tf.placeholder(tf.float32,[None,128,128,1])/255
y = tf.placeholder(tf.float32,[None,2])

#四个参数：长，宽，单个过滤器深度，过滤器个数
weights = {'wc1': tf.Variable(tf.random_normal([5,5,1,64],stddev=0.05)),
           'wc2': tf.Variable(tf.random_normal([5,5,64,128],stddev=0.05)),
           'wc3': tf.Variable(tf.random_normal([5,5,128,512],stddev=0.05)),
           'wd1': tf.Variable(tf.random_normal([16*16*512,1024],stddev=0.05)),
           'wd2': tf.Variable(tf.random_normal([1024,2],stddev=0.05))
           }
#b值(特征值),偏移量
biases = {'bc1': tf.Variable(tf.random_normal([64],stddev=0.05)),
          'bc2': tf.Variable(tf.random_normal([128],stddev=0.05)),
          'bc3': tf.Variable(tf.random_normal([512],stddev=0.05)),
          'bd1': tf.Variable(tf.random_normal([1024],stddev=0.05)),
          'bd2': tf.Variable(tf.random_normal([2],stddev=0.05))
          }

def Forward_conv(input,weights,biases,keepratio):
    #输入的批量数据处理
    input_r = tf.reshape(input,shape=[-1,128,128,1])
    conv_1 = tf.nn.conv2d(input=input_r, filter=weights['wc1'], strides=[1, 1, 1, 1], padding='SAME')
    conv_1 = tf.nn.relu(tf.add(conv_1, biases['bc1']))
    pool_1 = tf.nn.max_pool(value=conv_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # 将池化层合理缩减，去掉一部分神经节点，防止过拟合，这里意思是将pool1层保留百分比为keepratio的节点
    _pool1_drl = tf.nn.dropout(pool_1, keepratio)
    conv_2 = tf.nn.conv2d(input=_pool1_drl, filter=weights['wc2'], strides=[1, 1, 1, 1], padding='SAME')
    conv_2 = tf.nn.relu(tf.add(conv_2, biases['bc2']))
    pool_2 = tf.nn.max_pool(value=conv_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    _pool2_drl = tf.nn.dropout(pool_2, keepratio)

    conv_3 = tf.nn.conv2d(input=_pool2_drl, filter=weights['wc3'], strides=[1, 1, 1, 1], padding='SAME')
    conv_3 = tf.nn.relu(tf.add(conv_3, biases['bc3']))
    pool_3 = tf.nn.max_pool(value=conv_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    _pool3_drl = tf.nn.dropout(pool_3, keepratio)

    densel = tf.reshape(_pool3_drl, [-1, weights['wd1'].get_shape().as_list()[0]])
    fcl = tf.nn.sigmoid(tf.add(tf.matmul(densel, weights['wd1']), biases['bd1']))
    fcl_drl = tf.nn.dropout(fcl, keepratio)
    out = tf.add(tf.matmul(fcl_drl, weights['wd2']), biases['bd2'])
    return out
#获取数据集
covid = rd.creat_x_database('.\\grey_covid',128,128)
non_covid = rd.creat_x_database('.\\grey_non',128,128)
dataSet = np.vstack((covid,non_covid))
#设定标签
covid_label = creat_label(covid.shape[0],2,[0,1])
non_covid_label = creat_label(non_covid.shape[0],2,[1,0])
label = np.vstack((covid_label,non_covid_label))
#获取最终数据集
# x_train,x_test,y_train,y_test = train_test_split(dataSet,label,test_size=0.1,random_state=0,shuffle=True)

pre = Forward_conv(x,weights,biases,0.8)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = pre,labels = y))
# cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pre+ 1e-10), reduction_indices=[1]))

optimizer = tf.train.AdamOptimizer(0.00001).minimize(cost)
p = tf.equal(tf.argmax(y,1),tf.argmax(pre,1))
accuracy = tf.reduce_mean(tf.cast(p,tf.float32))

###########################################################################
sess = tf.Session()
sess.run(tf.global_variables_initializer())
avg_cost = 0
for j in range(0,1000):
   x_train, x_test, y_train, y_test = train_test_split(dataSet, label, test_size=0.2, random_state=0,shuffle=True)
   print(j)
   avg_cost = 0
   for i in range(0,3):
        k = i*179
        x_train1 = [x_train[m] for m in range(k,k+179)]
        y_train1 = [y_train[m] for m in range(k,k+179)]
        sess.run(optimizer, feed_dict={x: x_train1, y: y_train1})
        avg_cost += sess.run(cost,feed_dict={x: x_train1, y: y_train1})/3
        # avg_cost += tf.reduce_mean(sess.run(cost, feed_dict={x: x_train1, y: y_train1}))
        # print(avg_cost)
        # training_acc = sess.run(accuracy, feed_dict={x: x_train, y: y_train})
        # print('训练数据精度:', training_acc)
        test_acc = sess.run(accuracy, feed_dict={x: x_test, y: y_test})
        print('测试数据精度:', test_acc)
   print('损失值',avg_cost)