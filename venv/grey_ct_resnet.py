import tensorflow.compat.v1 as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import grey_picture as rd
import os,sys
from sklearn.model_selection import train_test_split
import time
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
tf.disable_eager_execution()
def weight_variable(shape):
    # 正态分布，标准差为0.1，默认最大为1，最小为-1，均值为
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def creat_label(length,classfication_value,one_hot_value):
    #创建一个适当大小的矩阵来接收
    array=np.arange(length*classfication_value).reshape(length,classfication_value)
    for i in range(0,length):
        array[i]=one_hot_value #这里采用one hot值来区别合格与不合格
    return array


def identity_block(X_input, kernel_size, in_filter, out_filters, stage, block, training):
    """
    Implementation of the identity block as defined in Figure 3

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    kernel_size -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    training -- train or test

    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    block_name = 'res' + str(stage) + block
    f1, f2, f3 = out_filters
    with tf.variable_scope(block_name):
        X_shortcut = X_input

        # first
        W_conv1 =  weight_variable([1, 1, in_filter, f1])
        X = tf.nn.conv2d(X_input, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
        X = tf.layers.batch_normalization(X, axis=3, training=training)
        X = tf.nn.relu(X)

        # second
        W_conv2 =  weight_variable([kernel_size, kernel_size, f1, f2])
        X = tf.nn.conv2d(X, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
        X = tf.layers.batch_normalization(X, axis=3, training=training)
        X = tf.nn.relu(X)

        # third

        W_conv3 =  weight_variable([1, 1, f2, f3])
        X = tf.nn.conv2d(X, W_conv3, strides=[1, 1, 1, 1], padding='VALID')
        X = tf.layers.batch_normalization(X, axis=3, training=training)

        # final step
        add = tf.add(X, X_shortcut)
        add_result = tf.nn.relu(add)

    return add_result

def convolutional_block( X_input, kernel_size, in_filter,
                            out_filters, stage, block, training, stride=2):
        """
        Implementation of the convolutional block as defined in Figure 4

        Arguments:
        X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        kernel_size -- integer, specifying the shape of the middle CONV's window for the main path
        filters -- python list of integers, defining the number of filters in the CONV layers of the main path
        stage -- integer, used to name the layers, depending on their position in the network
        block -- string/character, used to name the layers, depending on their position in the network
        training -- train or test
        stride -- Integer, specifying the stride to be used

        Returns:
        X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
        """

        # defining name basis
        block_name = 'res' + str(stage) + block
        with tf.variable_scope(block_name):
            f1, f2, f3 = out_filters

            x_shortcut = X_input
            #first
            W_conv1 =  weight_variable([1, 1, in_filter, f1])
            X = tf.nn.conv2d(X_input, W_conv1,strides=[1, stride, stride, 1],padding='VALID')
            X = tf.layers.batch_normalization(X, axis=3, training=training)
            X = tf.nn.relu(X)

            #second
            W_conv2 =  weight_variable([kernel_size, kernel_size, f1, f2])
            X = tf.nn.conv2d(X, W_conv2, strides=[1,1,1,1], padding='SAME')
            X = tf.layers.batch_normalization(X, axis=3, training=training)
            X = tf.nn.relu(X)

            #third
            W_conv3 =  weight_variable([1,1, f2,f3])
            X = tf.nn.conv2d(X, W_conv3, strides=[1, 1, 1,1], padding='VALID')
            X = tf.layers.batch_normalization(X, axis=3, training=training)

            #shortcut path
            W_shortcut =  weight_variable([1, 1, in_filter, f3])
            x_shortcut = tf.nn.conv2d(x_shortcut, W_shortcut, strides=[1, stride, stride, 1], padding='VALID')

            #final
            add = tf.add(x_shortcut, X)
            add_result = tf.nn.relu(add)

        return add_result


x = tf.placeholder(tf.float32,[None,128,128,1])/255
y = tf.placeholder(tf.float32,[None,2])
training = tf.placeholder(tf.bool, name='training')

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

# def Forward_conv(input,weights,biases,keepratio):
#     #输入的批量数据处理
#     input_r = tf.reshape(input,shape=[-1,128,128,1])
#     conv_1 = tf.nn.conv2d(input=input_r, filter=weights['wc1'], strides=[1, 1, 1, 1], padding='SAME')
#     conv_1 = tf.layers.batch_normalization(conv_1,axis=3,training=True)
#     conv_1 = tf.nn.relu(tf.add(conv_1, biases['bc1']))
#     pool_1 = tf.nn.max_pool(value=conv_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#     # 将池化层合理缩减，去掉一部分神经节点，防止过拟合，这里意思是将pool1层保留百分比为keepratio的节点
#     _pool1_drl = tf.nn.dropout(pool_1, keepratio)
#     conv_2 = tf.nn.conv2d(input=_pool1_drl, filter=weights['wc2'], strides=[1, 1, 1, 1], padding='SAME')
#     conv_2 = tf.layers.batch_normalization(conv_2, axis=3, training=True)
#     conv_2 = tf.nn.relu(tf.add(conv_2, biases['bc2']))
#     pool_2 = tf.nn.max_pool(value=conv_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#     _pool2_drl = tf.nn.dropout(pool_2, keepratio)
#
#     conv_3 = tf.nn.conv2d(input=_pool2_drl, filter=weights['wc3'], strides=[1, 1, 1, 1], padding='SAME')
#     conv_3 = tf.layers.batch_normalization(conv_3, axis=3, training=True)
#     conv_3 = tf.nn.relu(tf.add(conv_3, biases['bc3']))
#     pool_3 = tf.nn.max_pool(value=conv_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#     _pool3_drl = tf.nn.dropout(pool_3, keepratio)
#
#     densel = tf.reshape(_pool3_drl, [-1, weights['wd1'].get_shape().as_list()[0]])
#     fcl = tf.nn.sigmoid(tf.add(tf.matmul(densel, weights['wd1']), biases['bd1']))
#     fcl_drl = tf.nn.dropout(fcl, keepratio)
#     out = tf.add(tf.matmul(fcl_drl, weights['wd2']), biases['bd2'])
#     return out
#获取数据集

def deepnn(x_input):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:

    Returns:
    """
    x = tf.reshape(x_input,shape=[-1,128,128,1])
    with tf.variable_scope('reference'):
        # stage 1
        # 第一个kernel为[7,7,3,64]
        w_conv1 = weight_variable([5, 5, 1, 64])
        # 步距为2
        x = tf.nn.conv2d(x, w_conv1, strides=[1, 1, 1, 1], padding='SAME')
        x = tf.layers.batch_normalization(x, axis=3, training=training)
        x = tf.nn.relu(x)
        x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME')
        # assert (x.get_shape() == (x.get_shape()[0], 64, 64, 64))

        # stage 2
        x = convolutional_block(x, 3, 64, [64, 64, 256], 2, 'a', training, stride=1)
        x = identity_block(x, 3, 256, [64, 64, 256], stage=2, block='b', training=training)
        x = identity_block(x, 3, 256, [64, 64, 256], stage=2, block='c', training=training)

        # stage 3
        #一个conv_block块+三个id_block块
        x = convolutional_block(x, 3, 256, [128, 128, 512], 3, 'a', training)
        x = identity_block(x, 3, 512, [128, 128, 512], 3, 'b', training=training)
        x = identity_block(x, 3, 512, [128, 128, 512], 3, 'c', training=training)
        x = identity_block(x, 3, 512, [128, 128, 512], 3, 'd', training=training)

        # stage 4
        x = convolutional_block(x, 3, 512, [256, 256, 1024], 4, 'a', training)
        x = identity_block(x, 3, 1024, [256, 256, 1024], 4, 'b', training=training)
        x = identity_block(x, 3, 1024, [256, 256, 1024], 4, 'c', training=training)
        x = identity_block(x, 3, 1024, [256, 256, 1024], 4, 'd', training=training)
        x = identity_block(x, 3, 1024, [256, 256, 1024], 4, 'e', training=training)
        x = identity_block(x, 3, 1024, [256, 256, 1024], 4, 'f', training=training)

        # stage 5
        x = convolutional_block(x, 3, 1024, [512, 512, 2048], 5, 'a', training)
        x = identity_block(x, 3, 2048, [512, 512, 2048], 5, 'b', training=training)
        x = identity_block(x, 3, 2048, [512, 512, 2048], 5, 'c', training=training)
        #平均池化层
        x = tf.nn.avg_pool(x, [1, 2, 2, 1], strides=[1, 1, 1, 1], padding='VALID')

        flatten = tf.layers.flatten(x)
        x = tf.layers.dense(flatten, units=50, activation=tf.nn.relu)
        # Dropout - controls the complexity of the model, prevents co-adaptation of
        # features.
        with tf.name_scope('dropout'):
            keep_prob = 0.8
            x = tf.nn.dropout(x, keep_prob)

        logits = tf.layers.dense(x, units=2, activation=tf.nn.softmax)
        return logits

covid = rd.creat_x_database('.\\grey_covid_power',128,128)
non_covid = rd.creat_x_database('.\\grey_non_power',128,128)
dataSet = np.vstack((covid,non_covid))
#设定标签
covid_label = creat_label(covid.shape[0],2,[0,1])
non_covid_label = creat_label(non_covid.shape[0],2,[1,0])
label = np.vstack((covid_label,non_covid_label))
#获取最终数据集
# x_train,x_test,y_train,y_test = train_test_split(dataSet,label,test_size=0.1,random_state=0,shuffle=True)

pre = deepnn(x)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = pre,labels = y))
# cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pre+ 1e-10), reduction_indices=[1]))

optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
p = tf.equal(tf.argmax(y,1),tf.argmax(pre,1))
accuracy = tf.reduce_mean(tf.cast(p,tf.float32))

###########################################################################
sess = tf.Session()
sess.run(tf.global_variables_initializer())
x_train,x_test,y_train,y_test = train_test_split(dataSet,label,test_size=0.1,random_state=0)
avg_cost = 0
for j in range(0,1000):
   state = np.random.get_state()
   np.random.shuffle(x_train)
   np.random.set_state(state)
   np.random.shuffle(y_train)
   print(j)
   avg_cost = 0
   for i in range(0,3):
        k = i*179
        x_train1 = [x_train[m] for m in range(k,k+179)]
        y_train1 = [y_train[m] for m in range(k,k+179)]
        sess.run(optimizer, feed_dict={x: x_train1, y: y_train1, training:True})
        avg_cost += sess.run(cost,feed_dict={x: x_train1, y: y_train1, training: False})/3
        # avg_cost += tf.reduce_mean(sess.run(cost, feed_dict={x: x_train1, y: y_train1}))
        # print(avg_cost)
        # training_acc = sess.run(accuracy, feed_dict={x: x_train, y: y_train})
        # print('训练数据精度:', training_acc)
        test_acc = sess.run(accuracy, feed_dict={x: x_test, y: y_test, training: False})
        train_acc = sess.run(accuracy, feed_dict={x: x_train1, y: y_train1, training: False})
        print('测试数据精度:', test_acc)
        print('训练数据精度:', train_acc)
   print('损失值',avg_cost)