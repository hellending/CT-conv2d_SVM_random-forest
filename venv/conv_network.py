#两个卷积+池化，两个全连接层，一个输出层
import tensorflow.compat.v1 as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
tf.disable_eager_execution()
minist = input_data.read_data_sets('../../../Desktop/data/', one_hot=True)
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])
#k值(卷积核)
#四个参数：长，宽，单个过滤器深度，过滤器个数
weights = {'wc1': tf.Variable(tf.random_normal([3,3,1,64],stddev=0.1)),
           'wc2': tf.Variable(tf.random_normal([3,3,64,128],stddev=0.1)),
           'wd1': tf.Variable(tf.random_normal([7*7*128,1024],stddev=0.1)),
           'wd2': tf.Variable(tf.random_normal([1024,10],stddev=0.1))
           }
#b值(特征值),偏移量
biases = {'bc1': tf.Variable(tf.zeros([64])),
          'bc2': tf.Variable(tf.zeros([128])),
          'bd1': tf.Variable(tf.zeros([1024])),
          'bd2': tf.Variable(tf.zeros([10]))
          }
#前向传播算法(务必把池化函数和卷积函数的四个参数分别搞懂，具体结构和计算过程与普通的神经网络差距不大)
def Forward_conv(input,weights,biases,keepratio):
  #batch_size,长,宽,深度（正常图片为3，灰度图为1）
  input_r = tf.reshape(input,shape=[-1,28,28,1])
  #第一层卷积层
    #strides batch_size,长,宽,深度的步长，池化层大小为2*2每次移动两格，不重叠，将数据量大小减半
  conv_1 = tf.nn.conv2d(input=input_r,filter=weights['wc1'],strides=[1,1,1,1],padding='SAME')
  #输出层之前的真实层的输出总是经过激活函数修饰的
  conv_1 = tf.nn.relu(tf.add(conv_1,biases['bc1']))
  pool_1 = tf.nn.max_pool(value=conv_1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
  #将池化层合理缩减，去掉一部分神经节点，防止过拟合，这里意思是将pool1层保留百分比为keepratio的节点
  _pool1_drl = tf.nn.dropout(pool_1,keepratio)
  conv_2 = tf.nn.conv2d(input=_pool1_drl,filter=weights['wc2'],strides=[1,1,1,1],padding='SAME')
  conv_2 = tf.nn.relu(tf.add(conv_2,biases['bc2']))
  pool_2 = tf.nn.max_pool(value=conv_2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
  _pool2_drl = tf.nn.dropout(pool_2,keepratio)
  densel = tf.reshape(_pool2_drl,[-1,weights['wd1'].get_shape().as_list()[0]])
  fcl = tf.nn.relu(tf.add(tf.matmul(densel,weights['wd1']),biases['bd1']))
  fcl_drl = tf.nn.dropout(fcl,keepratio)
  #总算输出了
  out = tf.add(tf.matmul(fcl_drl,weights['wd2']),biases['bd2'])
  return out

#反向传播算法（我这里是直接照抄普通神经网络的（softmax交叉熵+梯度下降算法优化））
pre = Forward_conv(x,weights,biases,0.8)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=pre))
optimizer = tf.train.AdamOptimizer(0.0001).minimize(cost)
p = tf.equal(tf.argmax(y,1),tf.argmax(pre,1))
accuracy = tf.reduce_mean(tf.cast(p,tf.float32))

#####################################################################
sess = tf.Session()
sess.run(tf.global_variables_initializer())
training_epoches = 15
batch_size = 16
display_step = 1

for epoch in range(training_epoches):
    avg_cost = 0
    num_batch = int(minist.train.num_examples/batch_size)
    for i in range(num_batch):
        x_train,y_train = minist.train.next_batch(batch_size)
        sess.run(optimizer,feed_dict={x: x_train,y: y_train})
        avg_cost += sess.run(cost,feed_dict={x: x_train,y: y_train})/batch_size

    # if epoch%display_step==0:
        training_acc = sess.run(accuracy,feed_dict={x: x_train,y: y_train})
        test_acc = sess.run(accuracy,feed_dict={x: minist.test.images,y: minist.test.labels})
        print('训练数据精度:',training_acc,'测试数据精度:',test_acc)
    print('损失值:',avg_cost)