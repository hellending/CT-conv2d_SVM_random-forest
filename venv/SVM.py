import sklearn.svm as svm
import numpy as np
from sklearn.model_selection import train_test_split
import grey_picture as rd
import tensorflow as tf

def creat_label(length,classfication_value,one_hot_value):
    #创建一个适当大小的矩阵来接收
    array=np.arange(length*classfication_value).reshape(length,classfication_value)
    for i in range(0,length):
        array[i]=one_hot_value #这里采用one hot值来区别合格与不合格
    return array
 #获取数据集
covid = rd.creat_x_database('.\\grey_covid',128,128)
non_covid = rd.creat_x_database('.\\grey_non',128,128)
dataSet = np.vstack((covid,non_covid))
#设定标签
covid_label = creat_label(covid.shape[0],1,1)
non_covid_label = creat_label(non_covid.shape[0],1,0)
label = np.vstack((covid_label,non_covid_label))

clf = svm.SVC(gamma='auto',kernel='poly',C=0.8,random_state=2,probability=True,degree=2)
# 获取最终数据集
x_train,x_test,y_train,y_test = train_test_split(dataSet,label,test_size=0.2)
x_train = np.array(x_train,dtype=float).reshape(-1,128*128)
x_test = np.array(x_test,dtype=float).reshape(-1,128*128)
y_train = np.array(y_train,dtype=float)
y_test = np.array(y_test,dtype=float)
clf.fit(x_train,y_train)
# for i in range(100):
# x_train, x_test, y_train, y_test = train_test_split(dataSet, label, test_size=0.2, random_state=0)
# x_train = np.array(x_train, dtype=float).reshape(-1, 128 * 128)
# x_test = np.array(x_test, dtype=float).reshape(-1, 128 * 128)
# y_train = np.array(y_train, dtype=float)
# y_test = np.array(y_test, dtype=float)
# clf.fit(x_train, y_train)
pre = clf.predict(x_train)
accuracy = 0
for k in range(0, len(pre)):
    if (pre[k] == y_train[k]):
        accuracy += 1
accuracy /= len(pre)
print(accuracy)
accuracy = 0
pre = clf.predict(x_test)
for j in range(0,len(pre)):
    if(pre[j]==y_test[j]):
     accuracy+=1
accuracy/=len(pre)
print(accuracy)