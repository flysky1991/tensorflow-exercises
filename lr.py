# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 21:01:31 2017

@author: Administrator
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


#参数设置
learning_rate = 0.01   #学习率
training_epochs = 1000 #训练轮数
display_step = 50      #训练结果展示比例1:50

#生成训练数据
train_X = np.linspace(-1,1,200)
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.2

n_samples = train_X.shape[0]

#图输入
X = tf.placeholder("float")
Y = tf.placeholder("float")

#初始化变量w和b
W = tf.Variable(np.random.randn(),name="weight")
b = tf.Variable(np.random.randn(),name="bias")
 
#定义线性模型
pred = tf.add(tf.multiply(X,W),b)
 
#均方差(MSE)
cost = tf.reduce_sum(tf.pow(pred-Y,2)) / (2 * n_samples)
 
#创建优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
 
#初始化变量
#由tf.initialize_all_variables改为tf.global_variables_initializer
init = tf.global_variables_initializer()
 
#启动图
with tf.Session() as sess:
    sess.run(init)
     
    #适用于所有训练数据
    for epoch in range(training_epochs):
        for (x,y) in zip(train_X,train_Y):
            sess.run(optimizer,feed_dict={X:x,Y:y})
        #
        if (epoch+1) % display_step == 0:
            c = sess.run(cost,feed_dict={X:train_X,Y:train_Y})
            print("Epoch:",'%04d' % (epoch+1),"cost=","{:.9f}".format(c),\
                  "W=",sess.run(W),"b=",sess.run(b))
    print("Optimization Finished!")
    training_cost = sess.run(cost,feed_dict={X:train_X,Y:train_Y})
    print("Training cost=",training_cost,"W=",sess.run(W),sess.run(b),'\n')
    
    #图形展示
    plt.plot(train_X,train_Y,'ro',label='Original data')
    plt.plot(train_X,sess.run(W) * train_X + sess.run(b),label='Fitted line')
    plt.legend()
    plt.show()
    
    #生成测试数据
    test_X = np.linspace(-1,1,100)
    test_Y = 2 * test_X + np.random.randn(*test_X.shape) * 0.2
    
    print("Testing...(Mean square loss Comparsion)")
    testing_cost = sess.run(
            tf.reduce_sum(tf.pow(pred - Y,2)) / (2 * test_X.shape[0]),
            feed_dict = {X:test_X,Y:test_Y})
    print("Testing cost=",testing_cost)
    print("Absolute mean square loss difference:",abs(training_cost - testing_cost))
    
    plt.plot(test_X,test_Y,'bo',label='Testing data')
    plt.plot(train_X,sess.run(W) * train_X + sess.run(b),label='Fitted line')
    plt.legend()
    plt.show()
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
