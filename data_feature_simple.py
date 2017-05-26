#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 23:57:34 2017

@author: wuzhenglin
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy
import random

feature_num = 5
feature_sum_num = 100
n_input = 5000
n_class = 1

n_hidden_1 = 10
n_hidden_2 = 1

learning_rate = 0.001
batch_size = 1000
display_step = 1
training_epochs = n_input 

Alpha = 0.6
keep_p = 0.5

num = 1
itr = 50#train data itr times

x_train = np.random.random([n_input, feature_sum_num])

a = []
for i in range(0, feature_sum_num):
    a.append(i)
    i = i + 1

idx = random.sample(a, feature_num)

idx = sorted(idx)

w = np.random.random(size = feature_num) * 10

y_train = scipy.dot(x_train[:, idx], w) / feature_num
y_train = y_train.reshape(1, -1)


train_loss = []
train_acc = []

judge_0 = tf.zeros([n_input], tf.float32) 

x = tf.placeholder(tf.float32, [None, feature_sum_num])
y = tf.placeholder(tf.float32, [n_class, None])

weight_lasso = tf.random_normal([feature_sum_num], 0, 0.3)
bias_lasso = tf.Variable(tf.random_normal([feature_sum_num], 0, 0.05))
output_lasso = x * weight_lasso + bias_lasso
output_lasso = tf.cast(output_lasso, tf.float32)

layer_0 = tf.nn.relu(output_lasso)

def subsqr(pred, y):
    sub = tf.abs(tf.subtract(pred, y))
    sub_sqr = tf.square(sub)   
    return sub_sqr          
        

weight_1 = tf.cast(tf.Variable(tf.random_normal([feature_sum_num, n_hidden_1], 0, 0.3)), tf.float32)
bias_1 = tf.Variable(tf.random_normal([n_hidden_1], 0, 0.05))
layer_1 = tf.add(tf.matmul(layer_0, weight_1), bias_1)
layer_1 = tf.nn.relu(layer_1)

weight_2 = tf.cast(tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], 0, 0.3)), tf.float32)
bias_2 = tf.Variable(tf.random_normal([n_hidden_2], 0, 0.05))
layer_2 = tf.add(tf.matmul(layer_1, weight_2), bias_2)
layer_2 = tf.nn.relu(layer_2)

weight_out = tf.cast(tf.Variable(tf.random_normal([n_hidden_2, n_class], 0, 0.3)), tf.float32)
bias_out = tf.Variable(tf.random_normal([n_class], 0, 0.05))
out_layer = tf.add(tf.matmul(layer_2, weight_out), bias_out)
pred = tf.nn.relu(out_layer)

pred = tf.reshape(pred, [1, batch_size])

weight_lasso_sum = tf.reduce_sum(weight_lasso,[0, 0])
weight_lasso_sum_abs = tf.abs(weight_lasso_sum)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(y, pred)), [0, 1]) + tf.multiply(Alpha, weight_lasso_sum_abs))  
optimize = tf.train.AdamOptimizer(learning_rate).minimize(loss)

inaccuracy = subsqr(pred, y)
in_accuracy = tf.reduce_mean(tf.cast(inaccuracy, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    step = 1
    
    while num < itr + 1:
        
        while step * batch_size - 1 < training_epochs:
        
            x_train_n = x_train[(step - 1)*batch_size: step * batch_size, :]
                  
            y_train_n = y_train[:, (step - 1)*batch_size : step * batch_size]
            y_train_np = y_train_n.reshape(1, -1)
        
            feed = {x: x_train_n, y: y_train_np}
        
            sess.run(optimize, feed_dict = feed)
        
            if step % display_step == 0:
            
                los, inacc = sess.run([loss, in_accuracy], feed_dict = feed)
            
                train_acc.append(inacc)
                train_loss.append(los)
            
               
            
#                print("step: %d  loss: %.9f  TRAIN_InaCCURACY: %.3f"  % (step, los, inacc))
                weight_1_tensor = sess.run(weight_1, feed_dict = feed)
                print weight_1_tensor
            
            
            step = step + 1
        
        print "完成第", num
        num = num + 1
        step = 1
    
    print "*******************************************************"
    weight_lasso_tensor = sess.run(weight_lasso, feed_dict = feed)
    print "1.weight_lasso:\n", weight_lasso_tensor
    
    diff_tensor = tf.setdiff1d(weight_lasso_tensor, judge_0, name=None)
    diff, diff_idx = sess.run(diff_tensor, feed_dict = feed)
    print "2.Selected feature:\n", diff_idx
    
    print "3.Real feature:\n", idx
    print "*******************************************************"
    
    
    

#plt.subplot(211)
plt.plot(train_loss, 'r')
#plt.title('lr=%f, te=%d, bs=%d, acc=%f' % (learning_rate, training_epochs, batch_size, acc))
plt.xlabel("epochs")
plt.ylabel("Training loss")
plt.grid(True)

#plt.subplot(212)
#plt.plot(train_acc, 'r')
#plt.xlabel("epochs")
#plt.ylabel('Training Inaccuracy')
##plt.ylim(0.0, 1)
#plt.grid(True)

plt.show()

