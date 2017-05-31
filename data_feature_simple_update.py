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

#from calculate_like_cross import like_crossproduct

feature_num = 4
feature_sum_num = 10
n_input = 100
n_class = 1

n_hidden_1 = 4
n_hidden_2 = 2
n_hidden_3 = 1

learning_rate = 0.01
batch_size = n_input
display_step = 1
training_epochs = n_input 

Alpha = 0.85
keep_p = 0.6

num = 1
#iteration times
itr = 200 


def like_crossproduct(matrix, vector):
    print "Enter-> Calculate"
    vector = tf.expand_dims(vector, 0) 
#    tensor_shape = matrix.get_shape()
#    matrix_shape_0 = tensor_shape[0].value
#    matrix_shape_1 = tensor_shape[1].value
    matrix_shape_0 = n_input
    matrix_shape_1 = feature_sum_num
    new_matrix = tf.zeros([matrix_shape_0, matrix_shape_1])
    
    #Slice the Matrix
    for matrix_row in range(matrix_shape_0):
        
        matrix_slice = tf.slice(matrix, [matrix_row, 0], [1, matrix_shape_1])
        temporary_matrix = tf.concat([vector, matrix_slice], 0)
        
        print "Enter-> Slice"           
        
        #Calculate the #value                       
        for matrix_rank_1 in range(matrix_shape_1):
            
            delta_shape = [matrix_shape_0, matrix_shape_1]
            delta_indice = [[matrix_row, matrix_rank_1]]
            
            result = tf.constant(0.)
            add_or_sub_count = 1
            add_not_sub = 1
            sub_not_add = -1
            print "Enter-> Calculate Vector"
            #Loop to calculate
            for matrix_rank_2 in range(matrix_shape_1 - 1):
                #if it is the First
                if matrix_rank_1 == 0:
                    
                    
                    if matrix_rank_2 == 0:
                        continue
                    fir = temporary_matrix[0][matrix_rank_2] * temporary_matrix[1][matrix_rank_2 + 1]
                    
                    sec = temporary_matrix[0][matrix_rank_2 + 1] * temporary_matrix[1][matrix_rank_2]

                    res = fir - sec
                    
                    if add_or_sub_count % 2 == 0:
                        res = res * sub_not_add
                        add_or_sub_count = add_or_sub_count + 1
                    
                    else:
                        res = res * add_not_sub
                        add_or_sub_count = add_or_sub_count + 1
                    
                    result = result + res
                
                #if it is the last    
                elif matrix_rank_1 == matrix_shape_1 - 1:
                    
                    
                    if matrix_rank_2 == matrix_shape_1 - 1 - 1:
                        continue
                    fir = temporary_matrix[0][matrix_rank_2] * temporary_matrix[1][matrix_rank_2 + 1]
                    
                    sec = temporary_matrix[0][matrix_rank_2 + 1] * temporary_matrix[1][matrix_rank_2]

                    res = fir - sec
                    
                    if add_or_sub_count % 2 == 0:
                        res = res * sub_not_add
                        add_or_sub_count = add_or_sub_count + 1
                    
                    else:
                        res = res * add_not_sub
                        add_or_sub_count = add_or_sub_count + 1
                    
                    result = result + res
                #if it is in the middle
                else:
                    
                    
                    if matrix_rank_2 == matrix_rank_1:
                        continue
                    
                    if matrix_rank_2 + 1 == matrix_rank_1:
                        
                        fir = temporary_matrix[0][matrix_rank_2] * temporary_matrix[1][matrix_rank_2 + 1 + 1]
                    
                        sec = temporary_matrix[0][matrix_rank_2 + 1 + 1] * temporary_matrix[1][matrix_rank_2]

                        res = fir - sec
                    
                        if add_or_sub_count % 2 == 0:
                            res = res * sub_not_add
                            add_or_sub_count = add_or_sub_count + 1
                    
                        else:
                            res = res * add_not_sub
                            add_or_sub_count = add_or_sub_count + 1
                    
                        result = result + res
                    
                    else:
                        fir = temporary_matrix[0][matrix_rank_2] * temporary_matrix[1][matrix_rank_2 + 1]
                    
                        sec = temporary_matrix[0][matrix_rank_2 + 1] * temporary_matrix[1][matrix_rank_2]
                        
                        res = fir - sec
                    
                        if add_or_sub_count % 2 == 0:
                            res = res * sub_not_add
                            add_or_sub_count = add_or_sub_count + 1
                    
                        else:
                            res = res * add_not_sub
                            add_or_sub_count = add_or_sub_count + 1
                    
                        result = result + res
                
                        
            delta_value = tf.expand_dims(result, 0)             
                     
            delta = tf.SparseTensor(delta_indice, delta_value, delta_shape)
            
            new_matrix = new_matrix + tf.sparse_tensor_to_dense(delta)
            
            
        
    print "like_cross Finish!"
    return  new_matrix 

def subsqr(pred, y):
    sub = tf.abs(tf.subtract(pred, y))
    sub_sqr = tf.square(sub)   
    return sub_sqr  

#train data
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

#plot matrix
train_loss = []
train_acc = []

judge_0 = tf.zeros([feature_sum_num], tf.float32) 

x = tf.placeholder(tf.float32, [None, feature_sum_num])
y = tf.placeholder(tf.float32, [n_class, None])

weight_lasso = tf.random_normal([feature_sum_num], 0, 0.3)
bias_lasso = tf.Variable(tf.random_normal([feature_sum_num], 0, 0.1))
print "Begin"
output_lasso_no_bias = like_crossproduct(x, weight_lasso)
output_lasso = output_lasso_no_bias + bias_lasso
output_lasso = tf.cast(output_lasso, tf.float32)

layer_0 = tf.nn.relu(output_lasso)

        
        
#hidden_1
weight_1 = tf.cast(tf.Variable(tf.random_normal([feature_sum_num, n_hidden_1], 0, 0.3)), tf.float32)
bias_1 = tf.Variable(tf.random_normal([n_hidden_1], 0, 0.05))
layer_1 = tf.add(tf.matmul(layer_0, weight_1), bias_1)
layer_1 = tf.nn.relu(layer_1)
#hidden_2
weight_2 = tf.cast(tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], 0, 0.3)), tf.float32)
bias_2 = tf.Variable(tf.random_normal([n_hidden_2], 0, 0.05))
layer_2 = tf.add(tf.matmul(layer_1, weight_2), bias_2)
layer_2 = tf.nn.relu(layer_2)
#hidden_3
weight_3 = tf.cast(tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3], 0, 0.3)), tf.float32)
bias_3 = tf.Variable(tf.random_normal([n_hidden_3], 0, 0.05))
layer_3 = tf.add(tf.matmul(layer_2, weight_3), bias_3)
layer_3 = tf.nn.relu(layer_3)
#output_layer
weight_out = tf.cast(tf.Variable(tf.random_normal([n_hidden_3, n_class], 0, 0.3)), tf.float32)
bias_out = tf.Variable(tf.random_normal([n_class], 0, 0.05))
out_layer = tf.add(tf.matmul(layer_3, weight_out), bias_out)
pred = tf.nn.relu(out_layer)

pred = tf.reshape(pred, [1, batch_size])

#weight_lasso SUM
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
            
               
                
                print("step: %d  loss: %.9f  TRAIN_InaCCURACY: %.3f"  % (step, los, inacc))
                weightlasso_tensor = sess.run(weight_lasso, feed_dict = feed)
#                print "weight_lasso", weightlasso_tensor
                
                if num % 10 ==0:
                    
                    plt.subplot(211)
                    plt.plot(weightlasso_tensor, 'b')
                    plt.xlabel("epochs")
                    plt.ylabel("weight_lasso")
                    plt.grid(True)
            
            
            step = step + 1
        
        print "Finish #", num
        num = num + 1
        step = 1
    
    print "*******************************************************"
    
    weight_lasso_tensor = sess.run(weight_lasso, feed_dict = feed)
    print "1.weight_lasso:\n", weight_lasso_tensor
#    diff_tensor = tf.setdiff1d(weight_lasso_tensor, judge_0, name=None)
#    diff, diff_idx = sess.run(diff_tensor, feed_dict = feed)
#    print "2.Selected feature(weight_lasso):\n", diff_idx

 
    print "2.Real feature(s):\n", idx
          
    rank_sum_tensor = tf.cast(tf.reduce_sum(layer_0, 0), tf.float32)
    rank_sum = sess.run(rank_sum_tensor, feed_dict = feed)
    diff_tensor_ = tf.setdiff1d(rank_sum_tensor, judge_0, name=None)  
    diff_, diff_idx_ = sess.run(diff_tensor_, feed_dict = feed)
    print "3.From layer_0, guess the unselected feature(s):\n", diff_idx_
    
    print "*******************************************************"
    
plt.subplot(212)
plt.plot(train_loss, 'r')
plt.xlabel("epochs")
plt.ylabel("Training loss")
plt.grid(True)

plt.show()

