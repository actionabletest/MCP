#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 16:42:34 2018

@author: qq
"""

import keras
from keras.models import load_model
from keras.models import model_from_json
import h5py  #导入工具包  
import numpy as np  
from keras.datasets import mnist
from keras.datasets import cifar10
from keras.models import Model,Input,load_model

'''
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32').reshape(-1,28,28,1)
x_test = x_test.astype('float32').reshape(-1,28,28,1)

x_train = x_train / 255
x_test = x_test / 255

#y_test[2]=1
#返回不通过的测试用例list
def find_notpass(model,image=x_test,test_label=y_test):
    pred=model.predict(image)
    pred=list(map(lambda x:np.argmax(x),pred))
    notpasslist=[]
    for i in range(len(image)):
        if pred[i]!=test_label[i]:
            notpasslist.append(i)
    #print 'notpass:',len(notpasslist)
    return notpasslist
'''
#之前的问题是，当更新max时，把此时的max扔了，这个时候应该置为second
def find_second(act):
    max_=0
    second_max=0
    index=0
    max_index=0
    for i in range(10):
        if act[i]>max_:
            max_=act[i]
            max_index=i
            
    for i in range(10):
        if i==max_index:
            continue
        if act[i]>second_max:#第2大加一个限制条件，那就是不能和max_一样
            second_max=act[i]
            index=i
    ratio=1.0*second_max/max_
    #print 'max:',max_index
    return index,ratio#ratio是第二大输出达到最大输出的百分比


#最大/第二大，比值越大，说明离边界越远。越接近越好
#我们排序优先挑比值小的，靠近边界的
def order_output(act_layers):
    order_lst=[]
    ratio_lst=[]
    for i in range(len(act_layers)):
        act=act_layers[i]
        __,ratio = find_second(act)      
        ratio_lst.append(ratio)
    
    tmp_ratio_lst=ratio_lst[:]
    order_ratio_lst=[]
    while len(tmp_ratio_lst) > 0:
        min_ratio=min(tmp_ratio_lst)
        order_lst.append(ratio_lst.index(min_ratio))
        order_ratio_lst.append(min_ratio)
        tmp_ratio_lst.remove(min_ratio)
    return order_lst,order_ratio_lst



#不需要pass
def get_bound_data_mnist(model,x_test,y_test,bound_ratio=10):
    x_tmp = x_test.astype('float32').reshape(-1,28,28,1)
    x_tmp = x_tmp / 255
    bound_data_lst =[]
    out_index=len(model.layers)-1
    model_layer=Model(inputs=model.input,outputs=model.layers[out_index].output)
  
    act_layers=model_layer.predict(x_tmp)
    #notpasslist=find_notpass(model)
    #print 'act_layers:',len(act_layers)
    for i in range(len(act_layers)):#此i只是choice_index序化后
        act=act_layers[i]
        index,ratio = find_second(act)     
        if ratio< bound_ratio :
            bound_data_lst.append(i) 
                #print index,y_test[i]   
    x_bound = np.zeros((len(bound_data_lst),28,28))
    y_bound = np.zeros((len(bound_data_lst),))

    for i in range(len(bound_data_lst)):
        x_bound[i] = x_test[bound_data_lst[i]]
        y_bound[i] = y_test[bound_data_lst[i]]
    return x_bound,y_bound

#pass的非边界值
def get_unbound_data_mnist(model,bound_ratio=10):
    unbound_data_lst =[]
    out_index=len(model.layers)-1
    model_layer=Model(inputs=model.input,outputs=model.layers[out_index].output)
  
    act_layers=model_layer.predict_on_batch(x_test)
    notpasslist=find_notpass(model)
    for i in range(len(act_layers)):#此i只是choice_index序化后
        act=act_layers[i]
        index,ratio = find_second(act)
        if ratio >= bound_ratio :
            if i not in notpasslist:
                unbound_data_lst.append(i)   
    return unbound_data_lst


#变异模型在边界值集合上的准确率
def accuracy_in_bound_data_mnist(mutated_model,bound_data_lst):
    test_part=x_test[bound_data_lst]#部分的测试用例
    label_part = y_test[bound_data_lst]
    pred=mutated_model.predict(test_part)
    pred=list(map(lambda x:np.argmax(x),pred))
    #notpasslist=find_notpass(model)
    acc=0
    for i in range(len(bound_data_lst)):
        if pred[i]==label_part[i]:
            acc=acc+1
    acc = 1.0*acc/len(bound_data_lst)
    return acc

#变异模型在非边界值集合上的准确率
def accuracy_in_unbound_data_mnist(mutated_model,unbound_data_lst):
    return accuracy_in_bound_data_mnist(mutated_model,unbound_data_lst)




#不仅要在边界，而且要pass
def get_bound_data_cifar(model,x_test,y_test,bound_ratio=10):
    bound_data_lst =[]
    out_index=len(model.layers)-1
    model_layer=Model(inputs=model.input,outputs=model.layers[out_index].output)
    cifar_X_test=x_test.astype('float32')
    cifar_X_test/=255
    act_layers=model_layer.predict_on_batch(cifar_X_test)
    for i in range(len(act_layers)):#此i只是choice_index序化后
        act=act_layers[i]
        index,ratio = find_second(act)
        if ratio< bound_ratio :
            bound_data_lst.append(i)   
            
            
    x_bound = np.zeros((len(bound_data_lst),32,32,3))
    y_bound = np.zeros((len(bound_data_lst),1))

    for i in range(len(bound_data_lst)):
        x_bound[i] = x_test[bound_data_lst[i]]
        y_bound[i] = y_test[bound_data_lst[i]]
    return x_bound,y_bound

#pass的非边界值
def get_unbound_data_cifar(model,bound_ratio=10):
    unbound_data_lst =[]
    out_index=len(model.layers)-1
    model_layer=Model(inputs=model.input,outputs=model.layers[out_index].output)
  
    act_layers=model_layer.predict_on_batch(cifar_X_test)
    notpasslist=find_notpass(model,image=cifar_X_test,test_label=cifar_Y_test)
    for i in range(len(act_layers)):#此i只是choice_index序化后
        act=act_layers[i]
        index,ratio = find_second(act)
        if ratio>= bound_ratio :
            if i not in notpasslist:
                unbound_data_lst.append(i)   
    return unbound_data_lst


#变异模型在边界值集合上的准确率
def accuracy_in_bound_data_cifar(mutated_model,bound_data_lst):
    test_part=cifar_X_test[bound_data_lst]#部分的测试用例
    label_part = cifar_Y_test[bound_data_lst]
    pred=mutated_model.predict(test_part)
    pred=list(map(lambda x:np.argmax(x),pred))
    #notpasslist=find_notpass(model)
    acc=0
    for i in range(len(bound_data_lst)):
        if pred[i]==label_part[i]:
            acc=acc+1
    acc = 1.0*acc/len(bound_data_lst)
    return acc


#变异模型在非边界值集合上的准确率
def accuracy_in_unbound_data_cifar(mutated_model,unbound_data_lst):
    return accuracy_in_bound_data_cifar(mutated_model,unbound_data_lst)
