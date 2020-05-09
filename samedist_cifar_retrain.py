#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 10:40:17 2019

@author: qq
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import keras
from keras.datasets import mnist
from keras.datasets import cifar10
from keras import optimizers
import random
import numpy as np
from keras.models import load_model
from matplotlib import pyplot as plt
import csv
from keras.utils import np_utils
from keras.models import Model,Input,load_model
from sa import fetch_dsa, fetch_lsa, get_sc
import pandas as pd
import argparse
import condition



(x_train, y_train), (x_test, y_test) = cifar10.load_data()  
x_train = x_train.astype("float32")
x_train = x_train / 255.0 

def retrain(x_target,y_test,origin_acc,model,args,layer_names,selectsize=100,attack='fgsm',measure='lsa',datatype='mnist'): 
    #y_train= npzfile['y_train']   
    target_lst=[]


    if measure=='SRS':
        x_select,y_select = select_rondom(selectsize,x_target,x_target,y_test)
    if measure=='MCP':    
        x_select,y_select = select_my_optimize(model,selectsize,x_target,y_test)
    if measure=='LSA':
        target_lst = fetch_lsa(model, x_train, x_target, attack, layer_names, args)
    if measure=='DSA':
        target_lst = fetch_dsa(model, x_train, x_target, attack, layer_names, args)

    if measure=='AAL':
        path= "./cifar_finalResults/cifar_"+attack+"_compound8_result.csv"
        csv_data = pd.read_csv(path,header=None)
        target_lst =[]
        for i in range(len(csv_data.values.T)):
            target_lst.append(csv_data.values.T[i])
    if measure=='CES': 
        tmpfile="./conditional/"+attack+"_cifar_"+str(selectsize)+".npy"
        if os.path.exists(tmpfile):
            indexlst = list(np.load(tmpfile))
        else:
            indexlst = condition.conditional_sample(model,x_target,selectsize)
            np.save(tmpfile,np.array(indexlst))
        x_select,y_select = select_from_index(selectsize,x_target,indexlst,y_test)
    elif measure not in ['SRS','MCP']:
        x_select,y_select = select_from_large(selectsize, x_target, target_lst,y_test)
        
    y_select = np_utils.to_categorical(y_select, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    model.compile(loss='categorical_crossentropy', optimizer="adadelta", metrics=['accuracy'])
    
    retrain_acc=0

    model.fit(x_select, y_select, batch_size=100, epochs=5, shuffle=True,verbose=1, validation_data=(x_target, y_test))
    score = model.evaluate(x_target, y_test,verbose=0)
    retrain_acc=score[1]
    
    return retrain_acc


def find_second(act):
    max_=0
    second_max=0
    sec_index=0
    max_index=0
    for i in range(10):
        if act[i]>max_:
            max_=act[i]
            max_index=i
            
    for i in range(10):
        if i==max_index:
            continue
        if act[i]>second_max:
            second_max=act[i]
            sec_index=i
    ratio=1.0*second_max/max_
    #print 'max:',max_index
    return max_index,sec_index,ratio


def select_my_optimize(model,selectsize,x_target,y_test):
    
    x = np.zeros((selectsize,32,32,3))
    y = np.zeros((selectsize,1))
    
    act_layers=model.predict(x_target)
    dicratio=[[] for i in range(100)]
    dicindex=[[] for i in range(100)]
    for i in range(len(act_layers)):
        act=act_layers[i]
        max_index,sec_index,ratio =find_second(act)#max_index 
        dicratio[max_index*10+sec_index].append(ratio)
        dicindex[max_index*10+sec_index].append(i)
    
    selected_lst = select_from_firstsec_dic(selectsize,dicratio,dicindex)
    #selected_lst,lsa_lst = order_output(target_lsa,select_amount)
    for i in range(selectsize):
        x[i] = x_target[selected_lst[i]]
        y[i] = y_test[selected_lst[i]]
        
    return x,y


def select_from_firstsec_dic(selectsize,dicratio,dicindex):
    selected_lst=[]
    tmpsize=selectsize
    
    noempty=no_empty_number(dicratio)
    #print(selectsize)
    #print(noempty)
    while selectsize>=noempty:
        for i in range(100):
            if len(dicratio[i])!=0:
                tmp=max(dicratio[i])
                j = dicratio[i].index(tmp)
                if tmp>=0.1:
                    selected_lst.append(dicindex[i][j])
                dicratio[i].remove(tmp)
                dicindex[i].remove(dicindex[i][j])
        selectsize=tmpsize-len(selected_lst)
        noempty=no_empty_number(dicratio)

    
    while len(selected_lst)!= tmpsize:
        max_tmp=[0 for i in range(selectsize)]
        max_index_tmp=[0 for i in range(selectsize)]
        for i in range(100):
            if len(dicratio[i])!=0:
                tmp_max=max(dicratio[i])
                if tmp_max>min(max_tmp):
                    index=max_tmp.index(min(max_tmp))
                    max_tmp[index]=tmp_max
                    #selected_lst.append()
                    #if tmp_max>=0.1:
                    max_index_tmp[index]=dicindex[i][dicratio[i].index(tmp_max)]
        if len(max_index_tmp)==0 and len(selected_lst)!= tmpsize:
            print('wrong!!!!!!')  
            break
        selected_lst=selected_lst+ max_index_tmp
    #print(selected_lst)
    assert len(selected_lst)== tmpsize
    return selected_lst


def no_empty_number(dicratio):
    no_empty=0
    for i in range(len(dicratio)):
        if len(dicratio[i])!=0:
            no_empty+=1
    return no_empty

def select_from_large(select_amount,x_target,target_lsa,y_test):
    x = np.zeros((select_amount,32,32,3))
    y = np.zeros((select_amount,1))
    
    selected_lst,lsa_lst = order_output(target_lsa,select_amount)
    #print(lsa_lst)
    #print(selected_lst)
    for i in range(select_amount):
        x[i] = x_target[selected_lst[i]]
        y[i] = y_test[selected_lst[i]]
    return x,y


def select_rondom(select_amount,x_target,target_lsa,y_test):
    x = np.zeros((select_amount,32,32,3))
    y = np.zeros((select_amount,1))
    
    selected_lst = np.random.choice(range(len(target_lsa)),replace=False,size=select_amount)
    #selected_lst,lsa_lst = order_output(target_lsa,select_amount)
    for i in range(select_amount):
        x[i] = x_target[selected_lst[i]]
        y[i] = y_test[selected_lst[i]]
    return x,y


def select_from_index(select_amount,x_target,indexlst,y_test):
    x = np.zeros((select_amount,32,32,3))
    y = np.zeros((select_amount,1))
    #print(indexlst)
    for i in range(select_amount):
        x[i] = x_target[indexlst[i]]
        y[i] = y_test[indexlst[i]]
    return x,y
    



def find_index(target_lsa,selected_lst,max_lsa):
    for i in range(len(target_lsa)):
        if max_lsa==target_lsa[i] and i not in selected_lst:
            return i
    return 0

#重新修改    
def order_output(target_lsa,select_amount):
    lsa_lst=[]
    
    tmp_lsa_lst=target_lsa[:]
    selected_lst=[]
    while len(selected_lst) < select_amount:
        max_lsa=max(tmp_lsa_lst)
        selected_lst.append(find_index(target_lsa,selected_lst,max_lsa))
        lsa_lst.append(max_lsa)
        tmp_lsa_lst.remove(max_lsa)
    return selected_lst,lsa_lst
    
def createdataset(attack,ratio=8):
    if attack in ['rotation','translation','shear','brightness','contrast','scale']:
        x_target=np.load('./imagetrans/cifar_'+attack+'.npy') 
    else:
        x_target=np.load('./adv/data/cifar/Adv_cifar_'+attack+'.npy')     
    if attack in ['rotation','translation','shear','brightness','contrast','scale']:
        x_target = x_target.astype("float32")
        x_target = (x_target / 255.0)
    
    model_path='./model/densenet_cifar10.h5df'
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    x_test = x_test.astype("float32")
    x_test = x_test / 255.0
    
    origin_lst = np.random.choice(range(10000),replace=False,size=ratio*1000)
    mutated_lst = np.random.choice(range(10000),replace=False,size=10000-ratio*1000)
           
    x_dest = np.append(x_test[origin_lst],x_target[mutated_lst],axis=0)
    y_dest = np.append(y_test[origin_lst],y_test[mutated_lst])
    np.savez('./adv/data/cifar/cifar_'+attack+'_compound9.npz',x_test=x_dest,y_test=y_dest)
    
    y_dest = np_utils.to_categorical(y_dest, 10)
    model=load_model(model_path)
    score = model.evaluate(x_dest, y_dest,verbose=0)
    #print('Test Loss: %.4f' % score[0])
    print('Before retrain, Test accuracy: %.4f'% score[1])
    return
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", "-d", help="Dataset", type=str, default="cifar")
    parser.add_argument(
        "--target",
        "-target",
        help="Target input set (test or adversarial set)",
        type=str,
        default="fgsm",
    )
    parser.add_argument(
        "--save_path", "-save_path", help="Save path", type=str, default="./tmp/"
    )
    parser.add_argument(
        "--batch_size", "-batch_size", help="Batch size", type=int, default=128
    )
    parser.add_argument(
        "--var_threshold",
        "-var_threshold",
        help="Variance threshold",
        type=int,
        default=1e-5,
    )
    parser.add_argument(
        "--upper_bound", "-upper_bound", help="Upper bound", type=int, default=2000
    )
    parser.add_argument(
        "--n_bucket",
        "-n_bucket",
        help="The number of buckets for coverage",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--num_classes",
        "-num_classes",
        help="The number of classes",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--is_classification",
        "-is_classification",
        help="Is classification task",
        type=bool,
        default=True,
    )

    args = parser.parse_args()

    baselines =['LSA','DSA','CES','MCP','SRS','AAL']
    operators =['fgsm','jsma','bim-a','bim-b','cw-l2','scale','rotation','translation','shear','brightness','contrast']

    model_path='./model/densenet_cifar10.h5df'

    for operator in operators: 

        model=load_model(model_path)
        npzfile=np.load('./adv/data/cifar/cifar_'+operator+'_compound8.npz')    
        y_test= npzfile['y_test']
        x_test= npzfile['x_test']
  
        x_target =x_test
        y_cat = np_utils.to_categorical(y_test, 10)
                            
        score = model.evaluate(x_target, y_cat,verbose=0)
        origin_acc=score[1]
        print('Before retrain, Test accuracy: %.4f'% origin_acc)
        
        for measure in baselines:
            layer_names= ['activation_40']
            resultfilename = './result/cifar_compound_'+operator+'.txt'
            result_to_write='' 
            result_to_write+=measure+':\n'
            for selectsize in [100,300,500,1000]:             
                result_to_write+='['
                for i in range(5):
                    print(operator)
                    print(measure)
                    print(selectsize)
                    print(i)
                    model=load_model(model_path)
                    retrain_acc = retrain(x_target,y_test,origin_acc,model, args,layer_names,selectsize,operator,measure)
                    #print('%0.4f' % origin_acc, '%0.4f' % retrain_acc)
                    result_to_write+=str(round(retrain_acc,4))+('' if i==4 else ',')
                result_to_write+='],\n'
            result_to_write+='\n'+'original acc: '+str(round(origin_acc,4))+'\n'
            with open(resultfilename,'a') as file_object:
                file_object.write(result_to_write)  