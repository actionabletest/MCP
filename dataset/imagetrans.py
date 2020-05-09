#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 15:53:43 2020

@author: qq
"""
import cv2
from keras.models import load_model
from imageio import imread, imsave
import numpy as np
import matplotlib.pyplot as plt

def image_translation(img, params):
    if not isinstance(params, list):
        params = [params, params]
    if len(img.shape)==2:
        rows, cols = img.shape
    else:
        rows, cols,ch = img.shape

    M = np.float32([[1, 0, params[0]], [0, 1, params[1]]])
    new_img = cv2.warpAffine(img, M, (cols, rows))
    return new_img

def image_scale(img, params):
    if not isinstance(params, list):
        params = [params, params]
    new_img = cv2.resize(img, None, fx=params[0], fy=params[1], interpolation=cv2.INTER_CUBIC)
    if len(img.shape)==2:#灰度图
        rows, cols = img.shape
        tmp_img = np.zeros((rows,cols))
        rows, cols = new_img.shape
        for i in range(rows):
            for j in range(cols):
                tmp_img[i][j]=new_img[i][j]
    else:      
        rows, cols,ch = img.shape
        tmp_img = np.zeros((rows,cols,ch))
        rows, cols,ch = new_img.shape
        for i in range(rows):
            for j in range(cols):
                for k in range(ch):
                    tmp_img[i][j][k]=new_img[i][j][k]
    
    return tmp_img

def image_shear(img, params):
    if len(img.shape)==2:
        rows, cols = img.shape
    else:
        rows, cols,ch = img.shape
    factor = params*(-1.0)
    M = np.float32([[1, factor, 0], [0, 1, 0]])
    new_img = cv2.warpAffine(img, M, (cols, rows))
    return new_img

def image_rotation(img, params):
    if len(img.shape)==2:
        rows, cols = img.shape
    else:
        rows, cols,ch = img.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), params, 1)
    new_img = cv2.warpAffine(img, M, (cols, rows))
    return new_img

def image_contrast(img, params):
    alpha = params
    new_img = cv2.multiply(img, np.array([alpha]))
    return new_img

def image_brightness(img, params):
    beta = params
    new_img = cv2.add(img, beta)
    return new_img
 
def saveimage(image,i,tran='shear',cmap='gray'):
    imagename="./bim-bfig/bim-b"+tran+str(i)+".jpg"
    if cmap=='gray':
        plt.imsave(imagename,image,vmin=0, vmax=255,format="jpg",cmap='gray')
    else:
        plt.imsave(imagename,image,vmin=0, vmax=255,format="jpg")
    return

def saveorigin_image(image,i,cmap='gray'):
    imagename="./fig/origin"+str(i)+".jpg"
    if cmap=='gray':
        plt.imsave(imagename,image,vmin=0, vmax=255,format="jpg",cmap='gray')
    else:
        plt.imsave(imagename,image,vmin=0, vmax=255,format="jpg")
    return

import scipy.io as sio
def getData(filename):
    load_data = sio.loadmat(filename)
    y = load_data['y']
    X = load_data['X'].transpose(3, 0, 1, 2)
    return X, y

if __name__ == "__main__":
    data ='mnist'
            
    if data=='test': 
      x_save =np.load('./adv/data/mnist/Adv_mnist_bim-b.npy')
      
      (x_train, y_train), (x_test, y_test) = mnist.load_data()
      x_save=x_save+0.5
      x_save=x_save*255.0
      x_save=x_save.astype('uint8')
      #y_test = np_utils.to_categorical(y_test, 10)
      for i in range(len(x_save)):
          saveimage(np.resize(x_save[i],(28,28)),i,'',cmap='gray')
    if data=='svhn':
        file = 'adv/data/svhn/svhn_test.mat'
        x_test, y_test = getData(filename=file)
        #x_test = x_test.astype('float32')
        #x_test =x_test/255.0
        y_test[y_test==10] = 0

        model_file="./model/model-svhn.h5df"
        model = load_model(model_file)        
        
        #rotation集，0.7222. 旋转30度
        #translation,先旋转30度，然后往右下平移3个像素：0.6390
        #shear1只拉伸0.4，精度.6304；shear2先旋转30度，然后拉伸0.4，精度0.7457
        #svhn_brightness2，先旋转30度，再增加亮度50：0.7288
        #contrast，先旋转30度，在增加对比度1.5：0.6899
        #scale，先旋转30度，再拉伸0.9：0.6969
        
        x_save = x_test.copy()
        #x_test = x_test.astype("float32")
        #x_test = x_test / 255.0
        for i in range(len(x_test)):
            #newimg = image_rotation(x_test[i], 30)
            #newimg = image_rotation(newimg, 10)
            #newimg= image_translation(newimg,3)
            newimg = image_shear(x_test[i],0.4)
            #newimg = image_rotation(newimg, 10)
            #newimg = image_brightness(newimg, 50)
            #newimg = image_contrast(x_test[i], 1.5)
            #newimg = image_scale(x_test[i],0.9)
            x_save[i]=newimg
            if i <20:
                #saveimage(newimg,i,'scale',cmap='COLOR')
                saveimage(newimg/255,i,'shear',cmap='COLOR')
                saveorigin_image(x_test[i],i)
        np.save("imagetrans/svhn_shear.npy",x_save)  
        #x_save =np.load("imagetrans/svhn_contrast1.npy")
        x_save = x_save.astype("float32")
        x_save = x_save / 255.0
        
        y_test = np_utils.to_categorical(y_test, 10)
        score = model.evaluate(x_save, y_test,verbose=0)
        print(score[1])
        
        
        
    if data=='mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_save = x_test.copy()
    
        for i in range(len(x_test)):
            #newimg = image_rotation(x_test[i], 15)
            newimg= image_translation(x_test[i],3)
            #newimg = image_shear(x_test[i],0.4)
            #newimg = image_rotation(newimg, 10)
            #newimg = image_brightness(newimg, 100)
            #newimg = image_contrast(newimg, 1.5)
            #newimg = image_scale(x_test[i],0.8)
            x_save[i]=newimg
            if i <0:
                saveimage(newimg,i,'color')
                saveorigin_image(x_test[i],i,'color')
      
        #np.save("imagetrans/mnist_brightness.npy",x_save)
        #x_save=np.load("imagetrans/mnist_brightness.npy") 
        np.save("imagetrans/mnist_translation.npy",x_save)
        model_path='./model/model_mnist.h5df'
        model=load_model(model_path)
        x_save = x_save.astype("float32").reshape(-1,28,28,1)
        x_save = (x_save / 255.0) - (1.0 - CLIP_MAX)
        
        y_test = np_utils.to_categorical(y_test, 10)
        
        score = model.evaluate(x_save, y_test,verbose=0)
        #print('Test Loss: %.4f' % score[0])
        print('After mutation, Test accuracy: %.4f'% score[1])
        
    if data=='cifar':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        #shear数据集，系数0.4，精度0.7080
        #contrast数据集，先旋转15度，再增加1.2倍的对比度：0.7545
        #translation数据集，先选择15度，后往右下2：0.7020
        #brightness数据集，增加100亮度，0.7576
        #rotation数据集15_7366
        #scale数据集，先旋转15度，再缩0.9：0.7424
        
        x_save = x_test.copy()
        #x_test = x_test.astype("float32")
        #x_test = x_test / 255.0
        for i in range(len(x_test)):
            newimg = image_rotation(x_test[i], 15)
            #newimg = image_rotation(newimg, 10)
            #newimg= image_translation(newimg,2)
            #newimg = image_shear(x_test[i],0.4)
            #newimg = image_rotation(newimg, 10)
            #newimg = image_brightness(x_test[i], 100)
            #newimg = image_contrast(newimg, 1.5)
            newimg = image_scale(newimg,0.9)
            x_save[i]=newimg
            if i <20:
                #saveimage(newimg,i,'scale',cmap='COLOR')
                saveimage(newimg/255.0,i,'scale',cmap='COLOR')
                saveorigin_image(x_test[i],i)
        np.save("imagetrans/cifar_scale.npy",x_save)      
        model_path='./model/densenet_cifar10.h5df'
        model=load_model(model_path)
        x_save = x_save.astype("float32")
        x_save = x_save / 255.0
        
        y_test = np_utils.to_categorical(y_test, 10)
        
        score = model.evaluate(x_save, y_test,verbose=0)
        #print('Test Loss: %.4f' % score[0])
        print('After mutation, Test accuracy: %.4f'% score[1])
