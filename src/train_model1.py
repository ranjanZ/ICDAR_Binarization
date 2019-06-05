import os,sys
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from keras.engine.topology import Layer
import numpy as np
from keras.layers import initializers,constraints
import tensorflow as tf
from keras.models import Sequential,Model
from keras.utils import conv_utils
from keras import backend as K
import numpy as np
from keras.layers import Dense, Dropout, Flatten,Input
from keras.layers import Conv2D, MaxPooling2D,Concatenate
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from keras_contrib.losses import DSSIMObjective
from keras.constraints import *
from keras.layers import Activation, Dense,Dropout
#import pandas
from utils import *
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import ModelCheckpoint
from morph_layers2D import *
from matplotlib.colors import ListedColormap
import keras

from generator import *
from keras.layers import *
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import disk,rectangle
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, UpSampling2D, Add, Dropout, Concatenate
import time
import numpy as np 
import cv2 as cv
from skimage.measure import compare_ssim







def cust_obj(y_true, y_pred): 

    loss1=DSSIMObjective(kernel_size=100)(y_true,y_pred)
    loss2=K.mean(K.abs(y_true-y_pred))
    loss=loss1+loss2
    return loss 


def DSSIM_RGB(y_true, y_pred):

    #loss1=DSSIMObjective(kernel_size=23)(y_true[:,:,:,:1],y_pred[:,:,:,:1])
    #loss2=DSSIMObjective(kernel_size=23)(y_true[:,:,:,1:2],y_pred[:,:,:,1:2])
    #loss3=DSSIMObjective(kernel_size=23)(y_true[:,:,:,2:3],y_pred[:,:,:,2:3])
    #loss=K.mean(loss1+loss2+loss3)
    #loss=(loss1+loss2+loss3)/3.0
    loss=DSSIMObjective(kernel_size=23)(y_true[:,:,:,:3],y_pred[:,:,:,:3])
    return loss



def model_DIBCO1(inp_shape=(256,256,3)):
    I=Input(inp_shape)
    #I=Input(shape=(None,None,1))

    z1=Dilation2D(2, (8,8),padding="same",strides=(1,1))(I)
    z2=Erosion2D(2, (8,8),padding="same",strides=(1,1))(I)
    z3=Concatenate()([z1,z2])
    z3=Conv2D(8,(1,1),padding='same')(z3)
    
    for j in range(1):
        z1=Dilation2D(2, (8,8),padding="same",strides=(1,1))(z3)
        z2=Erosion2D(2, (8,8),padding="same",strides=(1,1))(z3)
        z3=Concatenate()([z1,z2])
        z3=Conv2D(8,(1,1),padding='same')(z3)
      
    z1=Dilation2D(2, (8,8),padding="same",strides=(1,1))(z3)
    z2=Erosion2D(2, (8,8),padding="same",strides=(1,1))(z3)
    z3=Concatenate()([z1,z2])
    z3=Conv2D(1,(1,1),padding='same',activation='sigmoid')(z3)

    model=Model(inputs=[I],outputs=[z3])
    #model.compile(loss=DSSIMObjective(kernel_size=100), optimizer="adam",metrics=['mse',DSSIMObjective(kernel_size=100)])
    model.compile(loss=cust_obj, optimizer="adam",metrics=['mse',DSSIMObjective(kernel_size=100)])

    return model

def model_DIBCO(inp_shape=(256,256,3)):
    I=Input(inp_shape)
    #I=Input(shape=(None,None,1))

    z1=Dilation2D(4, (8,8),padding="same",strides=(1,1))(I)
    z2=Erosion2D(4, (8,8),padding="same",strides=(1,1))(I)
    z3=Concatenate()([z1,z2])
    z3=Conv2D(8,(1,1),padding='same',activation='linear')(z3)

    for j in range(2):
        z1=Dilation2D(3, (8,8),padding="same",strides=(1,1))(z3)
        z2=Erosion2D(3, (8,8),padding="same",strides=(1,1))(z3)
        z3=Concatenate()([z1,z2])
        z3=Conv2D(8,(1,1),padding='same',activation='linear')(z3)

    """
    z4=z3
    for j in range(4):
        z1=Dilation2D(2, (4,4),padding="valid",strides=(1,1))(z4)
        z2=Erosion2D(2, (4,4),padding="valid",strides=(1,1))(z4)
        z4=Concatenate()([z1,z2])
        z4=Conv2D(4,(1,1),strides=(2,2),padding='valid',activation='tanh')(z4)
        #z4=MaxPooling2D((2,2))(z4)
    z4=Flatten()(z4)
    z4=morph_layer(z4,10)
    z4=Dense(3,activation="sigmoid")(z4)
    """

    z1=Dilation2D(2, (8,8),padding="same",strides=(1,1))(z3)
    z2=Erosion2D(2, (8,8),padding="same",strides=(1,1))(z3)
    z3=Concatenate()([z1,z2])
    z3=Conv2D(1,(1,1),padding='same',activation='sigmoid')(z3)

    #z5=AGGLayer_DIBCO()([z3,z4,I])
    # model=Model(inputs=[I],outputs=[z5])
    model=Model(inputs=[I],outputs=[z3])
    #model.compile(loss=DSSIMObjective(kernel_size=24), optimizer="adam",metrics=['mse',DSSIMObjective(kernel_size=100)])
    model.compile(loss="mse", optimizer="adam",metrics=['mse',DSSIMObjective(kernel_size=100)])
    #model.compile(loss=cust_obj, optimizer="adam",metrics=['mse',DSSIMObjective(kernel_size=100),cust_obj,])

    return model
















def model_ISI(inp_shape=(256,256,3)):
    I=Input(inp_shape)
    #I=Input(shape=(None,None,1))

    z1=Dilation2D(2, (8,8),padding="same",strides=(1,1))(I)
    z2=Erosion2D(2, (8,8),padding="same",strides=(1,1))(I)
    z3=Concatenate()([z1,z2])
    z3=Conv2D(8,(1,1),padding='same',activation='linear')(z3)
    
    for j in range(1):
        z1=Dilation2D(2, (8,8),padding="same",strides=(1,1))(z3)
        z2=Erosion2D(2, (8,8),padding="same",strides=(1,1))(z3)
        z3=Concatenate()([z1,z2])
        z3=Conv2D(8,(1,1),padding='same',activation='linear')(z3)
      
    
    z4=z3
    for j in range(4):
        z1=Dilation2D(2, (4,4),padding="valid",strides=(1,1))(z4)
        z2=Erosion2D(2, (4,4),padding="valid",strides=(1,1))(z4)
        z4=Concatenate()([z1,z2])
        z4=Conv2D(4,(1,1),strides=(2,2),padding='valid',activation='tanh')(z4)
        #z4=MaxPooling2D((2,2))(z4)
    

    z4=Flatten()(z4)
    z4=morph_layer(z4,10)
    z4=Dense(3,activation="sigmoid")(z4)


    z1=Dilation2D(2, (8,8),padding="same",strides=(1,1))(z3)
    z2=Erosion2D(2, (8,8),padding="same",strides=(1,1))(z3)
    z3=Concatenate()([z1,z2])
    z3=Conv2D(1,(1,1),padding='same',activation='sigmoid')(z3)

    z5=AGGLayer()([z3,z4,I])
    model=Model(inputs=[I],outputs=[z5])
    #model=Model(inputs=[I],outputs=[z3,z4])
    model.compile(loss="mse", optimizer="adam",metrics=['mse',DSSIMObjective(kernel_size=100)])
    #model.compile(loss=DSSIM_RGB, optimizer="adam",metrics=['mse',DSSIMObjective(kernel_size=23),cust_obj,])

    return model



"""
inp = K.placeholder(shape=(1,256, 256, 3))
gt = K.placeholder(shape=(1,256, 256, 3))
output=DSSIMObjective(kernel_size=255)(inp,gt)
f_ssim=K.function(inputs=[inp,gt],outputs=[output])
f_ssim([t3[:1],t2[:1]])
#f_ssim([t3[:1,:,:,:1],t2[:1,:,:,:1]])
"""







#training of network
def train_model():
    #DIBCO DATA
    model=model_DIBCO(inp_shape=(256,256,3))
    imgen=ImageSequence_DIBCO(input_size=(256, 256),stride=(120,120),batch_size=4)

    model.load_weights("models/model_weights.h5")
    model.fit_generator(imgen,epochs=10000)
    model.save_weights("models/model_weights1.h5")

    

    #ISI-DATA
    model=model_ISI(inp_shape=(256,256,3))
    imgen=ImageSequence_ISI(input_size=(256, 256),stride=(120,120),batch_size=4)
    model.load_weights("./models/model_weights_isi.h5")
    model.fit_generator(imgen,epochs=100)
    model.save_weights("./models/model_weights_isi.h5")

    























