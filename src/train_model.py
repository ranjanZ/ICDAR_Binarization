import os,sys
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from keras.engine.topology import Layer
import numpy as np
from keras.layers import initializers,constraints
import tensorflow as tf
from keras.models import Sequential,Model
from keras.utils import conv_utils
from keras import backend as K
import numpy as np
from keras.layers import Dense, Dropout, Flatten,Input
from keras.layers import Conv2D, MaxPooling2D,Concatenate,Input
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from keras_contrib.losses import DSSIMObjective
from keras.constraints import *
from keras.layers import Activation, Dense,Dropout
#import pandas
#from utils import *
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
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, UpSampling2D, Add, Dropout, concatenate
import time
import numpy as np 
import cv2 as cv




from PIL import Image

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="2"




def model_DIBCO(inp_shape=(256,256,3)):
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
    model.compile(loss=DSSIMObjective(kernel_size=100), optimizer="adam",metrics=['mse',DSSIMObjective(kernel_size=100)])

    return model

def model_ISI(inp_shape=(256,256,3)):
    I=Input(shape=inp_shape)
    #I=Input(shape=(None,None,1))

    z1=Dilation2D(2, (8,8),padding="same",strides=(1,1))(I)
    z2=Erosion2D(2, (8,8),padding="same",strides=(1,1))(I)
    z3=Concatenate()([z1,z2])
    z3=Conv2D(8,(1,1),padding='same')(z3)
    
    for j in range(2):
        z1=Dilation2D(2, (8,8),padding="same",strides=(1,1))(z3)
        z2=Erosion2D(2, (8,8),padding="same",strides=(1,1))(z3)
        z3=Concatenate()([z1,z2])
        z3=Conv2D(8,(1,1),padding='same')(z3)
      
    z1=Dilation2D(2, (8,8),padding="same",strides=(1,1))(z3)
    z2=Erosion2D(2, (8,8),padding="same",strides=(1,1))(z3)
    z3=Concatenate()([z1,z2])
    z3=Conv2D(3,(1,1),padding='same',activation='sigmoid')(z3)

    model=Model(inputs=[I],outputs=[z3])
    model.compile(loss=DSSIMObjective(kernel_size=100), optimizer="adam",metrics=['mse',DSSIMObjective(kernel_size=100)])
    #model.compile(loss=DSSIM_RGB(kernel_size=100), optimizer="adam",metrics=['mse',DSSIMObjective(kernel_size=100)])

    return model


#training of network
def train_model(epochs=100,DIBCO=True,ISI=False):
    if(DIBCO==True):
        #DIBCO DATA
        model=model_DIBCO(inp_shape=(256,256,3))
        imgen=ImageSequence_DIBCO(input_size=(256, 256),stride=(120,120),batch_size=4)

        #model.load_weights("models/model_weights.h5")
        model.fit_generator(imgen,epochs=epochs)
        model.save_weights("models/model_weights_dibco.h5")
     
    if(ISI==True):
        #ISI-DATA
        model=model_ISI(inp_shape=(256,256,3))
        imgen=ImageSequence_ISI(input_size=(256, 256),stride=(120,120),batch_size=4)
        #model.load_weights("models/model_weights_isi.h5")
        model.fit_generator(imgen,epochs=epochs)
        model.save_weights("models/model_weights_isi.h5")

    









