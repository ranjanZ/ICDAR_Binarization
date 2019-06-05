# coding=utf-8
import skimage.transform
from keras.utils import Sequence
import numpy as np
import cv2
import glob
import pandas as pd
import os
from scipy import misc
from skimage.transform import  resize
import matplotlib.pyplot as plt


#DATA_PATH_DIBCO="../dataset/DIBCO/train/"
#DATA_PATH_ISI="/home/ranjan/work/morph_Network/ICDAR/dataset/ISI_letter/train/new/"

#DATA_PATH_DIBCO="/home/ranjan/work/morph_Network/ICDAR/dataset/DIBCO/train/others/"

DATA_PATH_DIBCO="/home/ranjan/work/morph_Network/ICDAR/dataset/ISI_letter/train/new/"


def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def read_image(file_path,mode="RGB"):
    Img=misc.imread(file_path,mode=mode)
    """
    r=int(Img.shape[0]/Img.shape[1])
    shape0=1000
    shape1=1000/r
    Img=misc.imresize(Img,(shape0,int(shape1))) 
    """
    #Img=rgb2gray(Img)/255.0
    return Img





class ImageSequence_DIBCO(Sequence):
    def __init__(self,  batch_size=4, input_size=(256, 256),stride=(100,100)):
        self.image_seq_path=DATA_PATH_DIBCO
        self.input_shape=input_size
        self.batch_size = batch_size
        self.epoch = 0
        self.SHAPE_Y=self.input_shape[0]
        self.SHAPE_X=self.input_shape[1]
        self.frames=sorted(os.listdir(self.image_seq_path+"X/"))
        self.STRIDE_X=stride[0]
        self.STRIDE_Y=stride[1]

    def __len__(self):
        #return (len(self.frames))
        return 100



    def __getitem__(self, idx):
        x_batch = []
        c=0

        LX=[]
        LY=[]
        D=os.listdir(self.image_seq_path+"X/")
        #d1=D[np.random.randint(len(D))]
        while(c<=self.batch_size):
            d1=D[np.random.randint(len(D))]
            I1=read_image(self.image_seq_path+"X/"+d1)
            I2=read_image(self.image_seq_path+"Y/"+d1[:-4]+"_gt."+d1[-3:],mode="L")
            #print I1.shape,I2.shape,d1
            
            a,b,_=I1.shape
            idx=np.r_[0:b-self.SHAPE_X:self.STRIDE_X]
            idy=np.r_[0:a-self.SHAPE_Y:self.STRIDE_Y]


            """
            for dx in idx:
                for dy in idy:
                    patchX=I1[dy:dy+self.SHAPE_X,dx:dx+self.SHAPE_Y,:]
                    patchY=I2[dy:dy+self.SHAPE_X,dx:dx+self.SHAPE_Y,:]
                    LX.append(patchX)
                    LY.append(patchY)
            """
            #print self.batch_size,c
            rdx=np.random.randint(len(idx))
            rdy=np.random.randint(len(idy))
            dx=idx[rdx]
            dy=idy[rdy]
            patchX=I1[dy:dy+self.SHAPE_X,dx:dx+self.SHAPE_Y,:]
            #patchY=I2[dy:dy+self.SHAPE_X,dx:dx+self.SHAPE_Y,:]
            #patchY=I2[dy:dy+self.SHAPE_X,dx:dx+self.SHAPE_Y,0]          #for bin input
            patchY=I2[dy:dy+self.SHAPE_X,dx:dx+self.SHAPE_Y]          #for bin input
            LX.append(patchX)
            LY.append(patchY)
            c=c+1


        LX=np.array(LX)/255.0
        LY=np.array(LY)/255.0
        LY=LY[:,:,:,np.newaxis]
	LY=LY.round()
        #print LX.shape,LY.shape

        return (LX,LY)

    def on_epoch_end(self):
        self.epoch += 1




class ImageSequence_ISI(Sequence):
    def __init__(self,  batch_size=4, input_size=(256, 256),stride=(100,100)):
        self.image_seq_path=DATA_PATH_ISI
        self.input_shape=input_size
        self.batch_size = batch_size
        self.epoch = 0
        self.SHAPE_Y=self.input_shape[0]
        self.SHAPE_X=self.input_shape[1]
        self.frames=sorted(os.listdir(self.image_seq_path+"X/"))
        self.STRIDE_X=stride[0]
        self.STRIDE_Y=stride[1]
        
    def __len__(self):
        #return (len(self.frames))
        return 100


    def __getitem__(self, idx):
        x_batch = []
        c=0

        LX=[]
        LY=[]
        D=os.listdir(self.image_seq_path+"X/")
        #d1=D[np.random.randint(len(D))]
        while(c<=self.batch_size):
            d1=D[np.random.randint(len(D))]
            I1=read_image(self.image_seq_path+"X/"+d1)
            I2=read_image(self.image_seq_path+"Y/"+d1[:-4]+"."+d1[-3:])
            
            
            a,b,_=I1.shape
            idx=np.r_[0:b-self.SHAPE_X:self.STRIDE_X]
            idy=np.r_[0:a-self.SHAPE_Y:self.STRIDE_Y]


            """
            for dx in idx:
                for dy in idy:
                    patchX=I1[dy:dy+self.SHAPE_X,dx:dx+self.SHAPE_Y,:]
                    patchY=I2[dy:dy+self.SHAPE_X,dx:dx+self.SHAPE_Y,:]
                    LX.append(patchX)
                    LY.append(patchY)
            """
            #print self.batch_size,c
            rdx=np.random.randint(len(idx))
            rdy=np.random.randint(len(idy))
            dx=idx[rdx]
            dy=idy[rdy]
            patchX=I1[dy:dy+self.SHAPE_X,dx:dx+self.SHAPE_Y,:]
            patchY=I2[dy:dy+self.SHAPE_X,dx:dx+self.SHAPE_Y,:]
            LX.append(patchX)
            LY.append(patchY)
            c=c+1


        LX=np.array(LX)/255.0
        LY=np.array(LY)/255.0
       
        print LX.shape,LY.shape

        return (LX,LY)

    def on_epoch_end(self):
        self.epoch += 1




"""
#from generator import *
A=ImageSequence(stride=(300,300))
t1,t2=A.__getitem__(3)
"""


