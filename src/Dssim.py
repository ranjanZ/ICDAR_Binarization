from __future__ import absolute_import
import keras_contrib.backend as KC
from keras import backend as K
from keras_contrib.losses import DSSIMObjective


def DSSIM_RGB(y_true, y_pred):

    #loss1=DSSIMObjective(kernel_size=23)(y_true[:,:,:,:1],y_pred[:,:,:,:1])
    #loss2=DSSIMObjective(kernel_size=23)(y_true[:,:,:,1:2],y_pred[:,:,:,1:2])
    #loss3=DSSIMObjective(kernel_size=23)(y_true[:,:,:,2:3],y_pred[:,:,:,2:3])
    #loss=K.mean(loss1+loss2+loss3)
    #loss=(loss1+loss2+loss3)/3.0
    loss=DSSIMObjective(kernel_size=23)(y_true[:,:,:,:3],y_pred[:,:,:,:3])
    return loss

