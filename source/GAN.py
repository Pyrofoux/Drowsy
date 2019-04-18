# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 12:03:48 2019

@author: Younès
"""

import numpy as np
import matplotlib
import json
from matplotlib import pyplot as plt
#np.random.seed(123)  # for reproducibility

 	
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement     = True
sess = tf.Session(config=config)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class GAN(object):
    
    def __init__(self, size = 8, channel = 1, kernelsize = 3, depth = 64):
        
        self.size = size
        self.channel = channel
        self.kernelsize = kernelsize
        self.depth = depth
        self.dropout = 0.4
        
        self.D = None #Discriminator
        self.G = None #Generator
        self.AM = None #Adversarial Model
        self.DM = None #Discriminator Model
        
        
    # (W−F+2P)/S+1 <-- formula for size
    def discriminator(self):
        
        if (self.D):
            return self.D
        
        
        self.D = Sequential();
        
        depth = self.depth
        dropout = self.dropout
        kernelSize = self.kernelsize
        
        # In: 28 x 28 x 1, depth = 1
        # Out: 14 x 14 x 1, depth=64
        
        input_shape = (self.size, self.size, self.channel)
        
        self.D.add(Conv2D(depth*1, kernelSize, strides=2, input_shape=input_shape,padding='same', activation=LeakyReLU(alpha=0.2)))
        self.D.add(Dropout(dropout))
        
        self.D.add(Conv2D(depth*2, kernelSize, strides=2, padding='same',activation=LeakyReLU(alpha=0.2)))
        self.D.add(Dropout(dropout))
        
        self.D.add(Conv2D(depth*4, kernelSize, strides=2, padding='same',activation=LeakyReLU(alpha=0.2)))
        self.D.add(Dropout(dropout))
        
        self.D.add(Conv2D(depth*8, kernelSize, strides=1, padding='same',activation=LeakyReLU(alpha=0.2)))
        self.D.add(Dropout(dropout))
        
        # Out: 1-dim probability
        self.D.add(Flatten())
        self.D.add(Dense(1))
        self.D.add(Activation('sigmoid'))
        
        #self.D.summary()
        
        return self.D
    
    def generator(self):
        
        
        if(self.G):
            return self.G
        
        self.G = Sequential()
        
        dropout = self.dropout
        depth = self.depth*4 
        dim = int(self.size/4) #Dépend de la taille des images
        kernelSize = self.kernelsize
        
        # In: 100
        # Out: dim x dim x depth
        
        self.G.add(Dense(dim*dim*depth, input_dim=100))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))
        self.G.add(Reshape((dim, dim, depth)))
        self.G.add(Dropout(dropout))
        
        
        # In: dim x dim x depth
        # Out: 2*dim x 2*dim x depth/2
        
        self.G.add(UpSampling2D())
        self.G.add(Conv2DTranspose(int(depth/2), kernelSize, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))
        
        self.G.add(UpSampling2D())
        self.G.add(Conv2DTranspose(int(depth/4), kernelSize, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))
        
        self.G.add(Conv2DTranspose(int(depth/8), kernelSize, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))
        
        
        # Out: 28 x 28 x 1 grayscale image [0.0,1.0] per pix
        
        self.G.add(Conv2DTranspose(1, kernelSize, padding='same'))
        self.G.add(Activation('sigmoid'))
        
        
       # def custom_activation(tensor):
            #return tf.keras.backend.round(tensor)

        
        #get_custom_objects().update({'custom_activation': Activation(custom_activation)})
        #self.G.add(Activation(custom_activation))
       

        #self.G.summary()
        
        return self.G
    
    
    
    def discriminatorModel(self):
         
        if self.DM:
            return self.DM
        
        #to display allocation
        #options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
        
        
        optimizer = RMSprop(lr=0.0002, decay=6e-8)
        self.DM = Sequential()
        self.DM.add(self.discriminator())
        self.DM.compile(loss='binary_crossentropy', optimizer=optimizer,\
            metrics=['accuracy'])
        return self.DM
    
    
    def adversarialModel(self):
         
        if self.AM:
            return self.AM
        
        #options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
        
        optimizer = RMSprop(lr=0.0001, decay=3e-8)
        self.AM = Sequential()
        self.AM.add(self.generator())
        self.AM.add(self.discriminator())
        self.AM.compile(loss='binary_crossentropy', optimizer=optimizer,\
            metrics=['accuracy'])
        return self.AM