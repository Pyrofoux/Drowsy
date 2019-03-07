# -*- coding: utf-8 -*-
import numpy as np
import matplotlib
import json
from matplotlib import pyplot as plt
np.random.seed(123)  # for reproducibility

 	
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop

from tensorflow.examples.tutorials.mnist import input_data


class GAN(object):
    
    def __init__(self, size = 8, channel = 1):
        
        self.size = size
        self.channel = channel
        
        
        self.D = None #Discriminator
        self.G = None #Generator
        self.AM = None #Adversarial Model
        self.DM = None #Discriminator Model
        
        
    # (W−F+2P)/S+1 <-- formula for size
    def discriminator(self):
        
        if (self.D):
            return self.D
        
        
        self.D = Sequential();
        
        depth = 64
        dropout = 0.4

        
        # In: 28 x 28 x 1, depth = 1
        # Out: 14 x 14 x 1, depth=64
        
        input_shape = (self.size, self.size, self.channel)
        
        self.D.add(Conv2D(depth*1, 5, strides=2, input_shape=input_shape,padding='same', activation=LeakyReLU(alpha=0.2)))
        self.D.add(Dropout(dropout))
        
        self.D.add(Conv2D(depth*2, 5, strides=2, padding='same',activation=LeakyReLU(alpha=0.2)))
        self.D.add(Dropout(dropout))
        
        self.D.add(Conv2D(depth*4, 5, strides=2, padding='same',activation=LeakyReLU(alpha=0.2)))
        self.D.add(Dropout(dropout))
        
        self.D.add(Conv2D(depth*8, 5, strides=1, padding='same',activation=LeakyReLU(alpha=0.2)))
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
        
        dropout = 0.4
        depth = 64*4 
        dim = int(self.size/4) #Dépend de la taille des images

        
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
        self.G.add(Conv2DTranspose(int(depth/2), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))
        
        self.G.add(UpSampling2D())
        self.G.add(Conv2DTranspose(int(depth/4), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))
        
        self.G.add(Conv2DTranspose(int(depth/8), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))
        
        
        # Out: 28 x 28 x 1 grayscale image [0.0,1.0] per pix
        
        self.G.add(Conv2DTranspose(1, 5, padding='same'))
        self.G.add(Activation('sigmoid'))
        
        #self.G.summary()
        
        return self.G
    
    def discriminatorModel(self):
         
        if self.DM:
            return self.DM
        
        optimizer = RMSprop(lr=0.0002, decay=6e-8)
        self.DM = Sequential()
        self.DM.add(self.discriminator())
        self.DM.compile(loss='binary_crossentropy', optimizer=optimizer,\
            metrics=['accuracy'])
        return self.DM
    
    
    def adversarialModel(self):
         
        if self.AM:
            return self.AM
        
        optimizer = RMSprop(lr=0.0001, decay=3e-8)
        self.AM = Sequential()
        self.AM.add(self.generator())
        self.AM.add(self.discriminator())
        self.AM.compile(loss='binary_crossentropy', optimizer=optimizer,\
            metrics=['accuracy'])
        return self.AM


class avatarGAN(object):
    
    def __init__(self):
        
        self.size = 8
        self.channel = 1

        #self.xTrain = input_data.read_data_sets("mnist",one_hot=True).train.images
        #self.xTrain = self.xTrain.reshape(-1, self.imgRows, self.imgCols, 1).astype(np.float32)

        data = json.loads(open("./data/sprites.json").read())
        self.xTrain = np.array(data)
        self.xTrain = self.xTrain.reshape(-1, self.size, self.size, 1).astype(np.int32)

        self.GAN = GAN(self.size, self.channel)
        self.discriminator =  self.GAN.discriminatorModel()
        self.adversarial = self.GAN.adversarialModel()
        self.generator = self.GAN.generator()
    
    
    def train(self, trainSteps=2000, batchSize=256, saveInterval=0):
        
        noiseInput = None
        
        if saveInterval>0:
            noiseInput = np.random.uniform(-1.0, 1.0, size=[16, 100])
            
            
        for i in range(trainSteps):
            
           
            noise = np.random.uniform(-1.0, 1.0, size=[batchSize, 100])
            
            trainImgs = self.xTrain[np.random.randint(0, self.xTrain.shape[0], size=batchSize), :, :, :]
            fakeImgs = self.generator.predict(noise)
            
            
            
            x = np.concatenate((trainImgs, fakeImgs))
            y = np.ones([2*batchSize, 1]) #1 for train image, 0 for fake
            y[batchSize:, :] = 0
            
            
            Dloss = self.discriminator.train_on_batch(x, y)

            y = np.ones([batchSize, 1]) 
            noise = np.random.uniform(-1.0, 1.0, size=[batchSize, 100])
            
            
            Aloss = self.adversarial.train_on_batch(noise, y)
            
            log_mesg = "%d: [D loss: %f, acc: %f]" % (i, Dloss[0], Dloss[1])
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, Aloss[0], Aloss[1])
            
            
            print(log_mesg)
            
            if saveInterval>0:
                
                if (i+1)%saveInterval==0:
                    print(log_mesg)
                    self.plotImgs(save2file=True, samples=noiseInput.shape[0],noise=noiseInput, step=(i+1))
                    
                    
    def plotImgs(self, save2file=False, fake=True, samples=16, noise=None, step=0):
        
            filename = 'digits.png'
            
            if fake:
                if noise is None:
                    noise = np.random.uniform(-1.0, 1.0, size=[samples, 100])
                else:
                    filename = "digits_%d.png" % step
                images = self.generator.predict(noise)
            else:
                i = np.random.randint(0, self.xTrain.shape[0], samples)
                images = self.xTrain[i, :, :, :]
    
            plt.figure(figsize=(10,10))
            for i in range(images.shape[0]):
                plt.subplot(4, 4, i+1)
                image = images[i, :, :, :]
                image = np.reshape(image, [self.size, self.size])
                plt.imshow(image, cmap='gray')
                plt.axis('off')
            plt.tight_layout()
            if save2file:
                plt.savefig(filename)
                plt.close('all')
            else:
                plt.show()                    
                    
    
    
if __name__ == '__main__':
    
    avgan = avatarGAN()
    #timer = ElapsedTimer()
    avgan.train(trainSteps=10000, batchSize=20, saveInterval=200) #100000 steps normalement et 500 interval
    #timer.elapsed_time()
    avgan.plotImgs(fake=True)
    #avgan.plotImgs(fake=False)
                    
                