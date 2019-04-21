# -*- coding: utf-8 -*-
import GAN 
import matplotlib
import json
from matplotlib import pyplot as plt
import numpy as np
import roomGeneticGenerator as genGen

class roomDisc(object):
    
    def __init__(self):
        
        self.size = 128
        self.channel = 1
        self.kernelSize= 48
        self.filename = 'rooms'
        self.depth = 16

        data = json.loads(open("../data/rooms.json").read())
        self.xTrain = np.array(data)
        self.xTrain = self.xTrain.reshape(-1, self.size, self.size, 1)

        print("Shape of data :", self.xTrain.shape)

        self.GAN = GAN.GAN( depth = self.depth, size = self.size, channel = self.channel, kernelsize = self.kernelSize)
        self.discriminator =  self.GAN.discriminatorModel()
    
    
    def setGenerator(self, generator):
        
        self.generator = generator
        
    
    
    def train(self, trainSteps=2000, batchSize=256, saveInterval=0):
        
        noiseInput = None
        
        if saveInterval>0:
            noiseInput = np.random.uniform(-1.0, 1.0, size=[16, 100])
            
            
        for i in range(trainSteps):
            
            
            trainImgs = self.xTrain[np.random.randint(0, self.xTrain.shape[0], size=batchSize), :, :, :]
 
            fakeImgs = self.generator.generateImages(number = batchSize, size = [self.size, self.size])
            fakeImgs = np.reshape(fakeImgs, (-1, self.size, self.size, 1))
            
            x = np.concatenate((trainImgs, fakeImgs))
            y = np.ones([2*batchSize, 1]) #100 for train image, 0 for fake
            y[batchSize:, :] = 0
            

            Dloss = self.discriminator.train_on_batch(x, y)

            log= "%d: [D loss: %f, acc: %f]" % (i, Dloss[0], Dloss[1])
            print(log)
            
            
            self.generator.trainStep()
            self.generator.showImg()
            
            
            
            
            if saveInterval>0:
                
                if (i+1)%saveInterval==0:
                    
                    self.save()
                    self.generator.save()
                                      
                    
    def save(self):
        self.discriminator.save_weights("./saves/room_discriminator.h5")
        
    
    def load(self):
        self.discriminator.load_weights("./saves/room_discriminator.h5")
    
if __name__ == '__main__':
    
    print("Started")
    
    roomDisc = roomDisc()
    roomGen = genGen.geneticGenerator()
    roomDisc.setGenerator(roomGen)
    roomGen.setDiscriminator(roomDisc)
    
    print("")
    print("Discriminator/GeneticGenerator couple created")
   
    #Later add load here
    
    
    #For archive purpose, save the first panel of images
    roomGen.savePanel()
    
    print("Starting Mutual Training")
    roomDisc.train(trainSteps=10000, batchSize=16, saveInterval=50)
                    
                