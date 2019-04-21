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
            #fakeImgs = np.random.uniform(-1.0, 1.0, size=[batchSize, 128, 128, 1])
            
            fakeImgs = self.generator.generateImages(number = batchSize, size = [self.size, self.size])
            fakeImgs = np.reshape(fakeImgs, (-1, self.size, self.size, 1))
            
            x = np.concatenate((trainImgs, fakeImgs))
            y = np.ones([2*batchSize, 1]) #1 for train image, 0 for fake
            y[batchSize:, :] = 0
            

            Dloss = self.discriminator.train_on_batch(x, y)

            log_mesg = "%d: [D loss: %f, acc: %f]" % (i, Dloss[0], Dloss[1])
            print(log_mesg)
            
            
            self.generator.trainStep()
            
            
            if saveInterval>0:
                
                if (i+1)%saveInterval==0:
                    
                    self.save()
                    
                    log_mesg = "%d: [D loss: %f, acc: %f]" % (i, Dloss[0], Dloss[1])
                    print(log_mesg)
                    self.plotImgs(save2file=True, samples=noiseInput.shape[0],noise=noiseInput, step=(i+1))
                    
                    
    def plotImgs(self, save2file=False, fake=True, samples=16, noise=None, step=0):
        
            
#            
#            path = "./generated/"
#        
#            if fake:
#                if noise is None:
#                    noise = np.random.uniform(-1.0, 1.0, size=[samples, 100])
#                else:
#                    filename = path+self.filename+"_%d.png" % step
#                images = self.generator.predict(noise)
#            else:
#                i = np.random.randint(0, self.xTrain.shape[0], samples)
#                images = self.xTrain[i, :, :, :]
#    
#            plt.figure(figsize=(10,10))
#            
#            
#            for i in range(images.shape[0]):
#                plt.subplot(4, 4, i+1)
#                image = images[i, :, :, :]
#                image = np.reshape(image, [self.size, self.size])
#                    
#                
#                plt.imshow(image, cmap='gray')
#                plt.axis('off')
#            plt.tight_layout()
#            
#            if save2file:
#                plt.savefig(filename)
#                plt.close('all')
#            #else
#            plt.show()     

            True               
                    
    def save(self):
        self.adversarial.save_weights("./weights/room_adversarial.h5")
        
    
    def load(self):
        self.adversarial.load_weights("./weights/room_adversarial.h5")
    
if __name__ == '__main__':
    
    print("Started")
    
    roomDisc = roomDisc()
    roomGen = genGen.geneticGenerator()
    roomDisc.setGenerator(roomGen)
    roomGen.setDiscriminator(roomDisc)
    
    print("")
    print("Discriminator/GeneticGenerator couple created")
    #rogan.load()
    roomDisc.train(trainSteps=10000, batchSize=, saveInterval=100)
    print("Training")
    #roomDisc.plotImgs(fake=True)
    #roomDisc.plotImgs(fake=False)
                    
                