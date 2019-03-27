# -*- coding: utf-8 -*-
import GAN from "GAN.py"



class avatarGAN(object):
    
    def __init__(self):
        
        self.size = 8
        self.channel = 1

        self.filename = 'avatars'


        data = json.loads(open("./data/avatars.json").read())
        self.xTrain = np.array(data)
        self.xTrain = self.xTrain-0.5
        self.xTrain = self.xTrain.reshape(-1, self.size, self.size, 1)

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
        
            
            
            filename = ""
        
            if fake:
                if noise is None:
                    noise = np.random.uniform(-1.0, 1.0, size=[samples, 100])
                else:
                    filename = self.filename+"_%d.png" % step
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
    avgan.plotImgs(fake=False)
                    
                