import numpy as np
import pickle
from matplotlib import pyplot as plt

class geneticGenerator(object):
    
    def __init__(self):
        
        
        self.popSize = 32
        
        self.randRate = 0.2  #% of entirely random individiuals in next gen
        self.eliteRate = 0.2 #% of elite in next gen
        self.mutRate = 0.15 #chance of mutation for each gene
        
        self.imgSize = [128, 128]
        
        self.currentPop = randPopulation(self.popSize) 
        self.gen = 0
        
        self.maxScore = 0
        self.minScore = 0
        self.medianScore = 0
        self.bestIndiv = randIndiv()
    
    
    def run(self, iterations = 1):
        
    
        
        
        #Evaluating population
        scores = self.evaluatePop(self.currentPop)
        
        #Ranking
        couples = sorted(zip(scores,self.currentPop), reverse = True, key = lambda x: x[0])
        
        self.maxScore = couples[0][0]
        self.minScore = couples[len(couples)-1][0]
        self.medianScore = np.median(scores)
        self.bestIndiv = couples[0][1]
        
        ranked = [rules for score,rules in couples]
        
        #Elite Selection
        elite = selectPop(ranked, self.eliteRate)

        #Next Generation
        self.currentPop = nextGen(elite, size = self.popSize, randRate = self.randRate)
        
        print("Gen %d - Score : [%f - %f - %f]" % (self.gen, self.maxScore, self.medianScore, self.minScore))
        
        
        self.gen += 1
        if(iterations <= 1):
            return self.currentPop
        else:
            return self.run(iterations-1)
    
        
    def generateImages(self, number = 1, size = [128, 128]):
        
        
        imgs = []
        
         
        for i in range(number):
            imgs.append(self.generateImage(size = size))
        
        return imgs
    
    def generateImage(self, size = [128, 128], iterations = 5, indiv = False):
        
        
        if indiv is False:
            indiv = self.currentPop[np.random.choice(range(len(self.currentPop)))]
        
        rules, mean, sigma = indiv
        
        
        #Initial Grid Generattion Selection here
        
        #Point to Point generation with deltas following Normal Distribution
        #grid = randGrid(size = size, mean = mean, sigma = sigma)
        
        #2D Normal Distribution grid
        #grid = randNormalGrid(size = size, mean = mean, sigma = sigma)
        
        #Unfiform distribution
        grid = randUniformGrid(size = size)
        
        grid = applyRulesGrid(rules, grid, iterations)
        return grid
    
    
    
    def showImg(self, size = [128, 128], iterations = 5, save = False):
        
        
        plt.figure(figsize=(10,10))
        img = self.generateImage(size = size, iterations = iterations)
        plt.imshow(img, cmap='gray', aspect='equal', shape=size)
        plt.axis('off')
        
        plt.imsave("./generated/lastShownRoom",img, cmap="gray")
        
        
        
        
    def setDiscriminator(self, network):
        self.discriminator = network.discriminator
        
    def evaluatePop(self, population):    
        
        #For each rules in population, apply them 5 times to a randGrid
        grids = [self.generateImage(indiv = indiv, size = self.imgSize) for indiv in population]
        shapedGrids = np.array(grids).reshape(-1, self.imgSize[0], self.imgSize[1],1) 
        scores = self.discriminator.predict(shapedGrids)
        shapedScores = scores.reshape(-1)
        
        
        return shapedScores

    
    
    
    def trainStep(self):
        
        iterations = 0
        
        self.run()
        
        #Stop trainning when results are really good (median > 0.7)
        #Force training when results are really poor (max < 0.1)
        #Max 50 iterations if not needed
        while( (iterations < 50 or self.maxScore < 0.1)  and self.medianScore < 0.7):     
            self.run()
            iterations += 1
    
    def evaluateImg(self, img):
        
        score = self.discriminator.predict(img.reshape(-1, self.size[0], self.size[1], 1))
        return score[0]
    
    
    def save(self):
        file = open('./saves/roomGenerator.txt','wb')
        
        disc = self.discriminator
        
        self.discriminator = "Set me up dynamically"
        pickle.dump(self, file)
        file.close()
        
        self.discriminator = disc
        
        #self.savePanel()
        self.saveBest()


    def load(self):
        file = open('./saves/roomGenerator.txt','rb')
        dataPickle = file.read()
        file.close()
        self = pickle.load(dataPickle)
        
    def savePanel(self):
        
        
        bestImg =  self.generateImage(indiv = self.bestIndiv)
        imgs = [bestImg] + self.generateImages(number = 8)
         
        plt.figure(figsize=(10,10))
        
        for i in range(9):
            plt.subplot(3, 3, i+1)
            image = imgs[i]
            plt.imshow(image, cmap='gray')
            plt.axis('off')
            
        plt.tight_layout()
        
        plt.savefig('./generated/rooms_%d' % self.gen,bbox_inches = 'tight', pad_inches = 0)
        plt.close('all')


    def saveBest(self):
        
        bestImg =  self.generateImage(indiv = self.bestIndiv)
        plt.imsave("./generated/best_room_%d" % self.gen,bestImg, cmap="gray")

def nextGen(elite, size = 200, randRate = 0.2):
    
    
    #Fill with elite
    gen = elite
    
    
    #Fill with random individuals
    for i in range(int(size*randRate)):
        
        new = randIndiv()
        gen.append(new)
    
    #Fill with mutations of elite
    for i in range(len(gen)-len(elite)):
        
        base = elite[np.random.choice(range(len(elite)))]
        mutant = mutateIndiv(base)
        gen.append(mutant)
    
    return gen
    
    

        
def selectPop(pop, eliteRate = 0.1):
    return pop[:int(len(pop)*eliteRate)]

def mutateIndiv(indiv, rate = 0.15):
    
    (rules, mean, sigma) = indiv
    
    
    #Mutate rules
    mutantRules = np.copy(rules)
    for y in range(len(rules)):
        for x in range(len(rules[y])):
            if(np.random.uniform() <= rate):
                mutantRules[y,x] = (mutantRules[y,x] == 0)*1

    
    #Mutate mean
    if(np.random.uniform() <= rate):
        mutantMean = mean + np.random.uniform(-20,20)
        
        if(mutantMean < 1):
            mutantMean = 1
        elif (mutantMean > 400):
            mutantMean = 400     
    else:
        mutantMean = mean
    
    
    #Mutate sigma
    if(np.random.uniform() <= rate):
        mutantSigma = sigma + np.random.uniform(-3,3)
        
        if(mutantSigma < 0):
            mutantSigma = 0
        elif (mutantSigma > 20):
            mutantSigma = 20     
    else:
        mutantSigma = sigma

    return (mutantRules, mutantMean, mutantSigma)

def applyRulesGrid(rules, grid, iterations = 1):
    
    
    nextGrid = np.zeros(grid.shape).astype(int)
    
    
    for y in range(len(grid)):
        for x in range(len(grid[y])):
            state = grid[y-1:y+2,x-1:x+2]
            #state = np.resize(state, (3,3))
            nextGrid[y,x] = applyRulesCell(rules, state, grid[y,x])
    
    if(iterations <= 1):
        return nextGrid
    else:
        return applyRulesGrid(rules, nextGrid, iterations -1)
    
    
def applyRulesCell(rules, state, value):
    
    
    
    neighbors = np.count_nonzero(state)-value
    return rules[value, neighbors]



def randNormalGrid(size = [128, 128], mean = 64, sigma = 32) :
    
    number = 160
    
    points = np.random.normal(mean,sigma,(number,2)).round().astype(int)
    
    grid = np.zeros(size).astype(int)
    
    for couple in points:
        
        x,y = couple
        
        if(x < 0):
            x = 0
        if(x >= size[0]):
            x = size[0]-1
        
        if(y < 0):
            y = 0
        if(y >= size[1]):
            y = size[1]-1
    
        grid[x,y] = 1
    
    return grid


def randGrid(size = [128, 128], mean = 64, sigma = 32) :
    
    
    flatSize = size[0]*size[1]
    
    
    flatGrid = np.zeros(flatSize).astype(int)
    
    current = 0
    
    while current < flatSize:
        
        delta = abs(round(np.random.normal(mean,sigma)))
        current += delta
        
        if current < flatSize:
            flatGrid[current] = 1
    
    return flatGrid.reshape(size)



def randIndiv():
    ruleSize = [2,9]
    
    #Rules of the cellular automata
    rules = np.random.choice([0, 1], size=ruleSize).astype(int)
    
    #Mean of the normal distribution used to generate the map
    mean = round(np.random.uniform(1,400))
    
    
    #Sigma of the normal distribution used to generate the map
    sigma = round(np.random.uniform(0,20))
    
    
    return (rules, mean, sigma)

def randPopulation(size = 100):
    return [randIndiv() for i in range(size)]

def randUniformGrid(size = [128, 128], probability = [99/100, 1/100]) :
    grid = np.random.choice([0, 1], size=size, p=probability).astype(int)
    return grid