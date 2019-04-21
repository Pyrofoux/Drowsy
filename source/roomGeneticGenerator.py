import numpy as np
import pickle
from matplotlib import pyplot as plt

class geneticGenerator(object):
    
    def __init__(self):
        
        
        self.popSize = 15 
        
        self.randRate = 0.2  #% of entirely random individiuals in next gen
        self.eliteRate = 0.2 #% of elite in next gen
        self.mutRate = 0.15 #chance of mutation for each gene
        
        self.imgSize = [128, 128]
        
        self.currentPop = randPopulation(self.popSize) 
        self.gen = 0
        
        self.maxScore = 0
        self.minScore = 0
        self.medianScore = 0
        self.bestIndiv = np.zeros([2,9])
    
    
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
    
    def generateImage(self, size = [128, 128], iterations = 5, rules = False):
        
        
        if rules is False:
            rules = self.currentPop[np.random.choice(range(len(self.currentPop)))]
        
        grid = randGrid(size = size)
        grid = applyRulesGrid(rules, grid, iterations)
        return grid
    
    def showImg(self, size = [128, 128], iterations = 5):
        
        img = self.generateImage(size = size, iterations = iterations)
        plt.figure(figsize=(10,10))
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        
    def setDiscriminator(self, network):
        self.discriminator = network.discriminator
        
    def evaluatePop(self, population):
    
        scores = [evaluate(population[i]) for i in range(len(population))]
        
        
        #For each rules in population, apply them 5 times to a randGrid
        grids = [applyRulesGrid(population[i],  randGrid(), 5)for i in range(len(population)) ]
        shapedGrids = np.array(grids).reshape(-1, self.imgSize[0], self.imgSize[1],1) 
        scores = self.discriminator.predict(shapedGrids)
        shapedScores = scores.reshape(-1)
        
        
        return shapedScores
    
    def trainStep(self):
        
        iterations = 10
        
        self.run()
        
        #Stop trainning when results are really good (median > 0.9)
        #Force training when results are really poor (max < 0.1)
        #Max 10 iterations if not needed
        while( (iterations > 1 or self.maxScore < 0.1)  and self.medianScore < 0.9):     
            self.run()
            iterations -= 1
    
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
        
        self.savePanel()


    def load(self):
        file = open('./saves/roomGenerator.txt','rb')
        dataPickle = file.read()
        file.close()
        self = pickle.load(dataPickle)
        
    def savePanel(self):
        
        
        bestImg =  self.generateImage(rules = self.bestIndiv)
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

def nextGen(elite, size = 200, randRate = 0.2):
    
    
    #Fill with elite
    gen = elite
    
    
    #Fill with random individuals
    for i in range(int(size*randRate)):
        
        new = randRules()
        gen.append(new)
    
    #Fill with mutations of elite
    for i in range(len(gen)-len(elite)):
        
        base = elite[np.random.choice(range(len(elite)))]
        mutant = mutateRules(base)
        gen.append(mutant)
    
    return gen
    
    

        
def selectPop(pop, eliteRate = 0.1):
    return pop[:int(len(pop)*eliteRate)]
    


    
    

def evaluate(rules):
    grid = randGrid()
    grid = applyRulesGrid(rules, grid, 5)
    
    
    #
    # INCLUDE NETWORK HERE
    #
    #
    
    return np.random.uniform()

def randRules():
    ruleSize = [2,9]
    rules = np.random.choice([0, 1], size=ruleSize).astype(int)
    return rules

def randGrid(size = [128, 128], probability = [99/100, 1/100]) :
    grid = np.random.choice([0, 1], size=size, p=probability).astype(int)
    return grid

def randPopulation(size = 100):
    return [randRules() for i in range(size)]

def mutateRules(rules, rate = 0.15):
    mutant = np.copy(rules)
    for y in range(len(rules)):
        for x in range(len(rules[y])):
            if(np.random.uniform() <= rate):
                mutant[y,x] = (mutant[y,x] == 0)*1

    return mutant





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



np.random
