import numpy as np
from matplotlib import pyplot as plt

class geneticGenerator(object):
    
    def __init__(self):
        
        self.eliteRate = 0.1
        self.mutRate = 0.15
        self.popSize = 10
        self.imgSize = [128, 128]
        
        self.currentPop = randPopulation(self.popSize) 
        self.gen = 0
        self.maxScore = None
        self.minScore = None
    
    
    def run(self, iterations = 1):
        
        
        print("Gen %d" % self.gen)
        
        
        #Evaluating population
        scores = self.evaluatePop(self.currentPop)
        
        #Ranking
        couples = sorted(zip(scores,self.currentPop), reverse = True, key = lambda x: x[0])
        
        self.maxScore = couples[0][0]
        self.minScore = couples[len(couples)-1][0]
        
        ranked = [rules for score,rules in couples]
        
        #Elite Selection
        elite = selectPop(ranked, self.eliteRate)

        #Next Generation
        self.currentPop = nextGen(elite, self.popSize)
        
        print("Score : [%d - %d]" % (self.maxScore, self.minScore))
        
        
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
    
    def generateImage(self, size = [128, 128], iterations = 5):
        
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
        
        self.run()
    

def nextGen(elite, size = 200):
    
    gen = elite
    
    for i in range(size-len(elite)):
        
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
         
    
    


def applyRulesCellPow2(rules, state):
    
    powers = 2 ** np.arange(9)
    bitIndex = 0
    index = 0
    for y in range(len(state)):
        for x in range(len(state[y])):
            bitIndex += state[y,x]*powers[index]
            index += 1
    
    return rules[int(bitIndex)]
    
    
def applyRulesCell(rules, state, value):
    
    neighbors = np.count_nonzero(state)-value
    return rules[value, neighbors]