import numpy as np
from matplotlib import pyplot as plt


def randRules():
    ruleSize = [2,9]
    rules = np.random.choice([0, 1], size=ruleSize).astype(int)
    return rules

def randGrid():
    mapSize = 128
    grid = np.random.choice([0, 1], size=[mapSize, mapSize], p=[99/100, 1/100]).astype(int)
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
    
    
    nextMap = np.zeros((mapSize, mapSize)).astype(int)
    
    
    for y in range(len(grid)):
        for x in range(len(grid[y])):
            state = grid[y-1:y+2,x-1:x+2]
            #state = np.resize(state, (3,3))
            nextMap[y,x] = applyRulesCell(rules, state, grid[y,x])
    
    if(iterations <= 1):
        return nextMap
    else:
        return applyRulesGrid(rules, nextMap, iterations -1)
         
    
    


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


plt.imshow(grid, cmap='gray')

plt.figure(figsize=(10,10))
plt.imshow(applyRulesGrid(rules, grid, 5), cmap='gray')
