# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 02:48:27 2016

@author: inoueharuki
"""


import time
import numpy as np
#from scipy.integrate import odeint
import numba
from deap import base
from deap import creator
from deap import tools

        
def evaluate(param, model):
    return model.evaluate(param)
    

class COPASIep(object):
    def __init__(self, objfun, numParents, numGeneration, lb, ub, origin=[], minvar=1.8e-8):
        self.objfun = objfun
        self.numParents = numParents
        self.numGeneration = numGeneration
        self.lb = lb
        self.ub = ub
        self.origin = np.array(origin)
        self.minvar = minvar
        self.numParam = len(lb)
        self.tau1 = 1/np.sqrt(2*self.numParam)
        self.tau2 = 1/np.sqrt(2*np.sqrt(self.numParam))
        self.numCreate = self.numParents - self.origin.shape[0]
        
    def create(self):
        parents = np.zeros([self.numCreate, self.numParam])
        for i in xrange(self.numParam):
            if self.lb[i] >= 0:   # 0<lb<ub
                scale = np.log10(self.ub[i]) - np.log10(self.lb[i])
                if scale < 1.8:   # linear scale
                    tempParam = self.lb[i] + (self.ub[i] - self.lb[i])*np.random.random([self.numCreate, 1])
                else:   # log scale
                    tempParam = np.power(10, np.log10(self.lb[i])+scale*np.random.random([self.numCreate, 1]))
            elif self.ub[i] > 0:   # lb<0<ub
                scale = np.log10(self.ub[i]) + np.log10(-self.lb[i])
                if scale < 3.6:   # linear scale
                    tempParam = self.lb[i] + (self.ub[i]-self.lb[i]*np.random.random([self.numCreate, 1]))
                else:   # log scale
                    mean = (self.lb[i] + self.ub[i])/2
                    sigma = mean/100
                    tempParam = np.zeros([self.numCreate,1])
                    for j in xrange(self.numCreate):
                        tempParam[j] = mean +sigma*np.random.random()
                        while tempParam[j] < self.lb[j] or tempParam[j] > self.ub[j]:
                            tempParam[j] = mean +sigma*np.random.random()
            else:   # lb<ub<0
                scale = np.log10(-self.lb[i]) - np.log10(-self.ub[i])
                if scale < 1.8:   # linear scale
                    tempParam = self.lb[i] + (self.ub[i] - self.lb[i])*np.random.random([self.numCreate, 1])
                else:
                    tempParam = -np.power(10, np.log10(-self.ub[i])) + scale*np.random.random([self.numCreate, 1])
            parents[:,i] = tempParam.T
        return parents
        
    def generate(self, parents, historyParents):
        lb = np.tile(self.lb, (self.numParents, 1))
        ub = np.tile(self.ub, (self.numParents, 1))
        sigma = historyParents * np.exp(np.tile(self.tau1*np.random.random(size=(self.numParents, 1)), (1, self.numParam))
                                                            + self.tau2*np.random.random(size=(self.numParents, self.numParam)))
        sigma[np.where(sigma < self.minvar)] = self.minvar
        historyChildren = sigma
        children = parents + historyChildren*np.random.random(size=(self.numParents, self.numParam))
        children = np.minimum(np.maximum(children,lb),ub)
        return children, historyChildren

    def select(self, scores):
        numPopulation = len(scores)
        numEnemy = np.floor(numPopulation/5).astype(int)
        population = np.tile(scores, (numEnemy, 1))
        challengers = np.array([np.random.choice(scores, size=numPopulation).tolist() for i in range(numEnemy)])
        randomTournament = population - challengers
        numWin = (randomTournament<0).sum(axis=0)
        ID = np.argsort(numWin)[::-1]
        return ID[0:self.numParents]
    
    @numba.jit
    def evolution(self):
        start = time.time()
        if len(self.origin) == 0:
            parents = self.create()
        else:
            parents = np.vstack((self.origin, self.create()))
        historyParents = parents/2
        report = np.zeros([self.numGeneration, 4])
        scoreParents = np.array(map(self.objfun, parents)) #use map
        bestScore = np.min(scoreParents)
        bestID = np.argmin(scoreParents)
        bestIndv = parents[bestID, :]
        meanScore = np.mean(scoreParents)
        lastScore = bestScore
        stall = 0
        lap = time.time() - start
        report[0, :] = np.array([bestScore, meanScore, lap, stall])
        print("Generation \t Best f(x) \t Mean f(x) \t lap (sec) \t StallGen")
        print('%s \t %s \t %s \t %s \t %s', (1, report[0,0], report[0,1], report[0,2], report[0,3]))
        
        for i in range(1, self.numGeneration):
            start = time.time()
            children, historyChildren = self.generate(parents, historyParents)
            scoreChildren = np.array(map(self.objfun, children))
            population = np.vstack((parents, children))
            scores = np.hstack((scoreParents, scoreChildren))
            history = np.vstack((historyParents, historyChildren))
            selectId = self.select(scores)
            parents = population[selectId, :]
            scoreParents = scores[selectId]
            historyParents = history[selectId,:]
            bestID = np.argmin(scoreParents)
            bestScore = scoreParents[bestID]
            bestIndv = parents[bestID,:]
            
            if bestScore >= lastScore:
                stall = stall + 1
            else:
                stall = 0
                lastScore = bestScore
            meanScore = np.mean(scoreParents)
            lap = time.time() - start
            report[i, :] = np.array([bestScore, meanScore, lap, stall])
            print ('%s \t %s \t %s \t %s \t %d' ,(i+1, report[i,0], report[i,1], report[i,2], report[i,3]))
        return bestIndv, bestScore, report
   

     
class GA(object):
    def __init__(self, modelObj, numParents, numGeneration, lb, ub):
        self.model = modelObj
        self.numParents = numParents
        self.numGeneration = numGeneration
        self.lb = lb
        self.ub = ub
    
    
    def evolution(self):
        start = time.time()
        CXPB, MUTPB= 0.3, 0.2
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)
        toolbox = base.Toolbox()
        toolbox.register("individual", tools.initRepeat, creator.Individual, np.random.random, self.model.numParam)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", evaluate, model=self.model) #toolbox.register("evaluate", models.objfunc(self.model.evaluate))
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutPolynomialBounded, indpb=1, eta=0, low=self.lb, up=self.ub)
        toolbox.register("select", tools.selTournament, tournsize=self.numParents/20)
        tools.mutGaussian()
        print("Start evolution")
        start = time.time()
        population = toolbox.population(n=self.numParents)
        fitnesses = list(map(toolbox.evaluate, population))
        report = np.zeros([self.numGeneration, 7])
        for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in population]
        lastScore = np.min(fits)
        meanScore = np.mean(fits)
        std = np.std(fits)
        lap = time.time() - start
        stall=0
        print("  Evaluated %i individuals" , len(population))
        print( 'Generation \t evaluate \t Best f(x) \t Mean f(x) \t std \t\t lap (sec) \t StallGen')
        print( '%d \t\t %d \t\t %10f \t %10f \t %10f \t %10f \t %d' ,(0, len(population), lastScore, meanScore, std, lap, stall))
        report[0, :] = np.array([0, len(population), lastScore, meanScore, std, lap, stall])
        
        # Begin the evolution        
        for generation in range(self.numGeneration):
            start = time.time()
            offspring = toolbox.select(population, len(population))
            offspring = list(map(toolbox.clone, offspring))
            
            for child1, child2 in zip(offspring[0::2], offspring[1::2]):
                # cross two individuals with probability CXPB
                if np.random.random() < CXPB:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
                    
            for mutant in offspring:
                # mutate an individual with probability MUTPB
                if np.random.random() < MUTPB:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
                
            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            # The population is entirely replaced by the offspring
            population[:] = offspring
            # Gather all the fitnesses in one list and print the stats
            fits = [ind.fitness.values[0] for ind in population]
            
            bestScore = min(fits)
            std = np.std(fits)
            if bestScore < lastScore:
                stall = 0
                lastScore = bestScore
            else:
                stall = stall + 1
                
                
                
            lap = time.time() - start
            print( '%d \t\t %d \t\t %10f \t %10f \t %10f \t %10f \t %d' ,(generation+1, len(invalid_ind), bestScore, np.mean(fits), std, lap, stall))
            report[0, :] = np.array([0, len(invalid_ind), bestScore, meanScore, std, lap, stall])        
        print("-- End of (successful) evolution --")
        best_ind = tools.selBest(population, 1)[0]
        print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
        return best_ind, best_ind.fitness.values, report