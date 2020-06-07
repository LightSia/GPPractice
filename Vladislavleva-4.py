import operator
import math
import random

import numpy
import matplotlib.pyplot as plt
import networkx as nx

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

def Sigma(inputFunc, inputCount):
    if(inputCount <= 0):
        return 0
    
    if(not isinstance(inputCount, int)):
        return 0

    if(not callable(inputFunc)):
        return 0

    sigmaTotal = 0
    for temp in range(1, inputCount+1):
        sigmaTotal += inputFunc(temp)
    return sigmaTotal

def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

def makeSquare(variable):
    return (variable ** 2)


def SigmaOrigin(x):
    return Sigma(makeSquare(x-3), 5)


pset = gp.PrimitiveSet("MAIN", 1)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(Sigma, 2)
pset.addPrimitive(makeSquare, 1)

pset.addTerminal(10)
pset.addTerminal(5)

pset.renameArguments(ARG0='x')
pset.renameArguments(ARG1='y')
pset.renameArguments(ARG2='z')
pset.renameArguments(ARG3='w')
pset.renameArguments(ARG4='o')

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=5)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

def evalSymbReg(individual, points):
    func = toolbox.compile(expr=individual)
    sqerrors = ((func(x) - SigmaOrigin(x))**2 for x in points)
    return math.fsum(sqerrors) / len(points),

def makeSet(bottom, up, count):
    diVFactor = (up - bottom) / count
    return [bottom + x * diVFactor for x in range(count)]

toolbox.register("evaluate", evalSymbReg, points=makeSet(0.05, 6.05, 1024))
toolbox.register("select", tools.selTournament, tournsize=20)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=1, max_=5)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

pop = toolbox.population(n=500)
hof = tools.HallOfFame(1)
stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
stats_size = tools.Statistics(len)
mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
mstats.register("avg", numpy.mean)
mstats.register("std", numpy.std)
mstats.register("min", numpy.min)
mstats.register("max", numpy.max)

pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 40, stats=mstats,
                                halloffame=hof, verbose=True)
