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


def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

def protectedPow(bottom, up):
    try:
        return math.pow(bottom, up)
    except ValueError:
        return 1

pset = gp.PrimitiveSet("MAIN", 1)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(protectedPow, 2)
pset.addTerminal(1)
pset.addTerminal(-4)

pset.renameArguments(ARG0='x')
pset.renameArguments(ARG1='y')

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=5)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

def evalSymbReg(individual, points):
    try:
        func = toolbox.compile(expr=individual)
        sqerrors = ((func(x) - ((1 / (1 + protectedPow(x, -4))) + (1 / (1 + protectedPow(x, -4)))))**2 for x in points)
        return math.fsum(sqerrors) / len(points),
    except:
        return 500000000/ len(points),

toolbox.register("evaluate", evalSymbReg, points=[-5, -4.6, -4.2, -3.8, -3.4, -3, -2.6, -2.2, -1.8, -1.4, -1, -0.6, -0.2, 0.2, 0.6, 1, 1.4, 1.8, 2.2, 3, 3.4, 3.8, 4.2, 4.6, 5])
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
