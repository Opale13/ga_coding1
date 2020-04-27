import random
import numpy as np

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

# https://www.geeksforgeeks.org/0-1-knapsack-problem-dp-10/
# Solution 220
NBR_ITEMS = 3
MAX_WEIGHT = 50
values = np.array([60, 100, 120])
weights = np.array([10, 20, 30])

# NBR_ITEMS = 5
# MAX_WEIGHT = 70
# values = np.array([60, 100, 120, 300, 10])
# weights = np.array([10, 20, 30, 50, 15])

# NBR_ITEMS = 10
# MAX_WEIGHT = 250
# values = np.array([random.randint(0, 80) for i in range(NBR_ITEMS)])
# weights = np.array([random.randint(0, 30) for i in range(NBR_ITEMS)])


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register('indices', random.randrange, 2)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.indices, NBR_ITEMS)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def fitness(individual):
    if sum(weights * np.array(individual)) > MAX_WEIGHT:
        return (0,)
    
    else:
        return (sum(values * np.array(individual)),)


# https://deap.readthedocs.io/en/master/api/tools.html
toolbox.register('evaluate', fitness)
toolbox.register('select', tools.selRoulette)
toolbox.register('mate', tools.cxUniform, indpb=1/NBR_ITEMS)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=1/NBR_ITEMS)


if __name__ == "__main__":
    NGEN = 500
    MU = 10
    LAMBDA = 100
    CXPB = 0.7
    MUTPB = 0.2

    population = toolbox.population(n=MU)

    hof = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register('avg', np.mean)
    stats.register('std', np.std)
    stats.register('min', np.min)
    stats.register('max', np.max)

    population, log = algorithms.eaMuPlusLambda(population, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, stats, halloffame=hof)

    print('Final population:\n', '\n '.join('{}: {}'.format([ind], ind.fitness.values[0]) for ind in population), '\n')
    print('Hall of Fame:\n', '{}: {}'.format([hof[0]], hof[0].fitness.values[0]))
    with open('tsp-ga-log.txt', 'w') as file:
        file.write(str(log))  