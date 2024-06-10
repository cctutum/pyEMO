import random
import numpy as np
from deap import base, creator, tools, algorithms

#%%

creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)


toolbox = base.Toolbox()
toolbox.register("attr_material", random.uniform, 0, 100)  
toolbox.register("attr_thickness", random.uniform, 0, 100)  
toolbox.register("attr_process", random.uniform, 0, 100)  

toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_material, toolbox.attr_thickness, toolbox.attr_process), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

#%%

def minimize_weight(individual):
    material, thickness, _ = individual
    return material * thickness * 0.01,  

def maximize_durability(individual):
    material, thickness, _ = individual
    return material + thickness - 50,  

def minimize_cost(individual):
    material, _, process = individual
    return material * process * 0.01,  

def temperature_resistance_constraint(individual):
    _, thickness, _ = individual
    return thickness >= 5  

def safety_margin_constraint(individual):
    _, thickness, _ = individual
    return thickness >= 5.5

#%%

def evaluate(individual):
    weight = minimize_weight(individual)[0]
    durability = maximize_durability(individual)[0]
    cost = minimize_cost(individual)[0]

    penalty = 0
    if not temperature_resistance_constraint(individual):
        penalty += 100  
    if not safety_margin_constraint(individual):
        penalty += 100  

  
    return weight + penalty, durability - penalty, cost + penalty

#%%

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=20, indpb=0.2)  
toolbox.register("select", tools.selNSGA2)

POP_SIZE = 300
MAX_GEN = 100
CXPB = 0.7
MUTPB = 0.3

#%%

def main():
    random.seed(64)
    pop = toolbox.population(n=POP_SIZE)
    hof = tools.ParetoFront()

    # Statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    # Run the algorithm
    algorithms.eaSimple(pop, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=MAX_GEN, 
                        stats=stats, halloffame=hof, verbose=True)

    # Return the hall of fame
    return hof

#%%

if __name__ == "__main__":
    hof = main()
    print("Best individual(s): ")
    for individual in hof:
        print(individual)
        
#%%
