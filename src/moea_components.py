from deap import creator, base, tools
from deap.base import Fitness, Toolbox
from deap.algorithms import eaMuPlusLambda
from deap.tools import ParetoFront, Statistics, Logbook, selNSGA2
from deap.tools import ParetoFront
import matplotlib.cm as cm
from random import randrange
import multiprocessing
import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np
import random


def genetic_algorithm(
    toolbox, guess_strategy, MU, LAMBDA, CXPB, MUTPB, NGEN, verbose=True, hack=True
):
    """
    Execute a genetic algorithm using the specified parameters and toolbox.

    :param toolbox: A DEAP toolbox containing the genetic algorithm's operators.
    :param guess_strategy: A strategy used to initialize individuals in the population.
    :param MU: The population size.
    :param LAMBDA: The number of offspring to produce.
    :param CXPB: The crossover probability.
    :param MUTPB: The mutation probability.
    :param NGEN: The number of generations to evolve.
    :param verbose: If true, output detailed log of the evolution process.
    :param hack: If true, use a hacked version of the evolution algorithm.
    :return: A tuple containing the final population, statistics, hall of fame,
             logbook, and additional fitness data.
    """

    pop = toolbox.population(n=MU)
    hof = ParetoFront()  # Retrieve the best non-dominated individuals of the evolution

    # Statistics created for compiling four different statistics over the generations
    stats = Statistics(lambda ind: ind.fitness.values)
    stats.register(
        "avg", np.mean, axis=0
    )  # Compute the statistics on each objective independently
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    if hack:
        _, logbook, all_generations, all_fitness = eaMuPlusLambda_hack(
            pop,
            toolbox,
            MU,
            LAMBDA,
            CXPB,
            MUTPB,
            NGEN,
            stats,
            halloffame=hof,
            verbose=verbose,
        )
        return pop, stats, hof, logbook, all_generations, all_fitness
    else:
        _, logbook = eaMuPlusLambda(
            pop,
            toolbox,
            MU,
            LAMBDA,
            CXPB,
            MUTPB,
            NGEN,
            stats,
            halloffame=hof,
            verbose=verbose,
        )
        return pop, stats, hof, logbook


def eaMuPlusLambda_hack(
    population,
    toolbox,
    mu,
    lambda_,
    cxpb,
    mutpb,
    ngen,
    stats=None,
    halloffame=None,
    verbose=__debug__,
):
    """
    Execute the evolutionary algorithm using a mu+lambda strategy with a hack for additional metrics.

    :param population: The current population of individuals.
    :param toolbox: A DEAP toolbox containing the evolutionary operators.
    :param mu: The number of individuals in the current population.
    :param lambda_: The number of offspring to produce.
    :param cxpb: The probability of crossover.
    :param mutpb: The probability of mutation.
    :param ngen: The number of generations to evolve.
    :param stats: Optional statistics to compile during the evolution.
    :param halloffame: Optional hall of fame to store the best individuals.
    :param verbose: If True, print detailed log information.
    :return: A tuple containing the final population, logbook,
             generated information, and fitness data.
    """

    logbook = Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]

    # Unpack the fitness and extra information
    fitnesses, extra_infos = zip(*toolbox.map(toolbox.evaluate, invalid_ind))
    # Assign both fitness and extra information to individuals
    for ind, fit, extra_info in zip(invalid_ind, fitnesses, extra_infos):
        ind.fitness.values = fit  # Assign the fitness values
        ind.extra_info = extra_info  # Store the additional information

        # Optional: Print the extra_info for debugging
        # print(f"Post-evaluation extra_info for individual: {ind.extra_info}")

    # Update the hall of fame if applicable
    if halloffame is not None:
        halloffame.update(population)

    # Compile statistics and log the initial generation
    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)

    if verbose:
        print(logbook.stream)

    # --- Hack ---
    all_generations = {}
    all_fitness = {}
    # ------------

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Vary the population
        offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

        # Unpack fitness and extra info
        fitnesses, extra_infos = zip(*toolbox.map(toolbox.evaluate, invalid_ind))

        # Assign fitness and extra info separately
        for ind, fit, extra_info in zip(invalid_ind, fitnesses, extra_infos):
            ind.fitness.values = fit  # Assign only fitness values
            ind.extra_info = extra_info  # Assign extra information to the individual

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # --- Hack ---
        all_generations[gen] = population + offspring
        all_fitness[gen] = fitnesses
        # ------------

        # Select the next generation population
        population[:] = toolbox.select(population + offspring, mu)

        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)

        if verbose:
            print(logbook.stream)

    return population, logbook, all_generations, all_fitness


import random


def varOr(population, toolbox, lambda_, cxpb, mutpb):
    """
    Generates offspring using a combination of crossover, mutation, and reproduction.

    :param population: The current population of individuals.
    :param toolbox: A DEAP toolbox containing the genetic operators.
    :param lambda_: The number of offspring to produce.
    :param cxpb: The probability of applying crossover.
    :param mutpb: The probability of applying mutation.
    :return: A list of offspring generated from the population.
    """

    assert (cxpb + mutpb) <= 1.0, (
        "The sum of the crossover and mutation probabilities must be smaller "
        "or equal to 1.0."
    )

    offspring = []

    for _ in range(lambda_):
        op_choice = random.random()

        if op_choice < cxpb:  # Apply crossover
            ind1, ind2 = map(toolbox.clone, random.sample(population, 2))
            ind1, ind2 = toolbox.mate(ind1, ind2)
            del ind1.fitness.values
            offspring.append(ind1)

        elif op_choice < cxpb + mutpb:  # Apply mutation
            ind = toolbox.clone(random.choice(population))
            (ind,) = toolbox.mutate(ind)
            del ind.fitness.values
            offspring.append(ind)

        else:  # Apply reproduction
            offspring.append(random.choice(population))

    return offspring


def get_valid_ind(icls, size_of_individual):
    """
    Generates a valid individual with a specified genome size.

    Each individual is represented as a list of binary values (0s and 1s),
    with a random retreat location marked as 2. Additionally, consecutive
    1s are replaced with 0s based on specific criteria.

    :param icls: The individual class to instantiate.
    :param size_of_individual: The desired size of the individual's genome.
    :return: An instance of the individual class with a valid genome.
    """

    # Generate a base list of binary values
    base_list = [random.choice([0, 1]) for _ in range(size_of_individual)]

    # Randomly assign a retreat location
    retreat_location = randrange(size_of_individual)
    base_list[retreat_location] = 2

    # Replace consecutive 1s with 0s based on specific criteria
    base_list = [
        base_list[i]
        if i == 0
        else 0
        if base_list[i] == 1 and base_list[i - 1] == 1
        else base_list[i]
        for i in range(len(base_list))
    ]

    # Create the individual from the genome
    individual = icls(base_list)

    return individual


def crossover_list(ind1, ind2, n_time_steps):
    """
    Performs a crossover between two individuals to produce two children.

    :param ind1: The first parent individual.
    :param ind2: The second parent individual.
    :param n_time_steps: The number of time steps in the individuals.
    :return: A tuple containing the two child individuals.
    """

    random_location = [i for i in range(n_time_steps)]
    crossover_location = random.choice(random_location)

    child1 = ind1[:crossover_location] + ind2[crossover_location:]
    child2 = ind2[:crossover_location] + ind1[crossover_location:]

    if child1.count(2) > 1:
        r_indices = [i for i, x in enumerate(child1) if x == 2]
        r_indices.pop(random.randrange(len(r_indices)))
        child1 = [
            random.choice([0, 1]) if i in r_indices else child1[i]
            for i in range(len(child1))
        ]

    if child2.count(2) > 1:
        r_indices = [i for i, x in enumerate(child2) if x == 2]
        r_indices.pop(random.randrange(len(r_indices)))
        child2 = [
            random.choice([0, 1]) if i in r_indices else child2[i]
            for i in range(len(child2))
        ]

    child1 = [
        child1[i]
        if i == 0
        else 0
        if child1[i] == 1 and child1[i - 1] == 1
        else child1[i]
        for i in range(len(child1))
    ]
    child2 = [
        child2[i]
        if i == 0
        else 0
        if child2[i] == 1 and child2[i - 1] == 1
        else child2[i]
        for i in range(len(child2))
    ]

    child1 = creator.Individual(child1)
    child2 = creator.Individual(child2)

    return child1, child2


def mutate_list(individual, min_gap):
    """
    Mutates an individual by potentially altering its genes according to defined rules.

    This function may replace '2' with a random choice of 0 or 1, handle multiple
    occurrences of '2', and allow '2' to move one position in either direction.

    :param individual: The individual to be mutated.
    :param min_gap: Not currently used; can be used for further mutation logic.
    :return: A tuple containing the mutated individual.
    """

    k = random.choice([0, 1])

    if k == 1 and individual.count(2) != 0:
        replace_index = individual.index(2)
        individual[replace_index] = random.choice([0, 1])

    if individual.count(2) > 1:
        r_indices = [i for i, x in enumerate(individual) if x == 2]
        r_indices.pop(random.randrange(len(r_indices)))
        individual = [
            random.choice([0, 1]) if i in r_indices else individual[i]
            for i in range(len(individual))
        ]

    if 2 in individual and random.random() < 0.1:
        index_of_2 = individual.index(2)

        if index_of_2 > 0 and index_of_2 < len(individual) - 1:
            # Randomly move the '2' to the next or previous step
            new_index = index_of_2 + random.choice([-1, 1])
        elif index_of_2 == 0:
            # '2' can only move forward
            new_index = index_of_2 + 1
        else:
            # '2' can only move backward
            new_index = index_of_2 - 1

        # Move the '2'
        individual[new_index] = 2
        individual[index_of_2] = random.choice([0, 1])

    individual = [
        individual[i]
        if i == 0
        else 0
        if individual[i] == 1 and individual[i - 1] == 1
        else individual[i]
        for i in range(len(individual))
    ]

    individual = creator.Individual(individual)

    return (individual,)
