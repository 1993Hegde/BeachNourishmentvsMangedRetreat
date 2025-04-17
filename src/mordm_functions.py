from SALib.sample import latin
import numpy as np
from scipy.stats import beta
from typing import Tuple, Dict, Any
import os
from decision_benchmarks import *
from moea_components import *


def pert_beta_params(
    b_min: float, b_most_likely: float, b_max: float
) -> Tuple[float, float]:
    """
    Calculates the parameters for a Beta distribution based on a PERT (Program Evaluation and Review Technique)
    distribution defined by a minimum, most likely, and maximum value.

    The PERT distribution is useful in project management for estimating the duration of tasks based on expert input.

    :param b_min: The minimum value of the distribution. This is the pessimistic estimate.
    :type b_min: float
    :param b_most_likely: The most likely value of the distribution. This is the optimistic estimate.
    :type b_most_likely: float
    :param b_max: The maximum value of the distribution. This is the most optimistic estimate.
    :type b_max: float
    :return: A tuple containing the calculated alpha and beta parameters for the Beta distribution.
    :rtype: Tuple[float, float]
    """
    alpha = 1 + 4 * (b_most_likely - b_min) / (b_max - b_min)
    beta_param = 1 + 4 * (b_max - b_most_likely) / (b_max - b_min)
    return alpha, beta_param


def generate_beach_nourishment_parameters_lhs_samples(
    dir_name: str, number_of_sows: int
) -> np.ndarray:
    """
    Generates Latin Hypercube Sampling (LHS) samples for beach nourishment parameters,
    including a parameter sampled from a Beta distribution based on specified bounds.

    The function creates LHS samples for various parameters related to beach nourishment,
    estimates the parameters for the 'b' parameter using a PERT approximation,
    and saves the resulting samples to a specified directory.

    :param dir_name: The name of the directory (within results_data) where the generated LHS samples will be saved.
    :type dir_name: str
    :param number_of_sows: The number of samples to generate using the Latin Hypercube Sampling method.
    :type number_of_sows: int
    :return: A 2D NumPy array containing the generated LHS samples along with 'b' values.
    :rtype: np.ndarray
    """
    # Define the problem (excluding 'b')
    problem = {
        "num_vars": 11,  # 12 total variables minus 'b'
        "names": [
            "epsilon",
            "phi_conc",
            "A",
            "alpha",
            "beta",
            "d",
            "eta",
            "l",
            "W",
            "xi",
            "delta",
        ],
        "bounds": [
            [0, 0.1],
            [0.7 * 193.8357, 1.3 * 193.8357],
            [0.7 * 669000, 1.3 * 669000],
            [1.0, 1.05],
            [0.95, 1.0],
            [-0.05, 0.05],
            [0.7 * 824, 1.3 * 824],
            [0.176, 0.264],
            [0.7 * 500000, 1.3 * 500000],
            [0, 0.1],
            [0.01, 0.07],
        ],
    }

    # Generate number_of_sows LHS samples for the other parameters
    lhs_samples = latin.sample(problem, number_of_sows)

    # Given values for 'b'
    b_min = 0
    b_most_likely = 0.0271 * 10**-3
    b_max = 0.113 * 10**-3

    # Estimate the alpha and beta using the PERT approximation
    alpha, beta_param = pert_beta_params(b_min, b_most_likely, b_max)

    # Generate the 'b' samples from the beta distribution
    b_samples = beta.rvs(
        alpha, beta_param, loc=b_min, scale=b_max - b_min, size=number_of_sows
    )

    # Combine 'b' with the LHS samples
    lhs_samples = np.column_stack((b_samples, lhs_samples))

    # Define the full path for the new directory inside the results_data folder
    results_dir = os.path.join("..", "results_data", dir_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Save the samples in the specified directory
    np.save(os.path.join(results_dir, "lhs_samples.npy"), lhs_samples)

    return lhs_samples


def evaluate_pathway_sow(
    individual: np.ndarray,
    pars: Dict[str, Any],
    num_objectives: int,  # New argument for number of objectives
) -> Tuple[float, float, ...]:  # Adjusted return type for flexibility
    """
    Evaluates the outcomes of a given strategy represented by an individual over a series of time periods.
    The function simulates the evolution of states based on actions taken at each time period,
    calculates associated costs and benefits, and returns accumulated benefits and costs for the strategy.
    :param individual: A NumPy array representing the actions taken by the town at each time period.
    :type individual: np.ndarray
    :param pars: A dictionary of parameters required for the simulation and calculations,
                 including "sim_length" and "delta".
    :type pars: Dict[str, Any]
    :param num_objectives: Number of objectives for evaluation, should be 2 or 5.
    :type num_objectives: int
    :return: A tuple containing the relevant accumulated metrics based on the number of objectives.
    :rtype: Tuple[float, float, ...]
    """
    # Validate number of objectives
    if num_objectives not in [2, 5]:
        return "Error: Current model has only been written for 2 and 5 objectives."

    initial_state = (0, 0, 0, 0)
    states_in_path = [initial_state]
    old_state = initial_state
    state_action = []
    sample_pars = pars

    for time_period in range(sample_pars["sim_length"] - 1):
        action = individual[time_period]
        combo = list(old_state) + [action]
        new_state = transition_observed_state(old_state, action, sample_pars)
        states_in_path.append(new_state)
        old_state = new_state
        state_action.append(combo)

    combo_final = old_state + [individual[-1]]
    state_action.append(combo_final)
    strategy_individual = np.array(state_action, dtype=object)

    # Calculate costs and benefits
    x_sow, V_sow, L_sow, E_sow = compute_coastal_variables(strategy_individual, pars)
    C_sow, nourish_cost_sow, relocate_cost_sow, damage_cost_sow = (
        compute_coastal_cost_metrics(
            strategy_individual, pars, x_sow, V_sow, L_sow, E_sow
        )
    )
    B_sow = compute_coastal_benefits(
        strategy_individual, pars, x_sow, V_sow, L_sow, E_sow
    )

    # Compute reliability and discounted values
    reliability_sow = np.count_nonzero(x_sow) / pars["sim_length"]
    discount_factor_sow = (1 + pars["delta"]) ** np.arange(pars["sim_length"])
    individual_benefits_sow = B_sow / discount_factor_sow
    individual_costs_sow = C_sow / discount_factor_sow
    individual_investment_costs_sow = (
        nourish_cost_sow + relocate_cost_sow
    ) / discount_factor_sow
    individual_damage_costs_sow = damage_cost_sow / discount_factor_sow
    individual_npv_sow = (B_sow - C_sow) / discount_factor_sow

    # Accumulate costs and benefits
    accumulated_costs_sow = np.cumsum(individual_costs_sow)[-1]
    accumulated_investment_costs_sow = np.cumsum(individual_investment_costs_sow)[-1]
    accumulated_damage_costs_sow = np.cumsum(individual_damage_costs_sow)[-1]
    accumulated_benefits_sow = np.cumsum(individual_benefits_sow)[-1]
    accumulated_npv_sow = np.cumsum(individual_npv_sow)[-1]

    total_costs_sum = accumulated_investment_costs_sow + accumulated_damage_costs_sow

    # Return based on number of objectives
    if num_objectives == 2:
        return (
            accumulated_benefits_sow,
            accumulated_costs_sow,
        )
    elif num_objectives == 5:
        return (
            accumulated_npv_sow,
            accumulated_benefits_sow,
            accumulated_investment_costs_sow,
            accumulated_damage_costs_sow,
            reliability_sow,
        )


def run_genetic_algorithm_on_configuration(
    parameter_set, guess_strategy, MU, LAMBDA, NGEN, CXPB, MUTPB
):
    """
    Runs a genetic algorithm configuration using the provided parameters.

    :param parameter_set: A dictionary containing simulation parameters.
    :param guess_strategy: An initial strategy to include in the population.
    :param MU: The population size.
    :param LAMBDA: The number of offspring to create.
    :param NGEN: The number of generations to evolve.
    :param CXPB: The probability of crossover.
    :param MUTPB: The probability of mutation.
    :return: A tuple containing the final population, hall of fame,
             hall of fame fitness, statistics, logbook, generations,
             and fitness data.
    """

    creator.create(
        "Fitness",
        base.Fitness,
        weights=(
            1.0,
            -1.0,
        ),
    )
    creator.create("Individual", list, fitness=creator.Fitness)

    size_of_individual = parameter_set["sim_length"]
    toolbox = base.Toolbox()

    toolbox.register(
        "individual", get_valid_ind, creator.Individual, size_of_individual
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    population = toolbox.population(n=MU)

    toolbox.register("evaluate", evaluate_pathway_sow, pars=parameter_set)
    toolbox.register("mate", crossover_list, n_time_steps=size_of_individual)
    toolbox.register("mutate", mutate_list, min_gap=parameter_set["minInterval"])
    toolbox.register("select", selNSGA2)

    cpu_count = multiprocessing.cpu_count()
    print(f"CPU count: {cpu_count}")

    pool = multiprocessing.Pool(cpu_count)
    toolbox.register("map", pool.map)

    print("Precomputation complete --- run genetic algorithm")

    pop, stats, hof, logbook, all_generations, all_fitness = genetic_algorithm(
        toolbox, guess_strategy, MU, LAMBDA, CXPB, MUTPB, NGEN, hack=True
    )

    pool.close()

    pool = multiprocessing.Pool(processes=cpu_count)
    async_results = [
        pool.apply_async(evaluate_pathway_sow, args=(i, parameter_set))
        for i in hof.items
    ]
    hof_fitness = [ar.get() for ar in async_results]

    pool.close()

    return pop, hof, hof_fitness, stats, logbook, all_generations, all_fitness
