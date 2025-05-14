from SALib.sample import latin
import numpy as np
from scipy.stats import beta
from typing import Tuple, Dict, Any
import os
import time
import pickle
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

    # Define the full path including the new folder inside results_data
    base_results_dir = os.path.join("..", "results_data", "baseline_optimization_runs")
    if not os.path.exists(base_results_dir):
        os.makedirs(base_results_dir)
    # Now the final directory including dir_name
    results_dir = os.path.join(base_results_dir, dir_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    # Save the samples in the specified directory
    np.save(os.path.join(results_dir, "lhs_samples.npy"), lhs_samples)
    return lhs_samples


def extract_states_of_the_world(dir_name):
    pwd = os.getcwd()
    base_path = os.path.join("..", "results_data", "baseline_optimization_runs")
    os.chdir(base_path)
    os.chdir(dir_name)
    param_samples = np.load("lhs_samples.npy")
    os.chdir(pwd)
    return param_samples


def evaluate_beach_nourishment_problem_on_strategy_best_guess_sow(
    individual: np.ndarray,
    pars: Dict[str, Any],  # New argument for number of objectives
) -> Tuple[float, float]:  # Adjusted return type for flexibility
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
    initial_state = (0, 0, 0, 0)
    time_horizon = pars["sim_length"]
    strategy_states = compute_state_action_transition(individual, initial_state, pars)
    (
        accumulated_npv_sow,
        accumulated_benefits_sow,
        accumulated_costs_sow,
        accumulated_investment_costs_sow,
        accumulated_damage_costs_sow,
        reliability_sow_new,
        bcr_sow_new,
        bcr_pass_fail,
    ) = beach_nourishment_problem(strategy_states, pars)
    return (
        accumulated_benefits_sow,
        accumulated_costs_sow,
    ), {
        "npv": accumulated_npv_sow,
        "damage_costs": accumulated_damage_costs_sow,
        "investment_costs": accumulated_investment_costs_sow,
        "reliability": reliability_sow_new,
        "bcr": bcr_sow_new,
        "bcr_pass_fail": bcr_pass_fail,
    }


def evaluate_beach_nourishment_problem_on_strategy_best_guess_sow_5_objectives(
    individual: np.ndarray,
    pars: Dict[str, Any],  # New argument for number of objectives
) -> Tuple[float, float]:  # Adjusted return type for flexibility
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
    initial_state = (0, 0, 0, 0)
    time_horizon = pars["sim_length"]
    strategy_states = compute_state_action_transition(individual, initial_state, pars)
    (
        accumulated_npv_sow,
        accumulated_benefits_sow,
        accumulated_costs_sow,
        accumulated_investment_costs_sow,
        accumulated_damage_costs_sow,
        reliability_sow_new,
        bcr_sow_new,
        bcr_pass_fail,
    ) = beach_nourishment_problem(strategy_states, pars)
    # Return based on number of objectives
    return (
        accumulated_npv_sow,
        accumulated_benefits_sow,
        accumulated_investment_costs_sow,
        accumulated_damage_costs_sow,
        reliability_sow_new,
    ), {
        "bcr": bcr_sow_new,
        "bcr_pass_fail": bcr_pass_fail,
    }


def compute_state_action_transition(strategy, initial_state, pars):
    """
    Compute the sequence of state-action pairs encountered by following a given strategy over a time horizon.

    This function takes an initial state and a strategy (sequence of actions), then iteratively computes
    the resulting states and pairs each state with the corresponding action taken. It returns an array of
    all state-action combinations over the specified time horizon.

    Parameters
    ----------
    strategy : list or array-like
        Sequence of actions to be followed at each time step. Length should be at least equal to time_horizon.
    initial_state : tuple or list
        The starting state of the system before any actions are taken.
    time_horizon : int
        The total number of time steps to simulate. The computed state-action sequence will have this length.

    Returns
    -------
    np.ndarray
        An array of shape (time_horizon, state_dimension + 1), where each row is a combination of a state vector
        concatenated with the action taken at that time step.
    """
    # Define initial state of the problem
    # initial_state = (0, 0, 0, 0)
    # This chunk below computes the state-action sequences encountered by following the action sequence in the strategy
    # Create an empty list to track all states that the strategy goes through
    states_in_path = []
    # Set the first state to be the initial state
    old_state = initial_state
    # Add the first state to the tracker
    states_in_path.append(initial_state)
    # Create an empty list of the state-action sequences that we'll have when we follow the strategy
    state_action = []
    # Loop through all the actions taken in the strategy and compute state-action combination sequences
    time_horizon = pars["sim_length"]
    for time_period in range(time_horizon - 1):
        # Find out what action is taken at time_period
        action = strategy[time_period]
        # Find the combination of state and action
        combo = list(old_state) + [action]
        # Use the state relationships to figure out what state we'll be in next
        new_state = transition_observed_state(old_state, action, pars)
        # Add new state to the list of states
        states_in_path.append(new_state)
        # Fix the previous state to compute the next state
        old_state = new_state
        # Add the state-action combination to the state-action combination sequence
        state_action.append(combo)
    # Find the final state-action combination
    combo_final = old_state + [strategy[-1]]
    # Add final combination to the state-action sequence tracker
    state_action.append(combo_final)
    # Find the last state after the action has taken place at the end of the time horizon
    final_state_after_action = state_action.copy()
    state_final = transition_observed_state(old_state, strategy[-1], pars)
    # Use this information
    state_final_combo = state_final + [0]
    final_state_after_action.append(state_final_combo)
    # Find the array of state action combinations procuced by the strategy
    strategy_states = np.array(final_state_after_action, dtype=object)
    return strategy_states


def beach_nourishment_problem(
    strategy_states,
    parameters,  # All parameters in the model
    b=0.0271 * 10**-3,  # Sea level rise co-efficient
    delta=0.035,  # Discount rate
    alpha=1.025,  # Co-efficient  : Influence of beach width on property valuation
    beta=0.975,  # Co-efficient : Infleunce of Sea Level on property valuation
    d=0,  # Development Rate
    A=669000,  # Initial Property Valuation
    W=5 * 10**5,  # Initial Valur of non structural value at risk
    epsilon=0.05,  # Co-efficient: Influence of increase in storm intensity with SLR
    xi=0.05,  # Co-efficient : Benefits from activities
    l=0.22,  # Tax rate in St. Lucie County
    eta=824,  # Land value ($1000/m)
    phi_conc=193.8357,  # Co-efficient : Concave Damage Function
):
    """ """
    # Assign values from sampling instead of default assignment
    parameters["b"] = b
    parameters["delta"] = delta
    parameters["alpha"] = alpha
    parameters["beta"] = beta
    parameters["d"] = d
    parameters["A"] = A
    parameters["W"] = W
    parameters["epsilon"] = epsilon
    parameters["xi"] = xi
    parameters["l"] = l
    parameters["eta"] = eta
    parameters["phi_conc"] = phi_conc
    # This chunk below computes the metrics of interest in our problem
    # Compute beach width, Property Valuation, SLE, expected erosion of strategy
    x_sow_new, V_sow_new, L_sow_new, E_sow_new = compute_coastal_state_variables(
        strategy_states, parameters
    )
    # Use information computed above to compute the costs total cost, nourishment cost, relocation cost and damage cost of
    # following the strategy in the current state of the world
    C_sow_new, nourish_cost_sow_new, relocate_cost_sow_new, damage_cost_sow_new = (
        compute_coastal_cost_metrics(
            strategy_states, parameters, x_sow_new, V_sow_new, L_sow_new, E_sow_new
        )
    )
    # Compute benefits due to recreation and development
    B_sow_new = compute_coastal_benefits(
        strategy_states, parameters, x_sow_new, V_sow_new, L_sow_new, E_sow_new
    )
    # Compute discount factor across the time horizon
    discount_factor_sow = (1 + parameters["delta"]) ** np.arange(
        parameters["sim_length"] + 1
    )
    # Compute new present value over time & the sum
    individual_npv_sow_new = (B_sow_new - C_sow_new) / discount_factor_sow
    accumulated_npv_sow = np.cumsum(individual_npv_sow_new)[-1]
    # Compute discounted benefits over time & sum them up
    individual_benefits_sow_new = B_sow_new / discount_factor_sow
    accumulated_benefits_sow = np.cumsum(individual_benefits_sow_new)[-1]
    # Compute total discounted costs over time and sum them up
    individual_costs_sow_new = C_sow_new / discount_factor_sow
    accumulated_costs_sow = np.cumsum(individual_costs_sow_new)[-1]
    # Compute individual investment costs over time
    individual_investment_costs_sow_new = (
        nourish_cost_sow_new + relocate_cost_sow_new
    ) / discount_factor_sow
    accumulated_investment_costs_sow = np.cumsum(individual_investment_costs_sow_new)[
        -1
    ]
    # Compute individual damage costs over time
    individual_damage_costs_sow_new = damage_cost_sow_new / discount_factor_sow
    accumulated_damage_costs_sow = np.cumsum(individual_damage_costs_sow_new)[-1]
    # Compute reliability of strategy for the state of the world - Number of time steps wherer there is a beach
    reliability_sow_new = np.count_nonzero(x_sow_new) / parameters["sim_length"]
    # Compute BCR of strategy in the state of the world
    bcr_sow_new = accumulated_benefits_sow / accumulated_costs_sow
    # Compute whether or not the BCR passes the 2.5 test
    if bcr_sow_new > 2.5:
        bcr_pass_fail = 1
    else:
        bcr_pass_fail = 0
    # Return all metrics of interest
    return (
        accumulated_npv_sow,
        accumulated_benefits_sow,
        accumulated_costs_sow,
        accumulated_investment_costs_sow,
        accumulated_damage_costs_sow,
        reliability_sow_new,
        bcr_sow_new,
        bcr_pass_fail,
    )


def calculate_beach_nourishment_metrics_strategy(
    strategy_states: np.ndarray, param_samples: np.ndarray, parameters: dict
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Runs the beach nourishment model across multiple Sobol parameter samples,
    computing outcome metrics for each sample.

    Parameters
    ----------
    strategy_states : np.ndarray
        The state-action sequence describing the strategy to evaluate.
    param_samples : np.ndarray
        A 2D array where each row is a sampled parameter set for the model.
        The columns correspond to parameters in the following order:
        b, epsilon, phi_conc, A, alpha, beta, d, eta, l, W, xi, delta.
    parameters : dict
        General model parameters dictionary used by the beach nourishment model.

    Returns
    -------
    Tuple of eight np.ndarray:
        npv_value_each_sow: NPV values over all parameter sets.
        benefits_each_sow: Total benefits for all samples.
        costs_each_sow: Total costs for all samples.
        investment_costs_each_sow: Investment costs for all samples.
        damage_costs_each_sow: Damage costs for all samples.
        reliability_each_sow: Reliability metric for all samples.
        bcr_each_sow: Benefit-Cost Ratio for all samples.
        bcr_pass_fail_each_sow: Binary indicator if BCR passes threshold, for all samples.
    """
    b_samples = param_samples[:, 0]
    epsilon_samples = param_samples[:, 1]
    phi_conc_samples = param_samples[:, 2]
    A_samples = param_samples[:, 3]
    alpha_samples = param_samples[:, 4]
    beta_samples = param_samples[:, 5]
    d_samples = param_samples[:, 6]
    eta_samples = param_samples[:, 7]
    l_samples = param_samples[:, 8]
    W_samples = param_samples[:, 9]
    xi_samples = param_samples[:, 10]
    delta_samples = param_samples[:, 11]
    # initialize arrays to store responses
    npv_value_each_sow = np.zeros(len(param_samples))
    benefits_each_sow = np.zeros(len(param_samples))
    costs_each_sow = np.zeros(len(param_samples))
    investment_costs_each_sow = np.zeros(len(param_samples))
    damage_costs_each_sow = np.zeros(len(param_samples))
    reliability_each_sow = np.zeros(len(param_samples))
    bcr_each_sow = np.zeros(len(param_samples))
    bcr_pass_fail_each_sow = np.zeros(len(param_samples))
    # run model across Sobol samples
    for i in range(0, len(param_samples)):
        # if i % 1000 == 0:
        #     print("Running sample " + str(i) + " of " + str(len(param_samples)))
        (
            npv_value_each_sow[i],
            benefits_each_sow[i],
            costs_each_sow[i],
            investment_costs_each_sow[i],
            damage_costs_each_sow[i],
            reliability_each_sow[i],
            bcr_each_sow[i],
            bcr_pass_fail_each_sow[i],
        ) = beach_nourishment_problem(
            strategy_states,
            parameters,  # All parameters in the model
            b=b_samples[i],  # Sea level rise co-efficient
            delta=delta_samples[i],  # Discount rate
            alpha=alpha_samples[
                i
            ],  # Co-efficient  : Influence of beach width on property valuation
            beta=beta_samples[
                i
            ],  # Co-efficient : Infleunce of Sea Level on property valuation
            d=d_samples[i],  # Development Rate
            A=A_samples[i],  # Initial Property Valuation
            W=W_samples[i],  # Initial Valur of non structural value at risk
            epsilon=epsilon_samples[
                i
            ],  # Co-efficient: Influence of increase in storm intensity with SLR
            xi=xi_samples[i],  # Co-efficient : Benefits from activities
            l=l_samples[i],  # Tax rate in St. Lucie County
            eta=eta_samples[i],  # Land value ($1000/m)
            phi_conc=phi_conc_samples[i],  # Co-efficient : Concave Damage Function
        )
    return (
        npv_value_each_sow,
        benefits_each_sow,
        costs_each_sow,
        investment_costs_each_sow,
        damage_costs_each_sow,
        reliability_each_sow,
        bcr_each_sow,
        bcr_pass_fail_each_sow,
    )


def stress_test_strategy_to_states_of_the_world(
    strategy: np.ndarray, initial_state: tuple, param_samples: np.ndarray, pars: dict
) -> Tuple[float, float, float, float, float, float, float, float]:
    """
    Evaluate the performance of a given strategy across multiple states of the world represented by parameter samples.

    Parameters
    ----------
    strategy : np.ndarray
        Array representing the sequence of actions (strategy) over time.
    initial_state : tuple
        The starting state before any actions are taken.
    param_samples : np.ndarray
        2D array where each row contains a set of sampled model parameters for simulation.
    pars : dict
        General parameter dictionary passed to the model functions.

    Returns
    -------
    Tuple containing aggregated metrics averaged across all sampled states of the world:
        - npv_across_sows : float
            Average net present value of the strategy.
        - benefits_across_sows : float
            Average accumulated benefits.
        - costs_across_sows : float
            Average accumulated costs.
        - investment_costs_across_sows : float
            Average investment costs.
        - damage_costs_across_sows : float
            Average damage costs.
        - reliability_across_sows : float
            Average reliability metric.
        - bcr_across_sows : float
            Average benefit-cost ratio.
        - sows_bcr_pass_number : float
            Fraction of states of the world where BCR passes the defined threshold.
    """

    time_horizon = len(strategy)

    # Compute the state-action sequences produced by the strategy
    strategy_states = compute_state_action_transition(strategy, initial_state, pars)

    # Calculate metrics for each sampled state of the world (parameter set)
    (
        npv_strategy_each_sow,
        benefits_strategy_each_sow,
        costs_strategy_each_sow,
        investment_costs_strategy_each_sow,
        damage_costs_strategy_each_sow,
        reliability_strategy_each_sow,
        bcr_strategy_each_sow,
        bcr_pass_fail_strategy_each_sow,
    ) = calculate_beach_nourishment_metrics_strategy(
        strategy_states, param_samples, pars
    )

    # Aggregate metrics by taking means over all parameter samples
    npv_across_sows = np.mean(npv_strategy_each_sow)
    benefits_across_sows = np.mean(benefits_strategy_each_sow)
    costs_across_sows = np.mean(costs_strategy_each_sow)
    investment_costs_across_sows = np.mean(investment_costs_strategy_each_sow)
    damage_costs_across_sows = np.mean(damage_costs_strategy_each_sow)
    reliability_across_sows = np.mean(reliability_strategy_each_sow)
    bcr_across_sows = np.mean(bcr_strategy_each_sow)

    # Proportion of states of the world passing the BCR threshold
    sows_bcr_pass_number = np.count_nonzero(bcr_pass_fail_strategy_each_sow) / len(
        bcr_pass_fail_strategy_each_sow
    )

    return (
        npv_across_sows,
        benefits_across_sows,
        costs_across_sows,
        investment_costs_across_sows,
        damage_costs_across_sows,
        reliability_across_sows,
        bcr_across_sows,
        sows_bcr_pass_number,
    )


def evaluate_individual(individual, param_samples, pars_dict, initial_state):
    """
    Evaluates a strategy individual across states of the world and returns fitness values and detailed metrics.

    Parameters
    ----------
    individual : np.ndarray
        Array representing the strategy (sequence of actions).
    param_samples : np.ndarray
        Parameter samples representing different states of the world.
    pars_dict : dict
        Dictionary of parameters required for the model.
    initial_state : tuple
        Initial system state before applying the strategy.

    Returns
    -------
    tuple
        (
            (benefits, costs),  # Fitness tuple: to be maximized/minimized accordingly
            dict with additional metrics:
                'npv', 'damage_costs', 'investment_costs', 'reliability', 'bcr', 'bcr_pass_fail'
        )
    """
    results = stress_test_strategy_to_states_of_the_world(
        individual, initial_state, param_samples, pars_dict
    )
    (
        npv,
        benefits,
        costs,
        investment_costs,
        damage_costs,
        reliability,
        bcr,
        bcr_pass_fail,
    ) = results

    return (benefits, costs), {
        "npv": npv,
        "damage_costs": damage_costs,
        "investment_costs": investment_costs,
        "reliability": reliability,
        "bcr": bcr,
        "bcr_pass_fail": bcr_pass_fail,
    }


def evaluate_individual_5_objectives(
    individual, param_samples, pars_dict, initial_state
):
    """
    Evaluates a strategy individual across states of the world and returns fitness values and detailed metrics.

    Parameters
    ----------
    individual : np.ndarray
        Array representing the strategy (sequence of actions).
    param_samples : np.ndarray
        Parameter samples representing different states of the world.
    pars_dict : dict
        Dictionary of parameters required for the model.
    initial_state : tuple
        Initial system state before applying the strategy.

    Returns
    -------
    tuple
        (
            (benefits, costs),  # Fitness tuple: to be maximized/minimized accordingly
            dict with additional metrics:
                'npv', 'damage_costs', 'investment_costs', 'reliability', 'bcr', 'bcr_pass_fail'
        )
    """
    results = stress_test_strategy_to_states_of_the_world(
        individual, initial_state, param_samples, pars_dict
    )
    (
        npv,
        benefits,
        costs,
        investment_costs,
        damage_costs,
        reliability,
        bcr,
        bcr_pass_fail,
    ) = results

    return (npv, benefits, investment_costs, damage_costs, reliability), {
        "costs": costs,
        "reliability": reliability,
        "bcr": bcr,
        "bcr_pass_fail": bcr_pass_fail,
    }


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

    toolbox.register(
        "evaluate",
        evaluate_beach_nourishment_problem_on_strategy_best_guess_sow,
        pars=parameter_set,
    )
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
        pool.apply_async(
            evaluate_beach_nourishment_problem_on_strategy_best_guess_sow,
            args=(i, parameter_set),
        )
        for i in hof.items
    ]
    hof_fitness = [ar.get() for ar in async_results]

    pool.close()

    return pop, hof, hof_fitness, stats, logbook, all_generations, all_fitness


def run_genetic_algorithm_on_configuration_5_objectives(
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
        weights=(1.0, 1.0, -1.0, -1.0, 1.0),
    )
    creator.create("Individual", list, fitness=creator.Fitness)

    size_of_individual = parameter_set["sim_length"]
    toolbox = base.Toolbox()

    toolbox.register(
        "individual", get_valid_ind, creator.Individual, size_of_individual
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    population = toolbox.population(n=MU)

    toolbox.register(
        "evaluate",
        evaluate_beach_nourishment_problem_on_strategy_best_guess_sow_5_objectives,
        pars=parameter_set,
    )
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
        pool.apply_async(
            evaluate_beach_nourishment_problem_on_strategy_best_guess_sow,
            args=(i, parameter_set),
        )
        for i in hof.items
    ]
    hof_fitness = [ar.get() for ar in async_results]

    pool.close()

    return pop, hof, hof_fitness, stats, logbook, all_generations, all_fitness


def run_genetic_algorithm_across_sows(
    param_samples: np.ndarray,
    pars_dict: Dict[str, Any],
    initial_state: tuple,
    strategy_seeds: List,
    MU: int,
    LAMBDA: int,
    NGEN: int,
    CXPB: float,
    MUTPB: float,
) -> Tuple:
    """
    Runs a genetic algorithm to optimize strategies across multiple states of the world (SOWs).

    Parameters
    ----------
    param_samples : np.ndarray
        Parameter samples defining the different states of the world for evaluation.
    pars_dict : dict
        Dictionary containing model parameters.
    initial_state : tuple
        Initial state before applying strategies.
    strategy_seeds : list
        List of initial guess strategies to seed the population.
    MU : int
        Population size.
    LAMBDA : int
        Number of offspring per generation.
    NGEN : int
        Number of generations for evolution.
    CXPB : float
        Crossover probability.
    MUTPB : float
        Mutation probability.

    Returns
    -------
    tuple
        (population, hall_of_fame, hall_of_fame_fitness, hall_of_fame_extra_info, stats, logbook, all_generations, all_fitness)
    """

    size_of_individual = pars_dict["sim_length"]

    # Define the custom fitness and individual classes
    # Fitness: maximize benefits (weight 1.0), minimize costs (weight -1.0)
    creator.create("Fitness", base.Fitness, weights=(1.0, -1.0))

    # Individual: use list with attached fitness & extra_info dictionary
    creator.create("Individual", list, fitness=creator.Fitness, extra_info={})

    toolbox = base.Toolbox()

    # Register individual generator (assuming get_valid_ind returns a valid strategy)
    toolbox.register(
        "individual", get_valid_ind, creator.Individual, size_of_individual
    )

    # Initialize population
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    population = toolbox.population(n=MU)

    # Replace the last MU individuals with seeds
    for guess_strategy in strategy_seeds:
        population.pop()
        population.insert(0, guess_strategy)

    # Register evaluation function with fixed parameters
    toolbox.register(
        "evaluate",
        evaluate_individual,
        param_samples=param_samples,
        pars_dict=pars_dict,
        initial_state=initial_state,
    )

    # Register genetic operators
    toolbox.register("mate", crossover_list, n_time_steps=size_of_individual)
    toolbox.register("mutate", mutate_list, min_gap=pars_dict.get("minInterval", 1))
    toolbox.register("select", tools.selNSGA2)

    # Setup multiprocessing pool, leaving 4 CPUs free
    cpu_count = max(multiprocessing.cpu_count() - 4, 1)
    print(f"Using {cpu_count} CPUs for multiprocessing")
    pool = multiprocessing.Pool(cpu_count)
    toolbox.register("map", pool.map)

    print("Precomputation complete --- running genetic algorithm...")

    # Run genetic algorithm (assumes you have a custom 'genetic_algorithm' function)
    pop, stats, hof, logbook, all_generations, all_fitness = genetic_algorithm(
        toolbox, strategy_seeds, MU, LAMBDA, CXPB, MUTPB, NGEN, hack=True
    )

    # Close multiprocessing pool
    pool.close()
    pool.join()

    # Extract fitness values and extra info from hall of fame individuals
    hof_fitness = [ind.fitness.values for ind in hof]
    hof_extra_info = [getattr(ind, "extra_info", {}) for ind in hof]

    return (
        pop,
        hof,
        hof_fitness,
        hof_extra_info,
        stats,
        logbook,
        all_generations,
        all_fitness,
    )


def run_genetic_algorithm_across_sows_5_objectives(
    param_samples: np.ndarray,
    pars_dict: Dict[str, Any],
    initial_state: tuple,
    strategy_seeds: List,
    MU: int,
    LAMBDA: int,
    NGEN: int,
    CXPB: float,
    MUTPB: float,
) -> Tuple:
    """
    Runs a genetic algorithm to optimize strategies across multiple states of the world (SOWs).

    Parameters
    ----------
    param_samples : np.ndarray
        Parameter samples defining the different states of the world for evaluation.
    pars_dict : dict
        Dictionary containing model parameters.
    initial_state : tuple
        Initial state before applying strategies.
    strategy_seeds : list
        List of initial guess strategies to seed the population.
    MU : int
        Population size.
    LAMBDA : int
        Number of offspring per generation.
    NGEN : int
        Number of generations for evolution.
    CXPB : float
        Crossover probability.
    MUTPB : float
        Mutation probability.

    Returns
    -------
    tuple
        (population, hall_of_fame, hall_of_fame_fitness, hall_of_fame_extra_info, stats, logbook, all_generations, all_fitness)
    """

    size_of_individual = pars_dict["sim_length"]

    # Define the custom fitness and individual classes
    # Fitness: maximize benefits (weight 1.0), minimize costs (weight -1.0)
    creator.create("Fitness", base.Fitness, weights=(1.0, 1.0, -1.0, -1.0, 1.0))

    # Individual: use list with attached fitness & extra_info dictionary
    creator.create("Individual", list, fitness=creator.Fitness, extra_info={})

    toolbox = base.Toolbox()

    # Register individual generator (assuming get_valid_ind returns a valid strategy)
    toolbox.register(
        "individual", get_valid_ind, creator.Individual, size_of_individual
    )

    # Initialize population
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    population = toolbox.population(n=MU)

    # Replace the last MU individuals with seeds
    for guess_strategy in strategy_seeds:
        population.pop()
        population.insert(0, guess_strategy)

    # Register evaluation function with fixed parameters
    toolbox.register(
        "evaluate",
        evaluate_individual_5_objectives,
        param_samples=param_samples,
        pars_dict=pars_dict,
        initial_state=initial_state,
    )

    # Register genetic operators
    toolbox.register("mate", crossover_list, n_time_steps=size_of_individual)
    toolbox.register("mutate", mutate_list, min_gap=pars_dict.get("minInterval", 1))
    toolbox.register("select", tools.selNSGA2)

    # Setup multiprocessing pool, leaving 4 CPUs free
    cpu_count = max(multiprocessing.cpu_count() - 4, 1)
    print(f"Using {cpu_count} CPUs for multiprocessing")
    pool = multiprocessing.Pool(cpu_count)
    toolbox.register("map", pool.map)

    print("Precomputation complete --- running genetic algorithm...")

    # Run genetic algorithm (assumes you have a custom 'genetic_algorithm' function)
    pop, stats, hof, logbook, all_generations, all_fitness = genetic_algorithm(
        toolbox, strategy_seeds, MU, LAMBDA, CXPB, MUTPB, NGEN, hack=True
    )

    # Close multiprocessing pool
    pool.close()
    pool.join()

    # Extract fitness values and extra info from hall of fame individuals
    hof_fitness = [ind.fitness.values for ind in hof]
    hof_extra_info = [getattr(ind, "extra_info", {}) for ind in hof]

    return (
        pop,
        hof,
        hof_fitness,
        hof_extra_info,
        stats,
        logbook,
        all_generations,
        all_fitness,
    )


def run_baseline_runs_2objectives(
    dir_name,
    param_samples,
    pars,
    initial_state,
    MU: int,
    LAMBDA: int,
    NGEN: int,
    CXPB: float,
    MUTPB: float,
    seeds="Yes",
):
    nourish_whenever_possible = [
        1 if i % 2 == 1 else 0 for i in range(pars["sim_length"])
    ]

    othing_always = [0] * pars["sim_length"]
    pwd = os.getcwd()
    (
        npv_nourish_when_possible,
        benefits_nourish_when_possible,
        costs_nourish_when_possible,
        investment_costs_nourish_when_possible,
        damage_costs_nourish_when_possible,
        reliability_nourish_when_possible,
        bcr_nourish_when_possible,
        sows_bcr_pass_nourish_when_possible,
    ) = stress_test_strategy_to_states_of_the_world(
        nourish_whenever_possible, initial_state, param_samples, pars
    )
    (
        npv_do_nothing_always,
        benefits_do_nothing_always,
        costs_do_nothing_always,
        investment_costs_do_nothing_always,
        damage_costs_do_nothing_always,
        reliability_do_nothing_always,
        bcr_do_nothing_always,
        sows_bcr_pass_do_nothing_always,
    ) = stress_test_strategy_to_states_of_the_world(
        do_nothing_always, initial_state, param_samples, pars
    )
    if seeds == "Yes":
        strategy_seeds = []
        strategy_seeds.append(nourish_whenever_possible)
        strategy_seeds.append(do_nothing_always)
        strategy_seeds.append(
            [
                0,
                0,
                0,
                1,
                0,
                0,
                1,
                0,
                0,
                1,
                0,
                0,
                1,
                0,
                0,
                1,
                0,
                0,
                1,
                0,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
            ]
        )
    else:
        strategy_seeds = []
    print("#########################################################")
    print("Running two objective optimization neglecting uncertainty")
    print("#########################################################")
    start_time = time.time()
    # Go one folder above the current directory
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
    # Define the path to 'results_data/baseline_optimization_runs'
    target_dir = os.path.join(parent_dir, "results_data", "baseline_optimization_runs")
    # Define the folder to check/create
    final_folder = "2objective_no_uncertainty"
    final_path = os.path.join(target_dir, final_folder)
    # Create '2objective_no_uncertainty' if it doesn't exist
    if not os.path.exists(final_path):
        os.makedirs(final_path)
    # Change current working directory to the final path
    os.chdir(final_path)
    print("Current directory:", os.getcwd())
    pop, hof, hof_fitness, stats, logbook, all_generations, all_fitness = (
        run_genetic_algorithm_on_configuration(
            pars, strategy_seeds, MU, LAMBDA, NGEN, CXPB, MUTPB
        )
    )
    with open("pop.pkl", "wb") as file:
        pickle.dump(pop, file)

    with open("hof.pkl", "wb") as file:
        pickle.dump(hof, file)

    with open("hof_fitness.pkl", "wb") as file:
        pickle.dump(hof_fitness, file)

    # with open("stats.pkl", "wb") as file:
    #     pickle.dump(stats, file)

    with open("logbook.pkl", "wb") as file:
        pickle.dump(logbook, file)

    with open("all_generations.pkl", "wb") as file:
        pickle.dump(all_generations, file)

    with open("all_fitness.pkl", "wb") as file:
        pickle.dump(all_fitness, file)
    os.chdir(pwd)
    end_time = time.time()
    time_diff = end_time - start_time
    print("Time to run five objective optimization neglecting uncertainty:", time_diff)
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    print("Finished running two objective optimization neglecting uncertainty")
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    print("Transitioning......Transitioning......Transitioning........")
    print("##########################################################")
    print("Running two objective optimization considering uncertainty")
    print("##########################################################")
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
    target_dir = os.path.join(parent_dir, "results_data", "baseline_optimization_runs")
    # Define the folder to check/create
    final_folder = "2objective_considering_uncertainty"
    final_path = os.path.join(target_dir, final_folder)
    # Create '2objective_no_uncertainty' if it doesn't exist
    if not os.path.exists(final_path):
        os.makedirs(final_path)
    # Change current working directory to the final path
    os.chdir(final_path)
    print("Current directory:", os.getcwd())
    start_time = time.time()
    (
        pop_across_sow,
        hof_across_sow,
        hof_fitness_across_sow,
        hof_extra_across_sow,
        stats_across_sow,
        logbook_across_sow,
        all_generations_across_sow,
        all_fitness_across_sow,
    ) = run_genetic_algorithm_across_sows(
        param_samples,
        pars,
        initial_state,
        strategy_seeds,
        MU,
        LAMBDA,
        NGEN,
        CXPB,
        MUTPB,
    )
    end_time = time.time()
    time_diff = end_time - start_time
    # 1. Save population across SoWs
    with open("pop_across_sow.pkl", "wb") as file:
        pickle.dump(pop_across_sow, file)
    # 2. Save Hall of Fame across SoWs
    with open("hof_across_sow.pkl", "wb") as file:
        pickle.dump(hof_across_sow, file)
    # 3. Save Hall of Fame fitness across SoWs
    with open("hof_fitness_across_sow.pkl", "wb") as file:
        pickle.dump(hof_fitness_across_sow, file)
    # 4. Save extra information related to Hall of Fame
    with open("hof_extra_across_sow.pkl", "wb") as file:
        pickle.dump(hof_extra_across_sow, file)
    # 5. Save statistics across SoWs
    # with open('stats_across_sow.pkl', 'wb') as file:
    #     pickle.dump(stats_across_sow, file)
    # 6. Save logbook across SoWs
    with open("logbook_across_sow.pkl", "wb") as file:
        pickle.dump(logbook_across_sow, file)
    # 7. Save all generations across SoWs
    with open("all_generations_across_sow.pkl", "wb") as file:
        pickle.dump(all_generations_across_sow, file)
    # 8. Save all fitness across SoWs
    with open("all_fitness_across_sow.pkl", "wb") as file:
        pickle.dump(all_fitness_across_sow, file)
    output_data_nourishment = {
        "npv_nourish_when_possible": npv_nourish_when_possible,
        "benefits_nourish_when_possible": benefits_nourish_when_possible,
        "costs_nourish_when_possible": costs_nourish_when_possible,
        "investment_costs_nourish_when_possible": investment_costs_nourish_when_possible,
        "damage_costs_nourish_when_possible": damage_costs_nourish_when_possible,
        "reliability_nourish_when_possible": reliability_nourish_when_possible,
        "bcr_nourish_when_possible": bcr_nourish_when_possible,
        "sows_bcr_pass_nourish_when_possible": sows_bcr_pass_nourish_when_possible,
    }
    do_nothing_always_data = {
        "npv_do_nothing_always": npv_do_nothing_always,
        "benefits_do_nothing_always": benefits_do_nothing_always,
        "costs_do_nothing_always": costs_do_nothing_always,
        "investment_costs_do_nothing_always": investment_costs_do_nothing_always,
        "damage_costs_do_nothing_always": damage_costs_do_nothing_always,
        "reliability_do_nothing_always": reliability_do_nothing_always,
        "bcr_do_nothing_always": bcr_do_nothing_always,
        "sows_bcr_pass_do_nothing_always": sows_bcr_pass_do_nothing_always,
    }
    with open("nourish_when_possible.pkl", "wb") as file:
        pickle.dump(output_data_nourishment, file)
    with open("do_nothing.pkl", "wb") as file:
        pickle.dump(do_nothing_always_data, file)
    os.chdir(pwd)
    print("Time to run two objective optimization considering uncertainty:", time_diff)
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    print("Finished running two objective optimization considering uncertainty")
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    # print("Running five objective objective optimization neglecting uncertainty")
    # print("Running five objective objective optimization considering uncertainty")
    return MU, LAMBDA, NGEN, CXPB, MUTPB


def run_baseline_runs_5objectives(
    dir_name,
    param_samples,
    pars,
    initial_state,
    MU: int,
    LAMBDA: int,
    NGEN: int,
    CXPB: float,
    MUTPB: float,
    seeds="Yes",
):
    nourish_whenever_possible = [
        1 if i % 2 == 1 else 0 for i in range(pars["sim_length"])
    ]
    do_nothing_always = [0] * pars["sim_length"]
    pwd = os.getcwd()
    (
        npv_nourish_when_possible,
        benefits_nourish_when_possible,
        costs_nourish_when_possible,
        investment_costs_nourish_when_possible,
        damage_costs_nourish_when_possible,
        reliability_nourish_when_possible,
        bcr_nourish_when_possible,
        sows_bcr_pass_nourish_when_possible,
    ) = stress_test_strategy_to_states_of_the_world(
        nourish_whenever_possible, initial_state, param_samples, pars
    )
    (
        npv_do_nothing_always,
        benefits_do_nothing_always,
        costs_do_nothing_always,
        investment_costs_do_nothing_always,
        damage_costs_do_nothing_always,
        reliability_do_nothing_always,
        bcr_do_nothing_always,
        sows_bcr_pass_do_nothing_always,
    ) = stress_test_strategy_to_states_of_the_world(
        do_nothing_always, initial_state, param_samples, pars
    )
    if seeds == "Yes":
        strategy_seeds = []
        strategy_seeds.append(nourish_whenever_possible)
        strategy_seeds.append(do_nothing_always)
    else:
        strategy_seeds = []
    print("#########################################################")
    print("Running five objective optimization neglecting uncertainty")
    print("#########################################################")
    start_time = time.time()
    # Go one folder above the current directory
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
    # Define the path to 'results_data/baseline_optimization_runs'
    target_dir = os.path.join(parent_dir, "results_data", "baseline_optimization_runs")
    # Define the folder to check/create
    final_folder = "5objective_no_uncertainty"
    final_path = os.path.join(target_dir, final_folder)
    # Create '2objective_no_uncertainty' if it doesn't exist
    if not os.path.exists(final_path):
        os.makedirs(final_path)
    # Change current working directory to the final path
    os.chdir(final_path)
    print("Current directory:", os.getcwd())
    pop, hof, hof_fitness, stats, logbook, all_generations, all_fitness = (
        run_genetic_algorithm_on_configuration_5_objectives(
            pars, strategy_seeds, MU, LAMBDA, NGEN, CXPB, MUTPB
        )
    )
    with open("pop.pkl", "wb") as file:
        pickle.dump(pop, file)

    with open("hof.pkl", "wb") as file:
        pickle.dump(hof, file)

    with open("hof_fitness.pkl", "wb") as file:
        pickle.dump(hof_fitness, file)

    # with open("stats.pkl", "wb") as file:
    #     pickle.dump(stats, file)

    with open("logbook.pkl", "wb") as file:
        pickle.dump(logbook, file)

    with open("all_generations.pkl", "wb") as file:
        pickle.dump(all_generations, file)

    with open("all_fitness.pkl", "wb") as file:
        pickle.dump(all_fitness, file)
    os.chdir(pwd)
    end_time = time.time()
    time_diff = end_time - start_time
    print("Time to run five objective optimization neglecting uncertainty:", time_diff)
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    print("Finished running five objective optimization neglecting uncertainty")
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    print("Transitioning......Transitioning......Transitioning........")
    print("##########################################################")
    print("Running five objective optimization considering uncertainty")
    print("##########################################################")
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
    target_dir = os.path.join(parent_dir, "results_data", "baseline_optimization_runs")
    # Define the folder to check/create
    final_folder = "5objective_considering_uncertainty"
    final_path = os.path.join(target_dir, final_folder)
    # Create '2objective_no_uncertainty' if it doesn't exist
    if not os.path.exists(final_path):
        os.makedirs(final_path)
    # Change current working directory to the final path
    os.chdir(final_path)
    print("Current directory:", os.getcwd())
    start_time = time.time()
    (
        pop_across_sow,
        hof_across_sow,
        hof_fitness_across_sow,
        hof_extra_across_sow,
        stats_across_sow,
        logbook_across_sow,
        all_generations_across_sow,
        all_fitness_across_sow,
    ) = run_genetic_algorithm_across_sows_5_objectives(
        param_samples,
        pars,
        initial_state,
        strategy_seeds,
        MU,
        LAMBDA,
        NGEN,
        CXPB,
        MUTPB,
    )
    end_time = time.time()
    time_diff = end_time - start_time
    # 1. Save population across SoWs
    with open("pop_across_sow.pkl", "wb") as file:
        pickle.dump(pop_across_sow, file)
    # 2. Save Hall of Fame across SoWs
    with open("hof_across_sow.pkl", "wb") as file:
        pickle.dump(hof_across_sow, file)
    # 3. Save Hall of Fame fitness across SoWs
    with open("hof_fitness_across_sow.pkl", "wb") as file:
        pickle.dump(hof_fitness_across_sow, file)
    # 4. Save extra information related to Hall of Fame
    with open("hof_extra_across_sow.pkl", "wb") as file:
        pickle.dump(hof_extra_across_sow, file)
    # 5. Save statistics across SoWs
    # with open('stats_across_sow.pkl', 'wb') as file:
    #     pickle.dump(stats_across_sow, file)
    # 6. Save logbook across SoWs
    with open("logbook_across_sow.pkl", "wb") as file:
        pickle.dump(logbook_across_sow, file)
    # 7. Save all generations across SoWs
    with open("all_generations_across_sow.pkl", "wb") as file:
        pickle.dump(all_generations_across_sow, file)
    # 8. Save all fitness across SoWs
    with open("all_fitness_across_sow.pkl", "wb") as file:
        pickle.dump(all_fitness_across_sow, file)
    output_data_nourishment = {
        "npv_nourish_when_possible": npv_nourish_when_possible,
        "benefits_nourish_when_possible": benefits_nourish_when_possible,
        "costs_nourish_when_possible": costs_nourish_when_possible,
        "investment_costs_nourish_when_possible": investment_costs_nourish_when_possible,
        "damage_costs_nourish_when_possible": damage_costs_nourish_when_possible,
        "reliability_nourish_when_possible": reliability_nourish_when_possible,
        "bcr_nourish_when_possible": bcr_nourish_when_possible,
        "sows_bcr_pass_nourish_when_possible": sows_bcr_pass_nourish_when_possible,
    }
    do_nothing_always_data = {
        "npv_do_nothing_always": npv_do_nothing_always,
        "benefits_do_nothing_always": benefits_do_nothing_always,
        "costs_do_nothing_always": costs_do_nothing_always,
        "investment_costs_do_nothing_always": investment_costs_do_nothing_always,
        "damage_costs_do_nothing_always": damage_costs_do_nothing_always,
        "reliability_do_nothing_always": reliability_do_nothing_always,
        "bcr_do_nothing_always": bcr_do_nothing_always,
        "sows_bcr_pass_do_nothing_always": sows_bcr_pass_do_nothing_always,
    }
    with open("nourish_when_possible.pkl", "wb") as file:
        pickle.dump(output_data_nourishment, file)
    with open("do_nothing.pkl", "wb") as file:
        pickle.dump(do_nothing_always_data, file)
    os.chdir(pwd)
    print("Time to run five objective optimization neglecting uncertainty:", time_diff)
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    print("Finished running five objective optimization considering uncertainty")
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    # print("Running five objective objective optimization neglecting uncertainty")
    # print("Running five objective objective optimization considering uncertainty")
    return MU, LAMBDA, NGEN, CXPB, MUTPB


# def run_baseline_runs_5objectives(
#     dir_name,
#     param_samples,
#     pars,
#     initial_state,
#     MU: int,
#     LAMBDA: int,
#     NGEN: int,
#     CXPB: float,
#     MUTPB: float,
#     seeds="Yes",
# ):
#     nourish_whenever_possible = [
#         1 if i % 2 == 1 else 0 for i in range(pars["sim_length"])
#     ]
#     do_nothing_always = [0] * pars["sim_length"]
#     pwd = os.getcwd()
#     (
#         npv_nourish_when_possible,
#         benefits_nourish_when_possible,
#         costs_nourish_when_possible,
#         investment_costs_nourish_when_possible,
#         damage_costs_nourish_when_possible,
#         reliability_nourish_when_possible,
#         bcr_nourish_when_possible,
#         sows_bcr_pass_nourish_when_possible,
#     ) = stress_test_strategy_to_states_of_the_world(
#         nourish_whenever_possible, initial_state, param_samples, pars
#     )
#     (
#         npv_do_nothing_always,
#         benefits_do_nothing_always,
#         costs_do_nothing_always,
#         investment_costs_do_nothing_always,
#         damage_costs_do_nothing_always,
#         reliability_do_nothing_always,
#         bcr_do_nothing_always,
#         sows_bcr_pass_do_nothing_always,
#     ) = stress_test_strategy_to_states_of_the_world(
#         do_nothing_always, initial_state, param_samples, pars
#     )
#     if seeds == "Yes":
#         strategy_seeds = []
#         strategy_seeds.append(nourish_whenever_possible)
#         strategy_seeds.append(do_nothing_always)
#     else:
#         strategy_seeds = []
#     print("#########################################################")
#     print("Running five objective optimization neglecting uncertainty")
#     print("#########################################################")
#     start_time = time.time()
#     # Go one folder above the current directory
#     parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
#     # Define the path to 'results_data/baseline_optimization_runs'
#     target_dir = os.path.join(parent_dir, "results_data", "baseline_optimization_runs")
#     # Define the folder to check/create
#     final_folder = "2objective_no_uncertainty"
#     final_path = os.path.join(target_dir, final_folder)
#     # Create '2objective_no_uncertainty' if it doesn't exist
#     if not os.path.exists(final_path):
#         os.makedirs(final_path)
#     # Change current working directory to the final path
#     os.chdir(final_path)
#     print("Current directory:", os.getcwd())
#     pop, hof, hof_fitness, stats, logbook, all_generations, all_fitness = (
#         run_genetic_algorithm_on_configuration_5_objectives(
#             parameter_set, guess_strategy, MU, LAMBDA, NGEN, CXPB, MUTPB
#         )
#     )
#     end_time = time.time()
#     total_time = end_time - start_time
#     os.chdir(pwd)
#     print("Time to run five objective optimization neglecting uncertainty:", total_time)
#     print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
#     print("Finished running five objective optimization neglecting uncertainty")
#     print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
#     print("Transitioning......Transitioning......Transitioning........")
#     print("##########################################################")
#     print("Running five objective optimization considering uncertainty")
#     print("##########################################################")
#     print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
#     print("Finished running five objective optimization considering uncertainty")
#     print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
#     # print("Running five objective objective optimization neglecting uncertainty")
#     # print("Running five objective objective optimization considering uncertainty")
#     return MU, LAMBDA, NGEN, CXPB, MUTPB
