import sys
import os

sys.path.insert(0, "../src")  # Adjust the path to import from the src directory
from mordm_functions import *
from decision_benchmarks import *
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
from tqdm import tqdm
import random
import numpy as np
import pickle

dir_name = "100_time_steps_concave_b_beta_SALib_10000_sow"
# number_of_sows = 10000
# lhs_samples = generate_beach_nourishment_parameters_lhs_samples(
#     dir_name, number_of_sows
# )
all_parameters = model_instance_with_best_guess_values_of_uncertainties(1)
base_path = os.path.join("..", "results_data")
# File paths
file1 = os.path.join(base_path, "npv_maximizing_strategy.pickle")
file2 = os.path.join(base_path, "usace_strategy.pickle")

# Reading the pickle files
with open(file1, "rb") as f:
    npv_maximizing_strategy = pickle.load(f)

with open(file2, "rb") as f:
    usace_strategy = pickle.load(f)

print("NPV mazimizing Strategy", npv_maximizing_strategy)
print("USACE Strategy", usace_strategy)
max_npv_benefits, max_npv_costs = (
    evaluate_beach_bourishment_problem_on_strategy_best_guess_sow(
        npv_maximizing_strategy, all_parameters, 2
    )
)
print("Max NPV benefits:", max_npv_benefits, "Max NPV costs:", max_npv_costs)
usace_benefits, usace_costs = (
    evaluate_beach_bourishment_problem_on_strategy_best_guess_sow(
        usace_strategy, all_parameters, 2
    )
)
print("USACE benefits:", usace_benefits, "USACE costs:", usace_costs)
