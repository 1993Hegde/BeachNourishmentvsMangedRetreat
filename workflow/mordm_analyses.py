import sys

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
import sys

dir_name = "100_time_steps_concave_b_beta_SALib_10000_sow"
number_of_sows = 10000
lhs_samples = generate_beach_nourishment_parameters_lhs_samples(
    dir_name, number_of_sows
)
