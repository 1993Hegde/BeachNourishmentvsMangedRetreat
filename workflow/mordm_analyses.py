import sys

sys.path.insert(0, "../src")  # Adjust the path to import from the src directory
from mordm_functions import *


dir_name = "100_time_steps_concave_b_beta_SALib_10000_sow"
number_of_sows = 10000
lhs_samples = generate_beach_nourishment_parameters_lhs_samples(
    dir_name, number_of_sows
)
