import sys
import os

sys.path.insert(0, "../src")  # Adjust the path to import from the src director
from plot_paper_figures import *
from mordm_functions import *
from decision_benchmarks import *

baseline_directory = "100_time_steps_concave_b_beta_SALib_10000_sow"
uncertain_param_samples = extract_states_of_the_world(baseline_directory)
all_parameters = model_instance_with_best_guess_values_of_uncertainties(1)
initial_state = (0, 0, 0, 0)


(
    max_npv_benefits_across_sow,
    max_npv_costs_across_sow,
    usace_benefits_across_sow,
    usace_costs_across_sow,
) = compare_2d_pareto_fronts_with_and_without_uncertainty(
    baseline_directory, uncertain_param_samples, all_parameters, initial_state
)
