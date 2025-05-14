import sys
import os

sys.path.insert(0, "../src")  # Adjust the path to import from the src director
from plot_paper_figures import *
from mordm_functions import *
from decision_benchmarks import *

baseline_directory = "100_time_steps_concave_b_beta_SALib_10000_sow"
uncertain_param_threshold_satisficings = extract_states_of_the_world(baseline_directory)
all_parameters = model_instance_with_best_guess_values_of_uncertainties(1)
initial_state = (0, 0, 0, 0)


(npv_maximizing_strategy, usace_strategy) = (
    compare_2d_pareto_fronts_with_and_without_uncertainty(
        baseline_directory,
        uncertain_param_threshold_satisficings,
        all_parameters,
        initial_state,
    )
)
# generate_multi_axis_parallel_plots_with_satisficing_strategies(baseline_directory)
threshold_satisficing_strategy, robustness_satisficing_strategy = (
    identify_satisfycing_strategies(baseline_directory)
)
threshold_satisficing_strategy_states = compute_state_action_transition(
    threshold_satisficing_strategy, initial_state, all_parameters
)
npv_maximizing_strategy_states = compute_state_action_transition(
    npv_maximizing_strategy, initial_state, all_parameters
)
usace_strategy_states = compute_state_action_transition(
    usace_strategy, initial_state, all_parameters
)
robustness_satisficing_strategy_states = compute_state_action_transition(
    robustness_satisficing_strategy, initial_state, all_parameters
)
(
    SA_npv_usace,
    SA_reliability_usace,
    SA_costs_usace,
    SA_benefits_usace,
    SA_npv_npvmax,
    SA_reliability_npvmax,
    SA_costs_npvmax,
    SA_benefits_npvmax,
    SA_costs_threshold_satisficing,
    SA_benefits_threshold_satisficing,
    SA_npv_threshold_satisficing,
    SA_reliability_threshold_satisficing,
) = compute_sobol_for_different_metrics(
    usace_strategy_states,
    npv_maximizing_strategy_states,
    threshold_satisficing_strategy_states,
    all_parameters,
)

plot_radial_chord_plot_npv_reliability(
    SA_npv_threshold_satisficing,
    SA_reliability_threshold_satisficing,
    SA_npv_usace,
    SA_reliability_usace,
)
plot_strategies(
    usace_strategy,
    npv_maximizing_strategy,
    threshold_satisficing_strategy,
    robustness_satisficing_strategy,
)
plot_decision_convergence(baseline_directory)
orig = os.getcwd()
os.chdir(os.path.join("..", "results_data"))

with (
    open("threshold_satisficing_strategy.pickle", "wb") as f1,
    open("robustness_satisficing_strategy.pickle", "wb") as f2,
):
    pickle.dump(threshold_satisficing_strategy, f1)
    pickle.dump(robustness_satisficing_strategy, f2)

os.chdir(orig)
