import sys

sys.path.insert(0, "../src")  # Adjust the path to import from the src directory
from decision_benchmarks import *
import os
import json

slr_scenario = 1
pars = baseline_model_instance_with_default_parameters(slr_scenario)
initial_state = [0, 0, 0, 0]
# Find the strategy that maximizes for NPV neglecting uncertainty
(
    optS_npvmax,
    x_final_npvmax,
    v_final_npvmax,
    L_final_npvmax,
    C_final_npvmax,
    B_final_npvmax,
    accumulated_npv_npvmax,
    strategy_npvmax,
) = solve_cutler_et_al_ddp(pars, initial_state)
npv_maximizing_strategy = optS_npvmax[:, 4]
# Mimic the strategy used by the USACE
# slr_scenario = 0 ; discount rate = 3.5% ; horizon = 100 years
# Actual horizon is 50 years. However, we want to use the same horizon for comparison
slr_scenario = 0
pars = baseline_model_instance_with_default_parameters(slr_scenario)
pars["delta"] = 0.035  # Discount rate of 3.5%
# Find approximation of USACE startegy
(
    optS_usace,
    x_final_usace,
    v_final_usace,
    L_final_usace,
    C_final_usace,
    B_final_usace,
    accumulated_npv_usace,
    strategy_usace,
) = solve_army_corps_bcr_max(pars, initial_state)
usace_strategy = optS_usace[:, 4]
# Construct the paths: move one folder up, then into 'results_data'
# Two lines to ensure the 'results_data' directory one level up exists:
folder_path = os.path.join("..", "results_data")
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
npv_file_path = os.path.join("..", "results_data", "npv_maximizing_strategy.json")
usace_file_path = os.path.join("..", "results_data", "usace_strategy.json")
# Save each list as a JSON file
with open(npv_file_path, "w") as f:
    json.dump(npv_maximizing_strategy, f, indent=4)

with open(usace_file_path, "w") as f:
    json.dump(usace_strategy, f, indent=4)

print(f"Saved NPV strategy to {npv_file_path}")
print(f"Saved USACE strategy to {usace_file_path}")
