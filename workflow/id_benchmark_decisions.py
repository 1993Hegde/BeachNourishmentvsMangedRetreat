import sys
sys.path.insert(0, '../src')  # Adjust the path to import from the src directory
from decision_benchmarks import *
import pickle

slr_scenario = 1
pars = baseline_model_instance_with_default_parameters(slr_scenario)
initial_state = [0, 0, 0, 0]
optS, x_final, v_final, L_final, C_final, B_final, accumulated_npv, strategy = solve_cutler_et_al_ddp(pars, initial_state)
cutler_et_al_strategy = optS[:,4]
print(x_final)
print(cutler_et_al_strategy)