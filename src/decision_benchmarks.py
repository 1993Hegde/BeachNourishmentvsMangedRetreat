import numpy as np
import math
from scipy.stats import poisson
import matplotlib.pyplot as plt
from itertools import product
import quantecon as qe

def baseline_model_instance_with_default_parameters(slr_scenario: int) -> dict:
    '''
    This function initializes and returns a dictionary of input parameters 
    for the sea level rise (SLR) model based on the specified SLR scenario.

    :param slr_scenario: An integer representing the sea level rise scenario. 
                         Must be one of the following values:
                         - 0: Low sea level rise
                         - 1: Medium sea level rise
                         - 2: High sea level rise
                         - -1: Default or unspecified scenario
    :type slr_scenario: int
    :raises ValueError: If the slr_scenario is not one of the allowed values.
    :return: A dictionary containing the initialized model parameters.
    :rtype: dict
    '''
    if slr_scenario not in [0, 1, 2, -1]:
        raise ValueError(
            "slr_scenario must be one of the following values: 0 (low), "
            "1 (medium), 2 (high), or -1 (default)."
        )
    
    pars = {}
    pars["scenario"] = slr_scenario
    pars["a"] = 2.355 * 10**-3  # Historical sea level change (m/yr)
    if slr_scenario == 0:
        pars["b"] = 0  # Low sea level rise acceleration
    elif slr_scenario == 1:
        pars["b"] = 0.0271 * 10**-3  # Intermediate sea level rise acceleration
    elif slr_scenario == 2:
        pars["b"] = 0.113 * 10**-3  # High sea level rise acceleration
    else:
        pars["b"] = 0
        pars["a"] = 0

    # ...existing code for initializing other parameters...
    pars["x_nourished"] = 6.096  # Nourished beach width (m)
    pars["x_crit"] = 0  # Beach width nourishment trigger (m)
    pars["mu"] = 1  # (x_nourished-x_crit)/x_nourished; nourished portion of beach 
    pars["init_rate"] = -0.0792  # Historical shoreline change rate (m/yr)
    pars["theta"] = 0.1  # Exponential erosion rate
    pars["r"] = 70.04  # Slope of active profile
    pars["H"] = pars["init_rate"] + pars["r"] * pars["a"]
    '''Initial conditions'''
    pars["tau_init"] = 0  # Initial years since nourishment
    pars["v_init"] = 6690000  # Initial value at risk
    pars["x_init"] = pars["x_nourished"]  # Initial beach width (m)
    '''Time parameters'''
    pars["deltaT"] = 1  # Time step (yr)
    pars["Ti"] = 2020  # Year of simulation start
    pars["sim_length"] = 100  # Simulation Length (yr)
    pars["Tswitch"] = 1  # Time when relocation becomes an option
    pars["T"] = np.inf  # Time horizon
    '''Expected storm induced erosion'''
    pars["lambda"] = 0.35  # Storm Frequency
    pars["m"] = 1.68  # GEV location parameter
    pars["sigma"] = 4.24  # GEV scale parameter
    pars["k"] = 0.277  # GEV shape parameter
    meanGEV = pars["m"] + pars["sigma"] * (
        (math.gamma(1 - pars["k"]) - 1) / pars["k"]
    )
    p = poisson.pmf(np.arange(1, 5), mu=pars["lambda"])
    pars["E"] = 0
    for n in np.arange(1, 5):
        M = 0
        for i in range(1, n + 1):
            M = M + meanGEV / i
        pars["E"] += 0.1 * p[n - 1] * M  # Annual expected storm erosion
    pars["epsilon"] = 0  # Increase in storm erosion with SLR
    '''Property value parameters'''
    pars["d"] = 0
    pars["alpha"] = (1 + 0.01)**3.2808  # Property value increase due to 1m 
                                        # increase in beach width
    pars["beta"] = 0.5  # Property value decrease due to 1m increase in sea level
    pars["A"] = 669000
    pars["v_init"] = pars["A"] * (pars["alpha"]**(pars["x_init"]))  # Baseline 
                                                                    # property value
    pars["W"] = 5 * 10**5  # Non-structural value at risk
    '''Benefit and cost parameters'''
    pars["delta"] = 0.0265
    pars["eta"] = 824  # Land value ($1000/m), assumes $14/sq ft and 5470 m 
                       # of beach length
    pars["l"] = 0.22  # St. Lucie County general fund tax rate
    pars["c1"] = 12000  # Fixed cost of nourishment ($1000), assumes $14 
                        # million per nourishment, c2=350
    pars["c2"] = 350  # Variable cost of nourishment ($1000/m), assumes 
                      # $9.55/m^3, 5470 m of beach length, and 224,000 m^3 
                      # per 6.096 m nourishment
    pars["xi"] = 0  # Exponential increase in c2 as time progresses (0 means 
                    # cost is autonomous)
    pars["constructionCosts"] = 0
    pars["Cfunc"] = "concave"
    pars["phi_exp"] = 5.6999  # Sea level base for proportion damaged
    pars["phi_lin"] = 61.3951
    pars["phi_conc"] = 193.8357
    pars["phi_poly"] = 3.7625
    pars["kappa"] = 1.2  # Beach width base for proportion damaged
    pars["D0"] = 5.4 * 10**-3  # Expected proportion damaged when width = 0 
                               # sea level = 0
    '''Relocation parameters'''
    pars["relocationDelay"] = 1  # Years after decision is made that 
                                 # relocation occurs
    pars["rho"] = 1  # Proportion of property value spent to relocate
    '''Feasibility constraints'''
    pars["minInterval"] = 2  # Minimum amount of time between two 
                             # renourishments
    '''Max and min values for uncertainty and sensitivity analysis as reported in Cutler et al'''
    pars["x_nourishedMin"] = 0.8 * pars["x_nourished"]
    pars["x_nourishedMax"] = 1.2 * pars["x_nourished"]
    pars["Hmin"] = -0.2
    pars["Hmax"] = 0.2
    pars["thetaMin"] = 0.8 * pars["theta"]
    pars["thetaMax"] = 1.2 * pars["theta"]
    pars["rMin"] = 0.8 * pars["r"]
    pars["rMax"] = 1.2 * pars["r"]
    pars["Emin"] = 0.8 * pars["E"]
    pars["Emax"] = 1.2 * pars["E"]
    pars["dMin"] = -0.05
    pars["dMax"] = 0.05
    pars["alphaMin"] = 1
    pars["alphaMax"] = 1.2
    pars["betaMin"] = 0.1
    pars["betaMax"] = 1
    pars["c1Min"] = 0.8 * pars["c1"]
    pars["c1Max"] = 1.2 * pars["c1"]
    pars["c2Min"] = 0.8 * pars["c2"]
    pars["c2Max"] = 1.2 * pars["c2"]
    pars["xiMin"] = 0
    pars["xiMax"] = 0.05
    pars["etaMin"] = 0.8 * pars["c1"]
    pars["etaMax"] = 1.2 * pars["c1"]
    pars["kappaMin"] = 1 + 0.8 * (pars["kappa"] - 1)
    pars["kappaMax"] = 1 + 1.2 * (pars["kappa"] - 1)
    pars["D0Min"] = 0.8 * pars["D0"]
    pars["D0Max"] = 1.2 * pars["D0"]
    pars["deltaMin"] = 0.01
    pars["deltaMax"] = 0.07
    pars["relocationDelayMin"] = 1
    pars["relocationDelayMax"] = 10
    pars["bMin"] = 0
    pars["bMax"] = 0.113 * 10**-3
    pars["epsilonMin"] = 0
    pars["epsilonMax"] = 0.1
    pars["AMin"] = 0.1 * pars["A"]
    pars["AMax"] = 3 * pars["A"]
    pars["muMin"] = 0.5
    pars["muMax"] = 1
    pars["WMin"] = 2 * 10**5
    pars["TMin"] = 10
    pars["Tmax"] = 10000
    return pars