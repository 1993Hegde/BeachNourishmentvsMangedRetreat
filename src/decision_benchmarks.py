import numpy as np
import math
from typing import Dict, Any, Tuple, List, Optional
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

def compute_coastal_state_variables(
    X: np.ndarray,
    pars: Dict[str, Any],
    E: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    This function calculates the beach width, property valuation, sea-level rise, 
    and storm-induced erosion across different states in the model.

    :param X: A 2D numpy array of shape (n, 4) representing the state-action matrix.
        Data types of columns in X (float):
        - X[:, 0]: tau (time of last nourishment)
        - X[:, 1]: t   (current time)
        - X[:, 2]: R   (relocation time)
        - X[:, 3]: nourishing (binary indicator)
    :type X: np.ndarray

    :param pars: A dictionary containing model parameters with keys:
        - 'a' (float)
        - 'b' (float)
        - 'Ti' (int or float)
        - 'r' (float)
        - 'H' (float)
        - 'E' (float, optional)
        - 'epsilon' (float)
        - 'mu' (float)
        - 'theta' (float)
        - 'x_nourished' (float)
        - 'relocationDelay' (float)
        - 'd' (float)
        - 'A' (float)
        - 'alpha' (float)
        - 'beta' (float)
    :type pars: Dict[str, Any]

    :param E: (Optional) A 1D numpy array or None, representing storm-induced 
              erosion values. If not provided, it will be computed within the 
              function.
    :type E: Optional[np.ndarray]

    :return: A tuple containing:
             - x: 1D numpy array of computed beach width values.
             - V: 1D numpy array of computed property valuation values.
             - L: 1D numpy array of computed sea-level rise values.
             - E: 1D numpy array of storm-induced erosion values (either 
                  provided or computed).
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    """

    # Handle the default for E if None is provided
    if E is None:
        E = np.array([])

    # Convert X to float
    X = X.astype(float)

    # Unpack columns of X
    t = X[:, 1]
    tau = X[:, 0]
    R = X[:, 2]
    nourishing = X[:, 3]

    # Compute sea-level rise (L) and its integral (L_int)
    L = pars["a"] * t + pars["b"] * (t**2 + 2 * t * (pars["Ti"] - 1992))
    L = np.around(L, 4)
    L_int = 0.5 * pars["a"] * t**2 + pars["b"] * (
        t**3 / 3 + t**2 * (pars["Ti"] - 1992)
    )
    L_int = np.round(L_int, 4)

    # Compute storm-induced erosion (E) if not provided
    if "E" in pars and E.size == 0:
        E = pars["E"] * tau + pars["epsilon"] * L_int
        E = np.round(E, 4)

    # Compute gamma_erosion
    condition = (tau > 0) & (t > 0) & (t >= tau)
    gamma_erosion = np.zeros_like(t)
    L_t = pars["a"] * t + pars["b"] * (t**2 + 2 * t * (pars["Ti"] - 1992))
    L_tau = pars["a"] * (t - tau) + pars["b"] * (
        (t - tau) ** 2 + 2 * (t - tau) * (pars["Ti"] - 1992)
    )
    same_tau_t = tau == t
    diff_tau_t = tau != t

    gamma_erosion[condition & same_tau_t] = (
        pars["r"] * L_t[condition & same_tau_t]
        - pars["H"] * t[condition & same_tau_t]
    )
    gamma_erosion[condition & diff_tau_t] = (
        pars["r"] * (L_t[condition & diff_tau_t] - L_tau[condition & diff_tau_t])
        - pars["H"] * tau[condition & diff_tau_t]
    )

    # Compute beach width (x)
    term1 = pars["x_nourished"] * nourishing
    term2 = (
        (1 - pars["mu"]) * pars["x_nourished"]
        + pars["mu"] * np.exp(-pars["theta"] * tau) * pars["x_nourished"]
        - gamma_erosion
        - E
    )
    term2 = np.maximum(term2, 0) * (1 - nourishing)
    x = term1 + term2

    # Compute property valuation (V)
    condition = R <= pars["relocationDelay"]
    V = np.where(
        condition,
        ((1 + pars["d"]) ** t) * pars["A"] * (pars["alpha"] ** x) * (pars["beta"] ** L),
        0,
    )
    V = np.round(V, 4)

    return x, V, L, E


def compute_coastal_cost_metrics(
    X: np.ndarray,
    pars: Dict[str, Any],
    x: np.ndarray,
    V: np.ndarray,
    L: np.ndarray,
    E: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    This function calculates various cost metrics (nourishment cost, relocation cost,
    damage cost) and their total sum for a coastal model. It uses pre-computed
    beach width (`x`), property valuation (`V`), sea-level rise (`L`), and 
    erosion (`E`).

    :param X: A 2D numpy array of shape (n, 5) representing the state-action matrix:
        - X[:, 0] = tau (time of last nourishment)
        - X[:, 1] = t   (current time)
        - X[:, 2] = R   (relocation time)
        - X[:, 3] = nourishing (binary indicator)
        - X[:, 4] = A   (binary indicator for action/decision)
      This must be of type float or castable to float.

    :type X: np.ndarray

    :param pars: A dictionary containing model parameters with keys such as:
        - "c1" (float): fixed cost of nourishment
        - "c2" (float): additional cost factor for nourishment
        - "xi" (float): cost escalation rate 
        - "constructionCosts" (float): construction costs for nourishment
        - "rho" (float): fraction of valuation lost upon relocation
        - "Cfunc" (str): type of damage function ("linear", "exponential", "concave", or "polynomial")
        - "D0" (float): base damage cost
        - "phi_lin" (float): linear cost parameter
        - "phi_exp" (float): exponential cost parameter
        - "phi_conc" (float): concave cost parameter
        - "phi_poly" (float): polynomial cost exponent
        - "kappa" (float): decay parameter for damage as function of beach width
        - "W" (float): cost term to add if relocation hasn't happened yet
        - "minInterval" (float or int): minimum interval constraint for nourishment feasibility
        - "relocationDelay" (float or int): time at which relocation happens
        - "x_nourished" (float): desired post-nourishment beach width 
      and any other relevant keys.
    :type pars: Dict[str, Any]

    :param x: 1D numpy array of shape (n,) representing the pre-computed beach widths.
    :type x: np.ndarray

    :param V: 1D numpy array of shape (n,) representing the pre-computed property valuations.
    :type V: np.ndarray

    :param L: 1D numpy array of shape (n,) representing the pre-computed sea-level rise values.
    :type L: np.ndarray

    :param E: 1D numpy array of shape (n,) representing the pre-computed erosion values (not used in this function directly, 
              but included if needed for consistency/future use).
    :type E: np.ndarray

    :return: A tuple containing:
             - C (np.ndarray): total cost across all items (nourish, relocate, damage) plus feasibility cost
             - nourishCost (np.ndarray): nourishment cost
             - relocateCost (np.ndarray): relocation cost
             - damageCost (np.ndarray): damage cost
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    """

    # Unpack columns of X
    tau = X[:, 0]
    t = X[:, 1]
    R = X[:, 2]
    nourishing = X[:, 3]  # Not used directly in this function
    A = X[:, 4]           # Binary action indicator

    # Precompute the cost factor c2 at current time t
    c2 = pars["c2"] * (1 + pars["xi"]) ** t

    # === 1) Nourishment Cost ===
    # Costs if 'A == 1' (i.e., a nourishment decision is made)
    # x_nourished - x gives the additional width to be nourished 
    # Multiplying that by c2 yields the "width-based" cost 
    nourishCost = (
        pars["c1"] * np.ones_like(x)
        + c2 * (pars["x_nourished"] * np.ones_like(x) - x)
        + pars["constructionCosts"] * np.ones_like(x)
    ) * (A == 1)

    # === 2) Relocation Cost ===
    # Occurs if 'R == pars["relocationDelay"]'
    # R is the time at which relocation is triggered
    relocateCost = pars["rho"] * V * (R == pars["relocationDelay"])

    # === 3) Damage Cost ===
    # Depends on the specified damage function
    # If no valid function is specified, raise an error.
    if pars["Cfunc"] == 'linear':
        damageCost = (
            pars["D0"] * (1 + L * pars["phi_lin"])
            / (pars["kappa"] ** x)
        ) * (V + pars["W"] * (R < pars["relocationDelay"]))

    elif pars["Cfunc"] == 'exponential':
        damageCost = (
            pars["D0"] * (pars["phi_exp"] ** L)
            / (pars["kappa"] ** x)
        ) * (V + pars["W"] * (R < pars["relocationDelay"]))

    elif pars["Cfunc"] == 'concave':
        damageCost = (
            pars["D0"] * (1 + pars["phi_conc"] * (1 - np.exp(-L)))
            / (pars["kappa"] ** x)
        ) * (V + pars["W"] * (R < pars["relocationDelay"]))

    elif pars["Cfunc"] == 'polynomial':
        damageCost = (
            pars["D0"] * (1 + L) ** pars["phi_poly"]
            / (pars["kappa"] ** x)
        ) * (V + pars["W"] * (R < pars["relocationDelay"]))

    else:
        raise ValueError("Unrecognized damage function. Must be one of "
                         "['linear', 'exponential', 'concave', 'polynomial'].")

    # === 4) Feasibility Cost (penalizing infeasible nourishment timing) ===
    # If 'A == 1' and 'tau < (pars["minInterval"] - 1)', cost is infinite.
    infeasible = (A == 1) & (tau < (pars["minInterval"] - 1))
    feasibilityCost = np.zeros_like(infeasible, dtype=float)
    feasibilityCost[infeasible] = np.inf

    # === 5) Total Cost ===
    C = nourishCost + relocateCost + damageCost + feasibilityCost

    return C, nourishCost, relocateCost, damageCost

def compute_coastal_benefits(
    X: np.ndarray,
    pars: Dict[str, Any],
    x: np.ndarray,
    V: np.ndarray,
    L: np.ndarray,
    E: np.ndarray
) -> np.ndarray:
    """
    This function calculates the total benefits (from location and beach 
    amenities) for a coastal model using precomputed values.

    :param X: A 2D numpy array (state-action matrix) of shape (n, k).
              Although not used directly in this function, it is included 
              for consistency with the overall model interface.
    :type X: np.ndarray

    :param pars: A dictionary containing model parameters with keys:
        - "l"   (float): factor for location-based benefit
        - "eta" (float): factor for beach-based benefit
      Additional keys may exist for other purposes in the full model.
    :type pars: Dict[str, Any]

    :param x: 1D numpy array of shape (n,) representing the precomputed 
              beach widths.
    :type x: np.ndarray

    :param V: 1D numpy array of shape (n,) representing the precomputed 
              property valuations.
    :type V: np.ndarray

    :param L: 1D numpy array of shape (n,) representing the precomputed 
              sea-level rise values. It is not used in this function but 
              is included for consistency.
    :type L: np.ndarray

    :param E: 1D numpy array of shape (n,) representing the precomputed 
              erosion values. It is not used in this function but is included 
              for consistency.
    :type E: np.ndarray

    :return: A 1D numpy array of shape (n,) where each element is the total 
             benefit (location benefit plus beach benefit).
    :rtype: np.ndarray
    """

    # Compute location-based benefits
    B_location = pars["l"] * V

    # Compute beach-based benefits
    B_beach = pars["eta"] * x

    # Total benefits
    B = B_location + B_beach

    return B

def transition_observed_state(
    old_state: List[int],
    action: int,
    pars: Dict[str, Any]
) -> List[int]:
    """
    Computes the new state of the coastal system given the old state 
    and an action.

    The state is assumed to be a 4-element list:
        - state[0]: Time (in discrete steps) since the last nourishment (tau).
        - state[1]: Current simulation time step (t).
        - state[2]: Relocation status (e.g., 0 = not relocating, 1 = planning 
                    to relocate, 2 = relocated).
        - state[3]: Nourishment indicator (e.g., 0 = not nourishing, 1 = nourishing).

    Actions (integer) might be defined as:
        - 0: Do nothing.
        - 1: Nourish.
        - 2: Relocate (or move toward relocation if not already in progress).

    The function updates elements of the state vector to reflect the
    chosen action, then ensures neither time since last nourishment (state[0])
    nor the current time step (state[1]) exceed the simulation length defined 
    in `pars["sim_length"]`.

    :param old_state: The current state of the system, a list of four integers.
    :type old_state: List[int]

    :param action: An integer representing the chosen action:
                   0 = do nothing, 1 = nourish, 2 = relocate.
    :type action: int

    :param pars: A dictionary of model parameters, 
                 which must include at least:
                 - "sim_length" (int): The maximum time step for the simulation.
    :type pars: Dict[str, Any]

    :return: The new state of the system (a list of four integers).
    :rtype: List[int]
    """

    # Initialize the new state with placeholders
    new_state = [0, 0, 0, 0]

    # 1) Increment current simulation time (old_state[1]) by 1.
    new_state[1] = old_state[1] + 1

    # 2) If relocation status was 1 (planning to relocate), move it to 2 (relocated).
    if old_state[2] == 1:
        new_state[2] = 2

    # 3) Transition logic based on action
    if action == 0:
        # Action 0: Do nothing
        new_state[0] = old_state[0] + 1   # Increase time since last nourishment
        new_state[2] = old_state[2]      # Keep relocation status
        new_state[3] = 0                 # Not nourishing

    elif action == 1:
        # Action 1: Nourish
        new_state[0] = 0                 # Reset time since last nourishment
        new_state[2] = old_state[2]      # Keep relocation status
        new_state[3] = 1                 # Nourishing

    elif action == 2:
        # Action 2: Relocate (or plan to relocate)
        new_state[0] = old_state[0] + 1  # Increase time since last nourishment
        new_state[3] = 0                 # Not nourishing

        # If currently not relocating (0), switch to planning to relocate (1)
        if old_state[2] == 0:
            new_state[2] = 1
        else:
            # Otherwise, carry forward whatever relocation status was there
            new_state[2] = old_state[2]

    # Ensure if relocation status was 1 in old_state, it's set to 2 in new_state.
    # (Duplicating the check here in case of multiple logic paths.)
    if old_state[2] == 1:
        new_state[2] = 2

    # 4) Enforce simulation length constraints
    new_state[0] = min(new_state[0], pars["sim_length"])
    new_state[1] = min(new_state[1], pars["sim_length"])

    return new_state