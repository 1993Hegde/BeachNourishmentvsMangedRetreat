import numpy as np
import math
from typing import Dict, Any, Tuple, List, Optional
from scipy.stats import poisson
import matplotlib.pyplot as plt
from itertools import product
from scipy.sparse import lil_matrix, csr_matrix
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


def build_sparse_transition_matrix(
    S: List[Tuple[int, ...]],
    A: List[int],
    X: np.ndarray,
    pars: Dict[str, Any]
) -> Tuple[csr_matrix, List[int], List[int]]:
    """
    Constructs a sparse transition matrix Q for a coastal model or MDP-like system, 
    where each row corresponds to a (state, action) pair and each column corresponds 
    to a resulting state.

    :param S: A list of all possible states in the system. Each state is typically 
              a tuple, e.g. (tau, t, relocation_status, nourishment_indicator).
    :type S: List[Tuple[int, ...]]

    :param A: A list of possible actions (e.g., [0, 1, 2] for do nothing, nourish, relocate).
    :type A: List[int]

    :param X: A 2D NumPy array where each row combines state and action into a single 
              vector. For example, if each state is four-dimensional and there's a 
              single action appended, each row might be length 5.
    :type X: np.ndarray

    :param pars: A dictionary of parameters for the model. Must include necessary 
                 keys for the function compute_new_state to work, e.g. "sim_length".
    :type pars: Dict[str, Any]

    :return: 
        - Q (csr_matrix): A sparse boolean matrix of shape (len(S)*len(A), len(S)), 
          where entry (i, j) = True indicates that from row i's (state, action) pair, 
          the system transitions to state j.
        - s_indices (List[int]): The index of the state in S for each row i of Q.
        - a_indices (List[int]): The index of the action in A for each row i of Q.

    :rtype: Tuple[csr_matrix, List[int], List[int]]
    """
    # We use LIL format for efficient assignment, then convert to CSR for faster operations.
    Q = lil_matrix((len(S) * len(A), len(S)), dtype=bool)

    s_indices = []
    a_indices = []

    for s in S:
        for a in A:
            s_index = S.index(s)
            a_index = A.index(a)
            s_indices.append(s_index)
            a_indices.append(a_index)

            # Construct the combined state-action row
            x_a_combo = list(s) + [a]
            
            # Find the corresponding row in X
            # np.all(..., axis=1) returns a boolean array which we use to locate the index
            x_index_arr = np.where(np.all(X == x_a_combo, axis=1))[0]
            if x_index_arr.size == 0:
                raise ValueError(f"State-action combination {x_a_combo} not found in X.")
            x_index = x_index_arr[0]

            # Compute the next state given the old state s and action a
            s_prime = tuple(compute_new_state(s, a, pars))

            # Find index of the new state in S
            if s_prime not in S:
                raise ValueError(f"Resulting state {s_prime} not found in the list of states S.")
            s_prime_index = S.index(s_prime)

            # Mark the transition
            Q[x_index, s_prime_index] = True

    # Convert to CSR for more efficient row operations
    Q = Q.tocsr()

    return Q, s_indices, a_indices


def simulate_mdp_trajectory(
    initial_state_index: int,
    S: List[Tuple[int, ...]],
    Ix: np.ndarray,
    X: np.ndarray,
    pars: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulates a Markov Decision Process (MDP) trajectory for a specified 
    number of time steps. The policy is implicitly defined by the array Ix, 
    which indicates the row in X (a state-action matrix) to use for each 
    state index.

    At each time step t:
      1. The function looks up the current state index (Si[t-1]).
      2. It retrieves an action from the row X[ Xi[t-1] ] (i.e., the last 
         element in that row is assumed to be the action).
      3. It computes the next state using compute_new_state.
      4. It then finds the index of this new state in S.
      5. It sets Xi[t] to the row index from Ix corresponding to that new state.

    :param initial_state_index: The index of the initial state in S.
    :type initial_state_index: int

    :param S: A list of all possible states in the system. Each element is a 
              tuple (or list) describing a unique state, e.g. 
              (tau, t, relocation_status, nourishment_indicator).
    :type S: List[Tuple[int, ...]]

    :param Ix: A NumPy array (or list) of length len(S), where Ix[i] gives 
               the row index in X that should be used for state i. 
               This effectively encodes a policy: for state i, use X[ Ix[i] ].
    :type Ix: np.ndarray

    :param X: A 2D NumPy array (or matrix) whose rows typically represent 
              a state-action combination. The last column is assumed to 
              represent the action. E.g., X[j, :-1] might be a state, and 
              X[j, -1] is the action.
    :type X: np.ndarray

    :param pars: A dictionary of parameters for the MDP simulation, which 
                 must include:
                 - "sim_length" (int): The number of time steps to simulate.
                 Other keys may be needed for compute_new_state.
    :type pars: Dict[str, Any]

    :return: 
        - Xi (np.ndarray): An array of length sim_length containing the indices 
          into X (the state-action matrix) used at each time step.
        - Si (np.ndarray): An array of length sim_length containing the indices 
          into S (the list of possible states) for each time step.
    :rtype: Tuple[np.ndarray, np.ndarray]
    """

    sim_length = pars["sim_length"]

    # Arrays to store the trajectory of state and state-action indices
    Si = np.zeros(sim_length, dtype=int)
    Xi = np.zeros(sim_length, dtype=int)

    # Initialize with the provided starting state index
    Si[0] = initial_state_index
    # The initial "policy" selection in X comes from Ix for that state
    Xi[0] = Ix[initial_state_index]

    for t in range(1, sim_length):
        # Current state (index) from the previous time step
        s_current_index = Si[t - 1]
        # Row in X that encodes the action for this state
        x_index = Xi[t - 1]

        # Extract the action from X. We assume the last element in X[x_index] is the action
        action = X[x_index, -1]

        # Convert the current state (a tuple) to a list (if needed) for compute_new_state
        old_state = list(S[s_current_index][0:-1])  # or simply list(S[s_current_index]) if that's appropriate
        
        # Compute the next state
        new_state = transition_observed_state(old_state, action, pars)

        # Convert next state to a tuple so we can look it up in S
        new_state_tuple = tuple(new_state)
        
        # Find the index of the new state in S
        s_next_index = S.index(new_state_tuple)
        Si[t] = s_next_index

        # Determine the row in X that corresponds to this new state under our policy
        Xi[t] = Ix[s_next_index]

    return Xi, Si

def solve_DDP_return_NPV_action_sequence(
    R: np.ndarray,
    Q_sparse: csr_matrix,
    pars: Dict[str, Any],
    S: List[Tuple[int, int, int, int]],
    X: np.ndarray,
    S0i: int,
    s_indices: List[int],
    a_indices: List[int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Solves a Discrete Dynamic Programming (DDP) model (via value iteration),
    simulates the resulting optimal policy, and returns key metrics including 
    net present value (NPV).

    :param R: A 1D NumPy array of shape (n_state_action,) representing the 
              immediate reward for each (state, action) pair.
    :type R: np.ndarray

    :param Q_sparse: A sparse transition probability matrix (CSR format) of 
                     shape (n_state_action, n_states). Each row corresponds 
                     to a (state, action) pair, and each column corresponds 
                     to the next-state index.
    :type Q_sparse: csr_matrix

    :param pars: A dictionary of model parameters required by quantecon and 
                 downstream functions, which must include:
                 - "delta" (float): discount rate
                 - "sim_length" (int): number of time steps in the simulation
                 Additional parameters may be needed by xVLE, compute_cost, 
                 and compute_benefits.
    :type pars: Dict[str, Any]

    :param S: A list of all possible states in the model. Each state is 
              typically a tuple (tau, time, relocation_status, nourished_indicator).
    :type S: List[Tuple[int, int, int, int]]

    :param X: A 2D NumPy array where each row corresponds to a (state, action) 
              combination. For instance, X[i, :-1] might be a state, and 
              X[i, -1] the corresponding action.
    :type X: np.ndarray

    :param S0i: The index (in S) of the initial state.
    :type S0i: int

    :param s_indices: A list mapping each row of Q_sparse (and R) to the 
                      appropriate state index in S.
    :type s_indices: List[int]

    :param a_indices: A list mapping each row of Q_sparse (and R) to the 
                      appropriate action index in your action set.
    :type a_indices: List[int]

    :return: A tuple containing:
        1. optS (np.ndarray): The (state, action) pairs chosen over the 
           simulation horizon.
        2. x_final (np.ndarray): Beach width at each time step.
        3. v_final (np.ndarray): Property valuation at each time step.
        4. L_final (np.ndarray): Sea-level rise at each time step.
        5. C_final (np.ndarray): Total cost at each time step.
        6. B_final (np.ndarray): Total benefit at each time step.
        7. accumulated_npv (float): The total accumulated net present value 
           over the simulation horizon.
        8. strategy (np.ndarray): The action chosen at each time step (last 
           column of optS).
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, 
                  np.ndarray, np.ndarray, float, np.ndarray]
    """

    # 1) Build and solve the DDP using quantecon
    beta = 1 / (1 + pars["delta"])
    ddp = qe.markov.DiscreteDP(R, Q_sparse, beta, s_indices, a_indices)
    results = ddp.solve(method='value_iteration')

    # 2) Construct the optimal (state, action) array from the policy
    S_array = np.array(S)
    sigma_array = np.array(results.sigma).reshape(len(results.sigma), 1)
    results.Xopt = np.hstack([S_array, sigma_array])

    # 3) Convert the (state, action) combos to the corresponding rows in X
    results.Ixopt = []
    for row in results.Xopt:
        # row[:-1] is the state, row[-1] is the action
        idx = np.where(np.all(X == row, axis=1))[0][0]
        results.Ixopt.append(idx)
    results.Ixopt = np.array(results.Ixopt)
    Ix = results.Ixopt

    # 4) Simulate from the initial state using the computed policy
    Xi, Si = mdpsim_inf(S0i, S, Ix, X, pars)
    optS = X[Xi]  # Full (state, action) pairs over time

    # 5) Compute relevant coastal model metrics
    x_final, v_final, L_final, E_final = compute_coastal_state_variables(optS, pars)
    C_final, nourish_cost_final, relocate_cost_final, damage_cost_final = compute_coastal_cost_metrics(optS, pars,x_final, v_final, L_final, E_final)
    B_final = compute_coastal_benefits(optS, pars,x_final, v_final, L_final, E_final)

    # 6) Discount future costs/benefits and compute net present value
    df = [(1 + pars["delta"]) ** i for i in range(pars["sim_length"])]
    individual_benefits = [B_final[i] / df[i] for i in range(pars["sim_length"])]
    individual_costs = [C_final[i] / df[i] for i in range(pars["sim_length"])]
    individual_npv = [(B_final[i] - C_final[i]) / df[i] for i in range(pars["sim_length"])]

    # 7) Get the final accumulated NPV and action strategy
    accumulated_npv = np.cumsum(individual_npv)[-1]
    strategy = optS[:, -1]

    return optS, x_final, v_final, L_final, C_final, B_final, accumulated_npv, strategy


def solve_cutler_et_al_ddp(
    pars: Dict[str, Any], 
    initial_state: Tuple[int, int, int, int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Sets up and solves the coastal DDP model using the approach described 
    by "Cutler et al." (as referenced in your code base). This involves:
      1. Defining the state space (tau, time, relocation, nourished).
      2. Defining the action space (0, 1, 2).
      3. Creating all (state, action) pairs (X) and the base set of states (S).
      4. Building a transition matrix (Q_sparse).
      5. Computing costs (C), benefits (B), and immediate rewards (R).
      6. Solving the DDP via `solve_DDP_return_NPV_action_sequence`.
      7. Returning the optimal path and relevant metrics.

    :param pars: A dictionary of parameters needed to define the model. 
                 Must include:
                 - "sim_length" (int)
                 - "deltaT" (int): The increment in tau/time to build the state space
                 - "delta" (float): Discount rate
                 Additional keys may be needed by other functions 
                 (xVLE, compute_cost, compute_benefits).
    :type pars: Dict[str, Any]

    :param initial_state: A tuple defining the initial state in the form 
                          (tau, time, relocation, nourished).
    :type initial_state: Tuple[int, int, int, int]

    :return: A tuple containing the same outputs as solve_DDP_return_NPV_action_sequence:
        1. optS (np.ndarray): The (state, action) pairs chosen over the 
           simulation horizon.
        2. x_final (np.ndarray): Beach width at each time step.
        3. v_final (np.ndarray): Property valuation at each time step.
        4. L_final (np.ndarray): Sea-level rise at each time step.
        5. C_final (np.ndarray): Total cost at each time step.
        6. B_final (np.ndarray): Total benefit at each time step.
        7. accumulated_npv (float): The total accumulated net present value 
           over the simulation horizon.
        8. strategy (np.ndarray): The action chosen at each time step (last 
           column of the resulting path).
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, 
                  np.ndarray, np.ndarray, float, np.ndarray]
    """

    # 1. Define action space
    A = [0, 1, 2]

    # 2. Define the discrete values for tau (time since last nourishment)
    #    and time (current simulation year/step)
    tau = [i for i in range(0, pars["sim_length"] + 1, pars["deltaT"])]
    time = [i for i in range(0, pars["sim_length"] + 1, pars["deltaT"])]

    # 3. Define possible states for relocation and nourishment indicators
    relocation = [0, 1, 2]
    nourished = [0, 1]

    # 4. Build the full set of (state, action) combos: X, and just states: S
    X = list(product(tau, time, relocation, nourished, A))
    S = list(product(tau, time, relocation, nourished))

    # Convert to NumPy arrays for downstream use
    X_pr = np.array(X)

    # 5. Build the sparse transition matrix
    Q_sparse, s_indices, a_indices = build_sparse_transition_matrix(S, A, X_pr, pars)

    # 6. Compute the coastal metrics for each (state, action) in X
    #    Here xVLE and compute_cost/compute_benefits can handle either list or array,
    #    as long as shapes match the function definitions.
    x_vals, v_vals, L_vals, E_vals = compute_coastal_state_variables(X_pr, pars)
    C_vals, nourish_cost, relocate_cost, damage_cost = compute_coastal_cost_metrics(X, pars)
    B_vals = compute_coastal_benefits(X, pars)

    # Convert X back to NumPy array for consistent indexing in the solver
    X_np = np.array(X)

    # 7. Build the immediate reward array R = B - C
    R = np.array([B_vals[i] - C_vals[i] for i in range(len(B_vals))])

    # 8. Locate the initial state's index in S
    S0i = S.index(tuple(initial_state))

    # 9. Solve and simulate the DDP
    (
        optS, 
        x_final, 
        v_final, 
        L_final, 
        C_final, 
        B_final, 
        accumulated_npv, 
        strategy
    ) = solve_DDP_return_NPV_action_sequence(
        R, Q_sparse, pars, S, X_np, S0i, s_indices, a_indices
    )

    return optS, x_final, v_final, L_final, C_final, B_final, accumulated_npv, strategy