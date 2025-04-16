from SALib.sample import latin
import numpy as np
from scipy.stats import beta
from typing import Tuple
import os


def pert_beta_params(
    b_min: float, b_most_likely: float, b_max: float
) -> Tuple[float, float]:
    """
    Calculates the parameters for a Beta distribution based on a PERT (Program Evaluation and Review Technique)
    distribution defined by a minimum, most likely, and maximum value.

    The PERT distribution is useful in project management for estimating the duration of tasks based on expert input.

    :param b_min: The minimum value of the distribution. This is the pessimistic estimate.
    :type b_min: float
    :param b_most_likely: The most likely value of the distribution. This is the optimistic estimate.
    :type b_most_likely: float
    :param b_max: The maximum value of the distribution. This is the most optimistic estimate.
    :type b_max: float
    :return: A tuple containing the calculated alpha and beta parameters for the Beta distribution.
    :rtype: Tuple[float, float]
    """
    alpha = 1 + 4 * (b_most_likely - b_min) / (b_max - b_min)
    beta_param = 1 + 4 * (b_max - b_most_likely) / (b_max - b_min)
    return alpha, beta_param


def generate_beach_nourishment_parameters_lhs_samples(
    dir_name: str, number_of_sows: int
) -> np.ndarray:
    """
    Generates Latin Hypercube Sampling (LHS) samples for beach nourishment parameters,
    including a parameter sampled from a Beta distribution based on specified bounds.

    The function creates LHS samples for various parameters related to beach nourishment,
    estimates the parameters for the 'b' parameter using a PERT approximation,
    and saves the resulting samples to a specified directory.

    :param dir_name: The name of the directory (within results_data) where the generated LHS samples will be saved.
    :type dir_name: str
    :param number_of_sows: The number of samples to generate using the Latin Hypercube Sampling method.
    :type number_of_sows: int
    :return: A 2D NumPy array containing the generated LHS samples along with 'b' values.
    :rtype: np.ndarray
    """
    # Define the problem (excluding 'b')
    problem = {
        "num_vars": 11,  # 12 total variables minus 'b'
        "names": [
            "epsilon",
            "phi_conc",
            "A",
            "alpha",
            "beta",
            "d",
            "eta",
            "l",
            "W",
            "xi",
            "delta",
        ],
        "bounds": [
            [0, 0.1],
            [0.7 * 193.8357, 1.3 * 193.8357],
            [0.7 * 669000, 1.3 * 669000],
            [1.0, 1.05],
            [0.95, 1.0],
            [-0.05, 0.05],
            [0.7 * 824, 1.3 * 824],
            [0.176, 0.264],
            [0.7 * 500000, 1.3 * 500000],
            [0, 0.1],
            [0.01, 0.07],
        ],
    }

    # Generate number_of_sows LHS samples for the other parameters
    lhs_samples = latin.sample(problem, number_of_sows)

    # Given values for 'b'
    b_min = 0
    b_most_likely = 0.0271 * 10**-3
    b_max = 0.113 * 10**-3

    # Estimate the alpha and beta using the PERT approximation
    alpha, beta_param = pert_beta_params(b_min, b_most_likely, b_max)

    # Generate the 'b' samples from the beta distribution
    b_samples = beta.rvs(
        alpha, beta_param, loc=b_min, scale=b_max - b_min, size=number_of_sows
    )

    # Combine 'b' with the LHS samples
    lhs_samples = np.column_stack((b_samples, lhs_samples))

    # Define the full path for the new directory inside the results_data folder
    results_dir = os.path.join("..", "results_data", dir_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Save the samples in the specified directory
    np.save(os.path.join(results_dir, "lhs_samples.npy"), lhs_samples)

    return lhs_samples
