import numpy as np
from typing import Tuple

hyperparameters = {
    "learning_rate_function": lambda x: 1.0 / np.sqrt(x),
}

def gaussian_pertubation(individuals, learning_rate):
    """Gaussian pertubation.

    :param individual: Individual to be pertubated.
    :param mu: Mean of the Gaussian distribution.
    :param sigma: Standard deviation of the Gaussian distribution.
    :returns: A tuple of one individual.
    """
    new_individual = []

    for individual in individuals:
        x = individual[0]
        size = x.shape[0]

        sigma = individual[1]
        factors = np.exp(learning_rate * np.random.normal(0, 1, size))
        new_sigma = sigma * factors
    
        new_x = x + new_sigma * np.random.normal(0, 1, size)
    
        new_individual.append([new_x, new_sigma])
    
    return new_individual