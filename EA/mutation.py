import numpy as np
from typing import Tuple

hyperparameters = {
    "learning_rate_function": lambda x: 1.0 / np.sqrt(x),
}

def gaussian_pertubation(individual: Tuple[np.array, np.array]):
    """Gaussian pertubation.

    :param individual: Individual to be pertubated.
    :param mu: Mean of the Gaussian distribution.
    :param sigma: Standard deviation of the Gaussian distribution.
    :returns: A tuple of one individual.
    """
    x = individual[0]
    size = x.shape[0]
    learning_rate = hyperparameters["learning_rate_function"](size)
    print("learning_rate ", learning_rate)
    sigma = individual[1]
    factors = np.exp(learning_rate * np.random.normal(0, 1, size))
    new_sigma = sigma * factors
    
    new_x = x + new_sigma * np.random.normal(0, 1, size)
    
    return (new_x, new_sigma)

    
def main():
    pass
    
    
if __name__ == "__main__":
    main()