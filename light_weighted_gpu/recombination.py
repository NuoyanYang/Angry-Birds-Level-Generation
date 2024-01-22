import numpy as np
from typing import Tuple, Sequence

hyperparameters = {
    "offspring_ratio": 3/4,
    "alpha": 0.5, # for arithmetic recombination
}

### non-arithmetic/discrete recombination

def n_point_crossover(individual_1: Tuple[np.array, np.array], 
                    individual_2: Tuple[np.array, np.array], 
                    n: int = -1, crossover_points: Sequence[int] = [None]) -> Tuple[Tuple[np.array, np.array], Tuple[np.array, np.array]]:
    """Generate offspring by n-point crossover.

    Args:
        individual_1 (Tuple[np.array, np.array]): _description_
        individual_2 (Tuple[np.array, np.array]): _description_
        n (int, optional): _description_. Defaults to -1.
        crossover_points (Sequence[int], optional): _description_. Defaults to [None].

    Returns:
        Tuple[Tuple[np.array, np.array], Tuple[np.array, np.array]]: _description_
    """
    x = individual_1[0]
    x_sigma = individual_1[1]
    y = individual_2[0]
    y_sigma = individual_2[1]
    size = x.shape[0]
    if n == -1: # random number of crossover points
        lower = 0
        upper = size // 2
        n = np.random.randint(lower, upper)
    
    if crossover_points[0] is None:
        # choose n random crossover points
        crossover_points = sorted(np.random.choice(size, n, replace=False))
    
    rest = sorted(set(range(size)) - set(crossover_points)) # the rest of the genes
    
    # create offspring
    offspring_1 = np.zeros(size)
    offspring_2 = np.zeros(size)
    offspring_1_sigma = np.zeros(size)
    offspring_2_sigma = np.zeros(size)
    
    # do crossover
    offspring_1[crossover_points] = y[crossover_points]
    offspring_2[crossover_points] = x[crossover_points]
    
    # do crossover for sigma
    offspring_1_sigma[crossover_points] = y_sigma[crossover_points]
    offspring_2_sigma[crossover_points] = x_sigma[crossover_points]
    
    offspring_1[rest] = x[rest]
    offspring_2[rest] = y[rest]
    offspring_1_sigma[rest] = x_sigma[rest]
    offspring_2_sigma[rest] = y_sigma[rest]
    
    return (offspring_1, offspring_1_sigma), (offspring_2, offspring_2_sigma)


### arithmetic/intermediate recombination
def simple_arithmetic_recomb(individual_1: Tuple[np.array, np.array], 
                    individual_2: Tuple[np.array, np.array], k: int, alpha: float = hyperparameters["alpha"]):
    """Generate offspring by simple arithmetic recombination.
    Args: individual_1, individual_2: Tuple[np.array, np.array], 
    k: crossover point. 
    Every gene after k of the 2 individuals will be applied the algorithm.
    """
    x = individual_1[0]
    y = individual_2[0]
    
    offspring_1 = np.zeros(x.shape[0])
    offspring_1[:k] = x[:k]
    
    offspring_2 = np.zeros(y.shape[0])
    offspring_2[:k] = y[:k]
    
    offspring_1[k:] = (1 - alpha) * x[k:] + alpha * y[k:]
    offspring_2[k:] = (1 - alpha) * y[k:] + alpha * x[k:]
    
    return (offspring_1, individual_1[1]), (offspring_2, individual_2[1])
    
    
def single_arithmetic_recomb(individual_1: Tuple[np.array, np.array],
                             individual_2: Tuple[np.array, np.array], k:int, alpha: float = hyperparameters["alpha"]):
    """Single point Arithmetic Recombination.
    Produces 2 offspring from 2 parents.

    Args:
        individual_1 (Tuple[np.array, np.array]): parent1
        individual_2 (Tuple[np.array, np.array]): parent2
        k (int): loci of allele to be recombined
        alpha (float, optional): ratio of parent1 and 2. Defaults to hyperparameters["alpha"].
    """
    x = individual_1[0]
    y = individual_2[0]
    
    offspring_1 = x.copy()
    offspring_2 = y.copy()
    
    offspring_1[k] = (1 - alpha) * x[k] + alpha * y[k]
    offspring_2[k] = (1 - alpha) * y[k] + alpha * x[k]
    
    return (offspring_1, individual_1[1]), (offspring_2, individual_2[1])

def whole_arithmetic_recomb(individual_1: Tuple[np.array, np.array], 
                    individual_2: Tuple[np.array, np.array], alpha: float = hyperparameters["alpha"]):
    """Generate offspring by whole arithmetic recombination.
    Args: individual_1, individual_2: Tuple[np.array, np.array], 
    """
    x = individual_1[0]
    y = individual_2[0]
    
    offspring_1 = (1 - alpha) * x + alpha * y
    offspring_2 = (1 - alpha) * y + alpha * x
    
    return (offspring_1, individual_1[1]), (offspring_2, individual_2[1])


def global_recomb(mating_pool: Sequence[Tuple[np.array, np.array]], offspring_size: int = -1):
    """Global Recombination.
    Produces offspring from mating pool. Each parent contribute 1 gene to the offspring.

    Args:
        mating_pool (Sequence[Tuple[np.array, np.array]]): mating pool
        offspring_size (int, optional): number of offspring to be produced. Defaults to -1. 
        If set to any positive number, offspring_size will be overriding the offspring_ratio in hyperparameters.

    Returns:
        Tuple[np.array, np.array]: offspring
    """
    if offspring_size < 0: # if not overridden
        offspring_size = int(len(mating_pool) * hyperparameters["offspring_ratio"])
    offspring = []
    
    individual_length = len(mating_pool[0][0])
    
    for _ in range(offspring_size):
        positions = [x for x in range(individual_length)]
        new_individual = (np.zeros(individual_length), np.zeros(individual_length))
        for _ in range(individual_length):
            parent = np.random.choice(mating_pool)
            locus = np.random.choice(positions)
            positions.remove(locus)
            new_individual[0][locus] = parent[0][locus]
            new_individual[1][locus] = parent[1][locus]
            offspring.append(new_individual)

    return offspring



def global_recombination(mating_pool, offspring_size, nstepsize_mode=True):

    # Require offspring_size to be even

    offspring = []
    
    individual_length = len(mating_pool[0][0])
    
    for _ in range(int(offspring_size / 2)):

        new_individual_1 = (np.zeros(individual_length), np.zeros(individual_length))
        new_individual_2 = (np.zeros(individual_length), np.zeros(individual_length))

        # X part
        for locus in range(individual_length):
            parents = np.random.choice(len(mating_pool), 2)
            new_individual_1[0][locus] = mating_pool[parents[0]][0][locus]
            new_individual_2[0][locus] = mating_pool[parents[1]][0][locus]

        # Stepsizes
        if nstepsize_mode:
            for locus in range(individual_length):
                parents = np.random.choice(len(mating_pool), 2)
                alpha = np.random.random()
                new_individual_1[1][locus] = mating_pool[parents[0]][1][locus] * alpha + mating_pool[parents[1]][1][locus] * (1 - alpha)
                new_individual_2[1][locus] = mating_pool[parents[1]][1][locus] * alpha +mating_pool[parents[0]][1][locus] * (1 - alpha)
        else:
            parents = np.random.choice(len(mating_pool), 2)
            alpha = np.random.random()
            new_individual_1[0] = mating_pool[parents[0]][1] * alpha + mating_pool[parents[1]][1] * (1 - alpha)
            new_individual_2[0] = mating_pool[parents[1]][1] * alpha + mating_pool[parents[0]][1] * (1 - alpha)

        offspring.extend([new_individual_1, new_individual_2])

    return offspring