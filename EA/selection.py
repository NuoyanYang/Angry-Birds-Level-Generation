import numpy as np
from typing import Tuple, Sequence

hyperparameters = {
    "select_size": 10 
}


def uniform(
    
    population: Sequence[Tuple[np.array, np.array]],
    select_size: int = hyperparameters["select_size"],
    allow_dup: bool = False
    
    ) -> Sequence[Tuple[np.array, np.array]]:
    
    if select_size > len(population) and not allow_dup:
        raise ValueError("select_size must be smaller than the population size.")
    
    selected = np.random.choice(len(population), select_size, replace=(not allow_dup))
    
    return [population[i] for i in selected]
    
