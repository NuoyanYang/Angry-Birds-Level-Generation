# Parent selection - overselection: Uniformly select non-winners to fill in the rest of mating pool
# Return indices

import random

# non_winner_size: int, number of individuals that lost the tournament
# mating_pool_size: int, size of the mating pool reserved for individuals that lost the tournament
def random_uniform(non_winner_size, mating_pool_size):
    """Random uniform selection"""

    # To uniformly select the non-winners
    selected_to_mate_index = []
    selected_to_mate_index = random.sample(range(0, non_winner_size), mating_pool_size)

    return selected_to_mate_index
