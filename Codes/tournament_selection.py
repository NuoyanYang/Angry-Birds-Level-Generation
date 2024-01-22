# Tournament selection no repeat

import random

def evaluate(individual):
    """
    :return: fitness
    """
    return 100*individual


def tournament_selection(population, winner_size, tournament_size):

    # Returned list
    selected_to_mate_index = []

    # Indices for the contestants to participate in tournament
    contestant_index = [_ for _ in range(0,len(population))]

    # Fitness dictionary: contestant index -> fitness
    index_to_fitness = {}

    for _ in range(winner_size):
        # Obtain indices of individuals to join the tournament
        tournament_index = random.sample(contestant_index, tournament_size)

        # Construct tournanment
        # tournament index -> fitness
        tournament_fitness = []
        for i in range(len(tournament_index)):
            cont_index = tournament_index[i]
            if cont_index in index_to_fitness.keys():
                tournament_fitness.append(index_to_fitness[cont_index])
            else:
                individual_fitness = evaluate(population[cont_index])
                tournament_fitness.append(individual_fitness)
                index_to_fitness[cont_index] = individual_fitness


        # Obtain all winners and randomly pick one.
        winner_pool = []
        max_fitness = max(tournament_fitness)
        for j in range(len(tournament_fitness)):
            if tournament_fitness[j] == max_fitness:
                winner_pool.append(tournament_index[j])

        winner_index = random.sample(winner_pool, 1)
        selected_to_mate_index.append(winner_index[0])

        # No replace winners
        contestant_index.remove(winner_index[0])

    return selected_to_mate_index

def main():
    population = range(20)
    winner_size = 5
    tournament_size = 2

    winners = tournament_selection(population, winner_size, tournament_size)
    print(winners)

main()