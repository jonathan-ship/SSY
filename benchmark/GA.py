import pygad
import random
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

from time import time

from environment.steelstockyard import Locating
from environment.plate import *


if __name__ == '__main__':
    np.random.seed(42)

    max_stack = 30
    num_pile = 20

    num_plate = [100, 150, 200, 250, 300, 350, 400]
    num_instance = 10

    result_path = './benchmark/result/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    results = pd.DataFrame(index=num_plate, columns=["move", "time"])
    for num_p in num_plate:
        moves = []
        times = []
        for num_i in range(num_instance):
            df = pd.read_csv("./environment/data_plate{0}_{1}.csv".format(num_p, num_i))
            plates = [[]]
            for i, row in df.iterrows():
                plate = Plate(row['plate_id'], row['inbound'], row['outbound'])
                plates[0].append(plate)

            env = Locating(max_stack=max_stack, num_pile=num_pile, inbound_plates=plates)

            def fitness_function(solution, solution_index):
                env.step(solution)
                env._export_all_plates()
                fit = env.crane_move
                env.reset()
                return -fit

            last_fitness = 0
            def on_generation(ga_instance):
                global last_fitness
                print("Generation = {generation}".format(generation=ga_instance.generations_completed))
                print("Fitness    = {fitness}".format(
                    fitness=-ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]))

            start = time()

            inbound_GA = pygad.GA(num_generations=100000,
                                  num_parents_mating=2,
                                  fitness_func=fitness_function,
                                  num_genes=len(plates[0]),
                                  gene_type=int,
                                  sol_per_pop=10,
                                  init_range_low=0,
                                  init_range_high=num_pile - 1,
                                  parent_selection_type="tournament",
                                  crossover_type="two_points",
                                  mutation_type="random",
                                  mutation_probability=0.05,
                                  random_mutation_min_val=0,
                                  random_mutation_max_val=num_pile - 1,
                                  gene_space=list(range(num_pile)),
                                  on_generation=on_generation,
                                  save_best_solutions=True)
            inbound_GA.run()

            best_fitness = max(inbound_GA.best_solutions_fitness)
            best = inbound_GA.best_solutions[inbound_GA.best_solutions_fitness.index(best_fitness)]
            print("Parameters of the best solution : {solution}".format(solution=best))
            print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=-best_fitness))

            if inbound_GA.best_solution_generation != -1:
                print("Best fitness value reached after {best_solution_generation} generations.".format(
                    best_solution_generation=inbound_GA.best_solution_generation))
            finish = time()
            times.append(finish - start)
            moves.append(-best_fitness)
            print('time : ', finish - start)
            print('move: ', -best_fitness)

        time_avg = np.mean(times)
        move_avg = np.mean(moves)
        print('time_avg : ', time_avg)
        print('move_avg: ', move_avg)

        results.loc[num_p] = [move_avg, time_avg]

    results.to_csv(result_path + "final_GA.csv")