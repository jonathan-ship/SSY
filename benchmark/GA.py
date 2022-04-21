import os
import pygad
import random
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

from time import time
from environment.plate import *


class Locating_for_GA(object):  # 생성자에서 파일의 수, 최대 높이 등을 입력
    def __init__(self, num_pile=4, max_stack=4, inbound_plates=None, observe_inbounds=False, display_env=False):
        self.action_space = num_pile  # 가능한 action 수는 파일의 수로 설정
        self.max_stack = max_stack  # 한 파일에 적치 가능한 강재의 수
        self.empty = -1  # 빈 공간의 상태 표현 값
        self.stage = 0
        self.current_date = 0
        self.crane_move = 0
        self.plates = [[] for _ in range(num_pile)]  # 각 파일을 빈 리스트로 초기화
        self.n_features = max_stack * num_pile
        self.observe_inbounds = observe_inbounds
        self.done = False
        if inbound_plates:
            self.inbound_plates = inbound_plates
            self.inbound_clone = self.inbound_plates[:]
        else:
            self.inbound_plates = generate_schedule()
            self.inbound_clone = self.inbound_plates[:]

    def step(self, solution):
        for i in range(len(solution)):
            inbound = self.inbound_plates.pop(0)  # 입고 강재 리스트 가장 위에서부터 강재를 하나씩 입고
            if len(self.plates[solution[i]]) == self.max_stack:  # 적치 강재가 최대 높이를 초과하면 실패로 간주
                self.done = True
            else:
                self.plates[solution[i]].append(inbound)  # action 에 따라서 강재를 적치
                self.stage += 1

    def reset(self):
        self.inbound_plates = self.inbound_clone[:]
        self.plates = [[] for _ in range(self.action_space)]
        self.current_date = min(self.inbound_plates, key=lambda x: x.inbound).inbound
        self.crane_move = 0
        self.stage = 0
        self.done = False

    def _export_plates(self):
        for pile in self.plates:
            outbounds = []
            for i, plate in enumerate(pile):
                if plate.outbound <= self.current_date:
                    outbounds.append(i)
            if len(outbounds) > 0:
                self.crane_move += (len(pile) - outbounds[0] - len(outbounds))
            for index in outbounds[::-1]:
                del pile[index]

    def _export_all_plates(self):
        next_states = []
        if self.done == True:
            self.crane_move = 800
            return next_states
        while True:
            next_outbound_date = min(sum(self.plates, []), key=lambda x: x.outbound).outbound
            if next_outbound_date != self.current_date:
                self.current_date = next_outbound_date
                self._export_plates()
            if not sum(self.plates, []):
                break
        return next_states


if __name__ == '__main__':
    np.random.seed(42)

    max_stack = 30
    num_pile = 20

    num_plate = [100, 120, 140, 160, 180, 200]
    num_instance = 30

    result_path = './result/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    results = pd.DataFrame(index=num_plate, columns=["move", "time"])
    for num_p in num_plate:
        moves = []
        times = []
        for num_i in range(num_instance):
            print("the number of plates : {0} | test-{1}".format(num_p, num_i))
            df = pd.read_csv("./data/data_plate{0}_{1}.csv".format(num_p, num_i))
            plates = [[]]
            for i, row in df.iterrows():
                plate = Plate(row['plate_id'], row['inbound'], row['outbound'])
                plates[0].append(plate)

            env = Locating_for_GA(max_stack=max_stack, num_pile=num_pile, inbound_plates=plates[0])

            def fitness_function(solution, solution_index):
                env.step(solution)
                env._export_all_plates()
                fit = env.crane_move
                env.reset()
                return -fit

            last_fitness = 0
            def on_generation(ga_instance):
                global last_fitness
                # print("Generation = {generation}".format(generation=ga_instance.generations_completed))
                # print("Fitness    = {fitness}".format(
                #     fitness=-ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]))

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