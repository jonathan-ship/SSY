import os
import numpy as np
from time import time

from environment.steelstockyard import Locating
from environment.plate import *


def minimize_conflicts(s, env):
    s = s.reshape((env.max_stack, env.action_space))
    a_s = np.array([i for i in range(env.action_space)])
    idx_not_full = (np.min(s, axis=0) == -1)
    idx_empty = (np.max(s, axis=0) == -1)

    cond = (s == -1)
    s[cond] = np.max(s)

    Es = np.min(s, axis=0)
    idx_es = list(Es >= env.inbound_plates[0].outbound - env.inbound_plates[0].inbound)
    idx = np.logical_and(idx_not_full, idx_es)
    if sum(idx) == 0:
        idx_pos = idx_not_full
        a = np.random.choice(a_s[idx_pos])
    else:
        idx_pos = np.logical_or(idx_es, idx_empty)
        a = np.random.choice(a_s[idx_pos])

    return a


def delay_conflicts(s, env):
    s = s.reshape((env.max_stack, env.action_space))
    a_s = np.array([i for i in range(env.action_space)])
    idx_not_full = (np.min(s, axis=0) == -1)
    idx_empty = (np.max(s, axis=0) == -1)

    cond = (s == -1)
    s[cond] = np.max(s)

    Es = np.min(s, axis=0)
    idx_es = list(Es >= env.inbound_plates[0].outbound - env.inbound_plates[0].inbound)
    idx = np.logical_and(idx_not_full, idx_es)
    if sum(idx) == 0:
        idx_max_Es = (Es == max(Es[idx_not_full]))
        idx_pos = np.logical_or(np.logical_and(idx_max_Es, idx_not_full), idx_empty)
        a = np.random.choice(a_s[idx_pos])
    else:
        idx_pos = np.logical_or(np.logical_and(idx_es, idx_not_full), idx_empty)
        a = np.random.choice(a_s[idx_pos])

    return a

def flexibility_optimization(s, env):
    s = s.reshape((env.max_stack, env.action_space))
    a_s = np.array([i for i in range(env.action_space)])
    idx_not_full = (np.min(s, axis=0) == -1)
    idx_empty = (np.max(s, axis=0) == -1)

    cond = (s == -1)
    s[cond] = np.max(s)

    Es = np.min(s, axis=0)
    dF = (env.inbound_plates[0].outbound - env.inbound_plates[0].inbound) - Es
    idx_flex = (dF <= 0)
    idx = np.logical_and(idx_not_full, idx_flex)
    if sum(idx) == 0:
        idx_min_dF = (dF == min(dF[idx_not_full]))
        idx_pos = np.logical_or(np.logical_and(idx_min_dF, idx_not_full), idx_empty)
        a = np.random.choice(a_s[idx_pos])
    else:
        idx_max_dF = (dF == max(dF[idx]))
        idx_pos = np.logical_or(np.logical_and(idx_max_dF, idx_not_full), idx_empty)
        a = np.random.choice(a_s[idx_pos])

    return a


if __name__ == "__main__":
    np.random.seed(42)

    max_stack = 30
    num_pile = 20

    #algorithms = ["minimize_conflicts", "delay_conflicts", "flexibility_optimization"]
    algorithms = ["delay_conflicts"]
    num_plate = [100, 150, 200, 250, 300, 350, 400]
    num_instance = 10

    result_path = './result/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    for algo in algorithms:
        results = pd.DataFrame(index=num_plate, columns=["move", "time"])
        for num_p in num_plate:
            moves = []
            times = []
            for num_i in range(num_instance):
                df = pd.read_csv("./data/data_plate{0}_{1}.csv".format(num_p, num_i))
                plates = [[]]
                for i, row in df.iterrows():
                    plate = Plate(row['plate_id'], row['inbound'], row['outbound'])
                    plates[0].append(plate)

                env = Locating(max_stack=max_stack, num_pile=num_pile, inbound_plates=plates)

                start = time()
                s = env.reset()

                while True:
                    if algo == "minimize_conflicts":
                        a = minimize_conflicts(s, env)
                    elif algo == "delay_conflicts":
                        a = delay_conflicts(s, env)
                    elif algo == "flexibility_optimization":
                        a = flexibility_optimization(s, env)
                    s1, r, d = env.step(a)
                    s = s1

                    if d:
                        finish = time()
                        times.append(finish - start)
                        moves.append(env.crane_move)
                        break
            time_avg = np.mean(times)
            move_avg = np.mean(moves)
            results.loc[num_p] = [move_avg, time_avg]

        results.to_csv(result_path + "final_{0}.csv".format(algo))