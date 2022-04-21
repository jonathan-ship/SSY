import os
import tensorflow as tf

from time import time
from agent.a3c.helper import *
from agent.a3c.network import AC_Network
from environment.steelstockyard import Locating
from environment.plate import *


if __name__ == '__main__':
    np.random.seed(42)

    max_stack = 30
    num_pile = 20

    observe_inbounds = True
    if observe_inbounds:
        s_shape = (max_stack, num_pile + 1)
    else:
        s_shape = (max_stack, num_pile)
    a_size = num_pile

    model_path = './results/models/%d-%d' % s_shape

    num_plate = [100, 120, 140, 160, 180, 200]
    num_test = 30

    result_path = './results/test/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    results = pd.DataFrame(index=num_plate, columns=["move", "time"])
    for num_p in num_plate:
        moves = []
        times = []
        for num_i in range(num_instance):
            df = pd.read_csv("../../benchmark/data/data_plate{0}_{1}.csv".format(num_p, num_i))
            plates = [[]]
            for i, row in df.iterrows():
                plate = Plate(row['plate_id'], row['inbound'], row['outbound'])
                plates[0].append(plate)

            env = Locating(max_stack=max_stack, num_pile=num_pile, inbound_plates=plates,
                                observe_inbounds=observe_inbounds, display_env=False)

            tf.reset_default_graph()
            with tf.Session() as sess:
                network = AC_Network(s_shape, a_size, 'global', None)
                ckpt = tf.train.get_checkpoint_state(model_path)
                saver = tf.train.Saver(max_to_keep=5)
                saver.restore(sess, ckpt.model_checkpoint_path)

                s = env.reset()
                start = time()
                #episode_frames = []
                rnn_state = network.state_init

                while True:
                    a_dist, v, rnn_state = sess.run(
                        [network.policy, network.value, network.state_out],
                        feed_dict={network.inputs: [s],
                                   network.state_in[0]: rnn_state[0],
                                   network.state_in[1]: rnn_state[1]})

                    # s_temp = s.reshape((env.max_stack, env.action_space + 1))[:, 1:]
                    # a_s = np.array([i for i in range(env.action_space)])
                    # idx_full = (np.min(s_temp, axis=0) != -1)
                    # a_dist[0][idx_full] = 0
                    # a_dist[0] = a_dist[0] / (sum(a_dist[0]))

                    a = np.random.choice(a_dist[0], p=a_dist[0])
                    a = np.argmax(a_dist == a)

                    s1, r, d = env.step(a)
                    s = s1

                    if not d:
                        pass
                        #episode_frames.append(s1)
                    else:
                        finish = time()
                        times.append(finish - start)
                        moves.append(env.crane_move)
                        #save_gif(episode_frames, s_shape, '-result', 'a3c')
                        break

        time_avg = np.mean(times)
        move_avg = np.mean(moves)
        results.loc[num_p] = [move_avg, time_avg]

    results.to_csv(result_path + "final_RL.csv")