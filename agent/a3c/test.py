import os
import tensorflow as tf

from agent.a3c.helper import *
from agent.a3c.network import AC_Network
from environment.steelstockyard import Locating
from environment.plate import *


if __name__ == '__main__':
    inbounds = import_plates_schedule_by_week('../../environment/data/SampleData.csv')

    max_stack = 10
    num_pile = 8

    observe_inbounds = True
    if observe_inbounds:
        s_shape = (max_stack, num_pile + 1)
    else:
        s_shape = (max_stack, num_pile)
    a_size = num_pile

    model_path = '../../models/a3c/%d-%d' % s_shape

    env = Locating(max_stack=max_stack, num_pile=num_pile, inbound_plates=inbounds,
                        observe_inbounds=observe_inbounds, display_env=False)

    tf.reset_default_graph()
    with tf.Session() as sess:
        network = AC_Network(s_shape, a_size, 'global', None)
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver = tf.train.Saver(max_to_keep=5)
        saver.restore(sess, ckpt.model_checkpoint_path)

        s = env.reset()
        episode_frames = []
        rnn_state = network.state_init

        while True:
            a_dist, v, rnn_state = sess.run(
                [network.policy, network.value, network.state_out],
                feed_dict={network.inputs: [s],
                           network.state_in[0]: rnn_state[0],
                           network.state_in[1]: rnn_state[1]})
            a = np.random.choice(a_dist[0], p=a_dist[0])
            a = np.argmax(a_dist == a)

            s1, r, d = env.step(a)

            if not d:
                episode_frames.append(s1)
            else:
                save_gif(episode_frames, s_shape, '-result', 'a3c')
                break

            s = s1
