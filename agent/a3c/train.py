import threading
import multiprocessing
import os
import numpy as np
import tensorflow as tf
import scipy.signal

from agent.a3c.network import AC_Network
from agent.a3c.helper import *
from environment.steelstockyard import Locating
from environment.plate import *

from random import choice
from time import sleep
from time import time

# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope,to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


# Discounting function used to calculate discounted returns.
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


class Worker():
    def __init__(self, env, name, s_shape, a_size, trainer, model_path, summary_path, global_episodes):
        self.env = env
        self.name = "worker_" + str(name)
        self.number = name
        self.s_shape = s_shape
        self.model_path = model_path
        self.summary_path = summary_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter(self.summary_path + "/train_" + str(self.number))

        # Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_AC = AC_Network(s_shape, a_size, self.name, trainer)
        self.update_local_ops = update_target_graph('global', self.name)

    def train(self, rollout, sess, gamma, bootstrap_value):
        rollout = np.array(rollout)
        observations = rollout[:, 0]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        next_observations = rollout[:, 3]
        values = rollout[:, 5]

        # Here we take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted returns.
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus, gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages, gamma)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        feed_dict = {self.local_AC.target_v: discounted_rewards,
                     self.local_AC.inputs: np.vstack(observations),
                     self.local_AC.actions: actions,
                     self.local_AC.advantages: advantages,
                     self.local_AC.state_in[0]: self.batch_rnn_state[0],
                     self.local_AC.state_in[1]: self.batch_rnn_state[1]}
        v_l, p_l, e_l, g_n, v_n, self.batch_rnn_state, _ = sess.run([self.local_AC.value_loss,
                                                                     self.local_AC.policy_loss,
                                                                     self.local_AC.entropy,
                                                                     self.local_AC.grad_norms,
                                                                     self.local_AC.var_norms,
                                                                     self.local_AC.state_out,
                                                                     self.local_AC.apply_grads],
                                                                    feed_dict=feed_dict)
        return v_l / len(rollout), p_l / len(rollout), e_l / len(rollout), g_n, v_n

    def work(self, max_episode_length, max_episode, gamma, sess, coord, saver):
            episode_count = sess.run(self.global_episodes)
            total_steps = 0
            print("Starting worker " + str(self.number))
            with sess.as_default(), sess.graph.as_default():
                while not coord.should_stop() and episode_count < max_episode:
                    sess.run(self.update_local_ops)
                    episode_buffer = []
                    episode_values = []
                    episode_frames = []
                    episode_reward = 0
                    episode_step_count = 0
                    d = False

                    s = self.env.reset()
                    episode_frames.append(s)
                    rnn_state = self.local_AC.state_init
                    self.batch_rnn_state = rnn_state
                    while True:
                        # Take an action using probabilities from policy network output.
                        a_dist, v, rnn_state = sess.run(
                            [self.local_AC.policy, self.local_AC.value, self.local_AC.state_out],
                            feed_dict={self.local_AC.inputs: [s],
                                       self.local_AC.state_in[0]: rnn_state[0],
                                       self.local_AC.state_in[1]: rnn_state[1]})
                        a = np.random.choice(a_dist[0], p=a_dist[0])
                        a = np.argmax(a_dist == a)

                        s1, r, d = self.env.step(a)
                        if not d:
                            episode_frames.append(s1)
                        else:
                            s1 = s

                        episode_buffer.append([s, a, r, s1, d, v[0, 0]])
                        episode_values.append(v[0, 0])

                        episode_reward += r
                        s = s1
                        total_steps += 1
                        episode_step_count += 1

                        # If the episode hasn't ended, but the experience buffer is full, then we
                        # make an update step using that experience rollout.
                        if len(episode_buffer) == 30 and d != True and episode_step_count != max_episode_length - 1:
                            # Since we don't know what the true final return is, we "bootstrap" from our current
                            # value estimation.
                            v1 = sess.run(self.local_AC.value,
                                          feed_dict={self.local_AC.inputs: [s],
                                                     self.local_AC.state_in[0]: rnn_state[0],
                                                     self.local_AC.state_in[1]: rnn_state[1]})[0, 0]
                            v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, sess, gamma, v1)
                            episode_buffer = []
                            sess.run(self.update_local_ops)
                        if d:
                            break

                    self.episode_rewards.append(episode_reward)
                    self.episode_lengths.append(episode_step_count)
                    self.episode_mean_values.append(np.mean(episode_values))

                    # Update the network using the episode buffer at the end of the episode.
                    if len(episode_buffer) != 0:
                        v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, sess, gamma, 0.0)

                    # Periodically save gifs of episodes, model parameters, and summary statistics.
                    if episode_count % 5 == 0 and episode_count != 0:
                        if self.name == 'worker_0' and episode_count % 20 == 0:
                            save_gif(episode_frames, self.s_shape, episode_count, 'a3c')
                        if episode_count % 250 == 0 and self.name == 'worker_0':
                            saver.save(sess, self.model_path + '/model-' + str(episode_count) + '.cptk')
                            print("Saved Model at episode %d" % episode_count)

                        mean_reward = np.mean(self.episode_rewards[-5:])
                        mean_length = np.mean(self.episode_lengths[-5:])
                        mean_value = np.mean(self.episode_mean_values[-5:])
                        summary = tf.Summary()
                        summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                        summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                        summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                        summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                        summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                        summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                        summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                        summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
                        self.summary_writer.add_summary(summary, episode_count)

                        self.summary_writer.flush()
                    if self.name == 'worker_0':
                        sess.run(self.increment)
                    episode_count += 1


if __name__ == '__main__':
    inbounds = import_plates_schedule_by_week('../../environment/data/SampleData.csv')
    max_episode_length = 300
    max_episode = 50000
    gamma = .99  # discount rate for advantage estimation and reward discounting

    max_stack = 10
    num_pile = 8

    observe_inbounds = True
    if observe_inbounds:
        s_shape = (max_stack, num_pile + 1)
    else:
        s_shape = (max_stack, num_pile)
    a_size = num_pile

    load_model = False
    model_path = '../../models/a3c/%d-%d' % s_shape
    frame_path = '../../frames/a3c/%d-%d' % s_shape
    summary_path = '../../summary/a3c/%d-%d' % s_shape
    tf.reset_default_graph()

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    if not os.path.exists(frame_path):
        os.makedirs(frame_path)

    if not os.path.exists(summary_path):
        os.makedirs(summary_path)

    with tf.device("/cpu:0"):
        global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
        trainer = tf.train.AdamOptimizer(learning_rate=5e-5)
        master_network = AC_Network(s_shape, a_size, 'global', None)  # Generate global network
        num_workers = multiprocessing.cpu_count()  # Set workers to number of available CPU threads
        workers = []
        # Create worker classes
        for i in range(num_workers):
            locating = Locating(max_stack=max_stack, num_pile=num_pile, inbound_plates=inbounds,
                                observe_inbounds=observe_inbounds, display_env=False)
            workers.append(Worker(locating, i, s_shape, a_size, trainer, model_path, summary_path, global_episodes))
        saver = tf.train.Saver(max_to_keep=5)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        if load_model:
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(model_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

        # This is where the asynchronous magic happens.
        # Start the "work" process for each worker in a separate threat.
        worker_threads = []
        for worker in workers:
            worker_work = lambda: worker.work(max_episode_length, max_episode, gamma, sess, coord, saver)
            t = threading.Thread(target=(worker_work))
            t.start()
            sleep(0.5)
            worker_threads.append(t)
        coord.join(worker_threads)