#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import tensorlayer as tl
import time
import Env
import random
from tensorlayer.layers import *

tf.logging.set_verbosity(tf.logging.DEBUG)
tl.logging.set_verbosity(tl.logging.DEBUG)

env = Env.Env()

def create_standard_model():
    #Create the model
    inputs = tf.placeholder(shape=[None, 11], dtype=tf.float32)
    net = InputLayer(inputs, name='observation')
    net = DenseLayer(net, 120, act=tf.nn.relu,  W_init=tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32), name='q_d_s')
    net = DenseLayer(net, 120, act=tf.nn.relu,  W_init=tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32), name='q_t_s')
    net = DenseLayer(net, 120, act=tf.nn.relu,  W_init=tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32), name='q_e_s')
    net = DenseLayer(net, 3, act=tf.nn.softmax, W_init=tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32),  name='q_a_s')
    return net, inputs

def create_conv_model():
    #Create the model
    inputs = tf.placeholder(shape=[None, 22, 22], dtype=tf.float32)
    net = InputLayer(inputs, name='observation')
    net = Conv1d(net, 5, 1, act=tf.nn.relu, padding='VALID', name='conv1_1')
    net = MaxPool1d(net, 2, padding='VALID', name='pool1')
    net = Conv1d(net, 3, 2, act=tf.nn.relu, padding='VALID', name='conv1_2')
    net = MaxPool1d(net, 2, padding='VALID', name='pool2')
    net = FlattenLayer(net, name='flatten')
    net = DenseLayer(net, 120, act=tf.nn.relu,  W_init=tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32), name='q_d_s')
    net = DenseLayer(net, 120, act=tf.nn.relu,  W_init=tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32), name='q_t_s')
    net = DenseLayer(net, 120, act=tf.nn.relu,  W_init=tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32), name='q_e_s')
    net = DenseLayer(net, 3, act=tf.nn.softmax, W_init=tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32),  name='q_a_s')
    return net, inputs

conv_model = True

if conv_model:
    net, inputs = create_conv_model()
    state_size = 484
else:
    net, inputs = create_standard_model()
    state_size = 11

y = net.outputs

nextQ = tf.placeholder(shape=[None, 3], dtype=tf.float32)
loss = tl.cost.mean_squared_error(nextQ, y)  # tf.reduce_sum(tf.square(nextQ - y))
train_op = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(loss)

lambd = .9  # decay factor
e = 0.9 #randomness, decays over time
batch_size = 1000

experiences = np.empty((0, 3), dtype=int)
init_states = np.empty((0, state_size))
target_states = np.empty((0, state_size))

saver = tf.train.Saver()
train_new = True

with tf.Session() as sess:

    if not train_new:
        saver.restore(sess, "tmp/model.ckpt")

    tl.layers.initialize_global_variables(sess)
    net.print_layers()
    net.print_params()

    loss_list = list()
    start_time = time.time()
    for i in range(200):

        if len(experiences) > batch_size * 10:
            experiences = experiences[batch_size:]
        j = 0

        while(True):
            j += 1
            env.reward = 0

            # Reshapes depend on the used model
            if conv_model:
                s = env.matrix.reshape(1, 22, 22)
            else:
                s = env.get_features().reshape(1, 11)

            ## Choose an action by greedily (with e chance of random action) from the Q-network
            # allQ = net.predict(s)
            allQ = sess.run(y, feed_dict={inputs: s})
            # a = sess.run(tf.argmax(allQ))
            a = np.argmax(allQ)
            ## e-Greedy Exploration !!! sample random action
            if np.random.rand(1) < e:
                a = random.randint(0,2)
            ## Get new state and reward from environments
            r, dead = env.take_action(a)
            ## Obtain the Q' values by feeding the new state through our network
            if conv_model:
                s1 = env.matrix.reshape(1, 22, 22)
            else:
                s1 = env.get_features()
            # s1 = env.matrix.reshape(22, 22)
            state_vars = np.array([a, r, dead])

            Q1 = sess.run(y, feed_dict={inputs: s1})
            maxQ1 = np.max(Q1)  # in Q-Learning, policy is greedy, so we use "max" to select the next action.
            allQ[0][a] = r + lambd * maxQ1


            _, losss = sess.run([train_op, loss], {inputs: s, nextQ: allQ})

            if j < batch_size:
                init_states = np.append(init_states, s.reshape(1, state_size), axis=0)
                target_states = np.append(target_states, s1.reshape(1, state_size), axis=0)
                experiences = np.vstack((experiences, np.array([a, r, dead])))

            if dead:
                break

        print("LENGTH::: ", env.score)

        env.reset_env()

        # Experience replay
        if len(experiences) > batch_size:
            # Take a sample of size batch_size
            index = np.random.choice(experiences.shape[0], batch_size, replace=False)
            init_state_sample = init_states[index, :]
            target_state_sample = target_states[index, :]
            experiences_sample = experiences[index, :]

            if conv_model:
                init_state_sample = init_state_sample.reshape(1000, 22, 22)
                target_state_sample = target_state_sample.reshape(1000, 22, 22)


            targetQ = sess.run([y], feed_dict={inputs: init_state_sample})
            targetQ = np.squeeze(np.array(targetQ))
            batch_loss = 0
            Q1 = sess.run(y, feed_dict={inputs: target_state_sample}) # Get q values of target state
            maxQ1 = np.expand_dims(np.amax(Q1, axis=1), axis=1)
            actions = experiences_sample[:, 0] # Get the actions array
            rewards = np.expand_dims(experiences_sample[:, 1], axis=1) # Get the rewards array

            # Update q-values  of action with reward and  lambda
            targetQ[np.arange(targetQ.shape[0]), actions] = rewards[np.arange(rewards.shape[0]), 0] + lambd * maxQ1[np.arange(maxQ1.shape[0]), 0]
            _, losss = sess.run([train_op, loss], {inputs: init_state_sample, nextQ: targetQ})

        if i % 10 == 0:
            e -= 0.1
            print("EPOCH NUM: ", i, " Took: ", time.time() - start_time)
            start_time = time.time()

    save_path = saver.save(sess, "tmp/model.ckpt")
    print("Model saved in path: %s" % save_path)
