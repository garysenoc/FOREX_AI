from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import pandas as pd
import os
import sys

sys.path.append('environment')
from env import ForexEnv

from tf_agents.environments import utils
from tf_agents.environments import tf_py_environment

from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import q_network
from tf_agents.utils import common

from learn import learningHelper
import matplotlib.pyplot as plt


def train(model,epoch):
    model.train_agent(epoch)


def main():


    tf.random.set_seed(12)
    tf.print(tf.config.list_physical_devices('GPU') )
    tf.compat.v1.enable_v2_behavior()

    environment = ForexEnv(is_evaluation=True)
    utils.validate_py_environment(environment, episodes=3)

    print('action_spec:', environment.action_spec())
    print('time_step_spec.observation:', environment.time_step_spec().observation)
    print('time_step_spec.step_type:', environment.time_step_spec().step_type)
    print('time_step_spec.discount:', environment.time_step_spec().discount)
    print('time_step_spec.reward:', environment.time_step_spec().reward)

    train_env = tf_py_environment.TFPyEnvironment(ForexEnv())
    eval_env = tf_py_environment.TFPyEnvironment(ForexEnv(is_evaluation=True))

    start = 1000
    goal = 1050

    # fig, ax = plt.subplots()
    # plt.axhline(y = start,color="brown",label="Start")
    # plt.axhline(y = goal,color="blue",label="Goal")
    # ax.plot(environment.reward_list, color = 'green', label = 'Rewards')
    # ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    # # plt.show()

    learning_rate = 1e-3  

    #network configuration
    fc_layer_params = (40,)

    # as we are using dictionary in our enviroment, we will create preprocessing layer
    preprocessing_layers = {
        'price': tf.keras.layers.Flatten(),
        'pos': tf.keras.layers.Dense(2)
        }
    preprocessing_combiner = tf.keras.layers.Concatenate(axis=-1)

    #create a q_net
    q_net = q_network.QNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        preprocessing_layers=preprocessing_layers,
        preprocessing_combiner=preprocessing_combiner,
        fc_layer_params=fc_layer_params)

    #create optimizer
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

    #create a global step coubter
    #train_step_counter = tf.Variable(0)
    global_step = tf.compat.v1.train.get_or_create_global_step()

    #create agent
    agent = dqn_agent.DqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        #train_step_counter=train_step_counter)
        train_step_counter=global_step)

    agent.initialize()

    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    agent.train = common.function(agent.train)

    magent = learningHelper(train_env=train_env, test_env=eval_env, agent=agent, global_step=global_step, collect_episodes = 10000,
    eval_interval=5, verbose=0, batch_size=64, chkpdir='./fc_chkp/')
    magent.restore_check_point()

    train(magent,20)
#     magent.train_agent(1)
  

    #magent.train_agent_with_avg_ret_condition(100, 10000, 100)
    magent.store_check_point()
    magent.restore_check_point()
    magent.save_policy()
    
if __name__ == "__main__":
    main()
