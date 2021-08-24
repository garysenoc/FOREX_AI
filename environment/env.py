from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import numpy as np
import pandas as pd
import constants
import time
from datetime import datetime
from get_data import fetch_data

from tf_agents.environments import py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import common



#environment
class ForexEnv(py_environment.PyEnvironment):

    # used for update enviroment observation step
    # to updating observation step, only this function should be used
    def __update_state(self, init=False):
        # state is a dictionary wich consist of price data and position state
        # TODO: Enable refetch of candle on every step
        if not init and not self.is_evaluation:    
            time.sleep(constants.SLEEP_TIME)
            self.price_data = fetch_data()
            
        self._state = {
            'price' : self.price_data.values.astype(np.float32),
            'pos' : self.active_position.astype(np.int32)
            }
    
    def append_reward(self, reward):
        if not self.is_evaluation:
            with open('rewards.txt', 'a') as file:
                file.write(f'{reward}\n')

    def __init__(self, is_evaluation=False):
        
        self.price_data = fetch_data()
        self.episode_index = 1
        self.is_evaluation = is_evaluation
        
        self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=3, name='action')

        # [0, 0] no open position
        # [0, 1] long position
        # [1, 0] short position
        self._observation_spec = {
            'price':array_spec.BoundedArraySpec(shape=self.price_data.shape, dtype=np.float32, minimum=0, name='obs_price'),
            'pos':array_spec.BoundedArraySpec(shape=(2,), dtype=np.int32, minimum=0, maximum=1, name='obs_pos')
        }


                
        self._episode_ended = False
        self.reward_list = []
        
        self.position = constants.NOTHELD_POS
        
        # environment variables
        self._current_iter_index = 0
        self.spread = constants.SPREAD
        self.pip_value = constants.PIP_VALUE
        self.stop_loss = constants.STOP_LOSS
        self.position_price = 0.0
        self.position_stop_loss = 0.0
        
        self.reward_terminator = constants.LOWER_BOUNDARY
        self.reward = constants.REWARD # initial reward value (USD)
        
        self.max_iters = constants.MAX_ACTIONS
        self.reward_multiplier = 1000
                
        self.active_position = np.array([0, 0])
        
        self.__update_state(init=True)
       
    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._episode_ended = False
        self._current_iter_index = 0
        self.spread = constants.SPREAD
        self.pip_value = constants.PIP_VALUE
        self.stop_loss = constants.STOP_LOSS
        self.position_price = 0.0
        self.position_stop_loss = 0.0
        self.reward = constants.REWARD

        self.__reset_position()
        self.__update_state()

        return ts.restart(self._state)    

    def __reset_position(self):
        self._position = constants.NOTHELD_POS
        self._activePosition = np.array([0,0]).astype(np.int32)

    def _step(self, action):

        if self._episode_ended:
            return self._reset()
                
        self._current_iter_index += 1
        
        index = self._current_iter_index
                        
        bid_close_price = self.price_data.tail(1).Close.values[0]
        
        if not self.is_evaluation:
            print(bid_close_price)
        
        # proposal algorithm
        if (self.position == constants.NOTHELD_POS):
            if (action == constants.ACTION_BUY):
                self.active_position[0] = 1.0 
                
                self.position = constants.LONG_POS
                self.position_price = bid_close_price + self.spread
                self.position_stop_loss = bid_close_price - self.stop_loss
                self.reward += 5
            elif (action == constants.ACTION_SELL):
                self.active_position[1] = 1.0
                
                self.position = constants.SHORT_POS
                self.position_price = bid_close_price - self.spread
                self.position_stop_loss = bid_close_price + self.stop_loss
                self.reward += 5
            elif (action == constants.ACTION_CLOSE):
                self.reward -= (self.pip_value * self.stop_loss * self.reward_multiplier)
            elif (action == constants.ACTION_SKIP):
                self.reward += 5
        
        elif (self.position == constants.LONG_POS):
            if action in (constants.ACTION_BUY, constants.ACTION_SELL):
                self.reward -= (self.pip_value * self.stop_loss * self.max_iters)
            elif (action == constants.ACTION_CLOSE):
                self.reward += (bid_close_price - self.position_price) * self.pip_value * self.reward_multiplier
                self.position = constants.NOTHELD_POS
                self.active_position = np.array([0,0])
                self.position_price = 0.0
                self.position_stop_loss = 0.0
            elif (action == constants.ACTION_SKIP):
                  self.reward += 5   
            if self.position_stop_loss >= bid_close_price:
                    self.reward -= (self.pip_value * self.stop_loss * self.max_iters)
                    self.position = constants.NOTHELD_POS
                    self.position_price = 0.0
                    self.position_stop_loss = 0.0
               
            
        elif (self.position == constants.SHORT_POS):
            if action in (constants.ACTION_BUY, constants.ACTION_SELL):
                self.reward -= (self.pip_value * self.stop_loss * self.reward_multiplier)
            elif action == constants.ACTION_CLOSE:
                self.reward += (self.position_price - bid_close_price) * self.pip_value * self.reward_multiplier
                self.position = constants.NOTHELD_POS
                self.active_position = np.array([0,0])
                self.position_price = 0.0
                self.position_stop_loss = 0.0
            elif action == constants.ACTION_SKIP:
                self.reward += 5
                if self.position_stop_loss <= bid_close_price:
                    self.reward -= (self.pip_value * self.stop_loss * self.max_iters)
                    self.position = constants.NOTHELD_POS
                    self.position_price = 0.0
                    self.position_stop_loss = 0.0
       
        
        self.reward = round(self.reward, 5)
        self.__update_state()
#         print(f'Iteration #{index}-> Reward = {self.reward}')
#         print(f'Position Price = {self.position_price}, position_stop_loss = {self.position_stop_loss}\n')
        if not self.is_evaluation:
            print(f'Iteration #{self._current_iter_index} ended ... Time: {datetime.now()}, Pos = {self.active_position}, Reward = {self.reward}, Action = {action}',)
            print(f'self._state')
        if self.reward < 100 or self._current_iter_index >= self.max_iters:
            self._episode_ended = True
        
            if self._episode_ended:
                print(f'Episode #{self.episode_index} ended. . .')
                self.episode_index += 1
                # print time
                print(datetime.now())
                # write reward in text file
                self.append_reward(reward=self.reward)
                return ts.termination(self._state, self.reward)
        else:
            return ts.transition(self._state, reward=self.reward, discount=1.0)
    
