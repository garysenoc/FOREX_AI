  
from tf_agents.metrics import tf_py_metric as tf_p_m
from tf_agents.metrics import py_metric as py_m
import tensorflow as tf
import constants

class SumOfRewards(py_m.PyStepMetric):
    def __init__(self, name='SumOfRewards'):
        super(py_m.PyStepMetric, self).__init__(name)
        self.rewards = []
        self.actions = []
        self.sum_rew = constants.FLOAT_ZERO
        self.reset()
    
    def reset(self):
        self.rewards = [] 
        self.actions = []
        self.sum_rew = constants.FLOAT_ZERO


    def call(self, trajectory):
        if(trajectory.is_first()):
            self.reset()
        
        self.rewards += trajectory.reward
        self.actions += trajectory.action

        if(trajectory.is_last()):      
            print(self.rewards)
            print(self.actions)
            
            
    def result(self):
        return tf.math.reduce_sum(self.rewards)

class TFSumOfRewards(tf_p_m.TFPyMetric):

  def __init__(self, name='SumOfRewards', dtype=tf.float32):
    py_m = SumOfRewards()

    super(TFSumOfRewards, self).__init__(
        py_m=py_m, name=name, dtype=dtype)