import numpy as np
import os
import gym
import time as timer
import mujoco_py
from mujoco_py import MjViewer
from gym import utils
from gym.utils import seeding
from gym import sapces, error
from utils.utils_simulation import MujocoEnv

class RelocateEnv(MujocoEnv, utils.ezpickle):
    def __init__(self, N):
        '''
            initialization
            match env xml file with # of objects
        '''
        ...

    def step(self, a):
        '''
            reward shaping
        '''
        ...
    
    def get_obs(self):
        '''
            partially observable state
            image
            env.sim.render(width = frame_size[0], height = frame_size[1], mode = 'offscreen', camera_name = camera_name, device_id = 0)
        '''
        ...
    
    def reset(self):
        '''
            reset objects and hand
        '''
        ...
    
    def get_env_state(self):
        '''
            fully observed state
            pos
        '''
        ...
    
    def set_env_state(self):
        '''
            set pos of objects and hand
        '''
        ...

    def set_num_of_objects(self, N):
        '''
            change # of objects
        '''
        self.__init__(N)
        return self
    

