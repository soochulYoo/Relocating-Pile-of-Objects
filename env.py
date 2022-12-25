import gym_relocate
import gym
from utils.utils_simulation import GymEnv

def make_env(env_name):
    env = GymEnv(env_name)
    return env