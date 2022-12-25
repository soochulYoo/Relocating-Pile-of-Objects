import os
from os import path
from mujoco_py import load_model_from_path, MjSim
from gym import error, spaces
from gym.utils import seeding
import numpy as np
import gym


def get_sim(model_path):
    '''
        get simulation from model path (XML)
    '''
    if model_path.startswith("/"):
        fullpath = model_path
    else:
        fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
    if not path.exists(fullpath):
        raise IOError("File %s does not exist" % fullpath)
    model = load_model_from_path(fullpath)
    return MjSim(model)

class EnvSpec():
    def __init__(self, obs_dim, act_dim, horizon):
        self.observation_dim = obs_dim
        self.action_dim = act_dim
        self.horizon = horizon

class GymEnv():
    '''
        wrapper for gym env
    '''
    def __init__(self, env, env_kwargs = None, obs_mask = None, act_repeat = 1, *args, **kwargs):
        if type(env) == str:
            env = gym.make(env)
        elif isinstance(env, gym.Env):
            env = env
        elif callable(env):
            env = env(**env_kwargs)
        else:
            print("Unsupported environment format")
            raise AttributeError
        self.env = env
        self.env_id = env.spec.id
        self.act_repeat = act_repeat
        try:
            self._horizon = env.spec.max_episode_steps
        except AttributeError:
            self._horizon = env.spec._horizon
        try:
            self._action_dim = self.env.env.action_dim
        except AttributeError:
            self._action_dim = self.env.action_space.shape[0]

        try:
            self._observation_dim = self.env.env.obs_dim
        except AttributeError:
            self._observation_dim = self.env.observation_space.shape[0]
        self.spec = EnvSpec(self._observation_dim, self._action_dim, self._horizon)
        self.obs_mask = np.ones(self._observation_dim) if obs_mask is None else obs_mask

    @property
    def action_dim(self):
        return self._action_dim

    @property
    def observation_dim(self):
        return self._observation_dim

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def horizon(self):
        return self._horizon
    
    def reset(self, seed = None):
        try:
            self.env._elapsed_steps = 0
            return self.env.env.reset_model(seed=seed)
        except:
            if seed is not None:
                self.set_seed(seed)
            return self.env.reset()

    def step(self, action):
        action = action.clip(self.action_space.low, self.action_space.high)
        if self.act_repeat == 1: 
            obs, cum_reward, done, info = self.env.step(action)
        else:
            cum_reward = 0.0
            for i in range(self.act_repeat):
                obs, reward, done, info = self.env.step(action)
                cum_reward += reward
                if done: break
        return self.obs_mask * obs, cum_reward, done, info
    
    def render(self):
        try:
            self.env.env.render_frames = True
            self.env.env.mj_render()
        except:
            self.env.render()
    
    def set_seed(self, seed=123):
        try:
            self.env.seed(seed)
        except AttributeError:
            self.env._seed(seed)
    
    def get_obs(self):
        try:
            return self.obs_mask * self.env.get_obs()
        except:
            return self.obs_mask * self.env.env._get_obs()

    def get_env_state(self):
        try:
            return self.env.get_env_state()
        except:
            raise NotImplementedError

    def set_env_state(self, state_dict):
        try:
            self.env.set_env_state(state_dict)
        except:
            raise NotImplementedError


class MujocoEnv(gym.Env):
    def __init__(self, model_path = None, frame_skip = 1, sim = None):
        if sim is None:
            self.sim = get_sim(model_path)
        else:
            self.sim = sim
        self.data = self.sim.data
        self.model = self.sim.model
        self.frame_skip = frame_skip
        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }
        self.render_frames = False
        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()
        try:
            observation, _reward, done, _info = self.step(np.zeros(self.model.nu))
        except NotImplementedError:
            observation, _reward, done, _info = self._step(np.zeros(self.model.nu))
        assert not done
        self.obs_dim = np.sum([o.size for o in observation]) if type(observation) is tuple else observation.size

        bounds = self.model.actuator_ctrlrange.copy()
        low = bounds[:, 0]
        high = bounds[:, 1]
        self.action_space = spaces.Box(low, high, dtype=np.float32)

        high = np.inf*np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.seed()
    
    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def do_simulation(self, ctrl, n_frames):
        for i in range(self.model.nu):
            self.sim.data.ctrl[i] = ctrl[i]
        for _ in range(n_frames):
            self.sim.step()
            if self.mujoco_render_frames is True:
                self.mj_render()
