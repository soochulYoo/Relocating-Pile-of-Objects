'''
    sampling: roll out
    computing log likelihoods: returns, advantages, log likelihoods
    computing gradient: surrogate loss, Hessian vector product, conjugate gradient, kl distribution

    plot training log: (epochs, returns)
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np


def loss_func():
    '''
        Return policy criterion type of torch.nn  loss function
    '''
    policy_criterion = nn.CrossEntropyLoss(reduction='none')
    return policy_criterion

def optim_func(model, learning_rate):
    '''
        Return optimizer 
    '''
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    return optimizer

def roll_out(num_traj, env, policy, max_timestep, seed):
    '''
        sample trajectories
        input: env, num_traj, policy, max_timestep
        output: trajectory list
    '''
    trajectories = []
    for ep in range(num_traj):
        seed = seed + ep
        env.set_seed(seed)
        np.random.seed(seed)

        observations = []
        actions = []
        rewards = []

        obs = env.reset()
        done = False
        t = 0

        while t < max_timestep and done != True:
            action = policy.get_action(obs) # action sampling
            next_obs, r, done, info = env.step(action)
            observations.append(obs)
            actions.append(action)
            rewards.append(r)
            obs = next_obs
            t += 1

        trajectory = dict(
            observations = np.array(observations),
            actions = np.array(actions),
            rewards = np.array(rewards),
        )
        trajectories.append(trajectory)
    del(env)
    return trajectories

def sample_trajectories():
    '''
        return trajectory list
    '''

def discount_sum(reward_list, gamma):
    '''
        return discounted sum of rewards
        input: reward list
        output: sum of cumulative rewards
    '''
    reward_sum = 0.0
    reward_sum_list = []
    for t in range(len(reward_list)-1, -1, -1):
        reward_sum = reward_list[t] + gamma * reward_sum
        reward_sum_list.append(reward_sum)
    # reverse    
    return np.array(reward_sum_list[::-1])


def compute_returns(trajectories, gamma):
    '''
        compute returns
        input: trajectory list, gamma
        output: returns
    '''
    for trajectory in trajectories:
        trajectory['return'] = discount_sum(trajectory['rewards'], gamma)


def compute_advantages():
    '''
        compute advantages
        input: trajectory list, gamma, baseline
        output: advantage
    '''


def compute_CPI(observations, actions, advantages, policy):
    '''
        conservative policy iteration
        return surrogate loss
        input: agent - obs, actions, advantages
        output: log likelihood ratio x advantages
    '''
    advantage_variable = Variable(torch.from_numpy(advantages).float(), requires_grad = True)
    Likelihood_Ratio = policy.Likelihood_Ratio(observations, actions)
    surrgate_loss = torch.mean(Likelihood_Ratio * advantage_variable)
    return surrgate_loss


def compute_VPG(observations, actions, advantages, policy):
    '''
        vanilla policy gradient
        input: agent + demo - obs, actions, advantages
        output: gradient of surrogate loss
    '''
    surrogate_loss = compute_CPI(observations, actions, advantages, policy)
    vanilla_policy_gradient = torch.autograd.grad(surrogate_loss, policy.trainable_params)
    vanilla_policy_gradient = np.concatenate([gradient.contiguous().view(-1).data.numpy() for gradient in vanilla_policy_gradient])
    return vanilla_policy_gradient

def build_Hessian():
    '''

    '''
def compute_Hessian():
    '''
        Hessian -> Fisher Information matrix
        input: policy
        output: Fisher Information matrix of policy
    '''

def compute_conjugate_gradient():
    '''
    
    '''

