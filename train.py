import os
import time
import argparse
import yaml
import numpy as np
import torch
import torch.multiprocessing as mp
import time

from model import STL, RelocatePolicy, RelocateBaseline
from env import make_env

from utils.utils_train import *


def train(cfg, args):
    '''
        read cfg file and pretrained log
        pass cfg file values to STL class
        STL proceed bc, ft and dc using own methods
    '''
    ############################
    # Get train configuration  #
    ############################
    device = torch.device('cuda' if cfg['device'] == 'cuda' and torch.cuda.is_available() else 'cpu')
    batch_size = cfg['train']['batch_size']
    sample_size = cfg['train']['sample_size']
    learning_rate = cfg['train']['learning_rate']
    epochs = cfg['train']['epochs']
    sample_epoch = cfg['train']['sample_epoch']
    buffer_size = cfg['train']['buffer_size']
    validation_epoch = cfg['train']['validation_epoch']
    num_processes = cfg['train']['num_processes']
    model_path = cfg['train']['model_path']
    progress_path = cfg['train']['progress_path']
    hidden_dim = cfg['model']['hidden_dim']
    env_name = cfg['env']['env_name']
    seed = cfg['train']['seed']

    origin_demo_path = args.demo # change path of demo path and train it


    ############################
    #      Train settings      #
    ############################
    # Use multi process
    # # from utils.utils_train import multi_process

    env = make_env(env_name)
    start_epoch = 1
    policy = RelocatePolicy(obs_dim = env.spec.observation_dim, action_dim = env.spec.action_dim, hidden_dim = hidden_dim, seed = seed)
    baseline = RelocateBaseline(obs_dim = env.spec.observation_dim, action_dim = env.spec.action_dim, hidden_dim = hidden_dim, seed = seed)
    policy_criterion = loss_func()
    optimizer = optim_func()
    agent = STL(demo = origin_demo_path, policy = policy, epochs = epochs, batch_size = batch_size, lr = learning_rate, baseline = baseline, optimizer = optimizer, loss_func = policy_criterion, env = env)

    ############################
    #     behavior cloning     #
    ############################
    agent.Behavior_Cloning()

    ############################
    #       fine tuning        #
    ############################
    agent.DAPG()

    ############################
    #       create demo        #
    ############################
    agent.Create_Demo()

    # loop
    # origin_demo_path = new_demo_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/config.yaml', help='Path to config file')
    parser.add_argument('--demo', type=str, default='./demos/relocate-v0_demos.pickle', help='Path to demo file')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg, args)