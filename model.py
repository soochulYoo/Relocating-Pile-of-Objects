import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import time
from tqdm import tqdm
import os

# Policy
# ==============================================================================================
class Encoder_Policy_Network(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super(Encoder_Policy_Network, self).__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Encoder
        self.encoder_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.obs_dim[0]*self.obs_dim[1], self.hidden_dim[0]),
            nn.ELU(),
            nn.Linear(self.hidden_dim[0], self.hidden_dim[1]),
            nn.ELU()
        )
        # Policy
        self.policy_net = nn.Sequential(
            nn.Linear(self.hidden_dim[1], self.hidden_dim[2]),
            nn.ELU(),
            nn.Linear(self.hidden_dim[2], self.action_dim)
        )
    
    def forward(self, x):
        '''
            Return action output corresponding input states
        '''
        if x.is_cuda:
            x = x.to('cpu')
        else:
            x = x
        x = self.encoder_net(x)
        action_output = self.policy_net(x)
        return action_output


class RelocatePolicy:
    '''
        encoder and policy network
    '''
    def __init__(self, obs_dim, action_dim, hidden_dim, min_log_std = -3, init_log_std = 0, seed = None):
        super(RelocatePolicy, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.min_log_std = min_log_std
        # Set seed
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Policy Network
        self.model = Encoder_Policy_Network(obs_dim = self.obs_dim, action_dim = self.action_dim, hidden_dim = self.hidden_dim)
        
        # Policy Parameters
        # log std is column vector of log std of action
        self.log_std = Variable(torch.ones(self.action_dim) * init_log_std, requires_grad=True)
        self.trainable_params = list(self.model.parameters()) + [self.log_std]

        # Old Policy Network
        self.old_model = Encoder_Policy_Network(obs_dim = self.obs_dim, action_dim = self.action_dim, hidden_dim = self.hidden_dim)
        
        # Old Policy Parameters
        self.old_log_std = Variable(torch.ones(self.action_dim) * init_log_std)
        self.old_trainable_params = list(self.old_model.parameters()) + [self.log_std]
        # Copy 
        for idx, param in enumerate(self.old_trainable_params):
            param.data = self.trainable_params[idx].data.clone()

        # Parameter access card
        self.log_std_value = np.float64(self.log_std.data.numpy().ravel())
        self.parameter_shapes = [p.data.numpy().shape for p in self.trainable_params]
        self.parameter_sizes = [p.data.numpy().size for p in self.trainable_params]
        self.total_num_parameters = np.sum(self.parameter_sizes)
    
    
    # Utility Functions
    def get_parameter_values(self):
        '''
            get parameters of policy network
        '''
        parameters = np.concatenate([p.contiguous().view(-1).data.numpy() for p in self.trainable_params])
        return parameters.copy()
    
    def set_parameter_values(self, parameters_after_optim, set_new = True, set_old = True):
        if set_new:
            current_idx = 0
            for idx, param in enumerate(self.trainable_params):
                values = parameters_after_optim[current_idx: current_idx + self.parameter_sizes[idx]]
                values = values.reshape(self.parameter_shapes[idx])
                param.data = torch.from_numpy(values).float()
                current_idx += self.parameter_sizes[idx]
            self.trainable_params[-1].data = torch.clmap(self.trainable_params[-1], self.min_log_std).data
            self.log_std_value = np.float64(self.log_std.numpy().ravel())
        if set_old:
            current_idx = 0
            for idx, param in enumerate(self.old_trainable_params):
                values = parameters_after_optim[current_idx: current_idx + self.parameter_sizes[idx]]
                values = values.reshape(self.parameter_shapes[idx])
                param.data = torch.from_numpy(values).float()
                current_idx += self.parameter_sizes[idx]
            # Clip std
            self.old_trainable_params[-1].data = torch.clamp(self.old_trainable_params[-1], self.min_log_std).data
    
    # Main Functions
    def get_action(self, observation):
        '''
            sample action from policy
            it will be used in roll out method 
        '''
        observation = np.float32(observation.reshape(1, -1))
        obs_variable = Variable(torch.randn(self.obs_dim), requires_grad=False)
        obs_variable.data = torch.from_numpy(observation)
        mean_action = self.model(obs_variable).data.numpy().ravel()
        std_action = np.exp(self.log_std_value)
        epsilon = np.random.randn(self.action_dim) # Gaussian distribution N(0, 1^2)
        action = mean_action + std_action * epsilon # elementwise product
        return action

    def Mean_LogLikelihood(self, observations, actions, model, log_std):
        '''
            input: log std, action, policy, action dim, observation
        '''
        if type(observations) is not torch.Tensor:
            obs_var = Variable(torch.from_numpy(observations).float(), requires_grad=False)
        else:
            obs_var = observations
        if type(actions) is not torch.Tensor:
            act_var = Variable(torch.from_numpy(actions).float(), requires_grad=False)
        else:
            act_var = actions
        mean = model(obs_var)
        z = (act_var - mean) / torch.exp(log_std)
        LL = - 0.5 * torch.sum(z**2) - torch.sum(log_std) - 0.5 * self.action_dim * np.log(2*np.pi)
        return mean, LL
    
    def Likelihood_Ratio(self, observations, actions):
        '''
            Return Ratio of old and new policy likelihood of particular action
            it will be used for surrogate loss
        '''
        action, LL = self.Mean_LogLikelihood(observations, actions, self.model, self.log_std)
        old_action, old_LL = self.Mean_LogLikelihood(observations, actions, self.old_model, self.old_log_std)
        Likelihood_Ratio = torch.exp(LL - old_LL)
        return Likelihood_Ratio
    
    def Mean_KL(self, observations, actions):
        '''
            Return KL divergence of old policy and new policy
        '''
        action, LL = self.Mean_LogLikelihood(observations, actions, self.model, self.log_std)
        old_action, old_LL = self.Mean_LogLikelihood(observations, actions, self.old_model, self.old_log_std)
        KL = torch.sum(((old_LL - LL)**2 + self.old_log_std**2 - self.log_std**2) / (2*self.log_std**2 + 1e-8) + self.log_std - self.old_log_std, dim = 1)
        return KL

# ==============================================================================================

# Agent
# ==============================================================================================
class STL:
    '''
        Spiral Transfer Learning
        process: BC -> FT -> DC -> BC -> FT -> DC ...
    '''
    def __init__(self, demo, policy, epochs, batch_size, lr, base_line, seed, optimizer, loss_func, env, output_direction):
        '''
            input: policy network (model), cfg data
        '''
        self.demo = demo
        self.policy = policy
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = lr
        self.base_line = base_line
        self.seed = seed
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.env = env
        self.output_direction = output_direction
        self.save_logs = True

        self.num_iter = 200
        self.num_traj = 50


    def Behavior_Cloning(self):
        '''
            pre-train policy network using demo
            input: policy network, demo
            output: None

            process
            read demonstrations: observations, actions
            compute loss of policy action and demo action
            optimization
            update parameters
        '''
        demonstrations = self.demo
        policy = self.policy
        epochs = self.epochs # BC epochs
        batch_size = self.batch_size # BC batch size
        optimizer = self.optimizer # BC optimizer
        loss_func = nn.MSELoss() # BC loss

        observations = np.concatenate([demo['observations'] for demo in demonstrations])
        demo_actions = np.concatenate([demo['actions'] for demo in demonstrations])

        start_time = time.time()
        num_samples = observations.shape[0]

        for ep in tqdm(range(epochs)):
            for minibatch in range(int(num_samples / batch_size)):
                random_index = np.random.choice(num_samples, size = batch_size)
                optimizer.zero_grad()
                obs = observations[random_index]
                act = demo_actions[random_index]
                mean, loglikelihood = self.policy.Mean_LogLikelihood(observations=obs, actions=act, model=policy, log_std=policy.log_std)
                loss = -torch.mean(loglikelihood)
                loss.backward()
                optimizer.step()
        parameters_after_optim = policy.model.get_parameter_values()
        policy.mode.set_parameter_values(parameters_after_optim, set_new = True, set_old = True)
        


    
    def DAPG(self):
        '''
            fine tuning policy network
            input: policy network, demo
            output: None

            step
            - sample trajectories
            - compute vanilla policy gradient(REINFORCE)
            - pre-condtions gradient using the inverse of Fisher Information Matrix
            - compute Hessian Vector Product
            - compute conjugate gradient
            - update parameters
        '''

        demonstrations = self.demo
        policy = self.policy
        env = self.env
        seed = self.seed
        output_direction = self.output_direction

        # make save directory
        if os.path.isdir(output_direction) == False:
            os.mkdir(output_direction)
        os.chdir(output_direction)
        if os.path.isdir('iterations') == False: os.mkdir('iterations')
        if os.path.isdir('logs') == False and self.save_logs == True: os.mkdir('logs')

        for i in range(self.num_iter):
            print("ITERATION: %i " %i)

            N = self.num_traj

    
    def Create_Demo(self):
        '''
            create demo using trained policy
            output: demonstration of increased # of objects
        '''
        origin_demo = self.demo
        model = self.model



    





class RelocateBaseline(nn.Module):
    '''
        encoder and policy network
    '''
    def __init__(self, obs_dim, action_dim, hidden_dim, seed = None):
        super(RelocateBaseline, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.encoder_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.obs_dim[0]*self.obs_dim[1], self.hidden_dim[0]),
            nn.ELU(),
            nn.Linear(self.hidden_dim[0], self.hidden_dim[1]),
            nn.ELU()
        )
        self.policy_net = nn.Sequential(
            nn.Linear(self.hidden_dim[1], self.hidden_dim[2]),
            nn.ELU(),
            nn.Linear(self.hidden_dim[2], self.action_dim)
        )
    
    def forward(self, x):
        '''
            Return action output corresponding input states
        '''
        if x.is_cuda:
            x = x.to('cpu')
        else:
            x = x
        x = self.encoder_net(x)
        action_output = self.policy_net(x)
        return action_output