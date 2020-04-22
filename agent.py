import numpy as np
import random
from collections import namedtuple, deque

from model import FCNet

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5) # replay buffer size
BATCH_SIZE = 64        # minibatch size
GAMMA = 0.99           # discount factor
TAU = 1e-3             # soft update hyperparameter
LR = 5e-4              # learning rate
C = 4                  # update frequency
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DQNAgent:
    """ A DQN Agent which interacts and learns from the environment. """

    def __init__(self, state_size, action_size, seed):
        """
        
        Initializes a DQN Agent.

        params:
            - state_size (int)  : dimension of each state.
            - action_size (int) : dimension of each action.
            - seed (int)        : random seed.

        """

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # initialize the Q network
        self.qnet = FCNet(self.state_size, self.action_size, seed).to(device)
        # initialize the target Q network
        self.target_qnet = FCNet(self.state_size, self.action_size, seed).to(device)

        # create optimizer
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=LR)

        # create replay buffer
        self.buffer = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

        # initialize timestep for updates using C
        self.tstep = 0
    
    def step(self, state, action, reward, next_state, done):
        # save experiences in replay buffer
        self.buffer.push(state, action, reward, next_state, done)

        # Learn every C timesteps
        self.tstep = (self.tstep+1) % C

        if self.tstep == 0:

            # check if enough samples are available in buffer
            if len(self.buffer) > BATCH_SIZE:
                experiences = self.buffer.sample()
                self.learn(experiences, GAMMA)
    
    def learn(self, experiences, gamma):
        """
        Updates value params using batch of experience tuples.

        params:
            - experiences (Tuple[torch.Variable]) : (s, a, r, s', done) tuple.
            - gamma (float)                       : discount factor.
        """

        # unpack experiences
        s, a, r, ns, d = experiences

        # get expected q vals from qnet
        q_exp = self.qnet(s).gather(1, a)

        # get max Q vals for next state from target_qnet
        q_next = self.target_qnet(ns).detach().max(1)[0].unsqueeze(1)

        # compute Q vals for current state
        q_current = r + (gamma * q_next * (1 - d))

        # compute loss
        loss = F.smooth_l1_loss(q_exp, q_current) # huber loss
        # loss = F.mse_loss(q_exp, q_current)

        # minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


        ######################## Update Target Network ########################
        self.soft_update(TAU)
    
    def soft_update(self, tau):
        """
        Performs a soft update for the parameters.
        theta_target = tau * theta_local + (1 - tau) * theta_target
        
        params:
            - TAU (float) : interpolation parameter. 
        """

        for target_param, local_param in zip(self.target_qnet.parameters(), self.qnet.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)
    
    def act(self, state, eps=0.):
        """ 
        Returns actions for a given state as per current policy.

        params:
            - state (array like) : current state.
            - eps (float)        : epsilon for eps-greedy action selection.
        """

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        # set to eval mode
        self.qnet.eval()
        
        with torch.no_grad():
            # get action values
            act_vals = self.qnet(state)
        
        # turn back to train mode
        self.qnet.train()

        # epsilon greedy action selection
        if random.random() > eps:
            return np.argmax(act_vals.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))



        
        




class ReplayBuffer:
    """ Replay Buffer which stores experience tuples. """

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """ 
        Initializes a RB object.

        params:
            - action_size (int) : dimension of each action.
            - buffer_size (int) : max size of buffer.
            - batch_size (int)  : size of each training batch.
            - seed (int)        : random seed.
        
        """
        self.action_size = action_size
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        # creates the replay buffer
        self.buffer = deque(maxlen=buffer_size)
        # creates a namedtuple for experiences
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    
    def push(self, state, action, reward, next_state, done):
        """
        Adds new experience to the replay buffer.
        
        params:
            - state      : current state of the environment.
            - action     : action taken by the agent in the current state.
            - reward     : reward received for taking action in current state.
            - next_state : next state which the agent transitions to after taking action in the current state.
            - done       : flag which determines if the experience has ended.
        """
        # create namedtuple
        exp = self.experience(state, action, reward, next_state, done)
        # append to buffer
        self.buffer.append(exp)
    
    def sample(self):
        """
        Randomly samples a batch of experiences from the replay buffer.
        """

        # get experiences
        experiences = random.sample(self.buffer, k=self.batch_size)

        # stack the experiences into different torch tensors
        s_, a_, r_, ns_, d_ = [], [], [], [], []

        # use single loop instead of creating a generator for each
        for e in experiences:
            if e is not None:
                s_.append(e.state)
                a_.append(e.action)
                r_.append(e.reward)
                ns_.append(e.next_state)
                d_.append(e.done)
        
        states = torch.from_numpy(np.vstack(s_)).float().to(device)
        actions = torch.from_numpy(np.vstack(a_)).long().to(device)
        rewards = torch.from_numpy(np.vstack(r_)).float().to(device)
        next_states = torch.from_numpy(np.vstack(ns_)).float().to(device)
        dones = torch.from_numpy(np.vstack(d_).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        """
        Returns the current size of the replay buffer.
        """
        return len(self.buffer)
        