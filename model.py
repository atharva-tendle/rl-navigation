import torch
import torch.nn as nn
import torch.nn.functional as F

class FCNet(nn.Module):
    """
    This creates a Fully Connected Neural Network which acts a the Policy/Actor Model.

    """
    def __init__(self, state_size, action_size, seed):
        """ 
        Initialize params and build model.

        params:
            - state_size (int)  : dimension of each state.
            - action_size (int) : dimension of each action.
            - seed (int)        : random seed.
        """

        super(FCNet, self).__init__()
        self.seed = torch.manual_seed(seed)

        # input layer
        self.fc1 = nn.Linear(state_size, 64)
        # hidden layer
        self.fc2 = nn.Linear(64, 128)
        # hidden layer
        self.fc3 = nn.Linear(128, 64)
        # output layer
        self.fc4 = nn.Linear(64, action_size)
    
    def forward(self, state):
        """ Builds a NN which maps states to actions. """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)
