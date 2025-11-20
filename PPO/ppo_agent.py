from contextlib import nullcontext
from turtle import done
from PPO.ppo_training import state, value
from bandit import action, reward
import torch
from torch import nn
from torch.distributions import Normal


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"computing device: {device}")


class Actor(nn.module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor).__init__()
        self.fc1 = nn.Linear(state_dim, action_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.fc_std = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(self,x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        mean = self.tanh(self.fc_mean(x)) * 2
        std = self.softplus(self.fc_std(x)) + 1e-3

        return mean, std

    def select_action(self, s):
        with torch.no_grad():
            mu, sigma, self.forward(s) 
            normal_dist = Normal(mu,sigma)
            action = normal_dist.sample()
            action = action.clamp(-2.0,2.0)
            
        return action


class Critic(nn.module):
    def __init__(self, state_dim, hidden_dim=256):
        super(Critic).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        value = self.fc3(x)

        return value


class ReplayMemory:
    def __init__(self, batch_size):
        self.state_cap = []
        self.action_cap = []
        self.reward_cap = []
        self.value_cap = []
        self.done_cap = []
        self.BATCH_SIZE = batch_size

    def add_memo(self, state, action, reward, value,done):
        self.state_cap.append(state)
        self.action_cap.append(action)
        self.reward_cap.append(reward)
        self.value_cap .append(value)
        self.done_cap.append(done)

    def sample():
        pass
        



class PPOAgent:
    pass