import numpy as np
import torch
from collections import deque
from core.common_methods_agent import Common_Methods
import random

class DQNAgent(Common_Methods):
    def __init__(self, nn, n_actions, buffer_size, batch_size, epsilon, epsilon_min=0.01, epsilon_max = 0.9, gamma=0.99) :
        super().__init__(algo="dqn")
        if torch.cuda.is_available(): # CUDA NVIDIA
            self.device = torch.device("cuda")
            print(f"CUDA device available: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():  # MAC M1/M2/M3
            self.device = torch.device("mps")
        #elif torch.version.hip is not None:     # AMD ROCm
        #    self.device = torch.device("hip") # Only on Linux
        else:
            self.device = torch.device("cpu")
        
        self.nn = nn.to(self.device)
        self.n_actions = n_actions
        self.epsilon = epsilon  # Initial exploration probability
        self.epsilon_min = epsilon_min  # Minimum epsilon value
        self.epsilon_max = epsilon_max  # Maximum epsilon value
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma # Discount factor
        self.loss_fct = torch.nn.MSELoss()
    
    def getaction_dqn(self, state) :
        
        if np.random.rand() < self.epsilon :
            return np.random.randint(self.n_actions) # exploration
        else :
            # For single state input (cartpole)
            if state.dim() == 1:
                state = state.unsqueeze(0)
            with torch.no_grad() : # torch.no_grad to avoid unnecessary memory usage (no backward here)
                Q_values = self.nn.forward(state) # state already converted to tensor
            action = int(Q_values.argmax(dim=1).item()) # Use torch argmax and convert to int
            return action # No backpropagation yet, as we need the reward associated with the action
        
    def store_transition_dqn(self, state, action, reward, next_state, done) :
        self.memory.append((state, action, reward, next_state, done))
        
    def learn_dqn(self) :
        batch = random.sample(self.memory, self.batch_size)
        
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.cat(states, dim=0)
        next_states = torch.cat(next_states, dim=0)
        
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
        
        Q_values = self.nn(states).gather(1, actions) # self.nn(states) retrieves Q-values for each choice (0 or 1), then gather(1, actions) selects the Q-value based on the chosen action
        
        with torch.no_grad():
            max_next_Q = self.nn(next_states).max(1, keepdim=True)[0]  # Best future action (keepdim = True maintains shape (batch_size, 1))
            Q_targets = rewards + (1 - dones) * self.gamma * max_next_Q # Bellman equation calculating target Q-values
            
        loss = self.loss_fct(Q_targets, Q_values) # Loss calculation based on actual values (Q_values: values predicted by neural network) and best values (Q_targets: maximum values calculated with Bellman eq)
        
        self.nn.optimizer.zero_grad() # Reset gradients
        loss.backward() # Backpropagation
        self.nn.optimizer.step() # Weight update