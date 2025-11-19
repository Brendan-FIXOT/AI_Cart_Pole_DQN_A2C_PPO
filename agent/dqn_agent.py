import numpy as np
import torch
import torch.optim as optim
from collections import deque
from core.common_methods_agent import NeuralNetwork
from core.common_methods_agent import Common_Methods
import random

class DQNAgent(Common_Methods):
    def __init__(self, nn, buffer_size, batch_size, epsilon, epsilon_min=0.01, epsilon_max = 0.9, gamma=0.99) :
        super().__init__(algo="dqn")
        if torch.cuda.is_available(): # CUDA NVIDIA
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():  # MAC M1/M2/M3
            self.device = torch.device("mps")
        #elif torch.version.hip is not None:     # AMD ROCm
            #self.device = torch.device("hip") # Uniquement sur Linux
        else:
            self.device = torch.device("cpu")
        
        self.nn = nn.to(self.device)
        self.epsilon = epsilon  # Probabilité d'exploration initiale
        self.epsilon_min = epsilon_min  # Valeur minimale d'epsilon
        self.epsilon_max = epsilon_max  # Valeur maximale d'epsilon
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma # Facteur de réduction
        self.loss_fct = torch.nn.MSELoss()
    
    # ===============================
    # DQN methods
    # ===============================
    
    def getaction_dqn(self, state) :
        if np.random.rand() < self.epsilon :
            return np.random.choice([0,1]) # exploration
        else : 
            with torch.no_grad() : # torch.no_grad pour éviter de taper dans la mémoire inutilement (car pas de backward ici)
                Q_values = self.nn.forward(state) # state déjà passer en tensor
            action = int(np.argmax(Q_values)) # Pas np.max, car ici on veut l'index (0 ou 1). Besoin du int, sinon action serait un tensor
            return action # Pas de rétropropagation tout de suite, car on veut la récompense associé à l'action
        
    def store_transition_dqn(self, state, action, reward, next_state, done) :
        self.memory.append((state, action, reward, next_state, done))
        
    def learn_dqn(self) :
        batch = random.sample(self.memory, self.batch_size)
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
        
        Q_values = self.nn(states).gather(1, actions) # self.nn(states) récupère les Qvalues associés à chaque choix (0 ou 1), ensuite le gather(1, actions) choisi la Qvalues par rapport à l'action choisie
        
        with torch.no_grad():
            max_next_Q = self.nn(next_states).max(1, keepdim=True)[0]  # Meilleure action future (keepdim = True permet de garder la forme (batch_size, 1))
            Q_targets = rewards + (1 - dones) * self.gamma * max_next_Q # Equation de Bellman calculant la Qvalues cible
            
        loss = self.loss_fct(Q_targets, Q_values) # Calcul de la perte en fonction des valeurs réelles (Q_values : valeurs prédite par le réseau de neuronne) et des meilleurs valeurs (Q_targets : valeurs maximal calculer avec l'eq de Bellman)
        
        self.nn.optimizer.zero_grad() # Réinitialise les gradients
        loss.backward() # Rétropropagation
        self.nn.optimizer.step() # Mise à jour des poids