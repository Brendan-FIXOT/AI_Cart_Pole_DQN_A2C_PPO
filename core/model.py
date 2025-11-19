import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=2, mode="dqn", optimizer=optim.Adam, lr=1e-3):
        super(NeuralNetwork, self).__init__()
        self.mode = mode
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.optimizer = optimizer(self.parameters(), lr=lr)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        if self.mode in ["dqn", "critic"]:
            return x  # Return raw Q-values // or state values
        elif self.mode == "actor":
            return F.softmax(x, dim=-1) # Apply softmax to get action probabilities
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self, px=84, hidden_dim=64, output_dim=2, mode="dqn", optimizer=optim.Adam, lr=1e-3):
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.mode = mode
        """
        2 convolutional layers for image input (84x84x1)
        1st conv layer : 1 input channel (grayscale), 32 output channels, kernel size 3x3, padding 1
        2nd conv layer : 32 input channels, 64 output channels, kernel size 3x3, padding 1
        32/64 is good for small images like 84x84
        max pooling 2x2 after each conv layer to reduce spatial dimensions
        Then 3 fully connected layers
        """
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        final_px = px // 4
        self.fc1 = nn.Linear(64 * final_px * final_px, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        if self.mode in ["dqn", "critic"]:
            return x  # Return raw Q-values // or state values
        elif self.mode == "actor":
            return F.softmax(x, dim=-1) # Apply softmax to get action probabilities
        else:
            raise ValueError(f"Unknown mode: {self.mode}")