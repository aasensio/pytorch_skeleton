import torch
import torch.nn as nn

class Network(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Network, self).__init__()
        
        self.C1 = nn.Linear(input_size, hidden_size)
        self.C2 = nn.Linear(hidden_size, hidden_size)
        self.C3 = nn.Linear(hidden_size, output_size)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.C1(x)
        out = self.relu(out)
        out = self.C2(out)
        out = self.relu(out)
        out = self.C3(out)
            
        return out
    