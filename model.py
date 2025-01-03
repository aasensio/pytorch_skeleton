import torch
import torch.nn as nn

class Network(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(Network, self).__init__()
        
        self.C1 = nn.Linear(n_input, n_hidden)
        self.C2 = nn.Linear(n_hidden, n_hidden)
        self.C3 = nn.Linear(n_hidden, n_output)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.C1(x)
        out = self.relu(out)
        out = self.C2(out)
        out = self.relu(out)
        out = self.C3(out)
            
        return out
    