import torch
from torch import nn

# Using the model from Kachuee, M., Fazeli, S., & Sarrafzadeh, M. (2018, June). Ecg heartbeat classification: A deep transferable representation. In 2018 IEEE international conference on healthcare informatics (ICHI) (pp. 443-444). IEEE.
    
class RepeatableResBlock125(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.Conv1 = nn.Conv1d(32, 32, kernel_size=5, padding=2)
        self.Activ1 = nn.ReLU()
        self.Conv2 = nn.Conv1d(32, 32, kernel_size=5, padding=2)
        self.Activ2 = nn.ReLU()
        self.MPool1 = nn.MaxPool1d(kernel_size=3, stride=2)

    def forward(self, x):
 
        C = x
        x = self.Conv1(x)
        x = self.Activ1(x)
        x = self.Conv2(x)
        x = x + C
        x = self.Activ2(x)
        x = self.MPool1(x)
        
        return x

class ResNet125(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.initial_conv = nn.Conv1d(1, 32, kernel_size=5)
        
        self.ResBlocks = nn.ModuleList(RepeatableResBlock125() for i in range(5))
        
        self.fc1 = nn.Linear(64,64)
        self.FinalActiv = nn.ReLU()
        self.fc2 = nn.Linear(64,5)
        
    def forward(self, x):
        x = self.initial_conv(x)
        for i in range(5):
            x = self.ResBlocks[i](x)
        
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.FinalActiv(x)
        x = self.fc2(x)
        
        return x

# with 187 timepoints:

class RepeatableResBlock(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.Conv1 = nn.Conv1d(32, 32, kernel_size=5, padding=2)
        self.Activ1 = nn.ReLU()
        self.Conv2 = nn.Conv1d(32, 32, kernel_size=5, padding=2)
        self.Activ2 = nn.ReLU()
        self.MPool1 = nn.MaxPool1d(kernel_size=5, stride=2)

    def forward(self, x):
 
        C = x
        x = self.Conv1(x)
        x = self.Activ1(x)
        x = self.Conv2(x)
        x = x + C
        x = self.Activ2(x)
        x = self.MPool1(x)
        
        return x
    
class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.initial_conv = nn.Conv1d(1, 32, kernel_size=5)
        
        self.ResBlocks = nn.ModuleList(RepeatableResBlock() for i in range(5))
        
        self.fc1 = nn.Linear(64,64)
        self.FinalActiv = nn.ReLU()
        self.fc2 = nn.Linear(64,5)
        
    def forward(self, x):
        x = self.initial_conv(x)
        for i in range(5):
            x = self.ResBlocks[i](x)
        
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.FinalActiv(x)
        x = self.fc2(x)
        
        return x