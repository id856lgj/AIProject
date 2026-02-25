"""
Definizione del modello CNN per Fashion-MNIST
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FashionMNISTCNN(nn.Module):
    
    def __init__(self):
        super(FashionMNISTCNN, self).__init__()
        
        # Convoluzioni
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout
        self.dropout = nn.Dropout(0.25)
        
        # Dopo 3 pooling: 28x28 -> 14x14 -> 7x7 -> 3x3
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        
    def forward(self, x):
        # Block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(-1, 128 * 3 * 3)
        
        # Fully connected con dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


def count_parameters(model):
    """Conta i parametri del modello"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


### MAIN - test del modello
if __name__ == "__main__":
    test_model = FashionMNISTCNN()
    print(f"Modello creato: {test_model.__class__.__name__}")
    print(f"Parametri totali: {count_parameters(test_model):,}")
    
    test_input = torch.randn(1, 1, 28, 28)
    output = test_model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")