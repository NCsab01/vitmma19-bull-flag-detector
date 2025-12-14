import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import config

class FlagDataset(Dataset):
    def __init__(self, X, Y, lengths, augment=False):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.long)
        self.lengths = torch.tensor(lengths, dtype=torch.long)
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        y = self.Y[idx]
        l = self.lengths[idx]

        if self.augment:
            scale_factor = np.random.uniform(0.90, 1.10)
            x = x * scale_factor

            noise = torch.randn_like(x) * 0.05
            x = x + noise

        return x, y, l


def simple_collate(batch):
    (xx, yy, ll) = zip(*batch)
    
    x_tensor = torch.stack(xx)
    y_tensor = torch.stack(yy)
    l_tensor = torch.stack(ll)
    
    return x_tensor, y_tensor, l_tensor


class FlagClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3):
        super(FlagClassifier, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout),
            
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1) 
        
        x = self.features(x)
        
        x = self.global_pool(x)
        
        x = x.squeeze(-1) 
        
        logits = self.fc(x)
        
        return logits


def create_model(device):
    model = FlagClassifier(
        input_size=config.INPUT_SIZE,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        num_classes=config.NUM_CLASSES,
        dropout=config.DROPOUT
    ).to(device)
    return model