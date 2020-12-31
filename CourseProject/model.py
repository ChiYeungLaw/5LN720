import torch
import torch.nn as nn

class FNN(nn.Module):
    def __init__(self, num_class=62, num_char=4, dropout=0.2):
        super(FNN, self).__init__()
        self.num_class = num_class
        self.num_char = num_char
        self.proj1 = nn.Linear(3 * 120 * 40, 300)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.proj2 = nn.Linear(300, self.num_class * self.num_char)
    
    def forward(self, x):
        x = x.view(-1, 3 * 120 * 40)
        return self.proj2(self.dropout(self.relu(self.proj1(x))))

class CNN5(nn.Module):
    def __init__(self, num_class=62, num_char=4, dropout=0.2):
        super(CNN5, self).__init__()
        self.num_class = num_class
        self.num_char = num_char
        self.conv = nn.Sequential(
            # [bs, 3, 120, 40]
            nn.Conv2d(3, 16, 3, padding=(1,1)),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # [bs, 16, 60, 20]
            nn.Conv2d(16, 64, 3, padding=(1,1)),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # [bs, 64, 30, 10]
            nn.Conv2d(64, 128, 3, padding=(1,1)),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # [bs, 128, 15, 5]
            nn.Conv2d(128, 256, 3, padding=(1,1)),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # [bs, 256, 7, 2]
            nn.Conv2d(256, 512, 3, padding=(1,1)),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # [bs, 512, 3, 1]
        )
        self.dropout = nn.Dropout(p=dropout)
        self.proj = nn.Linear(512 * 3 * 1, self.num_class * self.num_char)
    
    def forward(self, x):
        x = self.dropout(self.conv(x))
        x = x.view(-1, 512 * 3 * 1)
        x = self.proj(x)
        return x

class CNN4(nn.Module):
    def __init__(self, num_class=62, num_char=4, dropout=0.2):
        super(CNN4, self).__init__()
        self.num_class = num_class
        self.num_char = num_char
        self.conv = nn.Sequential(
            # [bs, 3, 120, 40]
            nn.Conv2d(3, 16, 3, padding=(1,1)),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # [bs, 16, 60, 20]
            nn.Conv2d(16, 64, 3, padding=(1,1)),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # [bs, 64, 30, 10]
            nn.Conv2d(64, 128, 3, padding=(1,1)),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # [bs, 128, 15, 5]
            nn.Conv2d(128, 256, 3, padding=(1,1)),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # [bs, 256, 7, 2]
        )
        self.dropout = nn.Dropout(p=dropout)
        self.proj = nn.Linear(256 * 7 * 2, self.num_class * self.num_char)
    
    def forward(self, x):
        x = self.dropout(self.conv(x))
        x = x.view(-1, 256 * 7 * 2)
        x = self.proj(x)
        return x

class CNN3(nn.Module):
    def __init__(self, num_class=62, num_char=4, dropout=0.2):
        super(CNN3, self).__init__()
        self.num_class = num_class
        self.num_char = num_char
        self.conv = nn.Sequential(
            # [bs, 3, 120, 40]
            nn.Conv2d(3, 16, 3, padding=(1,1)),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # [bs, 16, 60, 20]
            nn.Conv2d(16, 64, 3, padding=(1,1)),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # [bs, 64, 30, 10]
            nn.Conv2d(64, 128, 3, padding=(1,1)),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # [bs, 128, 15, 5]
        )
        self.dropout = nn.Dropout(p=dropout)
        self.proj = nn.Linear(128 * 15 * 5, self.num_class * self.num_char)
    
    def forward(self, x):
        x = self.dropout(self.conv(x))
        x = x.view(-1, 128 * 15 * 5)
        x = self.proj(x)
        return x

class CNN2(nn.Module):
    def __init__(self, num_class=62, num_char=4, dropout=0.2):
        super(CNN2, self).__init__()
        self.num_class = num_class
        self.num_char = num_char
        self.conv = nn.Sequential(
            # [bs, 3, 120, 40]
            nn.Conv2d(3, 16, 3, padding=(1,1)),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # [bs, 16, 60, 20]
            nn.Conv2d(16, 64, 3, padding=(1,1)),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # [bs, 64, 30, 10]
        )
        self.dropout = nn.Dropout(p=dropout)
        self.proj = nn.Linear(64 * 30 * 10, self.num_class * self.num_char)
    
    def forward(self, x):
        x = self.dropout(self.conv(x))
        x = x.view(-1, 64 * 30 * 10)
        x = self.proj(x)
        return x

class CNN1(nn.Module):
    def __init__(self, num_class=62, num_char=4, dropout=0.2):
        super(CNN1, self).__init__()
        self.num_class = num_class
        self.num_char = num_char
        self.conv = nn.Sequential(
            # [bs, 3, 120, 40]
            nn.Conv2d(3, 16, 3, padding=(1,1)),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # [bs, 16, 60, 20]
        )
        self.dropout = nn.Dropout(p=dropout)
        self.proj = nn.Linear(16 * 60 * 20, self.num_class * self.num_char)
    
    def forward(self, x):
        x = self.dropout(self.conv(x))
        x = x.view(-1, 16 * 60 * 20)
        x = self.proj(x)
        return x