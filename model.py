import torch.nn as nn

class SmallCNN(nn.Module):
    def __init__(self, activation, num_classes=10, dropout=0.2):
        super().__init__()
        act = {
            "relu": nn.ReLU(inplace=True),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh()
        }[activation]

        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),  
            act,
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1),  
            act,
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, 64),  
            act,
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
