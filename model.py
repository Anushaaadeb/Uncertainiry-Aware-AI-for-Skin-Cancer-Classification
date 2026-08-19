import torch
import torch.nn as nn
import timm

from config import BACKBONE_NAME

class DualModeClassifier(nn.Module):
    def __init__(self, num_classes=7, dropout_rate=0.3, backbone_name=BACKBONE_NAME):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        in_features = self.backbone.num_features
        
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(in_features, 256)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.dropout1(x)
        x = self.relu(self.fc1(x))
        x = self.dropout2(x)
        return self.classifier(x)