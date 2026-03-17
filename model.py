import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# model.py
import torch
import torch.nn as nn
from torchvision import models

def build_model(num_classes, device):

    # Load pretrained ResNet50
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # Freeze all layers except the last block
    for name, param in model.named_parameters():
        if "layer4" not in name and "fc" not in name:
            param.requires_grad = False

    # Replace the final fully connected layer
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes)
    )

    return model.to(device)