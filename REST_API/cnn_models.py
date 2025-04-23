from torchvision import transforms

import torch

from torchvision.models import googlenet

import torch.nn as nn

import torchvision.models as models

# Load the trained U-Net model

num_classes = 4

resnet50_model = models.resnet50(pretrained=False)

resnet50_model.fc = nn.Sequential(
    nn.Linear(resnet50_model.fc.in_features, 512), # Első rejtett réteg
    nn.LeakyReLU(0.01),
    nn.Dropout(0.3),
    nn.Linear(512, 256), # Második rejtett réteg
    nn.LeakyReLU(0.01),
    nn.Dropout(0.3),
    nn.Linear(256, num_classes) # Kimeneti réteg
)

import timm

# Load pre-trained Inception-ResNet-v2 model
inception_resnet_model = timm.create_model('inception_resnet_v2', pretrained=False)
# Utolsó teljesen kapcsolt réteg (fc) cseréje saját osztályszámra
inception_resnet_model.classif = nn.Sequential(
    nn.Linear(inception_resnet_model.classif.in_features, 512), # Első rejtett réteg
    nn.LeakyReLU(0.01),
    nn.Dropout(0.3),
    nn.Linear(512, 256), # Második rejtett réteg
    nn.LeakyReLU(0.01),
    nn.Dropout(0.3),
    nn.Linear(256, num_classes) # Kimeneti réteg
)

googlenet_model = googlenet(weights='IMAGENET1K_V1') # Load pretrained weights
googlenet_model.fc = nn.Sequential(
    nn.Linear(googlenet_model.fc.in_features, 512), # Első rejtett réteg
    nn.LeakyReLU(0.01),
    nn.Dropout(0.3),
    nn.Linear(512, 256), # Második rejtett réteg
    nn.LeakyReLU(0.01),
    nn.Dropout(0.3),
    nn.Linear(256, num_classes) # Kimeneti réteg
)

resnet50_model = models.resnet50(pretrained=False)
resnet50_model.fc = nn.Sequential(
    nn.Linear(resnet50_model.fc.in_features, 512), # Első rejtett réteg
    nn.LeakyReLU(0.01),
    nn.Dropout(0.3),
    nn.Linear(512, 256), # Második rejtett réteg
    nn.LeakyReLU(0.01),
    nn.Dropout(0.3),
    nn.Linear(256, num_classes) # Kimeneti réteg
)
