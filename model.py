from dinov2.models.vision_transformer import vit_small, vit_base, vit_large, vit_giant2
from data import * 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

import os
import sys
import argparse
import random
import colorsys
import requests
from io import BytesIO
from matplotlib import gridspec
import skimage.io
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from tqdm import tqdm

from utils import parse_training_conf





def get_backbone(size):
  
    dino_path =  '/home/andrii/adient/dinov2'
    weights_path = os.path.join(dino_path, 'weights')
    weights_path = {
        'small' : [os.path.join(weights_path, 'dinov2_vits14_pretrain.pth'), vit_small],
        'base' : [os.path.join(weights_path, 'dinov2_vitb14_pretrain.pth'), vit_base],
        'large' : [os.path.join(weights_path, 'dinov2_vitl14_pretrain.pth'), vit_large],
        'giant' : [os.path.join(weights_path, 'dinov2_vitg14_pretrain.pth'), vit_giant2]
    }

    
    patch_size = 14

    backbone = weights_path[size][1]
    backbone = backbone(
        patch_size=patch_size,
        img_size=526,
        init_values=1.0,
        block_chunks=0
    )
    backbone.load_state_dict(torch.load(weights_path[size][0]))
    for p in backbone.parameters():
        p.requires_grad = False


    return backbone


class supConClas(nn.Module):
    def __init__(self, backbone, n_embeddings):
        super(supConClas, self).__init__()
        self.n_embeddings = n_embeddings
        self.backbone = backbone
        self.fc = nn.Linear(backbone.norm.normalized_shape[0], n_embeddings)

    def forward(self, features):
        features1 = self.backbone(features)
        features2 = F.normalize(features1, dim = 1)
        features2 = self.fc(features2)
        return features2


class clasifier(nn.Module):
    def __init__(self, backbone, num_classes, threshold = 0.5):
        super(clasifier, self).__init__()
        self.threshold = threshold
        self.num_classes = num_classes
        self.backbone = backbone 
        self.sigm = nn.Sigmoid()
        for name, param in self.backbone.named_parameters():
            param.requires_grad = False
            
        if num_classes == 1:
            self.head = nn.Linear(self.backbone.n_embeddings, 1)
        else:
            self.head = nn.Sequential(
                nn.Linear(self.backbone.n_embeddings, num_classes),
                nn.Softmax(dim = 1))
            
    def forward(self, x, return_predictions = False):
        features = self.backbone(x)
        preds = self.head(features)
        if return_predictions:
            if self.num_classes == 1:
                predictions = self.sigm(preds)
                predictions = (predictions > self.threshold).float()
            else:
                predictions =  torch.argmax(preds, dim=1)
            return preds, predictions
        else:
            return preds
        





if __name__ == '__main__':
    bc = get_backbone()

    model = supConClas(bc, 128)

    device = torch.device('cuda')

    input = torch.rand((5,3,224,224))

    input = input.to(device)
    model = model.to(device)

    output = model(input)
    print(output.shape)

    


