from network.loss import YoloLoss
from unittest import TestCase
import numpy as np
import torch

from network.dataset import VOCDataset
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os 


torch.manual_seed(13)
DATA_DIR = "data/pascal_voc_yolo/"
IMG_DIR = DATA_DIR+"images/"
LABEL_DIR = DATA_DIR+"labels/"
BATCH_SIZE = 4
NUM_WORKERS = 4
TRANSFORM = False
S = 7
B = 2
C = 20

 
torch.set_printoptions(linewidth=10000, edgeitems=160)

torch.manual_seed(13)
class TestLoss(TestCase):
    def setUp(self):
        self.model_loss = YoloLoss(7, 2, 20)
        dataset_args = dict(S=S, B=B, C=C, transform=TRANSFORM)
        self.dataset = VOCDataset(DATA_DIR+"8examples.csv", IMG_DIR, LABEL_DIR, **dataset_args)

    def test_fake_loss(self):
        output_shape = (2, 7, 7, 30)
        self.y = torch.abs(torch.zeros(output_shape))
        self.y[0, 2, 2, 3] = 1 
        self.y[0, 2, 2, 20] = 1 
        self.y[0, 2, 2, 21:25] = 0.5
        # -------------------------- 
        self.y[1, 5, 3, 3] = 1 
        self.y[1, 5, 3, 25] = 1 
        self.y[1, 5, 3, 26:30] = 0.25
        # -------------------------- 
        self.yhat = torch.abs(torch.zeros(output_shape))
        self.yhat[0, 2, 2, 3] = 1 
        self.yhat[0, 2, 2, 20] = 1 
        self.yhat[0, 2, 2, 21:25] = 0.75
        # -------------------------- 
        self.yhat[1, 5, 3, 3] = 1 
        self.yhat[1, 5, 3, 25] = 1 
        self.yhat[1, 5, 3, 26:30] = 0.2

        loss = self.model_loss(self.y.to("cuda"), self.yhat.to("cuda"))
        print(loss)

    def test_loss(self):
        
        img1, target = self.dataset.__getitem__(1)
        y = target.unsqueeze(0)
        yhat = y.clone()
        yhat[..., 21:23] += 0.05
        yhat[..., 26:28] += 0.6
        # print(y, "\n\n\n")
        # print(yhat, "\n\n\n")
        loss = self.model_loss(y.cuda(), yhat.cuda())
        print(loss)
