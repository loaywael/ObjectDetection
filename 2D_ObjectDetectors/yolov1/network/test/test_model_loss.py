from network.loss import YoloLoss
from unittest import TestCase
import numpy as np
import torch
from torch.utils.data import DataLoader
from network.transforms import img_transformer
from network.dataset import VOCDataset
from network.models import YoloResnetv1
from network.models import YoloDarknetv1
import matplotlib.pyplot as plt
import os 


torch.manual_seed(13)
DATA_DIR = "data/pascal_voc_yolo/"
IMG_DIR = DATA_DIR+"images/"
LABEL_DIR = DATA_DIR+"labels/"
# INPUT_SIZE = (3, 224, 224)
INPUT_SIZE = (3, 448, 448)
DEVICE = "cuda"
NUM_WORKERS = 4
BATCH_SIZE = 4
S = 7
B = 2
C = 20

 
torch.set_printoptions(linewidth=10000, edgeitems=160)
torch.manual_seed(13)

class TestModelLoss(TestCase):
    def setUp(self):
        # self.model = YoloDarknetv1(S=S, B=B, C=C).to(DEVICE)
        self.model = YoloResnetv1(INPUT_SIZE, S=S, B=B, C=C).to(DEVICE)
        self.model_loss = YoloLoss(S, B, C).to(DEVICE)
        dataset_args = dict(S=S, B=B, C=C)#, img_transformer=img_transformer)
        self.dataset = VOCDataset(DATA_DIR+"8examples.csv", IMG_DIR, LABEL_DIR, **dataset_args)
        dataloader_args = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, drop_last = False, shuffle=True)
        self.test_data = DataLoader(self.dataset, **dataloader_args)
        print("data size: ", len(self.test_data))

    def test_loss(self):
        imgs, targets = next(iter(self.test_data))
        print("imgs shape: ", imgs.shape)
        print("target shape: ", targets.shape)
        yhat = self.model(imgs.to(DEVICE))
        print("yhat shape: ", yhat.shape)
        # yhat = torch.rand(1, S, S, 30)
        # y = torch.rand(1, S, S, 30)
        loss = self.model_loss(yhat.to(DEVICE), targets.to(DEVICE))
        print("loss: ", loss)
        boxes, class_ids, scores = self.dataset.get_target_boxes(targets)
        for i in range(len(targets)):
            self.dataset.show_boxes(imgs[i], boxes[i], class_ids[i], scores[i])
