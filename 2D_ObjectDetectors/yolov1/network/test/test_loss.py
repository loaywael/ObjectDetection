from network.loss import YoloLoss
from unittest import TestCase
import numpy as np
import torch
torch.set_printoptions(linewidth=10000, edgeitems=160)

torch.manual_seed(13)
class TestLoss(TestCase):
    def test_loss(self):
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
        self.model_loss = YoloLoss(7, 2, 20)
        loss = self.model_loss(self.y.to("cuda"), self.yhat.to("cuda"))
        print(loss)
