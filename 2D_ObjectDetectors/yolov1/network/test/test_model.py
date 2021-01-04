from unittest import TestCase
import numpy as np
import torch
from network.model import Yolov1
from torchsummary import summary


torch.manual_seed(13)

class TestModel(TestCase):
    def setUp(self):
        self.x = torch.abs(torch.randint(0, 255, (2, 3, 448, 448)))/255.
        self.y = torch.abs(torch.randint(0, 1000, (2, 7, 7, 2, 25)))/1000.

        self.model = Yolov1(S=7, B=2, C=20)
        self.predictions = self.model(self.x)
        print(summary(self.model.to("cuda"), torch.Size([3, 448, 448])))

    def test_yolov1(self):
        prediction_shape = list(self.predictions.shape)
        self.assertEqual(prediction_shape, [2, 2450])
        # print(prediction_shape)

    # def test_yolo_loss(self):
    #     loss = self.model_loss(self.predictions, self.y)
    #     print(loss)
    