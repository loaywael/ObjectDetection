from unittest import TestCase
import numpy as np
import torch
from network.models import YoloResnetv1
from network.models import YoloDarknetv1
from torchsummary import summary


torch.manual_seed(13)
torch.set_printoptions(linewidth=10000, edgeitems=160)
DEVICE = "cuda"
C = 20
S = 7
B = 2
INPUT_SIZE = (3, 448, 448)


class TestModel(TestCase):
    def setUp(self):
        self.x = (torch.abs(torch.randint(0, 255, (2, *INPUT_SIZE)))/255.).to(DEVICE)
        self.y = (torch.abs(torch.randint(0, 1000, (2, 7, 7, 2, 25)))/1000.).to(DEVICE)
        # self.model = YoloDarknetv1(S=S, B=B, C=C).to(DEVICE)
        self.model = YoloResnetv1(S=S, B=B, C=C).to(DEVICE)
        self.predictions = self.model(self.x)
        print(summary(self.model.to(DEVICE), torch.Size(INPUT_SIZE)))
        print("pred_shape: ", self.predictions.shape)

    def test_yolov1(self):
        prediction_shape = list(self.predictions.shape)
        self.assertEqual(prediction_shape, [2, (5*B+C)*S*S])
        # print(prediction_shape)

    # def test_yolo_loss(self):
    #     loss = self.model_loss(self.predictions, self.y)
    #     print(loss)
    