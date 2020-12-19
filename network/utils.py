import torch
from cnn import Yolov1



def test_yolov1(S=7, B=2, C=20):
    net = Yolov1(grid_size=S, num_boxes=B, num_classes=C)
    x = torch.randn((2, 3, 448, 448))
    print(net(x).shape)


test_yolov1()