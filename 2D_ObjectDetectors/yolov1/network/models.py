import torch
import torch.nn as nn
from torchvision import models
from network.utils import eval_iou
from network.config import ARCH_CONFIG


class ConvBlock(nn.Module):
    """
    Darknet Convolution Block to be used to build the Darknet
    """
    def __init__(self, c_in, k, c_out, s, p, **kwargs):
        """
        Params
        ------
        c_in : (int)
            number of channels of the input feature maps
        k : (int)
            kernel/filter size
        c_out : (int)
            number of the filters of the convolution
        s : (int)
            stride of kernels
        p : (int)
            padding of feature maps

        """
        super(ConvBlock, self).__init__()
        kwargs["in_channels"] = c_in
        kwargs["kernel_size"] = k
        kwargs["out_channels"] = c_out
        kwargs["stride"] = s
        kwargs["padding"] = p
        kwargs["bias"] = False
        self.conv = nn.Conv2d(**kwargs)
        self.batchnorm = nn.BatchNorm2d(c_out)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))


class YoloDarknetv1(nn.Module):
    """
    YOLOv1 Original Architecture as the author paper proposed

    """
    def __init__(self, input_size=(448, 448, 3), S=7, B=2, C=20, **kwargs):
        super(YoloDarknetv1, self).__init__()
        self._arch = ARCH_CONFIG
        self.input_size = input_size[-1::-1]
        self.S, self.B, self.C = S, B, C
        self.darknet = self._build_darknet(self._arch)
        self.fcls = self._build_fcls(S, B, C, **kwargs)

    def forward(self, x):
        x = self.darknet(x)
        return self.fcls(torch.flatten(x, start_dim=1))
    
    def _build_darknet(self, layers_config):
        """
        Building the Darknet Backbone feature extraction network

        Params
        ------
        layers_config : (list)
            network layers configurations
        """
        layers = []
        # padding: same
        in_channels = self.input_size[0]
        for layer in layers_config:
            if type(layer) == tuple:
                (k, c, s, p) = layer
                layers.append(ConvBlock(in_channels, k, c, s, p))
                in_channels = c
            elif type(layer) == str:
                layers.append(nn.MaxPool2d(kernel_size=(2, 2), stride=2))
            elif type(layer) == list:
                # i.e. repeated blocks group
                *sub_blocks, n = layer
                for i in range(n):
                    for (k, c, s, p) in sub_blocks:
                        layers.append(ConvBlock(in_channels, k, c, s, p))
                        in_channels = c     # update next c_in = last c_out
        return nn.Sequential(*layers)

    @staticmethod
    def _build_fcls(grid_size, num_boxes, num_classes):
        """
        Building the fully connected output layers

        Params
        ------
        grid_size: (int)
            number of grid cells per axe of the input image
        num_boxes: (int)
            number of anchor boxes per grid cell
        num_classes: (int)
            number of classes in the dataset
        
        """
        S, B, C = grid_size, num_boxes, num_classes
        output_layers = [
            nn.Flatten(), 
            nn.Linear(S*S*1024, 4096), 
            nn.LeakyReLU(0.1),
            nn.Linear(4096, S*S*B*(C+5)),     
            # to be reshaped later into (S, S, (C + B*5))
        ]
        return nn.Sequential(*output_layers)


class YoloResnetv1(nn.Module):
    """
    YOLOv1 Original Architecture as the author paper proposed

    """
    def __init__(self, input_size=(448, 448, 3), S=7, B=2, C=20, **kwargs):
        super(YoloResnetv1, self).__init__()
        self._arch = ARCH_CONFIG
        self.input_size = input_size[-1::-1]
        self.S, self.B, self.C = S, B, C
        self.resnet = models.resnet50(pretrained=True, progress=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])
        self.fcls = self._build_fcls(**kwargs)
        for param in self.resnet.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.resnet(x)
        print(x.shape)
        return self.fcls(x)
    
    def _build_fcls(self):
        """
        Building the fully connected output layers
        """
        S, B, C = self.S, self.B, self.C
        output_layers = [
            nn.Conv2d(2048, 1024, 3, bias=False),
            nn.BatchNorm2d(1024),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool2d((9, 9)), 
            nn.Conv2d(1024, (C+5*B), 1, bias=False),
            nn.Dropout(0.5),
            nn.Flatten()
        ]
        return nn.Sequential(*output_layers)
