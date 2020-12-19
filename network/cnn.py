import torch
import torch.nn as nn


# YOLOv1 Network Architecture
ARCH_CONFIG = [
    # k, c, s, p
    (7, 64, 2, 3), "M", 
    # -----------------
    (3, 192, 1, 1), "M", 
    # -----------------
    (1, 128, 1, 0), 
    (3, 256, 1, 1), 
    (1, 256, 1, 0), 
    (3, 512, 1, 1), "M", 
    # -----------------
    [(1, 256, 1, 0), (3, 512, 1, 1), 4], 
    (1, 512, 1, 0), 
    (3, 1024, 1, 1), "M", 
    # -----------------
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2], 
    (3, 1024, 1, 1), 
    (3, 1024, 2, 1), 
    # -----------------
    (3, 1024, 1, 1), 
    (3, 1024, 1, 1), 
]


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


class Yolov1(nn.Module):
    """
    YOLOv1 Original Architecture as the author paper proposed

    """
    def __init__(self, in_channels=3, **kwargs):
        super(Yolov1, self).__init__()
        self._arch = ARCH_CONFIG
        self.in_channels = in_channels
        self.darknet = self._build_darknet(self._arch)
        self.fcls = self._build_fcls(**kwargs)

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
        in_channels = self.in_channels
        for layer in layers_config:
            if type(layer) == tuple:
                (k, c, s, p) = layer
                layers.append(ConvBlock(in_channels, k, c, s, p))
                in_channels = c
            elif type(layer) == str:
                layers.append(nn.MaxPool2d(kernel_size=(2, 2), stride=2))
            elif type(layer) == list:
                # i.e. repeated block
                *sub_blocks, n = layer
                for i in range(n):
                    for j, (k, c, s, p) in enumerate(sub_blocks):
                        #if j > 0:   # if not the first sub-block
                            # update c_in for the current sub-block from the last sub-block
                            #in_channels = sub_blocks[j-1][1]
                        layers.append(ConvBlock(in_channels, k, c, s, p))
                        in_channels = c
        return nn.Sequential(*layers)

    def _build_fcls(self, grid_size, num_boxes, num_classes):
        S, B, C = grid_size, num_boxes, num_classes
        output_layers = [
            nn.Flatten(), 
            nn.Linear(S*S*1024, 4096), 
            nn.LeakyReLU(0.1),
            nn.Linear(4096, S*S*(C + B*5)),     
            # to be reshaped later into (S, S, (C + B*5))
        ]
        return nn.Sequential(*output_layers)

    
  