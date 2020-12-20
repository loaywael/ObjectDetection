import torch
import torch.nn as nn
from utils import intersection_over_union


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
                # i.e. repeated blocks group
                *sub_blocks, n = layer
                for i in range(n):
                    for (k, c, s, p) in sub_blocks:
                        layers.append(ConvBlock(in_channels, k, c, s, p))
                        in_channels = c     # update next c_in = last c_out
        return nn.Sequential(*layers)

    def _build_fcls(self, grid_size, num_boxes, num_classes):
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
            nn.Linear(4096, S*S*(C + B*5)),     
            # to be reshaped later into (S, S, (C + B*5))
        ]
        return nn.Sequential(*output_layers)

    
class YoloLoss(nn.Module):
    def __init__(self, S, B, C):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S
        self.B = B
        self.C = C
        self.p_noobj = 0.5
        self.coord = 5.0

    def forward(self, predictions, target):
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B*5)
        iou_scores = []
        # scoring anchor boxes
        for i in range(self.B):
            start_id = self.C+1 + i*4
            stop_id = start_id + (i+1)*4]
            iou_score = intersection_over_union(predictions[..., start_id:stop_id)
            iou_scores.append(iou_score.unsqueeze(dim=0))
        # filtering predicted anchor boxes
        iou_scores = torch.cat(iou_scores, dim=0)
        iou_max, bestbox_arg = torch.max(iou_scores, dim=0)
        objness = target[..., self.C].unsqueeze(3)
        # indexing box confidence, location
        start_id = self.C+1 + bestbox_arg*4
        stop_id = start_id + (bestbox_arg+1)*4]
        # gndtruth_box shape: (N, S, S, 4), where 4: x, y, w, h
        gndtruth_box = objness * target[..., self.C+1:] 
        gndtruth_box[..., 2:4] = torch.sqrt(gndtruth_box[..., 2:4])
        # p_anchor_box shape: (N, S, S, 4), where 4: x, y, w, h
        p_anchor_box = objness* predictions[..., start_id:stop_id]
        # prevent grads of being zero or box dims being negative value
        p_anchor_box[..., 2:4] = torch.sqrt(torch.abs(p_anchor_box[..., 2:4]+1e-6))
        p_anchor_box_sign = torch.sign(p_anchor_box[..., 2:4])
        p_anchor_box = p_anchor_box * p_anchor_box_sign
        # -------------- Box Coordinates Loss --------------
        # out should be flattened shape (N*S*S, 4)
        box_loc_loss = self.mse(
            torch.flatten(gndtruth_box, end_dim=-2),
            torch.flatten(p_anchor_box, end_dim=-2)
        )
        # --------------    Object Loss    --------------
        # flattned shape would be (N*S*S)
        objness_loss = self.mse(
            torch.flatten(objness * target[..., self.C]),
            torch.flatten(objness * predictions[..., start_id-1])
        )
        # --------------    No Object Loss    --------------
        no_objness_loss = self.mse(
            torch.flatten((1-objness) * target[..., self.C]),
            torch.flatten((1-objness) * predictions[..., start_id-1])
        )
        # --------------    Class Loss    --------------
        # (N, S, S, 20) --> (N*S*S, 20)
        class_loss = self.mse(
            torch.flatten(objness * target[..., :self.C], end_dim=-2),
            torch.flatten(objness * predictions[..., :self.C], end_dim=-2)
        )
        loss = (self.coord * box_loc_loss) + objness_loss
        loss +=  (self.coord * no_objness_loss) + class_loss
