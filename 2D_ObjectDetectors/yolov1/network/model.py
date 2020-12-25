import torch
import torch.nn as nn
from network.utils import eval_iou


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
    def __init__(self, input_size=(448, 448, 3), S=7, B=2, C=20, **kwargs):
        super(Yolov1, self).__init__()
        self._arch = ARCH_CONFIG
        self.input_size = input_size[-1::-1]
        # print(self.input_size)
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

    
class YoloLoss(nn.Module):
    def __init__(self, S, B, C):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S
        self.B = B
        self.C = C
        self.p_noobj = 0.5
        self.coord = 5.0

    def forward(self, predection_matrix, target_matrix):
        """
        Estimate the loss for boxes regression, confidence of predictions, 
        and classes category predictions.

        Params
        ------
        predection_matrix : (torch.tensor)
            output of the FCL layers of shape --> (N, S* S*B*(C+5))

        target_matrix : (torch.tensor)
            true target matrix of shape --> (N, S, S, B, C+5)

        Returns
        -------
        loss : (torch.tensor)
            the batch mean squared error of all grid cell boxes per image 
        """
        # reshaping predicted matrix from (N, S* S*B*(C+5)) --> (N, S, S, B, C+5)
        grid_shape = (-1, self.S, self.S, self.B, (self.C+5))
        predection_matrix = predection_matrix.reshape(grid_shape)
        anchor_boxes = predection_matrix[..., 21:25]
        target_boxes = target_matrix[..., 21:25]
        objness = target_matrix[..., self.C:self.C+1]
        # print("debugger >>> ", predection_matrix.shape)
        # print("debugger >>> ", anchor_boxes.shape)
        # print("debugger >>> ", target_boxes.shape)
        # print("debugger >>> ", objness.shape)
        # # scoring anchor boxes
        iou_scores = eval_iou(anchor_boxes, target_boxes)
        print("debugger >>> ", iou_scores.shape)
        # print("ious shape >>> ", iou_scores.shape)
        # filtering predicted anchor boxes
        bestboxes_args = torch.argmax(iou_scores, dim=-2).squeeze()
        bestboxes_args = bestboxes_args.reshape(-1)
        # print("maxargs >>>\t ", bestboxes_args.shape)
        print("debugger bestboxes_args >>> ", bestboxes_args.shape)
        # print("\n"*3, "objness >>> ", objness, "\n"*3)

        # indexing box confidence, location
        gndtruth_box = objness * target_boxes
        print("debugger >>> ", gndtruth_box.shape)
        gndtruth_box[..., 2:4] = torch.sqrt(gndtruth_box[..., 2:4])
        # anchor_box shape: (N, S, S, 4), where 4: x, y, w, h
        anchor_boxes = anchor_boxes.reshape(-1, 4)
        print("crazy >>> ", anchor_boxes.shape)
        anchor_box = objness.reshape(-1, 1)[bestboxes_args] * anchor_boxes[bestboxes_args]
        print("anchor_box >>> ", anchor_box.shape)
        print("target_box >>> ", gndtruth_box.shape)

        # prevent grads of being zero or box dims being negative value
        anchor_box_w_h = anchor_box[..., 2:4]
        anchor_box_w_h = torch.sign(anchor_box_w_h)*torch.sqrt(torch.abs(anchor_box_w_h+1e-6))
        anchor_box[..., 2:4] = anchor_box_w_h

        # -------------- Box Coordinates Loss --------------
        #           flattent(N, S, S, 4) --> (N*S*S, 4)
        box_loc_loss = self.mse(
            torch.flatten(gndtruth_box, end_dim=-2), 
            torch.flatten(anchor_box, end_dim=-2)
        )
        # print("gndtruth_box >>> ", gndtruth_box.shape)
        # print("anchor_box >>> ", anchor_box.shape)

        # --------------    Object Loss    --------------
        #           flattent(N, S, S, 1) --> (N*S*S, 1)
        p_box_score = objness * (bestboxes_args * predection_matrix[..., 25:26]
            + (1 - bestboxes_args) * predection_matrix[..., 20:21])
        objness_loss = self.mse(
            torch.flatten(objness * target_matrix[..., 20:21], end_dim=-2),
            torch.flatten(objness * p_box_score, end_dim=-2)
        )
        # print("true_objness_vals >>> ", target_matrix[..., 20:21].shape)
        # print("pred_objness_vals >>> ", p_box_score.shape)

        # --------------    No Object Loss    --------------
        #           flattent(N, S, S, 1) --> (N*S*S, 1)
        no_objness_loss = self.mse(
            torch.flatten((1-objness) * target_matrix[..., 20:21], end_dim=-2),
            torch.flatten((1-objness) * predection_matrix[..., 20:21], end_dim=-2)
        )
        no_objness_loss += self.mse(
            torch.flatten((1-objness) * target_matrix[..., 20:21], end_dim=-2),
            torch.flatten((1-objness) * predection_matrix[..., 25:26], end_dim=-2)
        )
        # print("true_no_objness_vals >>> ", ((1-objness)*target_matrix[..., 20:21]).shape)
        # print("pred_no_objness_vals >>> ", ((1-objness)*predection_matrix[..., 20:21]).shape)


        # --------------    Class Loss    --------------
        #           flattent(N, S, S, 20) --> (N*S*S, 20)
        class_loss = self.mse(
            torch.flatten(objness * target_matrix[..., :self.C], end_dim=-2),
            torch.flatten(objness * predection_matrix[..., :self.C], end_dim=-2)
        )
        # print("true_class >>> ", target_matrix[..., :self.C].shape)
        # print("pred_class >>> ", predection_matrix[..., :self.C].shape)

        loss = (self.coord * box_loc_loss) + objness_loss
        loss +=  (self.p_noobj * no_objness_loss) + class_loss

        return loss