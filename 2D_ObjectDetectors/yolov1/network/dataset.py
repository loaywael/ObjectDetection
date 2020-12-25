import os
import glob
import torch
import numpy as np
import pandas as pd
from PIL import Image
from network import DATA_PATH
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# torch.set_printoptions(precision=3, linewidth=300, profile="full")



class VOCDataset(torch.utils.data.Dataset):
    """

    Atrributes
    ----------
    annotations : (pandas.DataFrame) 
        csv_file of dataset --> (image.jpg, label.txt)
    
    img_dir : (str)
        directory of all the images

    label_dir : (str)
        directory of all the labels according to images
    """

    def __init__(self, csv_file, img_dir, label_dir, S=7, B=2, C=20, transform=False):
        """
        """
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        """
        Returns the length of the data generator
        """
        return len(self.annotations)

    @staticmethod
    def _norm_box_dims(box, S, i, j):
        """
        Normalising the boxes relative to their associated grid cell dimensions, 
        where each given box is normalised relative to its image dimensions

        Params
        ------
        box : (torch.tensor)
            normalised box relative to its image dimensions
            {cx, cy, w, h} of shape-- > (4, )

        S : (int)
            number of cells per axis of the image, 
            assuming ncells are equal in both x, y axes

        i : (int)
            grid cell y-axis index starts from 0

        j : (int)
            grid cell x-axis index starts from 0

        Return
        ------
        normed_box : (torch.tensor)
            normalised box relative to its cell dimensions
            {cx, cy, w, h} of shape-- > (4, )
        """
        cx, cy, bw, bh = box
        # relative to cell origin, i, j --> should start from 1
        cx, cy = S * cx - (j+1), S * cy - (i+1)     
        bw, bh = S * bw, S * bh               # equivalent cell steps
        return [cx, cy, bw, bh]

    @staticmethod
    def _denorm_box_dims(box, S, i, j):
        """
        Retrieving the box dimensions relative to the full image size, 
        instead of the cell dimensions.

        Params
        ------
        box : (torch.tensor)
            normalised box relative to its image dimensions
            {cx, cy, w, h} of shape-- > (4, )

        S : (int)
            number of cells per axis of the image, 
            assuming ncells are equal in both x, y axes

        i : (int)
            grid cell y-axis index starts from 0

        j : (int)
            grid cell x-axis index starts from 0

        Return
        ------
        denormed_box : (torch.tensor)
            denormed box relative to the full image dimensions
            {cx, cy, w, h} of shape-- > (4, )
        """
        cx, cy, bw, bh = box
        # i, j --> should start from 1
        cx, cy = (j+1 + cx)/S, (i+1 + cy)/S
        bw, bh = bw/S, bh/S
        return [cx, cy, bw, bh]

    @staticmethod
    def _denorm_batch_boxes_dims(boxes, S, i, j):
        """
        Retrieving the box dimensions relative to the full image size, 
        instead of the cell dimensions for a given batch of boxes.

        Params
        ------
        box : (torch.tensor)
            normalised box relative to its image dimensions
            {cx, cy, w, h} of shape-- > (N, 4)

        S : (int)
            number of cells per axis of the image, 
            assuming ncells are equal in both x, y axes

        i : (torch.tensor)
            grid cell y-axis indices starts from 0

        j : (torch.tensor)
            grid cell x-axis indices starts from 0

        Return
        ------
        denormed_boxes : (torch.tensor)
            denormed boxes relative to their full image dimensions 
            for a batch of images {cx, cy, w, h} of shape-- > (N, 4)
        """
        # i, j --> should start from 1
        i, j = i+1, j+1
        cx, cy, bw, bh = boxes.split(1, -1)
        cx, cy = (j + cx)/S, (i + cy)/S
        bw, bh = bw/S, bh/S
        return torch.hstack([cx, cy, bw, bh])


    def __getitem__(self, index):
        """
        Applys all the required preprocessing pipeline over a single image

        Params
        ------
        index : (int)
            the index of the image to be preprocessed

        Returns
        -------
        processed_img : (PIL.Image)
            the preprocessed image
        target_matrix : (troch.tensor)
            each box vector is [C0, ..., Cn, pc, cx, cy, w, h]
            true target matrix of shape --> (S, S, B, C+5)
        """
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        img = Image.open(img_path)
        annotations = []    # all boxes annotations in the image
        with open(label_path) as csv_f:
            boxes_annotations = csv_f.readlines()
            for box_label in boxes_annotations:
                annotations.append([eval(val) for val in box_label.strip().split()])
        annotations = torch.tensor(annotations)

        if self.transform:
            img, boxes = self.tansform(img, annotations)
        # target output shape --> (S, S, B, C+5)
        target_matrix = torch.zeros((self.S, self.S, self.B, self.C+5))
        # PIL image shape --> (w, h, c)
        for class_id, (*box) in annotations:
            cx, cy, w, h = box  # identify the box cell
            # i, j cell numbers in y, x starting from 0
            i, j = int(cy * self.S) - 1, int(cx * self.S) - 1
            # normalize dims relative to cell dims
            target_matrix[i, j, 0, int(self.C)] = 1    # conf_score
            target_matrix[i, j, 0, int(class_id)] = 1    # class_id
            normed_box = self._norm_box_dims(box, self.S, i, j)
            target_matrix[i, j, 0, self.C+1:self.C+5] = torch.tensor(normed_box)
        return img, target_matrix   
    
    @staticmethod
    def get_bboxes(y_matrix, threshold=0.5):
        """
        Takes target/prediction matrix and filters all the found boxes returning 
        their location, class_id, score and their grid cell indices.

        Params
        ------
        y_matrix : (torch.tensor)
            prediction/target matrix of shape --> (N, S, S, B, C+5)

        threshold : (float)
            boxes score threshold filter to keep the good boxes only

        Returns
        -------
        boxes : (torch.tensor)
            boxes location [cx, cy, bw, bh] of shape --> (N, M, 4)
            where N --> batch size, M --> no. of boxes found

        class_ids : (torch.tensor)
            class category id of shape --> (N, M, 1)

        scores : (torch.tensor)
            boxes score of shape --> (N, M, 1)

        grid_indices : (tuple)
            i, j grid indices of the found boxes of length --> (2)
            [i], [j] of similar shape of --> (N, M, 1)

        """
        pc_scores = y_matrix[..., 20:21]        # confidence scores of all boxes in all cells
        anchor_boxes = y_matrix[..., 21:25]     # box locations of all boxes in all cells
        classes_prob = y_matrix[..., :20]       # box classes probability of all boxes in all cells
        # -------------------------------------------------
        boxes_classes_scores = pc_scores * classes_prob  # boxes classes and scores (N, S, S, B, 20)
         # boxes score value and boxes class id of shape (N, S, S, 2)
        boxes_score, boxes_class = torch.max(boxes_classes_scores, dim=-1)   
        score_mask = boxes_score > threshold     # max filter mask
        # -------------------------------------------------
        scores = torch.masked_select(boxes_score, score_mask)   # index only the filter mask
        # select mask needs the input tensor, mask have equal dims and returns flattend tensor
        boxes = torch.masked_select(anchor_boxes, score_mask.unsqueeze(-1)).reshape(-1, 4)
        class_ids = torch.masked_select(boxes_class, score_mask)
        _, i, j, b = torch.where(score_mask == 1)   # retrieving the grid cell indices of each filtered box
        i, j = i.unsqueeze(-1), j.unsqueeze(-1)     # converting to col-vector to be used to denorm-boxes
        return boxes, class_ids, scores, [i, j, b]  

    @staticmethod
    def show_boxes(image, boxes):
        H, W, D = image.shape
        assert D == 3
        fig, ax = plt.subplots(1)
        plt.imshow(image)
        for (cx, cy, bw, bh) in boxes:
            cx, cy = cx * W, cy * H
            bw, bh = bw * W, bh * H
            rec = patches.Rectangle(
                (cx-bw//2, cy-bh//2), bw, bh, 
                edgecolor="yellow", 
                linewidth=3, facecolor="none"
            )
            rec = patches.Rectangle(
                (cx-bw//2, cy-(bh//2)-25), bw//4, 25, 
                edgecolor="yellow", 
                linewidth=3, facecolor="none"
            )
            ax.text(
                (cx-bw//2)+5, cy-(bh//2)-20, 
                bbox=dict(fill=False, edgecolor="r", linewidth=3)
            )
            ax.imshow(image)
            ax.add_patch(rec)
            plt.pause(0.0001)
        plt.show()
    
    # def resizing