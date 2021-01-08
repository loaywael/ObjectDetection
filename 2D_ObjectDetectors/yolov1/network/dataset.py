import os
import glob
import torch
import numpy as np
import pandas as pd
from PIL import Image
from network import DATA_PATH
import matplotlib.pyplot as plt
from torchvision import transforms
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

    def __init__(self, csv_file, img_dir, label_dir, S=7, B=2, C=20, 
        img_transformer=None, pair_transformer=None):
        """
        Params
        ------
        csv_file : (str)
            csv file containing img (name, lable)
        
        img_dir : (str)
            path to source images directory

        label_dir : (str)
            path to source labels directory

        S : (int)
            number of cells in each axe

        B : (int)
            number of boxes in each cell
        
        C : (int)
            number of classes to predict
        
        transform : (bool)
            apply image augmentation 
        
        """
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_transformer = img_transformer
        self.pair_transformer = pair_transformer
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        """
        Returns the length of the data generator
        """
        return len(self.annotations)

    def _norm_box_dims(self, box, i, j):
        """
        Normalising the boxes relative to their associated grid cell dimensions, 
        where each given box is normalised relative to its image dimensions

        Params
        ------
        box : (torch.tensor)
            normalised box relative to its image dimensions
            {cx, cy, w, h} of shape-- > (4, )

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
        cx, cy = self.S * cx - (j+1), self.S * cy - (i+1)     
        bw, bh = self.S * bw, self.S * bh               # equivalent cell steps
        return [cx, cy, bw, bh]

    def _denorm_box_dims(self, box, i, j):
        """
        Retrieving the box dimensions relative to the full image size, 
        instead of the cell dimensions.

        Params
        ------
        box : (torch.tensor)
            normalised box relative to its image dimensions
            {cx, cy, w, h} of shape-- > (4, )

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
        cx, cy = (j+1 + cx)/self.S, (i+1 + cy)/self.S
        bw, bh = bw/self.S, bh/self.S
        return [cx, cy, bw, bh]

    def _denorm_batch_boxes_dims(self, boxes, i, j):
        """
        Retrieving the box dimensions relative to the full image size, 
        instead of the cell dimensions for a given batch of boxes.

        Params
        ------
        box : (torch.tensor)
            normalised box relative to its image dimensions
            {cx, cy, w, h} of shape-- > (N, 4)

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
        cx, cy = (j + cx)/self.S, (i + cy)/self.S
        bw, bh = bw/self.S, bh/self.S
        return torch.hstack([cx, cy, bw, bh])

    def _get_cell_ids(self, box_center):
        """
        Locate the cell that contains the ceter point of the object bounding box

        Params
        ------
        box_center : (tuple)
            x, y location of the box midpoint

        Returns
        -------
        cell_id : (tuple)
            y, x indices of the cell
        """
        # i, j cell numbers in y, x starting from 0
        cx, cy = box_center
        i, j = int(cy * self.S) , int(cx * self.S) 
        return i, j

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
            true target matrix of shape --> (S, S, C+5*B)
        """
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        img = Image.open(img_path).resize((448, 448))
        annotations = []    # all boxes annotations in the image
        with open(label_path) as csv_f:
            boxes_annotations = csv_f.readlines()
            for box_label in boxes_annotations:
                annotations.append([eval(val) for val in box_label.strip().split()])
        annotations = torch.tensor(annotations)

        if self.img_transformer:
            img = self.img_transformer(img)
        if self.pair_transformer:
            img, annotations[:, 1:] = self.pair_transformer((img, annotations[:, 1:]))

        target_matrix = torch.zeros((self.S, self.S, self.C+5*self.B))
        # PIL image shape --> (w, h, c)
        for class_id, (*box) in annotations:
            cx, cy, w, h = box  # identify the box cell
            i, j = self._get_cell_ids((cx, cy))
            if not target_matrix[i, j, int(self.C)]:
            # normalize dims relative to cell dims
                target_matrix[i, j, int(self.C)] = 1    # conf_score
                target_matrix[i, j, int(class_id)] = 1    # class_id
                normed_box = self._norm_box_dims(box, i, j)
                target_matrix[i, j, self.C+1:self.C+5] = torch.tensor(normed_box)
        return img, target_matrix   
    
    def get_target_boxes(self, y_matrix, threshold=0.5):
        """
        Takes target/prediction matrix and filters all the found boxes returning 
        their location, class_id, score and their grid cell indices.

        Params
        ------
        y_matrix : (torch.tensor)
            prediction/target matrix of shape --> (N, S, S, C+5*B)

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
        all_boxes, all_classes, all_scores = [], [], []
        for i in range(len(y_matrix)):
            pc_scores = y_matrix[i, ..., 20:21]        # confidence scores of all boxes in all cells
            anchor_boxes = y_matrix[i, ..., 21:25]     # box locations of all boxes in all cells
            classes_prob = y_matrix[i, ..., :20]       # box classes probability of all boxes in all cells
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
            i, j = torch.where(score_mask == 1)   
            i, j = i.unsqueeze(-1), j.unsqueeze(-1)     # converting to col-vector to be used to denorm-boxes
            boxes = self._denorm_batch_boxes_dims(boxes, i, j)
            all_boxes.append(boxes)
            all_classes.append(class_ids)
            all_scores.append(scores)
        return all_boxes, all_classes, all_scores

    @staticmethod
    def restore_box_dim(box, img_size):
        """
        Denormalizing boxes back to its full size

        Params
        ------
        box : (torch.tensor)
            normalized box relative to its image size of shape (,4)
        
        img_size : (tuple)
            the width, height of the image containing the box

        Returns
        -------
        box : (torch.tensor)
            denormed box of full size
        """
        W, H = img_size
        cx, cy, bw, bh = box
        box[[0, 2]] *= W
        box[[1, 3]] *= H
        return box

    def show_boxes(self, image, boxes, class_ids=None, scores=None, img_size=None):
        """
        Draws the bounding boxes, class id, score for each given box over the image

        Params
        ------
        image : (PIL.Image)
            src image to draw over and show
        
        boxes : (torch.tensor), (numpy.ndarray), or (list) 
            the bounding boxes locations --> (M, 4)

        class_ids : (torch.tensor), or (int)
            the boxes class id --> (M, 1)

        scores : (torch.tensor), or (int)
            the boxes score --> (M, 1)

        img_size : (tuple)
            image size to display in
        """
        if not isinstance(image, (Image.Image, )):
            image = transforms.ToPILImage()(image)
        image = image.resize(img_size) if img_size else image
        fig, ax = plt.subplots(1)
        plt.imshow(image)
        for i, box in enumerate(boxes):
            (cx, cy, bw, bh) = VOCDataset.restore_box_dim(box.squeeze(), image.size)
            rec = patches.Rectangle(
                (cx-bw//2, cy-bh//2), bw, bh, 
                edgecolor="yellow", alpha=0.3,
                linewidth=2, facecolor=(0, 0, 0)
            )
            ax.add_patch(rec)
            rec = patches.Rectangle(
                (cx-bw//2, cy-bh//2), bw, bh, 
                edgecolor="yellow", alpha=0.3,
                linewidth=2, facecolor="none", linestyle="-"
            )
            if class_ids is not None and scores is not None:
                txt = ax.text(
                    (cx-bw//2), cy-(bh//2),  f"id: {int(class_ids[i]):0.2f} | pc: {float(scores[i])}", size=10, 
                    ha="left", va="top", alpha=1, color="black",
                    bbox=dict(facecolor="yellow", edgecolor="yellow", linewidth=1, alpha=0.3),
                )
            ax.add_patch(rec)
            ax.scatter(cx, cy, marker="+", s=50, c="yellow")
            ax.set_xticks(np.linspace(0, image.size[0], num=self.S+1))
            ax.set_yticks(np.linspace(0, image.size[0], num=self.S+1))
            fig.tight_layout()
            ax.grid(True)
        plt.show()
        plt.close()

    @staticmethod
    def plot_batch(X, boxes, class_ids=None, scores=None, img_size=None, max_cols=2):
        batch_size = len(X)
        fig = plt.figure(figsize=(9, 9))
        # fig.tight_layout()
        for i in range(batch_size):
            ax = fig.add_subplot((batch_size//max_cols)+1, max_cols, i+1)
            image = transforms.ToPILImage()(X[i])
            ax.imshow(image)
            # print(">>> ", boxes[i])
            for j, (cx, cy, bw, bh) in enumerate(boxes[i]):
                rec = patches.Rectangle(
                    (cx-bw//2, cy-bh//2), bw, bh, 
                    edgecolor="yellow", alpha=1.,
                    linewidth=1.5, facecolor="none", linestyle="-"
                )
                rec = patches.Rectangle(
                    (cx-bw//2, cy-bh//2), bw, bh, 
                    edgecolor="yellow", alpha=0.3,
                    linewidth=1.5, facecolor=(0, 0, 0)
                )
                if class_ids and scores:
                    txt = ax.text(
                        (cx-bw//2), cy-(bh//2),  f"id: {class_ids[i][j]:0.2f} | pc: {scores[i][j]}", size=10, 
                        ha="left", va="top", alpha=1, color="white",
                        bbox=dict(facecolor="green", edgecolor="black", linewidth=1, alpha=0.3),
                    )
                ax.add_patch(rec)
                ax.add_artist(plt.Circle((cx, cy), 5))
                ax.set_xticks(np.linspace(0, 448, num=8))
                ax.set_yticks(np.linspace(0, 448, num=8))
                ax.grid(True)
        plt.show()
