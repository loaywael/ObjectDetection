import os
import glob
import torch
import numpy as np
import pandas as pd
from PIL import Image
from network import DATA_PATH



class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_dir, label_dir, S=7, B=2, C=20, tansform=False):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.tansform = tansform
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        """
        Returns the length of the data generator
        """
        return len(self.annotations)

    def _norm_box_dims(self, box):
        """
        Normalising the boxes relative to their associated grid cell dimensions, 
        where each given box is normalised relative to its image dimensions

        Params
        ------
        box : (torch.tensor)
            normalised box relative to its image dimensions
            {class, cx, cy, w, h} of shape-- > (4, )

        Return
        ------
        normed_box : (torch.tensor)
            normalised box relative to its cell dimensions
            {class, cx, cy, w, h} of shape-- > (4, )
        """
        x, y, w, h = box
        cx, cy = self.S*x - j, self.S*y - i     # relative to cell origin
        w, h = self.S*w, self.S*h               # equivalent cell steps
        return [cx, cy, w, h]

    def _denorm_box_dim(self, box):
        pass

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
            true target matrix of shape --> (S, S, C+5*B)
        """
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        annotations = []
        with open(label_path) as f:
            for label in f.readlines():
                annotations.append([eval(val) for val in label.strip().split()])
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        img = Image.open("img_path")
        annotations = torch.tensor(annotations)
        if self.tansform:
            img, boxes = self.tansform(img, annotations)
        # target output shape --> (S, S, C+ 5*B)
        target_matrix = torch.zeros((self.S, self.S, self.C + 5*self.B))
        # PIL image shape --> (w, h, c)
        for class_id, box in annotations:
            # identify the box cell
            i, j = int(y * self.S), int(x * self.S)
            # normalize dims relative to cell dims
            normed_box = self._norm_box_dims(box)
            target_matrix[i, j, [self.C, class_id] = 1
            target_matrix[i, j, self.C+1:self.C+5] = torch.tensor(normed_box)
            
        return img, target_matrix   

# VOCDataset()