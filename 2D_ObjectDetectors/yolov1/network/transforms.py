import PIL
import torch
import random
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib import patches


class HorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.p = prob
        self.t = transforms.RandomHorizontalFlip(p=1.0)

    def __call__(self, items):
        r =  random.random()
        if r < self.p:
            img, boxes = items
            img = self.t(img)
            for idx, box in enumerate(boxes):
                boxes[idx, 0] = 1 - box[0]
            return img, boxes
        else:
            return items


class RandomRotation(object):
    def __init__(self, prob, degree):
        self.p = prob
        self.d = degree

    def __call__(self, items):
        r = random.random()
        img, boxes = items
        if r < self.p:
            img = img.rotate(self.d)
            print(r)
            degree = np.deg2rad(-self.d)
            rotation_matrix = np.array([
                [np.cos(degree), -np.sin(degree)],
                [np.sin(degree), np.cos(degree)]
            ])
            print(rotation_matrix.shape, boxes[:, :2].shape)
            boxes[:, :2] = torch.FloatTensor(boxes[:, :2].numpy() @ rotation_matrix)
        return img, boxes


img_transformer = transforms.Compose([
    transforms.Resize(224),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25),
    transforms.GaussianBlur(5, sigma=(0.1, 2.0)),
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.25, ratio=(0.01, 0.1))
])
