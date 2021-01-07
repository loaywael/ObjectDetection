import torch
import random
from torchvision import transforms
import matplotlib.pyplot as plt


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


img_transformer = transforms.Compose([
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25),
    transforms.RandomErasing(),
    transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
    transforms.ToTensor()
])


pair_transformer = HorizontalFlip(0.3)