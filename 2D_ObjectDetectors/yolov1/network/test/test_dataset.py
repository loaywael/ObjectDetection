from network.model import Yolov1
from torch.utils.data import DataLoader
from network.dataset import VOCDataset
from unittest import TestCase
import numpy as np
import torch
import os as os
import matplotlib.pyplot as plt
import matplotlib.patches as patches


torch.manual_seed(13)
DATA_DIR = "data/pascal_voc_yolo/"
IMG_DIR = DATA_DIR+"images/"
LABEL_DIR = DATA_DIR+"labels/"
BATCH_SIZE = 4
NUM_WORKERS = 4
TRANSFORM = False
S = 7
B = 2
C = 20

class TestModel(TestCase):
    def setUp(self):
        dataset_args = dict(S=S, B=B, C=C, transform=TRANSFORM)
        self.dataset = VOCDataset(DATA_DIR+"8examples.csv", IMG_DIR, LABEL_DIR, **dataset_args)
        # ------------------------
        dataloader_args = dict(
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            drop_last = False,
            shuffle=False
        )
        # self.test_data = DataLoader(self.dataset, **dataloader_args)
        # self.model = Yolov1(S=S, B=B, C=C)
        # print("data size: ", len(self.test_data))

    # def test_yolov1(self):
    #     prediction_shape = list(self.predictions.shape)
    #     self.assertEqual(prediction_shape, [2, 1470])
    #     print(prediction_shape)

    def test_norm_denorm_box(self):
        img = np.zeros((448, 448, 3), "uint8")
        w, h, c = img.shape
        # box1 = [0.5*w, 0.5*h, 0.5*w, 0.5*h]
        i, j = 4, 2
        box1 = [0.3, 0.7, 0.5, 0.5]
        normed_box1 = VOCDataset._norm_box_dims(box1, S, i, j)
        denormed_box1 = VOCDataset._denorm_box_dims(normed_box1, S, i, j)
        i, j = 3, 3
        box1 = [0.5, 0.5, 0.5, 0.5]

        normed_box1 = VOCDataset._norm_box_dims(box1, S, i, j)
        denormed_box1 = VOCDataset._denorm_box_dims(normed_box1, S, i, j)
        self.assertEqual(box1, denormed_box1)
        # print("---> ", "box1: ", box1)
        # print("---> ", "normed_box1: ", normed_box1)
        # print("---> ", "denormed_box1: ", denormed_box1)
        # normed_box1 = VOCDataset.show_boxes(img, [box1])


    # def test_get_box_shapes(self):
    #     img1, target1 = self.dataset.__getitem__(0)
    #     img2, target2 = self.dataset.__getitem__(1)
    #     target = torch.cat([target1.unsqueeze(0), target2.unsqueeze(0)])
    #     boxes, class_ids, scores, [i, j, b] = VOCDataset.get_bboxes(target)
    #     print(">>> ", boxes.shape, class_ids.shape, scores.shape)
    #     self.assertEqual(boxes.shape, (5, 4))
    #     self.assertEqual(class_ids.shape, (5,))
    #     self.assertEqual(scores.shape, (5,))
    #     img1.close()
    #     img2.close()

    def test_norm_denorm(self):
        img, target = self.dataset.__getitem__(1)
        target = target.unsqueeze(0)
        # print(target.shape)
        boxes, class_ids, scores, [i, j, b] = VOCDataset.get_bboxes(target)
        # print(boxes, "\n")
        boxes = VOCDataset._denorm_batch_boxes_dims(boxes, S, i, j)
        # print(boxes)
        # self.assertEqual(torch.round(boxes[0], 3).tolist(), [0.379, 0.567, 0.158, 0.381])
        # self.assertEqual(torch.round(boxes[1].tolist(), [0.339, 0.669, 0.402, 0.421]))
        # self.assertEqual(torch.round(boxes[2].tolist(), [0.555, 0.703, 0.078, 0.349]))
        # self.assertEqual(torch.round(boxes[3].tolist(), [0.612, 0.709, 0.084, 0.347]))
        img.close()
        # box = VOCDataset._denorm_box_dims(box, S, i, j)
        # VOCDataset.show_boxes(img, [denormed_box1])



    def test_dataset(self):
        pass
        # img, target = next(self.test_data)
        # for i, (img, target) in enumerate(self.test_data):
        #     print(img.shape, target.shape)
        # img, target = self.dataset.__getitem__(1)
        # # img.show()
        # # print(img.size, target.shape)
        # img.close()
        # for i, data in enumerate(self.test_data):
        #     print(self.test_data[i].shape)
        #     if i % 3 == 0: break
        # prediction_shape = list(self.predictions.shape)
        # self.assertEqual(prediction_shape, [2, 1470])
        # print(prediction_shape)


    # def test_yolo_loss(self):
    #     self.predictions = self.model(self.x)
    #     print(self.predictions.shape)
    