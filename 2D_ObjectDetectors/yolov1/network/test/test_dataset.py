from torch.utils.data import DataLoader
from network.dataset import VOCDataset
from network.transforms import pair_transformer
from network.transforms import img_transformer
from network.models import YoloDarknetv1
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from unittest import TestCase
import PIL
import numpy as np
import torch
import os


torch.set_printoptions(linewidth=10000, edgeitems=160)
torch.manual_seed(13)
DATA_DIR = "data/pascal_voc_yolo/"
IMG_DIR = DATA_DIR+"images/"
LABEL_DIR = DATA_DIR+"labels/"
BATCH_SIZE = 16
NUM_WORKERS = 4
S = 9
B = 2
C = 20

class TestModel(TestCase):
    def setUp(self):
        dataset_args = dict(S=S, B=B, C=C)
        # self.dataset = VOCDataset(DATA_DIR+"8examples.csv", IMG_DIR, LABEL_DIR, **dataset_args)
        self.dataset = VOCDataset(DATA_DIR+"100examples.csv", IMG_DIR, LABEL_DIR, **dataset_args)
        # ------------------------
        dataloader_args = dict(
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            drop_last = False,
            shuffle=True
        )
        self.test_data = DataLoader(self.dataset, **dataloader_args)
        print("data size: ", len(self.test_data))


    # def test_norm_denorm(self):
        # img1, target = self.dataset.__getitem__(2)
        # box = [0.07, 0.7598, 0.0360, 0.1744]
        # box = [0.408, 0.7278, 0.036, 0.1744]
        # i, j = self.dataset.get_cell_ids(box[:2])
        # normed = VOCDataset._norm_box_dims(box, S, i, j)
        # print("x_cell: ", j, "y_cell: ", i)
        # print("normed_box: ", normed)
        # denormed = VOCDataset._denorm_box_dims(normed, S, i, j)
        # print("denormed_box: ", denormed)

        # img2, target2 = self.dataset.__getitem__(1)
        # target = torch.cat([target1.unsqueeze(0), target2.unsqueeze(0)])
        # boxes = target[..., 21:25][-1]
        # print(boxes)
        # boxes, class_ids, scores = self.dataset.get_target_boxes(target)
        # print(">>> ", normed_boxes.shape, class_ids.shape, scores.shape)
        # self.assertEqual(boxes.shape, (5, 4))
        # self.assertEqual(class_ids.shape, (5,))
        # self.assertEqual(scores.shape, (5,))
        # img1.close()
        # img2.close()

    # def test_data_batch(self):
    #     data = iter(self.test_data)
    #     data_batch = next(data)
    #     print(data_batch[0].shape, data_batch[1].shape)
    #     # img1, target1 = self.dataset.__getitem__(0)
    #     # img2, target2 = self.dataset.__getitem__(1)
    #     # target = torch.cat([target1.unsqueeze(0), target2.unsqueeze(0)])
    #     img_batch, target_batch = data_batch
    #     boxes, class_ids, scores = self.dataset.get_target_boxes(target_batch)
    #     for i in range(len(img_batch)):
    #         self.dataset.show_boxes(img_batch[i], boxes[i], class_ids[i], scores[i])

        # print(boxes.shape, boxes.shape, scores.shape, class_ids.shape)
        # VOCDataset.plot_batch(img_batch, boxes, class_ids, scores)

        # np.testing.assert_array_almost_equal(
        #     boxes[0].tolist(), 
        #     [0.6410, 0.5706, 0.7180, 0.8408], decimal=3
        # )
        # np.testing.assert_array_almost_equal(
        #     boxes[1].tolist(), 
        #     [0.3790, 0.5667, 0.1580, 0.3813], decimal=3
        # )
        # np.testing.assert_array_almost_equal(
        #     boxes[2].tolist(), 
        #     [0.3390, 0.6693, 0.4020, 0.4213], decimal=3
        # )
        # np.testing.assert_array_almost_equal(
        #     boxes[3].tolist(), 
        #     [0.5550, 0.7027, 0.0780, 0.3493], decimal=3
        # )
        # np.testing.assert_array_almost_equal(
        #     boxes[4].tolist(), 
        #     [0.6120, 0.7093, 0.0840, 0.3467], decimal=3
        # )
        # print(denormed_boxes)
        # img1.close()
        # img2.close()
        

    def test_norm_denorm(self):
        imgsrc, target = self.dataset.__getitem__(1)
        target = target.unsqueeze(0)
        # print(target.shape)
        # self.
        # boxes, class_ids, scores = self.dataset.get_target_boxes(target)
        # print(boxes, "\n")
        # denormed_boxes = VOCDataset._denorm_batch_boxes_dims(boxes, S, i, j)
        # print(boxes)
        # np.testing.assert_array_almost_equal(
        #     denormed_boxes[0].toata_batch(self):
    #     data = iter(self.test_data)
    #     data_batch = next(data)
    #     print(data_batch[0].shape, data_batch[1].shape)
    #     # img1, target1 = self.dataset.__getitem__(0)
    #     # img2, target2 = self.dataset.__getitem__(1)
    #     # target = torch.cat([target1.unsqueeze(0), target2.unsqueeze(0)])
    #     img_batch, target_batch = data_batch
    #     boxes, class_ids, scores = self.dataset.get_target_boxes(target_batch)
    #     for i in range(len(img_batch)):
    #         self.dataslist(), 
        #     [0.379, 0.567, 0.158, 0.381], decimal=3
        # )
        # np.testing.assert_array_almost_equal(
        #     denormed_boxes[1].tolist(), 
        #     [0.339, 0.669, 0.402, 0.421], decimal=3
        # )
        # np.testing.assert_array_almost_equal(
        #     denormed_boxes[2].tolist(), 
        #     [0.555, 0.703, 0.078, 0.349], decimal=3
        # )
        # np.testing.assert_array_almost_equal(
        #     denormed_boxes[3].tolist(), 
        #     [0.612, 0.709, 0.084, 0.347], decimal=3
        # )
        org_size = (500, 375)
        org_size = None
        # VOCDataset.show_boxes(imgsrc, boxes, class_ids, scores, org_size)
        # imgsrc.close()

    def test_transforms(self):
        dataset_args = dict(S=S, B=B, C=C, 
            img_transformer=img_transformer, pair_transformer=pair_transformer)
        self.dataset = VOCDataset(DATA_DIR+"8examples.csv", IMG_DIR, LABEL_DIR, **dataset_args)
        imgsrc, target = self.dataset.__getitem__(1)
        boxes, class_ids, scores = self.dataset.get_target_boxes(target.unsqueeze(0))
        # print(boxes)
        # src_img = PIL.Image.open(IMG_DIR+"000032.jpg")
        # boxes = torch.tensor([
        #     [0.479, 0.464, 0.542, 0.373],
        #     [0.330, 0.375, 0.128, 0.124],
        #     [0.408, 0.727, 0.036, 0.174],
        #     [0.070, 0.759, 0.036, 0.174]
        # ])
        # hft = HorizontalFlip()
        # img, boxes = hft((src_img, boxes))
        self.dataset.show_boxes(imgsrc, boxes[0], class_ids[0], scores[0], img_size=(448, 448))