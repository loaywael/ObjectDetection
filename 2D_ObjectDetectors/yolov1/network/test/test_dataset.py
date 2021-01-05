from torch.utils.data import DataLoader
from network.dataset import VOCDataset
from network.model import Yolov1
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from unittest import TestCase
import numpy as np
import torch
import os


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
        # print("data size: ", len(self.test_data))


    # def test_get_box_shapes(self):
    #     img1, target1 = self.dataset.__getitem__(0)
    #     img2, target2 = self.dataset.__getitem__(1)
    #     target = torch.cat([target1.unsqueeze(0), target2.unsqueeze(0)])
    #     normed_boxes, class_ids, scores, [i, j, b] = VOCDataset.get_target_boxes(target)
    #     # print(">>> ", normed_boxes.shape, class_ids.shape, scores.shape)
    #     self.assertEqual(normed_boxes.shape, (5, 4))
    #     self.assertEqual(class_ids.shape, (5,))
    #     self.assertEqual(scores.shape, (5,))
    #     img1.close()
    #     img2.close()

    # def test_batch_norm_denorm(self):
    #     img1, target1 = self.dataset.__getitem__(0)
    #     img2, target2 = self.dataset.__getitem__(1)
    #     target = torch.cat([target1.unsqueeze(0), target2.unsqueeze(0)])
    #     normed_boxes, class_ids, scores, [i, j, b] = VOCDataset.get_target_boxes(target)
    #     denormed_boxes = VOCDataset._denorm_batch_boxes_dims(normed_boxes, S, i, j)
    #     np.testing.assert_array_almost_equal(
    #         denormed_boxes[0].tolist(), 
    #         [0.6410, 0.5706, 0.7180, 0.8408], decimal=3
    #     )
    #     np.testing.assert_array_almost_equal(
    #         denormed_boxes[1].tolist(), 
    #         [0.3790, 0.5667, 0.1580, 0.3813], decimal=3
    #     )
    #     np.testing.assert_array_almost_equal(
    #         denormed_boxes[2].tolist(), 
    #         [0.3390, 0.6693, 0.4020, 0.4213], decimal=3
    #     )
    #     np.testing.assert_array_almost_equal(
    #         denormed_boxes[3].tolist(), 
    #         [0.5550, 0.7027, 0.0780, 0.3493], decimal=3
    #     )
    #     np.testing.assert_array_almost_equal(
    #         denormed_boxes[4].tolist(), 
    #         [0.6120, 0.7093, 0.0840, 0.3467], decimal=3
    #     )
    #     # print(denormed_boxes)
    #     img1.close()
    #     img2.close()

    def test_norm_denorm(self):
        imgsrc, target = self.dataset.__getitem__(1)
        target = target.unsqueeze(0)
        print(target.shape)

        boxes, class_ids, scores, [i, j] = VOCDataset.get_target_boxes(target)
        # print(boxes, "\n")
        denormed_boxes = VOCDataset._denorm_batch_boxes_dims(boxes, S, i, j)
        # print(boxes)
        # np.testing.assert_array_almost_equal(
        #     denormed_boxes[0].tolist(), 
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
        VOCDataset.show_boxes(imgsrc, denormed_boxes, class_ids, scores, org_size)
        imgsrc.close()



    # def test_dataset(self):
    #     pass
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
    