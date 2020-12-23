import tqdm
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms.functional as FT
from network.dataset import VOCDataset
from network.model import Yolov1, YoloLoss
from network.utils import non_max_suppression
from network.utils import eval_iou, eval_mAP


SEED = 13
torch.manual_seed(SEED)
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
IOU_THRESHOLD = 0.5
CONF_THRESHOLD = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHT_DECAY = 0.0
EPOCHS = 30
NUM_WORKERS = 8
TRANSFORMS = False
LOAD_MODEL = False
LOAD_MODEL_FILE = ""
IMG_DIR = "data/images/"
LABEL_DIR = "data/labels/"


def train(model, optimizer, loss_fn, train_loader):
    loop = tqdm.tqdm(train_loader)
    mean_loss = []
    for batch_idx, (x, y) in enumerate(loop):
        x, y, = x.to(DEVICE, y.to(DEVICE))
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item())
    print(f"Mean loss: {sum(mean_loss)/len(mean_loss)}")


def main():
    yolo_args = dict(
        split_size=7, 
        num_boxes=2, 
        grid_size=7, 
        num_classes=20
    )
    adam_args = dict(
        model = Yolov1(**yolo_args).to(DEVICE)
        params=model.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY
    )
    optimizer = optim.Adam(**adam_args)
    loss_fn = YoloLoss(S=7, B=2, C=20)

    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, opt)
    
    data = "data/100examples.csv"
    dataset_args = dict(S=7, B=2, C=20, transforms=TRANSFORMS)
    dataloader_args = dict(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        drop_last = True,
        shuffle=False
    )

    train_dataset = VOCDataset(IMG_DIR, LABEL_DIR, **dataset_args)
    train_loader = DataLoader(dataloader_args)

    for epoch in range(EPOCHS):
        train(train_loader, model, optimizer, loss_fn)
        mean_avg_precision = eval_mAP(pred_boxes, target_boxes, iou_threshold=IOU_THRESHOLD)
        print(f"Training mAP: {mean_avg_precision}")
        if mean_avg_precision > 0.5:
            checkpoint = {
                "state_dict": model.state_dict(), 
                "optimizer": optimizer.state_dict()
            }
            save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)
            time.sleep(3)


if __name__ == "__main__":
    main()