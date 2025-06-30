import os
import json
import time
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import timm
import numpy as np
import csv
import wandb

BASE_DIR = "/home/cc/lvis"
IMG_SIZE = 224
PATCH_SIZE = 16
NUM_CLASSES = 1204
BATCH_SIZE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

log_file = os.path.join(BASE_DIR, "training_log.csv")
if not os.path.exists(log_file):
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "time", "train/total_loss", "train/cls_loss", "train/box_loss", "lr"])


transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ColorJitter(0.1, 0.1, 0.1),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3)
])

class ManualLVISDataset(Dataset):
    def __init__(self, image_root, ann_file, transforms=None):
        self.image_root = image_root
        self.transforms = transforms
        with open(ann_file, 'r') as f:
            data = json.load(f)
        #self.images = data['images']
        self.images = data['images'][:2000]
        allowed_image_ids = set(img['id'] for img in self.images)
        self.annotations = [ann for ann in data['annotations'] if ann['image_id'] in allowed_image_ids]
        #self.annotations = data['annotations']
        self.image_to_anns = {}
        for ann in self.annotations:
            self.image_to_anns.setdefault(ann['image_id'], []).append(ann)
        self.id_to_image = {img['id']: img for img in self.images}

    def __getitem__(self, index):
        img_info = self.images[index]
        img_id = img_info['id']
        file_name = img_info.get("file_name", f"{img_id:012d}.jpg")
        img_path = os.path.join(self.image_root, file_name)
        image = Image.open(img_path).convert("RGB")
        if self.transforms:
            image = self.transforms(image)
        anns = self.image_to_anns.get(img_id, [])
        return image, anns

    def __len__(self):
        return len(self.images)

def get_dataset(split, transform):
    img_dir = os.path.join(BASE_DIR, "images", f"{split}2017")
    ann_path = os.path.join(BASE_DIR, "annotations", f"lvis_v1_{split}.json")
    return ManualLVISDataset(img_dir, ann_path, transform)

train_dataset = get_dataset("train", transform)
val_dataset = get_dataset("val", transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=lambda x: tuple(zip(*x)), num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=lambda x: tuple(zip(*x)), num_workers=4)

class ViTDetectionModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        self.hidden_dim = self.vit.embed_dim
        self.cls_head = nn.Linear(self.hidden_dim, num_classes)
        self.reg_head = nn.Linear(self.hidden_dim, 4)

    def forward(self, x):
        B = x.shape[0]
        feats = self.vit.forward_features(x)[:, 1:]  # [B, P, D]
        cls_out = self.cls_head(feats)  # [B, P, C]
        reg_out = self.reg_head(feats)  # [B, P, 4]
        out = torch.cat([reg_out, cls_out], dim=-1)  # [B, P, 4 + C]
        return out

model = torch.compile(ViTDetectionModel(NUM_CLASSES).to(DEVICE))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total trainable parameters: {count_parameters(model):,}")

def get_patch_center_coords(img_size=224, patch_size=16):
    coords = []
    for y in range(0, img_size, patch_size):
        for x in range(0, img_size, patch_size):
            cx = x + patch_size // 2
            cy = y + patch_size // 2
            coords.append((cx, cy))
    return coords  

PATCH_CENTERS = get_patch_center_coords(IMG_SIZE, PATCH_SIZE)

def assign_gt_to_patches(boxes):
    """
    For each ground truth box, assign it to all patches whose centers fall inside the box.
    """
    bbox_targets = [None] * len(PATCH_CENTERS)
    for box in boxes:
        x1, y1, x2, y2 = box.tolist()
        for idx, (px, py) in enumerate(PATCH_CENTERS):
            if x1 <= px <= x2 and y1 <= py <= y2:
                bbox_targets[idx] = box
    return bbox_targets

def detection_loss(preds, targets):
    cls_loss_fn = nn.CrossEntropyLoss()
    reg_loss_fn = nn.SmoothL1Loss()
    total_cls_loss, total_reg_loss = 0.0, 0.0

    for i in range(len(preds)):
        pred = preds[i]  # shape: [P, 4 + C]
        gt = targets[i]
        boxes = gt['boxes'].to(DEVICE)
        labels = gt['labels'].to(DEVICE)

        if boxes.numel() == 0 or labels.numel() == 0:
            continue  

        assignments = assign_gt_to_patches(boxes)

        for patch_idx, box in enumerate(assignments):
            if box is None:
                continue 

            box_tensor = box.clone().detach().to(dtype=torch.float32, device=DEVICE)
            pred_box = pred[patch_idx, :4]
            pred_cls = pred[patch_idx, 4:]
            label = labels[0].clone().detach().to(dtype=torch.long, device=DEVICE)  # pick first label for demo

            if box_tensor.shape != pred_box.shape:
                continue

            total_reg_loss += reg_loss_fn(pred_box, box_tensor)
            total_cls_loss += cls_loss_fn(pred_cls.unsqueeze(0), label.unsqueeze(0))

    total_loss = total_cls_loss + total_reg_loss
    return total_loss, total_cls_loss.item(), total_reg_loss.item()


def preprocess_targets(targets):
    results = []
    for ann_list in targets:
        boxes, labels = [], []
        for ann in ann_list:
            x, y, w, h = ann["bbox"]
            boxes.append(torch.tensor([x, y, x + w, y + h]))
            labels.append(ann["category_id"])
        results.append({
            "boxes": torch.stack(boxes) if boxes else torch.zeros((0, 4)),
            "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
        })
    return results

def train_one_epoch(model, dataloader, optimizer, epoch):
    model.train()
    total_loss, total_cls_loss, total_reg_loss = 0.0, 0.0, 0.0
    start_time = time.time()
    for images, targets in dataloader:
        images = torch.stack(images).to(DEVICE)
        targets = preprocess_targets(targets)
        preds = model(images)
        loss, cls_loss, reg_loss = detection_loss(preds, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_cls_loss += cls_loss
        total_reg_loss += reg_loss
    n = len(dataloader)
    elapsed = time.time() - start_time

    row = [epoch, elapsed, total_loss/n, total_cls_loss/n, total_reg_loss/n, optimizer.param_groups[0]['lr']]
    file_exists = os.path.isfile('train_log.csv')
    with open('train_log.csv', mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['epoch', 'time', 'total_loss', 'cls_loss', 'box_loss', 'lr'])
        writer.writerow(row)
    print(f"Epoch {epoch} logged.")
    return total_loss / n

#@torch.no_grad()
#def evaluate(model, dataloader):
#    model.eval()
#    for images, targets in dataloader:
#        images = torch.stack(images).to(DEVICE)
#        preds = model(images)
#        print("Validation output shape:", preds.shape)
#        break  # just run one batch for quick check

from torchmetrics.detection.mean_ap import MeanAveragePrecision

@torch.no_grad()
def evaluate(model, dataloader):
    model.eval()
    metric = MeanAveragePrecision()

    for images, targets in dataloader:
        images = torch.stack(images).to(DEVICE)
        preds = model(images)

        preds_list = []
        for pred in preds:
            boxes = pred[:, :4]
            scores, labels = pred[:, 4:].max(dim=1)
            preds_list.append({
                "boxes": boxes.cpu(),
                "scores": scores.cpu(),
                "labels": labels.cpu()
            })
            
        targets_list = []
        for tgt in preprocess_targets(targets):
            targets_list.append({
                "boxes": tgt['boxes'],
                "labels": tgt['labels']
            })

        metric.update(preds_list, targets_list)

    res = metric.compute()
    print("mAP:", res["map"].item())
    print("mAP@50:", res["map_50"].item())
    print("Mean Recall@100:", res["mar_100"].item())
    print("mAP@75:", res["map_75"].item())
    print("mAP small/medium/large:", res["map_small"].item(), res["map_medium"].item(), res["map_large"].item())

with open(os.path.join(BASE_DIR, "annotations", "lvis_v1_train.json"), "r") as f:
    lvis_data = json.load(f)
id_to_class = {cat["id"]: cat["name"] for cat in lvis_data["categories"]}

checkpoint_path = os.path.join(BASE_DIR, "vit_lvis_model.pth")
if os.path.isfile(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    print(f"Loaded checkpoint from: {checkpoint_path}")
else:
    print("No checkpoint found, starting from scratch.")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

EPOCHS = 10

for epoch in range(1, EPOCHS + 1):
    avg_loss = train_one_epoch(model, train_loader, optimizer, epoch)
    print(f"[Epoch {epoch}] Average Train Loss: {avg_loss:.4f}")
    evaluate(model, val_loader)

SAVE_MODEL = True
if SAVE_MODEL:
    save_path = os.path.join(BASE_DIR, "vit_lvis_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
