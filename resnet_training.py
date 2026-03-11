'''
# Resnet50 + FPN Faster RCNN

import os
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm
import time

# === Dataset Loader
class CocoDroneDataset(CocoDetection):
    def __init__(self, img_folder, ann_file, transforms=None):
        super().__init__(img_folder, ann_file)
        self._transforms = transforms

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        boxes = []
        labels = []

        for obj in target:
            bbox = obj['bbox']
            x_min, y_min = bbox[0], bbox[1]
            x_max = x_min + bbox[2]
            y_max = y_min + bbox[3]
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(obj['category_id'])

        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor([idx])
        }

        if self._transforms:
            img = self._transforms(img)

        return img, target

# === Paths
DATASET_PATH = "resnet50_dataset"
TRAIN_IMAGES = os.path.join(DATASET_PATH, "train/images")
TRAIN_JSON = os.path.join(DATASET_PATH, "train/annotations/cleaned_annotations.json")
VALID_IMAGES = os.path.join(DATASET_PATH, "valid/images")
VALID_JSON = os.path.join(DATASET_PATH, "valid/annotations/annotations.json")
OUTPUT_DIR = "./fasterrcnn_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "checkpoint.pth")
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")

# === Transforms and Loaders
transform = T.Compose([T.ToTensor()])
train_dataset = CocoDroneDataset(TRAIN_IMAGES, TRAIN_JSON, transforms=transform)
val_dataset = CocoDroneDataset(VALID_IMAGES, VALID_JSON, transforms=transform)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# === Model Setup
model = fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.00025, momentum=0.9)

# === Resume Logic
start_epoch = 0
best_val_loss = float('inf')
patience = 3
no_improve_epochs = 0

if os.path.exists(CHECKPOINT_PATH):
    print("Resuming from checkpoint...")
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint['best_val_loss']
    no_improve_epochs = checkpoint['no_improve_epochs']

EPOCHS = 15
for epoch in range(start_epoch, EPOCHS):
    print(f"\n📘 Epoch {epoch + 1}/{EPOCHS} -------------------------------")
    start_time = time.time()

    # === Training Phase
    model.train()
    train_loss = 0.0
    for images, targets in tqdm(train_loader, desc=f"🔁 Training", leave=False):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        train_loss += losses.item()

    train_loss /= len(train_loader)
    print(f"✅ [Epoch {epoch+1}] Training Loss: {train_loss:.4f}")

    # === Validation Phase
    model.train()  # <-- Force model to return losses
    val_loss = 0.0
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="🔍 Validating", leave=False):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)  # Will return dict
            losses = sum(loss for loss in loss_dict.values())
            val_loss += losses.item()
    val_loss /= len(val_loader)
    print(f"📉 [Epoch {epoch+1}] Validation Loss: {val_loss:.4f}")


    # === Save Checkpoint
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
        'no_improve_epochs': no_improve_epochs
    }, CHECKPOINT_PATH)

    # === Early Stopping Logic
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve_epochs = 0
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"✅ Improvement: Saved best model at epoch {epoch+1}")
    else:
        no_improve_epochs += 1
        print(f"⚠️ No improvement for {no_improve_epochs} epoch(s).")

    # === Elapsed Time
    elapsed = time.time() - start_time
    print(f"⏱️ Time taken for epoch {epoch+1}: {elapsed:.2f} seconds")

    if no_improve_epochs >= patience:
        print("🛑 Early stopping triggered due to no improvement.")
        break
'''

# Resnet50 + FPN master script

import os
import time
import torch
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast

# === Dataset Loader
class CocoDroneDataset(CocoDetection):
    def __init__(self, img_folder, ann_file, transforms=None):
        super().__init__(img_folder, ann_file)
        self._transforms = transforms

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        boxes = []
        labels = []

        for obj in target:
            bbox = obj['bbox']
            x_min, y_min = bbox[0], bbox[1]
            x_max = x_min + bbox[2]
            y_max = y_min + bbox[3]
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(obj['category_id'])

        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor([idx])
        }

        if self._transforms:
            img = self._transforms(img)

        return img, target

# === Paths
DATASET_PATH = "resnet50_dataset"
TRAIN_IMAGES = os.path.join(DATASET_PATH, "train/images")
TRAIN_JSON = os.path.join(DATASET_PATH, "train/annotations/cleaned_annotations2.json")
VALID_IMAGES = os.path.join(DATASET_PATH, "valid/images")
VALID_JSON = os.path.join(DATASET_PATH, "valid/annotations/cleaned_annotations.json")
OUTPUT_DIR = "./fasterrcnn_output2"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "checkpoint.pth")
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")

# === Transforms and Loaders
train_transform = T.Compose([
    T.ToTensor(),
    T.RandomHorizontalFlip(0.5),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
])

val_transform = T.Compose([
    T.ToTensor()
])

train_dataset = CocoDroneDataset(TRAIN_IMAGES, TRAIN_JSON, transforms=train_transform)
val_dataset = CocoDroneDataset(VALID_IMAGES, VALID_JSON, transforms=val_transform)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# === Model Setup
model = fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 2  # drone vs background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.00025, momentum=0.9, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
scaler = GradScaler()

# === Resume Logic
start_epoch = 0
best_val_loss = float('inf')
patience = 3
no_improve_epochs = 0

if os.path.exists(CHECKPOINT_PATH):
    print("Resuming from checkpoint...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint['best_val_loss']
    no_improve_epochs = checkpoint['no_improve_epochs']

# === Loss Tracking for Plotting
train_loss_history = []
val_loss_history = []

# === Training Loop
EPOCHS = 15

for epoch in range(start_epoch, EPOCHS):
    print(f"\n📘 Epoch {epoch + 1}/{EPOCHS} -------------------------------")
    start_time = time.time()

    # === Training Phase
    model.train()
    train_loss = 0.0

    for images, targets in tqdm(train_loader, desc="🔁 Training", leave=False):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with autocast():
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += losses.item()

    train_loss /= len(train_loader)
    train_loss_history.append(train_loss)
    print(f"✅ [Epoch {epoch+1}] Training Loss: {train_loss:.4f}")

    # === Validation Phase
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="🔍 Validating", leave=False):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            val_loss += losses.item()

    val_loss /= len(val_loader)
    val_loss_history.append(val_loss)
    print(f"📉 [Epoch {epoch+1}] Validation Loss: {val_loss:.4f}")

    # === Scheduler Step
    scheduler.step(val_loss)

    # === Save Checkpoint
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
        'no_improve_epochs': no_improve_epochs
    }, CHECKPOINT_PATH)

    # === Early Stopping Logic
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve_epochs = 0
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"✅ Improvement: Saved best model at epoch {epoch+1}")
    else:
        no_improve_epochs += 1
        print(f"⚠️ No improvement for {no_improve_epochs} epoch(s).")

    elapsed = time.time() - start_time
    print(f"⏱️ Time taken for epoch {epoch+1}: {elapsed:.2f} seconds")

    if no_improve_epochs >= patience:
        print("🛑 Early stopping triggered due to no improvement.")
        break

# === Plotting Loss Curves after Training
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_loss_history)+1), train_loss_history, label='Training Loss', marker='o')
plt.plot(range(1, len(val_loss_history)+1), val_loss_history, label='Validation Loss', marker='x')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Curve')
plt.legend()
plt.grid()
plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"))
plt.show()


