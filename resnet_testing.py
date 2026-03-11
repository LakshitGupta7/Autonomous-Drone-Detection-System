# Resnet50 + Faster RCCN

import torch
import torchvision
import cv2
import os
import numpy as np
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Load the trained Faster R-CNN model
def load_model(checkpoint_path="fasterrcnn_output/best_model.pth", num_classes=2):
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model

# Initialize model once
model = load_model()

# Detection for a single image file
def run_detection_image(image_path, score_thresh=0.60):
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ ERROR: Cannot read {image_path}")
        return []
    return run_detection_frame(image, score_thresh)

# Detection for video or live frame
def run_detection_frame(frame, score_thresh=0.60):
    image_tensor = F.to_tensor(frame).to("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        outputs = model([image_tensor])[0]

    detections = []
    for box, score in zip(outputs["boxes"], outputs["scores"]):
        if score >= score_thresh:
            x1, y1, x2, y2 = box.tolist()
            detections.append((x1, y1, x2, y2, score.item()))
    return detections

# Same function reused for real-time webcam
def run_detection_live(frame):
    return run_detection_frame(frame)