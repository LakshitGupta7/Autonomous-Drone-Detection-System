
# Resnet50 + Faster RCNN

# pipeline_image.py

import cv2
import os
import numpy as np
import joblib
from skimage.feature import hog
from yolo_testing import run_yolo_detection
from resnet_testing import run_detection_image  # ✅ Updated
# Arduino

# ARduino

# ✅ Paths
output_dir = "results/images"
cropped_dir = "cropped_objects/images"
svm_model_path = "svm/one_class_svm_for_drone.pkl"

# ✅ Create directories
os.makedirs(output_dir, exist_ok=True)
os.makedirs(cropped_dir, exist_ok=True)

# ✅ Load trained SVM model
svm_model = joblib.load(svm_model_path)

def extract_hog_features(image):
    image = cv2.resize(image, (60, 60))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)  
    return features

def test_svm(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None or image.size == 0:
            print(f"❌ Error: Could not read cropped image {image_path}")
            return False
        features = extract_hog_features(image).reshape(1, -1)  
        prediction = svm_model.predict(features)
        return prediction[0] == 1
    except Exception as e:
        print(f"⚠ SVM Error: {e}")
        return False

def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1p, y1p, x2p, y2p = box2
    xi1, yi1 = max(x1, x1p), max(y1, y1p)
    xi2, yi2 = min(x2, x2p), min(y2, y2p)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union_area = (x2 - x1) * (y2 - y1) + (x2p - x1p) * (y2p - y1p) - inter_area
    return inter_area / union_area if union_area != 0 else 0

def process_image(image_path):
    image_name = os.path.basename(image_path).split('.')[0]
    image = cv2.imread(image_path)
    if image is None:
        return None, "⚠ Error: Could not read image."

    # ✅ Run YOLO & Faster R-CNN (Replaces RetinaNet)
    yolo_detections = run_yolo_detection(image_path)
    fpn_detections = run_detection_image(image_path)  # 🔁 Changed from retinanet
    print("🔹 YOLO Detections: ", yolo_detections)
    print("🔹 Faster R-CNN Detections: ", fpn_detections)

    detections = yolo_detections + fpn_detections
    filtered_detections = []

    for i, (x1, y1, x2, y2, confidence) in enumerate(detections):
        cropped = image[int(y1):int(y2), int(x1):int(x2)]
        if cropped.size == 0:
            continue
        cropped_image_path = os.path.join(cropped_dir, f"{image_name}_crop_{i}.jpg")
        cv2.imwrite(cropped_image_path, cropped)
        print(f"✅ Saved Cropped Image: {cropped_image_path}")
        is_drone = test_svm(cropped_image_path)
        print(f"📢 SVM Result for {cropped_image_path}: {is_drone}")
        if is_drone:
            filtered_detections.append((x1, y1, x2, y2, confidence))

    # ✅ Merge overlapping boxes (IoU > 0.5)
    final_detections = []
    used_indices = set()
    for i, det1 in enumerate(filtered_detections):
        if i in used_indices:
            continue
        x1, y1, x2, y2, conf1 = det1
        largest_box = det1
        for j, det2 in enumerate(filtered_detections):
            if i == j or j in used_indices:
                continue
            if calculate_iou((x1, y1, x2, y2), (det2[0], det2[1], det2[2], det2[3])) > 0.5:
                largest_box = max(largest_box, det2, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))
                used_indices.add(j)
        final_detections.append(largest_box)

    confidence_scores = []
    for (x1, y1, x2, y2, confidence) in final_detections:
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(image, f"Drone ({confidence:.2f})", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        confidence_scores.append(confidence)

    final_image_path = os.path.join(output_dir, f"{image_name}_final.jpg")
    cv2.imwrite(final_image_path, image)

    if len(final_detections) == 0:
        return final_image_path, "✅ Detection Completed!\n❌ No Drone Detected", confidence_scores

    print("🔹 Final Detections: ", final_detections)
    return final_image_path, "✅ Detection Completed!\n✅ Drone Detected!", confidence_scores



