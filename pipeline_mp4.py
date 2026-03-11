
# ORIGINAL!!!!!!

import os
import cv2
import numpy as np
import joblib
from skimage.feature import hog
from yolo_testing import run_yolo_detection
from resnet_testing import run_detection_frame  
from PyQt5.QtCore import pyqtSignal
import time

# ✅ Paths
output_dir = "results/videos"
cropped_dir = "cropped_objects/videos"
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

def test_svm(image):
    try:
        features = extract_hog_features(image).reshape(1, -1)
        prediction = svm_model.predict(features)
        return prediction[0] == 1
    except Exception as e:
        print(f"⚠ Error processing cropped image: {e}")
        return False

def iou(box1, box2):
    x1, y1, x2, y2, _ = box1
    x1b, y1b, x2b, y2b, _ = box2
    xi1 = max(x1, x1b)
    yi1 = max(y1, y1b)
    xi2 = min(x2, x2b)
    yi2 = min(y2, y2b)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area_box1 = (x2 - x1) * (y2 - y1)
    area_box2 = (x2b - x1b) * (y2b - y1b)
    union = area_box1 + area_box2 - inter_area
    return inter_area / union if union > 0 else 0

def non_max_suppression(detections, iou_threshold=0.5):
    detections = sorted(detections, key=lambda x: x[4], reverse=True)
    final_detections = []

    while detections:
        best_box = detections.pop(0)
        detections = [box for box in detections if iou(best_box, box) < iou_threshold]
        final_detections.append(best_box)

    return final_detections

def process_video(video_path, log_callback=None, stop_flag_check=None, signal_callback=None, beep_callback=None):
    video_capture = cv2.VideoCapture(video_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if log_callback:
        print(f"🎬 Video has {total_frames} frames at {fps} FPS.")

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_video_path = os.path.join(output_dir, f"{video_name}_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    cropped_frame_dir = os.path.join(cropped_dir, f"{video_name}_frames")
    os.makedirs(cropped_frame_dir, exist_ok=True)

    frame_count = 1
    sample_frame = None

    while True:
        if stop_flag_check and stop_flag_check():
            if log_callback:
                time.sleep(1)
                log_callback("🛑 Video processing manually stopped..")
            break

        ret, frame = video_capture.read()
        if not ret:
            break

        if log_callback:
            print(f"\n📍 Processing frame {frame_count}/{total_frames}...")

        yolo_detections = run_yolo_detection(frame)
        fpn_detections = run_detection_frame(frame)  # ✅ Faster R-CNN inference

        print(f"🔵 YOLO detected {len(yolo_detections)} objects.")
        print(f"🟠 Faster R-CNN detected {len(fpn_detections)} objects.")

        all_detections = yolo_detections + fpn_detections
        svm_detections = []

        for idx, detection in enumerate(all_detections):
            x1, y1, x2, y2, confidence = detection
            cropped = frame[int(y1):int(y2), int(x1):int(x2)]
            if cropped.size == 0:
                continue

            cropped_image_path = f"{cropped_frame_dir}/{video_name}_frame{frame_count}_crop{idx}.jpg"
            cv2.imwrite(cropped_image_path, cropped)
            print(f"📸 Cropped image saved: {cropped_image_path}")

            if test_svm(cropped):
                svm_detections.append(detection)

        print(f"✅ SVM confirmed {len(svm_detections)} drone(s).")

        final_detections = non_max_suppression(svm_detections)
        print(f"✅ {len(final_detections)} final bounding box(es) after removing overlaps.")

        if log_callback:
            status = "✅Drone Detected" if final_detections else "❌No Drone Detected"
            log_callback(f"Frame {frame_count} -> {status}")

        if signal_callback:
            signal_callback(max([conf for *_, conf in final_detections]) if final_detections else -1)

        if final_detections and beep_callback:
            beep_callback()

        for (x1, y1, x2, y2, confidence) in final_detections:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"Drone ({confidence:.2f})", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if sample_frame is None:
            sample_frame = frame.copy()

        out.write(frame)
        frame_count += 1

    video_capture.release()
    out.release()
    print("\n🎬 Video processing complete!")
    print(f"🎥 Video saved: {output_video_path}")
    return output_video_path


# ORIGINAL!!!!


'''
import os
import cv2
import numpy as np
import joblib
from skimage.feature import hog
from yolo_testing import run_yolo_detection
from retinanet_testing import run_retinanet_detection_frame
from PyQt5.QtCore import pyqtSignal
import time

# ✅ Paths
output_dir = "results/videos"
cropped_dir = "cropped_objects/videos"
svm_model_path = "svm/one_class_svm_for_drone.pkl"

# ✅ Create directories
os.makedirs(output_dir, exist_ok=True)
os.makedirs(cropped_dir, exist_ok=True)

# ✅ Load trained SVM model
svm_model = joblib.load(svm_model_path)

# ✅ Function to extract HOG features
def extract_hog_features(image):
    image = cv2.resize(image, (60, 60))  # Resize to match training size
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
    return features

# ✅ Function to test SVM on a cropped image
def test_svm(image):
    try:
        features = extract_hog_features(image).reshape(1, -1)
        prediction = svm_model.predict(features)
        if prediction[0] == 1:
            print("✅ SVM: Drone detected!")
            return True
        else:
            print("❌ SVM: Not a drone, rejected.")
            return False
    except Exception as e:
        print(f"⚠ Error processing cropped image: {e}")
        return False

# ✅ Function to calculate IoU
def iou(box1, box2):
    x1, y1, x2, y2, _ = box1
    x1b, y1b, x2b, y2b, _ = box2
    xi1 = max(x1, x1b)
    yi1 = max(y1, y1b)
    xi2 = min(x2, x2b)
    yi2 = min(y2, y2b)
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    intersection = inter_width * inter_height
    area_box1 = (x2 - x1) * (y2 - y1)
    area_box2 = (x2b - x1b) * (y2b - y1b)
    union = area_box1 + area_box2 - intersection
    return intersection / union if union > 0 else 0

# ✅ Function to remove overlapping boxes
def non_max_suppression(detections, iou_threshold=0.5):
    detections = sorted(detections, key=lambda x: x[4], reverse=True)
    final_detections = []
    while detections:
        best_box = detections.pop(0)
        detections = [box for box in detections if iou(best_box, box) < iou_threshold]
        final_detections.append(best_box)
    return final_detections

# ✅ Main processing function
def process_video(video_path, log_callback=None, stop_flag_check=None, signal_callback=None, beep_callback=None, speed_callback=None):
    video_capture = cv2.VideoCapture(video_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if log_callback:
        print(f"🎬 Video has {total_frames} frames at {fps} FPS.")

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_video_path = os.path.join(output_dir, f"{video_name}_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    cropped_frame_dir = os.path.join(cropped_dir, f"{video_name}_frames")
    os.makedirs(cropped_frame_dir, exist_ok=True)

    previous_centers = {}
    frame_count = 1
    sample_frame = None

    while True:
        if stop_flag_check and stop_flag_check():
            if log_callback:
                time.sleep(1)
                log_callback("🛑 Video processing manually stopped..")
            break

        ret, frame = video_capture.read()
        if not ret:
            break

        if log_callback:
            print(f"\n📍 Processing frame {frame_count}/{total_frames}...")

        yolo_detections = run_yolo_detection(frame)
        retinanet_detections = run_retinanet_detection_frame(frame)

        print(f"🔵 YOLO detected {len(yolo_detections)} objects.")
        print(f"🟠 RetinaNet detected {len(retinanet_detections)} objects.")

        all_detections = yolo_detections + retinanet_detections
        svm_detections = []

        for idx, detection in enumerate(all_detections):
            x1, y1, x2, y2, confidence = detection
            cropped = frame[int(y1):int(y2), int(x1):int(x2)]
            if cropped.size == 0:
                print("⚠️ Skipping empty crop.")
                continue

            cropped_image_path = f"{cropped_frame_dir}/{video_name}_frame{frame_count}_crop{idx}.jpg"
            cv2.imwrite(cropped_image_path, cropped)
            print(f"📸 Cropped image saved: {cropped_image_path}")

            is_drone = test_svm(cropped)
            if is_drone:
                svm_detections.append(detection)

        print(f"✅ SVM confirmed {len(svm_detections)} drone(s).")

        final_detections = non_max_suppression(svm_detections)
        print(f"✅ {len(final_detections)} final bounding box(es) after removing overlaps.")

        if log_callback:
            if final_detections:
                log_callback(f"Frame {frame_count} -> ✅Drone Detected")
            else:
                log_callback(f"Frame {frame_count} -> ❌No Drone Detected")

        if signal_callback:
            if final_detections:
                max_conf = max([conf for (_, _, _, _, conf) in final_detections])
                signal_callback(max_conf)
            else:
                signal_callback(-1)

        if final_detections and beep_callback:
            beep_callback()

        speed_data = {}

        for idx, (x1, y1, x2, y2, confidence) in enumerate(final_detections):
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            if idx in previous_centers:
                prev_x, prev_y, prev_frame = previous_centers[idx]
                frame_diff = frame_count - prev_frame
                if frame_diff > 0:
                    pixel_distance = np.sqrt((center_x - prev_x) ** 2 + (center_y - prev_y) ** 2)
                    speed_pixels_per_frame = pixel_distance / frame_diff
                    speed_pixels_per_sec = speed_pixels_per_frame * fps

                    print(f"🚀 Speed of drone {idx}: {speed_pixels_per_sec:.2f} pixels/sec")
                    speed_data[idx] = speed_pixels_per_sec

                    cv2.putText(frame, f"Speed: {speed_pixels_per_sec:.2f}px/s",
                                (int(x1), int(y2) + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            previous_centers[idx] = (center_x, center_y, frame_count)

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"Drone ({confidence:.2f})", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if speed_callback:
            speed_callback(speed_data)

        if sample_frame is None:
            sample_frame = frame.copy()

        out.write(frame)
        frame_count += 1

    video_capture.release()
    out.release()
    print("\n🎬 Video processing complete!")
    print(f"🎥 Video saved: {output_video_path}")
    print("🔹 Final Detections: ", final_detections)
    return output_video_path
'''
