import cv2
import os
import numpy as np
import joblib
from skimage.feature import hog
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from yolo_testing import run_yolo_detection
from resnet_testing import run_detection_live  # Replaced RetinaNet

# Paths
svm_model_path = "svm/one_class_svm_for_drone.pkl"
RESULTS_DIR = "results/live"
CROPPED_DIR = "cropped_objects/live"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CROPPED_DIR, exist_ok=True)

# ✅ Load trained SVM model
svm_model = joblib.load(svm_model_path)

# ✅ OpenCV Optimization
cv2.setUseOptimized(True)
cv2.setNumThreads(4)

# ✅ HOG Feature Extractor
def extract_hog_features(image):
    image = cv2.resize(image, (60, 60))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
    return features

# ✅ Classifier for Cropped Image
def classify_with_svm(cropped_img):
    try:
        features = extract_hog_features(cropped_img).reshape(1, -1)
        prediction = svm_model.predict(features)
        return prediction[0] == 1
    except Exception as e:
        print(f"⚠ SVM Error: {e}")
        return False

# ✅ Webcam Feed
cap = cv2.VideoCapture(0)  # Use 0 for laptop webcam

FRAME_WIDTH, FRAME_HEIGHT = 640, 480
executor = ThreadPoolExecutor(max_workers=2)

def process_live_feed(frame, signal_callback=None, beep_callback=None, arduino_callback=None):
        
    if frame is None:
        return None, "❌ Empty frame received.", None

    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    # ✅ Get YOLO + Faster R-CNN Detections
    yolo_detections = run_yolo_detection(frame)
    fasterrcnn_detections = run_detection_live(frame)  

    yolo_detections = yolo_detections if yolo_detections else []
    fasterrcnn_detections = fasterrcnn_detections if fasterrcnn_detections else []

    all_detections = yolo_detections + fasterrcnn_detections

    final_detections = []
    log_entries = []
    frame_filename = None

    for x1, y1, x2, y2, conf in all_detections:
        if x2 <= x1 or y2 <= y1:
            print(f"⚠ Invalid box: {x1, y1, x2, y2}")
            continue

        cropped_obj = frame[int(y1):int(y2), int(x1):int(x2)]
        if cropped_obj.size == 0:
            continue

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        crop_path = os.path.join(CROPPED_DIR, f"{timestamp}.jpg")
        cv2.imwrite(crop_path, cropped_obj)

        future = executor.submit(classify_with_svm, cropped_obj)
        is_drone = future.result()

        if is_drone:
            final_detections.append((x1, y1, x2, y2, conf))
            log_entries.append(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 🛰️ Drone detected!")

    for x1, y1, x2, y2, conf in final_detections:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"Drone ({conf:.2f})", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if signal_callback:
        signal_callback(max([conf for *_, conf in final_detections]) if final_detections else -1)

    if final_detections:
        if beep_callback:
            beep_callback()
        if arduino_callback:
            arduino_callback()


    if final_detections:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        frame_filename = os.path.join(RESULTS_DIR, f"{timestamp}.jpg")
        cv2.imwrite(frame_filename, frame)

    log_message = "\n".join(log_entries) if log_entries else "❌ No drones detected."
    return frame, log_message, frame_filename

cap.release()
cv2.destroyAllWindows()
