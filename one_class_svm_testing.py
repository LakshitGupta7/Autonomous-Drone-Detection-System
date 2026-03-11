'''
# Single SVM
import os
import cv2
import numpy as np
import joblib
from skimage.feature import hog

# Load the trained SVM model
svm_model_path = "svm/one_class_svm_for_drone.pkl"
svm_model = joblib.load(svm_model_path)

# Function to extract HOG features
def extract_hog_features(image_path):
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError(f"⚠ Error: Could not read image {image_path}")
    
    image = cv2.resize(image, (60, 60))  # Resize to match training size
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)  
    return features

# Function to test an image using the trained SVM model
def test_svm(image_path):
    try:
        features = extract_hog_features(image_path).reshape(1, -1)  # Reshape to match SVM input
        prediction = svm_model.predict(features)  # Predict using One-Class SVM
        
        if prediction[0] == 1:
            print(f"✅ {image_path}: Detected as a DRONE ✅")
        else:
            print(f"❌ {image_path}: NOT a Drone ❌")
    
    except Exception as e:
        print(f"⚠ Error processing {image_path}: {e}")

# Example usage
custom_image_path = "D:/ADDS/cropped_objects/images/IMG-20250409-WA0041_crop_0.jpg"  # Change to your image path
test_svm(custom_image_path)
features = extract_hog_features(custom_image_path)
print("📌 Extracted Features in Manual:", features)
print("📌 Data Type in Manual:", features.dtype)
'''

import os
import cv2
import numpy as np
import joblib
from skimage.feature import hog

# === Load the binary SVM models with their HOG parameters ===
svm_models = [
    {"model": joblib.load("svm/svm_model_1.pkl"), "hog_params": {"pixels_per_cell": (8, 8), "cells_per_block": (2, 2)}},
    {"model": joblib.load("svm/svm_model_2.pkl"), "hog_params": {"pixels_per_cell": (8, 8), "cells_per_block": (2, 2)}},
    {"model": joblib.load("svm/svm_model_3.pkl"), "hog_params": {"pixels_per_cell": (8, 8), "cells_per_block": (2, 2)}},
    {"model": joblib.load("svm/svm_model_4.pkl"), "hog_params": {"pixels_per_cell": (8, 8), "cells_per_block": (2, 2)}},
]

# === Load the master one-class SVM model ===
master_model = joblib.load("svm/one_class_svm_for_drone.pkl")

# === Extract HOG features ===
def extract_hog_features(image, hog_params):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = hog(
        gray,
        pixels_per_cell=hog_params["pixels_per_cell"],
        cells_per_block=hog_params["cells_per_block"],
        visualize=False
    )
    return features

# === Main classification function using ensemble ===
def classify_with_ensemble(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Failed to read image: {image_path}")
        return

    image = cv2.resize(image, (60, 60))
    votes = []

    try:
        # Binary SVM votes
        for i, entry in enumerate(svm_models, 1):
            features = extract_hog_features(image, entry["hog_params"]).reshape(1, -1)
            pred = entry["model"].predict(features)[0]
            votes.append(pred)
            print(f"🧠 Model {i} Prediction: {pred}")

        # Master model vote
        master_hog_params = {"pixels_per_cell": (8, 8), "cells_per_block": (2, 2)}
        features = extract_hog_features(image, master_hog_params).reshape(1, -1)
        master_pred = master_model.predict(features)[0]
        votes.append(master_pred)
        print(f"👑 Master Model Prediction: {master_pred}")

        # Final decision
        final_vote = votes.count(1) >= 3
        print(f"🗳️ All Votes: {votes}")
        if final_vote:
            print(f"✅ {image_path}: Detected as a DRONE ✅")
        else:
            print(f"❌ {image_path}: NOT a Drone ❌")

    except Exception as e:
        print(f"⚠️ Error in classification: {e}")

# === Example usage ===
test_image = "D:/ADDS/cropped_objects/images/IMG-20250409-WA0041_crop_0.jpg"
classify_with_ensemble(test_image)
