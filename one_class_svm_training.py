import os
import cv2
import numpy as np
import joblib
from sklearn.svm import OneClassSVM
from skimage.feature import hog
from tqdm import tqdm  # Progress bar 

# Path to folder with cropped drone images (for training SVM)
drone_images_folder = "dataset/svm_dataset/images/train"

# Function to extract HOG features
def extract_features(image):
    image = cv2.resize(image, (60, 60))  # Resize to match SVM training
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)  # ✅ Fix: Removed visualize=True
    return features

# Prepare training data
train_features = []
image_files = os.listdir(drone_images_folder)

print("📌 Extracting features from images...")
for img_name in tqdm(image_files, desc="Processing Images", unit="img"):
    img_path = os.path.join(drone_images_folder, img_name)
    image = cv2.imread(img_path)
    if image is None:
        print(f"⚠ Warning: Could not read {img_path}")
        continue
    features = extract_features(image)
    train_features.append(features)

train_features = np.array(train_features)
print(f"\n✅ Feature extraction complete! {train_features.shape[0]} images processed.")

# Train One-Class SVM
print("\n🚀 Training One-Class SVM model...")
svm_model = OneClassSVM(kernel="rbf", gamma="scale", nu=0.001)
svm_model.fit(train_features)  # ✅ Only train once

# Save the trained model
os.makedirs("svm", exist_ok=True)  # Ensure directory exists
joblib.dump(svm_model, "svm/one_class_svm_for_drone.pkl")

print("\n✅ One-Class SVM training complete and saved as 'svm/one_class_svm_for_drone.pkl'!")
