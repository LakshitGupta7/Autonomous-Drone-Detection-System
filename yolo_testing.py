from ultralytics import YOLO

def run_yolo_detection(image_path, model_path="yolo/trained_yolo_updated.pt", conf=0.6):
    """
    Runs YOLO detection on the given image.

    Parameters:
    - image_path (str): Path to the image.
    - model_path (str): Path to the trained YOLO model.
    - conf (float): Confidence threshold.

    Returns:
    - detections (list): List of detected bounding boxes [(x1, y1, x2, y2, confidence)].
    """
    model = YOLO(model_path)
    results = model(image_path, save=False, conf=conf)

    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # Bounding box coordinates
            confidence = box.conf[0].item()  # Confidence score
            detections.append((x1.item(), y1.item(), x2.item(), y2.item(), confidence))

    return detections



