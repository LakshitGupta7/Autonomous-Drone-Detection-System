# if __name__ == '__main__':

#     from ultralytics import YOLO

#     # Load YOLO model
#     model = YOLO("yolov8s.pt")  # Using pre-trained YOLOv8 small model             

                 

#     # Train the YOLO model
#     model.train(data="dataset/yolo_dataset/data.yaml", epochs=30, batch=4, imgsz=640, workers=2, device="cuda")

#     # Save trained model
#     model.export(format="onnx")  # Save as ONNX
#     model.save("trained_yolo.pt")  # Save in PyTorch format

from ultralytics import YOLO

if __name__ == '__main__':
    # # Load previously trained model
    # model = YOLO("trained_yolo.pt")  # Your fine-tuned checkpoint on 25k images

    # # Train further on new 5k data
    # model.train(
    #     data="dataset/yolo_new/data_new.yaml",
    #     epochs=20,
    #     batch=2,  # Lowered to reduce memory load
    #     imgsz=640,
    #     workers=2,
    #     device="cuda",  # Use "cpu" if CUDA still fails
    #     name="yolo_finetune_new5k3",
    #     amp=False  # Disable mixed precision
    # )

    # Optional: Export or save if needed
    model = YOLO("E:/ADDS/yolov5/runs/detect/yolo_finetune_n ew5k3/weights/best.pt")
    model.export(format="onnx")
    model.save("trained_yolo_updated.pt")
