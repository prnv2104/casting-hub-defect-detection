from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(data="casting-dataset-kagglev1iyolo26.ndjson", epochs=30, imgsz=640, device="cpu")