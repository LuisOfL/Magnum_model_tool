from ultralytics import YOLO

model = YOLO("segmentation.pt")
print("Tarea configurada en el modelo:", model.task)