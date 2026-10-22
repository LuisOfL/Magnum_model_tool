from typing import Dict
import torch
from ultralytics import YOLO
from transformers import pipeline
from config import MODEL_PATH, SEG_MODEL_PATH, CLASSES_PATH

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"No se encontró el modelo de detección: {MODEL_PATH}")

if not SEG_MODEL_PATH.exists():
    raise FileNotFoundError(f"No se encontró el modelo de segmentación: {SEG_MODEL_PATH}")

USE_CUDA = torch.cuda.is_available()
PIPELINE_DEVICE = 0 if USE_CUDA else -1
YOLO_DEVICE = 0 if USE_CUDA else "cpu"

model = YOLO(str(MODEL_PATH))
seg_model = YOLO(str(SEG_MODEL_PATH))

depth_estimator = pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf",
    device=PIPELINE_DEVICE,
)

CLASS_NAMES: Dict[int, str] = {}
if CLASSES_PATH.exists():
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        CLASS_NAMES = {i: line.strip() for i, line in enumerate(f) if line.strip()}