import os
import yaml
import torch
from ultralytics import YOLO

def iniciar_entrenamiento():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    classes_path = os.path.join(BASE_DIR, "classes.txt")
    if os.path.exists(classes_path):
        with open(classes_path, "r", encoding="utf-8") as f:
            classes = [line.strip() for line in f if line.strip()]
    else:
        classes = ["rejilla"]

    names_dict = {i: name for i, name in enumerate(classes)}

    yaml_content = {
        "path": BASE_DIR,
        "train": "images",
        "val": "images",
        "names": names_dict
    }

    yaml_path = os.path.join(BASE_DIR, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_content, f, sort_keys=False)

    device_to_use = 0 if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Entrenando en: {device_to_use}")

    model = YOLO("yolov8n-seg.pt")

    model.train(
        data=yaml_path,
        epochs=150,
        patience=30,
        imgsz=1024,          # 👈 AUMENTAR RESOLUCIÓN: Permite que las varillas finas tengan grosor en píxeles
        batch=2,
        device=device_to_use,
        overlap_mask=False,  # 👈 Evita que las máscaras de varillas cercanas se deformen o fusionen

        # --- AUGMENTATION ---
        degrees=5.0,
        translate=0.02,
        scale=0.1,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,

        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,

        project="entrenamiento_rejilla",
        name="yolo_segmentation_rejilla_finas"
    )

    print("\n🚀 ¡Entrenamiento finalizado!")

if __name__ == "__main__":
    iniciar_entrenamiento()