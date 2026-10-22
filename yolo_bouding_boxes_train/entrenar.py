import os
from ultralytics import YOLO

def main():
    # ---------------------------------------------------------
    # 1. Configuración de rutas
    # ---------------------------------------------------------
    # Asegúrate de que este archivo .yaml apunte correctamente a tus 
    # carpetas train/val y tenga definidas tus 26 clases de helados.
    DATASET_YAML = "dataset.yaml"
    
    if not os.path.exists(DATASET_YAML):
        print(f"[ADVERTENCIA] No se encontró el archivo yaml en: {DATASET_YAML}")
        print("Por favor, actualiza la ruta 'DATASET_YAML' antes de ejecutar.")
        return

    print("--- Iniciando entrenamiento YOLOv8n-OBB optimizado ---")

    # ---------------------------------------------------------
    # 2. Cargar el modelo base preentrenado OBB (Nano)
    # ---------------------------------------------------------
    model = YOLO("yolov8n-obb.pt")

    # ---------------------------------------------------------
    # 3. Entrenamiento con hiperparámetros defensivos
    # ---------------------------------------------------------
    results = model.train(
        data=DATASET_YAML,
        epochs=150,                  # Más épocas para permitir que la rama del ángulo (theta) converja
        imgsz=640,                   # Resolución estándar (subir a 1024 si los empaques tienen texto muy chico)
        batch=8,                     # Ajustado para tarjetas con 6GB VRAM (como RTX 4050); subir a 16 si la VRAM lo permite
        device=0,                    # ID de la GPU (CUDA)
        
        # Optimizador y convergencia
        optimizer="AdamW",           # AdamW maneja mejor el decaimiento de pesos en datasets pequeños
        lr0=0.001,                   # Tasa de aprendizaje inicial conservadora
        lrf=0.01,                    # Factor final (lr0 * lrf)
        weight_decay=0.0005,
        
        # Aumentos de datos para compensar pocas imágenes por clase
        mosaic=1.0,                  # Mezcla 4 imágenes para dar contexto de múltiples productos
        mixup=0.15,                  # Transición suave entre clases para evitar sobreajuste
        copy_paste=0.1,              # Copia objetos sintéticamente para apoyar clases minoritarias
        
        # Geometría OBB (Rotación completa para productos en congeladores)
        degrees=180.0,               # Permite rotación en cualquier ángulo de inclinación
        shear=5.0,                   # Ligera distorsión geométrica
        scale=0.5,                   # Variación de escala (zoom in/out)
        fliplr=0.5,                  # Espejo horizontal
        flipud=0.0,                  # Evitar espejo vertical (los textos de cabeza confunden al clasificador)
        
        # Estabilización final
        close_mosaic=15,             # Desactiva mosaic en las últimas 15 épocas para afinar cajas y ángulos precisos
        save=True,
        plots=True,                  # Genera curvas de pérdida y matriz de confusión
        project="holanda_obb_project",
        name="yolov8n_obb_optimized"
    )

    print("\n--- Entrenamiento finalizado con éxito ---")

    # ---------------------------------------------------------
    # 4. Validación del mejor modelo obtenido
    # ---------------------------------------------------------
    print("Ejecutando validación en el conjunto de prueba...")
    metrics = model.val()
    
    print(f"Resultados globales:")
    print(f" - mAP50     : {metrics.box.map50:.4f}")
    print(f" - mAP50-95  : {metrics.box.map50_95:.4f}")

    # ---------------------------------------------------------
    # 5. Localización del peso final (`best.pt`)
    # ---------------------------------------------------------
    best_model_path = os.path.join("holanda_obb_project", "yolov8n_obb_optimized", "weights", "best.pt")
    if os.path.exists(best_model_path):
        print(f"\n[ÉXITO] El mejor modelo entrenado está guardado en:\n -> {best_model_path}")
        
        # Ejemplo opcional para hacer una predicción rápida con el modelo entrenado:
        # inference_model = YOLO(best_model_path)
        # inference_model.predict("ruta/a/imagen_prueba.jpg", save=True, conf=0.3)
    else:
        print("\n[AVISO] No se pudo localizar automáticamente el archivo best.pt en la ruta esperada.")

if __name__ == "__main__":
    main()