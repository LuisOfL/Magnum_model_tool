# 📐 Pipeline de Segmentación de Rejillas (`yolo_segmentation_train/`)

Este módulo está dedicado al entrenamiento y optimización del modelo de **segmentación de instancias (YOLOv8-Seg)** para detectar las rejillas, parrillas y canastillas divisorias de los congeladores de helados.

Detectar rejillas metálicas es un problema desafiante de visión por computadora porque las varillas son estructuras muy delgadas (de pocos píxeles de grosor) y repetitivas. Este módulo implementa técnicas específicas para evitar que las varillas se deformen o se fusionen en las máscaras generadas.

---

## 🗂️ Contenido del Directorio

| Archivo | Descripción |
| :--- | :--- |
| [`main.py`](file:///c:/Users/luigu/OneDrive/Escritorio/Ultimate_Model_Ice/yolo_segmentation_train/main.py) | Script autónomo que genera dinámicamente la configuración `data.yaml`, carga `yolov8n-seg.pt` e inicia el entrenamiento de alta resolución. |

---

## ⚙️ Decisiones Técnicas y Ajustes de Hiperparámetros

En [`main.py`](file:///c:/Users/luigu/OneDrive/Escritorio/Ultimate_Model_Ice/yolo_segmentation_train/main.py) se configuran parámetros críticos para resolver estructuras finas:

```python
model.train(
    data=yaml_path,
    epochs=150,
    patience=30,
    imgsz=1024,          # 👈 Alta resolución para varillas finas
    batch=2,             # 👈 Control de VRAM para imágenes de 1024px
    device=device_to_use,
    overlap_mask=False,  # 👈 Evita deformación o fusión entre alambres cercanos

    # Aumentación geométrica suave
    degrees=5.0,
    translate=0.02,
    scale=0.1,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.5,

    # Desactivar aumentaciones destructivas de textura
    mosaic=0.0,
    mixup=0.0,
    copy_paste=0.0,

    project="entrenamiento_rejilla",
    name="yolo_segmentation_rejilla_finas"
)
```

### Razonamiento de las Configuraciones:
1. **Resolución Alta (`imgsz=1024`)**:
   - A 640px estándar, los alambres metálicos de 2-4 mm de grosor se reducen a menos de 1 píxel, perdiéndose en las capas convolucionales intermedias. A 1024px, cada varilla cuenta con suficiente densidad de píxeles para ser segmentada con continuidad.
2. **`overlap_mask=False`**:
   - Impide que las máscaras de segmentación de dos barras o separadores contiguos se solapen o se unan artificialmente en un solo polígono deformado.
3. **Desactivación de Mosaicos (`mosaic=0.0`, `mixup=0.0`)**:
   - El aumento tipo mosaico corta y fragmenta las líneas rectas y cuadrículas de las rejillas, confundiendo la continuidad topológica de las parrillas. Se utilizan variaciones geométricas ligeras (`scale=0.1`, `degrees=5.0`).

---

## 📁 Preparación de los Datos

El script busca opcionalmente un archivo `classes.txt` en el mismo directorio. Si no existe, define automáticamente la clase por defecto `rejilla`.

Estructura esperada en la carpeta del dataset:
```
yolo_segmentation_train/
├── images/                # Fotografías con anotaciones en formato YOLO Segmentation
├── labels/                # Archivos .txt con las coordenadas poligonales normalizadas
├── classes.txt            # (Opcional) Nombre de las clases (ej: rejilla)
└── main.py                # Script de ejecución
```

Al ejecutarse, `main.py` genera automáticamente el archivo `data.yaml` con las rutas absolutas resueltas:
```yaml
path: <directorio_actual>
train: images
val: images
names:
  0: rejilla
```

---

## 🚀 Cómo Ejecutar el Entrenamiento

1. Asegúrate de tener instalado PyTorch con soporte CUDA y Ultralytics:
   ```bash
   pip install ultralytics torch torchvision pyyaml
   ```
2. Ejecuta el script:
   ```bash
   python main.py
   ```
3. Al finalizar las 150 épocas (o si se detiene por early stopping tras 30 épocas sin mejora gracias a `patience=30`), el mejor modelo se guardará en:
   ```
   entrenamiento_rejilla/yolo_segmentation_rejilla_finas/weights/best.pt
   ```
4. Este archivo de pesos se renombra a `segmentation.pt` y se coloca en [`app/backend/segmentation.pt`](file:///c:/Users/luigu/OneDrive/Escritorio/Ultimate_Model_Ice/app/backend/segmentation.pt) para su integración con el pipeline de inferencia.

---

## 🔗 Enlaces Relacionados
- [Volver al README Principal](file:///c:/Users/luigu/OneDrive/Escritorio/Ultimate_Model_Ice/README.md)
- [Backend y Motor de Profundidad (`app/backend/`)](file:///c:/Users/luigu/OneDrive/Escritorio/Ultimate_Model_Ice/app/backend/README.md)
- [Pipeline de Cajas Orientadas (`yolo_bouding_boxes_train/`)](file:///c:/Users/luigu/OneDrive/Escritorio/Ultimate_Model_Ice/yolo_bouding_boxes_train/README.md)
