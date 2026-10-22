# 📦 Pipeline de Cajas Orientadas YOLO OBB (`yolo_bouding_boxes_train/`)

Este directorio contiene los scripts y herramientas para la **curación de datos** y el **entrenamiento de modelos de detección de objetos orientados (Oriented Bounding Boxes - OBB)** con YOLOv8. 

En congeladores de helados, los empaques suelen colocarse inclinados, girados o amontonados. El uso de cajas delimitadoras orientadas permite estimar con exactitud el ángulo de rotación ($\theta$), la posición y el centro real de cada helado sin solapamientos falsos.

---

## 🗂️ Contenido del Directorio

| Archivo | Función Principal |
| :--- | :--- |
| [`entrenar.py`](file:///c:/Users/luigu/OneDrive/Escritorio/Ultimate_Model_Ice/yolo_bouding_boxes_train/entrenar.py) | Script de entrenamiento supervisado para **YOLOv8n-OBB** utilizando aumentaciones avanzadas y rotación geométrica completa. |
| [`wrapper.py`](file:///c:/Users/luigu/OneDrive/Escritorio/Ultimate_Model_Ice/yolo_bouding_boxes_train/wrapper.py) | Herramienta de desduplicación y saneamiento de imágenes basada en hash SHA-256 y nombres base. |

---

## 🧼 1. Curación y Desduplicación (`wrapper.py`)

Al recopilar miles de fotos en tiendas de conveniencia o supermercados, frecuentemente se introducen imágenes duplicadas o ya etiquetadas. `wrapper.py` previene la contaminación del dataset de entrenamiento comparando una carpeta de imágenes crudas con el conjunto de imágenes ya integradas a YOLO.

### Mecanismo de Verificación Dual:
1. **Coincidencia por Nombre**: Detecta si el nombre base del archivo ya existe en el conjunto anotado.
2. **Coincidencia por Hash Criptográfico (SHA-256)**: Si el nombre cambió pero el contenido del archivo es idéntico, calcula el hash binario en bloques de 8 KB para identificar duplicados exactos.
3. **Modo Seguro de Respaldo**: Por defecto (`BORRAR_DIRECTO = False`), los archivos duplicados se mueven a una subcarpeta `temp/` para revisión manual en lugar de eliminarse permanentemente.

### Ejecución:
```bash
python wrapper.py
```

---

## 🎯 2. Entrenamiento de YOLOv8n-OBB (`entrenar.py`)

El script configura un entrenamiento robusto con hiperparámetros diseñados específicamente para datasets pequeños y variabilidad de orientación en congeladores.

### Configuración y Parámetros Clave:

| Hiperparámetro | Valor | Justificación Técnica |
| :--- | :--- | :--- |
| **Modelo Base** | `yolov8n-obb.pt` | Versión Nano con soporte nativo de cajas orientadas ($\theta$). |
| **Resolución (`imgsz`)** | `640` | Equilibrio óptimo entre velocidad y precisión (ampliable a 1024 si hay textos diminutos). |
| **Batch Size (`batch`)** | `8` | Ajustado para tarjetas gráficas de 6 GB de VRAM (como NVIDIA RTX 4050). |
| **Optimizador** | `AdamW` | Mejor decaimiento de pesos y convergencia en datasets con clases desbalanceadas. |
| **Rotación (`degrees`)** | `180.0` | Permite rotación completa en 360° para detectar empaques en cualquier inclinación. |
| **Flip Vertical (`flipud`)** | `0.0` | **Desactivado** para evitar confundir las marcas al poner los logotipos y textos de cabeza. |
| **Aumentación** | `mosaic=1.0`<br>`mixup=0.15`<br>`copy_paste=0.1` | Combina múltiples escenas sintéticamente para enriquecer clases minoritarias. |
| **`close_mosaic`** | `15` épocas | Desactiva el mosaico en las últimas 15 épocas para afinar las esquinas y ángulos de las cajas. |

### Estructura Requerida de `dataset.yaml`:
Antes de iniciar, asegúrate de contar con el archivo de configuración del dataset con la estructura estándar de Ultralytics OBB:
```yaml
path: ../datasets/icecream_obb
train: images/train
val: images/val

names:
  0: magnum_clasica
  1: magnum_almendras
  2: cornetto_chocolate
  3: mordisko_original
  # ... resto de las clases
```

### Ejecución del Entrenamiento:
```bash
python entrenar.py
```

### Salida del Entrenamiento:
Al concluir, el script valida el modelo en el subconjunto de prueba imprimiendo las métricas $mAP_{50}$ y $mAP_{50-95}$, y genera los pesos entrenados en:
```
holanda_obb_project/yolov8n_obb_optimized/weights/best.pt
```
Este archivo `best.pt` resultante es el que se traslada a [`app/backend/best.pt`](file:///c:/Users/luigu/OneDrive/Escritorio/Ultimate_Model_Ice/app/backend/best.pt) para su uso en producción.

---

## 🔗 Enlaces Relacionados
- [Volver al README Principal](file:///c:/Users/luigu/OneDrive/Escritorio/Ultimate_Model_Ice/README.md)
- [Pipeline de Segmentación de Rejillas (`yolo_segmentation_train/`)](file:///c:/Users/luigu/OneDrive/Escritorio/Ultimate_Model_Ice/yolo_segmentation_train/README.md)
