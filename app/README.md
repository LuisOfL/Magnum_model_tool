# 🖥️ Módulo de Aplicación y Dashboards de Usuario (`app/`)

Este directorio contiene las interfaces visuales (escritorio y web) para interactuar con el sistema **Smart Freezer Audit**, así como el microservicio de backend.

---

## 📁 Contenido del Directorio

| Archivo / Carpeta | Tipo | Descripción |
| :--- | :--- | :--- |
| [`app.py`](file:///c:/Users/luigu/OneDrive/Escritorio/Ultimate_Model_Ice/app/app.py) | Python Script | Aplicación de escritorio nativa desarrollada con el framework **Flet** (Flutter para Python). |
| [`index.html`](file:///c:/Users/luigu/OneDrive/Escritorio/Ultimate_Model_Ice/app/index.html) | Web Frontend | Dashboard web responsive moderno estilizado con **Tailwind CSS** y **Lucide Icons**. |
| [`temp.py`](file:///c:/Users/luigu/OneDrive/Escritorio/Ultimate_Model_Ice/app/temp.py) | Python Script | Script utilitario para inspección rápida de metadatos y tareas (`model.task`) de pesos YOLO. |
| [`backend/`](file:///c:/Users/luigu/OneDrive/Escritorio/Ultimate_Model_Ice/app/backend) | Subdirectorio | Servidor FastAPI con los modelos de visión, profundidad métrica y lógica de auditoría. |

---

## 🌟 Características de las Interfaces

Tanto la aplicación de escritorio como la versión web proporcionan una experiencia de usuario completa y sincronizada:

1. **Carga de Fotografías**: Soporte para formatos `JPG`, `JPEG` y `PNG` tomadas cenitalmente o en ángulo hacia el congelador.
2. **Visualización de Inferencia en 3 Capas**:
   - **Detección (Boxes)**: Cajas delimitadoras orientadas con etiquetas de los productos y porcentajes de confianza.
   - **Segmentación (Polígonos)**: Delineado de las rejillas metálicas y separadores del congelador.
   - **Mapa de Profundidad (Inferno)**: Mapa térmico generado con `Depth Anything V2` que representa visualmente la cercanía y profundidad de cada zona.
3. **Resumen General de Inventario**:
   - Lista agrupada por SKU con el conteo de caras visibles detectadas y el stock total estimado.
4. **Desglose Detallado por Detección**:
   - Muestra para cada bounding box individual: ID, SKU detectado, confianza, capacidad máxima, unidades faltantes calculadas, stock estimado y comparación física de profundidad (Profundidad del SKU en cm vs. Profundidad de la Rejilla en cm).
5. **Selector Dinámico de Endpoint (en Web)**:
   - Permite cambiar la URL del servidor al vuelo (ejemplo: `http://127.0.0.1:8000/predict` para pruebas locales o túneles públicos como `https://xxxx.ngrok-free.dev/predict`).

---

## 🚀 Cómo Ejecutar

### 1. Iniciar la Aplicación de Escritorio (Flet)

Asegúrate de tener instalada la librería `flet`:
```bash
pip install flet requests
```

Ejecuta el script:
```bash
python app.py
```
> **Nota**: El backend debe estar ejecutándose previamente en el puerto 8000 para que la aplicación pueda procesar las fotos.

### 2. Iniciar el Dashboard Web (HTML / Tailwind)

No requiere compilación ni dependencias de Node.js. Puedes:
- Abrir el archivo `index.html` directamente en tu navegador (doble clic).
- O servirlo con un servidor estático rápido:
  ```bash
  python -m http.server 3000
  ```
  Y abrir `http://localhost:3000` en tu navegador.

---

## 🔗 Enlaces Relacionados
- [Documentación detallada del Backend (`backend/README.md`)](file:///c:/Users/luigu/OneDrive/Escritorio/Ultimate_Model_Ice/app/backend/README.md)
- [README principal del proyecto](file:///c:/Users/luigu/OneDrive/Escritorio/Ultimate_Model_Ice/README.md)
