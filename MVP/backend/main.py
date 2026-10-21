import base64
import io
from typing import Dict, List

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps

from config import REJILLA_KEYWORDS
from models import model, seg_model, YOLO_DEVICE, CLASS_NAMES
from rules import resolver_capacidad_sku, resolver_paso_profundidad
from depth_utils import (
    calcular_mapa_profundidad_cm,
    extraer_profundidad_bbox,
    extraer_coordenadas_rejillas,
    obtener_profundidad_rejilla_knn_pura,
)

app = FastAPI(title="Smart Freezer Audit API - Profundidad Métrica (cm)")

# Configuración de CORS para permitir solicitudes desde el frontend (S3 o Localhost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen válida.")

    try:
        image_bytes = await file.read()
        pil_img = ImageOps.exif_transpose(
            Image.open(io.BytesIO(image_bytes))
        ).convert("RGB")
        ancho, alto = pil_img.size

        # 1. Detección principal (Bounding Boxes de productos)
        resultado = model(pil_img, conf=0.25, device=YOLO_DEVICE)[0]

        # 2. Segmentación (Polígonos/Máscaras de rejilla)
        resultado_seg = seg_model(
            pil_img, 
            conf=0.10, 
            device=YOLO_DEVICE,
            retina_masks=True
        )[0]

        productos = []
        rejillas = []

        if resultado.boxes is not None:
            for indice, box in enumerate(resultado.boxes):
                clase_id = int(box.cls[0].item())
                nombre = CLASS_NAMES.get(
                    clase_id,
                    model.names.get(clase_id, f"Clase_{clase_id}"),
                )
                x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                deteccion = {
                    "id_deteccion": f"det_{indice + 1}",
                    "nombre": nombre,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "x_centro": (x1 + x2) / 2.0,
                    "y_centro": (y1 + y2) / 2.0,
                    "confianza": float(box.conf[0].item()),
                }

                if any(k in nombre.lower() for k in REJILLA_KEYWORDS):
                    rejillas.append(deteccion)
                else:
                    productos.append(deteccion)

        if resultado_seg.boxes is not None:
            masks = resultado_seg.masks.data.cpu().numpy() if resultado_seg.masks is not None else None
            
            for indice, box in enumerate(resultado_seg.boxes):
                clase_id = int(box.cls[0].item())
                nombre = seg_model.names.get(clase_id, f"Clase_{clase_id}")
                
                es_rejilla = any(k in nombre.lower() for k in REJILLA_KEYWORDS) or len(seg_model.names) == 1
                
                if es_rejilla:
                    x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                    mask = masks[indice] if masks is not None and indice < len(masks) else None
                    
                    if mask is not None and mask.shape != (alto, ancho):
                        mask = cv2.resize(mask, (ancho, alto), interpolation=cv2.INTER_NEAREST)

                    rejillas.append({
                        "id_deteccion": f"seg_{indice + 1}",
                        "nombre": nombre,
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "x_centro": (x1 + x2) / 2.0,
                        "y_centro": (y1 + y2) / 2.0,
                        "confianza": float(box.conf[0].item()),
                        "mask": mask
                    })

        # 3. Obtención del mapa de profundidad en centímetros (cm)
        depth_cm = calcular_mapa_profundidad_cm(pil_img)
        
        # 4. Construcción del k-NN espacial para rejillas
        knn_data = extraer_coordenadas_rejillas(rejillas, ancho, alto)

        detalle_detecciones = []
        resumen_por_sku: Dict[str, Dict] = {}

        # 5. Cálculo de faltantes en centímetros
        for prod in productos:
            nombre_sku = prod["nombre"]
            capacidad_max = resolver_capacidad_sku(nombre_sku)      # Retorna 12
            paso_profundidad_cm = resolver_paso_profundidad(nombre_sku) # Retorna 2.0 cm

            # Medición en cm
            profundidad_sku = extraer_profundidad_bbox(depth_cm, prod)
            profundidad_rejilla = obtener_profundidad_rejilla_knn_pura(
                depth_cm=depth_cm,
                producto=prod,
                knn_data=knn_data,
                k=20
            )

            if profundidad_sku is None or profundidad_rejilla is None:
                faltantes = 0
                inventario_estimado = 1
                diferencia_cm = 0.0
                estado = "sin_profundidad"
            else:
                # Diferencia real en centímetros
                diferencia_cm = abs(profundidad_sku - profundidad_rejilla)
                
                # Faltantes = Diferencia en cm / 2.0 cm por unidad
                faltantes = int(round(diferencia_cm / max(paso_profundidad_cm, 1e-6)))
                faltantes = int(np.clip(faltantes, 0, capacidad_max - 1))
                
                inventario_estimado = int(np.clip(capacidad_max - faltantes, 1, capacidad_max))
                estado = "estimado_exitoso"

            # Formatear el texto de distancias en cm para la auditoría en la UI
            if profundidad_sku is not None and profundidad_rejilla is not None:
                prof_sku_red = round(profundidad_sku, 1)
                prof_rej_red = round(profundidad_rejilla, 1)
                profundidad_texto = f"{prof_sku_red}cm / {prof_rej_red}cm"
            else:
                profundidad_texto = "N/A"

            detalle_detecciones.append({
                "id_deteccion": prod["id_deteccion"],
                "sku": nombre_sku,
                "confianza": round(prod["confianza"], 3),
                "profundidad_sku": round(profundidad_sku, 2) if profundidad_sku else None,
                "profundidad_rejilla": round(profundidad_rejilla, 2) if profundidad_rejilla else None,
                "profundidad_texto": profundidad_texto,
                "diferencia_cm": round(diferencia_cm, 2) if diferencia_cm else None,
                "capacidad_maxima": capacidad_max,
                "faltantes_calculados": faltantes,
                "inventario_estimado": inventario_estimado,
                "estado": estado,
            })

            if nombre_sku not in resumen_por_sku:
                resumen_por_sku[nombre_sku] = {
                    "etiqueta": nombre_sku,
                    "visibles": 0,
                    "stock_estimado": 0,
                }
            resumen_por_sku[nombre_sku]["visibles"] += 1
            resumen_por_sku[nombre_sku]["stock_estimado"] += inventario_estimado

        # Visualizaciones
        anotada_bgr = resultado.plot(boxes=True, masks=False)
        buffer_det = io.BytesIO()
        Image.fromarray(anotada_bgr[:, :, ::-1]).save(buffer_det, format="JPEG", quality=90)
        imagen_base64 = base64.b64encode(buffer_det.getvalue()).decode("utf-8")

        anotada_seg_bgr = resultado_seg.plot(
            boxes=False,
            masks=True,
            labels=False,
            probs=False
        )
        buffer_seg = io.BytesIO()
        Image.fromarray(anotada_seg_bgr[:, :, ::-1]).save(buffer_seg, format="JPEG", quality=90)
        imagen_seg_base64 = base64.b64encode(buffer_seg.getvalue()).decode("utf-8")

        # Generación y reversión visual del mapa de calor
        depth_vis = cv2.normalize(depth_cm, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_vis_invertido = cv2.bitwise_not(depth_vis) # Revertir el mapa de calor visual
        
        depth_colored = cv2.applyColorMap(depth_vis_invertido, cv2.COLORMAP_INFERNO)
        buffer_depth = io.BytesIO()
        Image.fromarray(cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)).save(buffer_depth, format="JPEG", quality=90)
        imagen_depth_base64 = base64.b64encode(buffer_depth.getvalue()).decode("utf-8")

        return {
            "status": "success",
            "imagen_base64": imagen_base64,
            "imagen_seg_base64": imagen_seg_base64,
            "imagen_depth_base64": imagen_depth_base64,
            "detecciones": list(resumen_por_sku.values()),
            "detalle_detecciones": detalle_detecciones,
            "total_skus_detectados": len(productos),
            "total_rejillas_detectadas": len(rejillas),
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Error en procesamiento de auditoría: {error}",
        ) from error


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)