import base64
import io
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool
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
from video_utils import crear_panoramica_desde_video, procesar_video_con_modelos

app = FastAPI(title="Smart Freezer Audit API - Profundidad Métrica (cm)")

BUILD = "v6-video-stream-models"

CARPETA_PROYECTO = Path(__file__).resolve().parent
INDEX_HTML = CARPETA_PROYECTO / "index.html"
if not INDEX_HTML.exists() and (CARPETA_PROYECTO.parent / "index.html").exists():
    INDEX_HTML = CARPETA_PROYECTO.parent / "index.html"
CARPETA_VIDEOS = CARPETA_PROYECTO / "static" / "videos"
CARPETA_VIDEOS.mkdir(parents=True, exist_ok=True)

# Configuración de CORS para permitir solicitudes desde el frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", include_in_schema=False)
@app.get("", include_in_schema=False)
@app.get("/index.html", include_in_schema=False)
async def frontend():
    if not INDEX_HTML.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No se encontró index.html en {INDEX_HTML}",
        )
    return FileResponse(INDEX_HTML, media_type="text/html")

@app.get("/health", include_in_schema=False)
async def health():
    return {"status": "ok"}

@app.get("/videos/{video_filename}")
async def get_video(video_filename: str):
    video_path = CARPETA_VIDEOS / video_filename
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video no encontrado")
    return FileResponse(video_path, media_type="video/mp4")

@app.post("/predict")
@app.post("/predict/")
@app.post("/{full_path:path}/predict")
@app.post("/{full_path:path}/predict/")
async def predict(
    file: UploadFile = File(...),
    numero_serie: str = Form("DEFAULT")
):
    content_type = file.content_type or ""
    filename = (file.filename or "").lower()
    es_video = content_type.startswith("video/") or filename.endswith((".mp4", ".avi", ".mov", ".webm", ".mkv"))
    es_imagen = content_type.startswith("image/") or filename.endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp"))

    if not (es_video or es_imagen):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen o video válido.")

    try:
        file_bytes = await file.read()
        
        if es_video:
            # Control de frames adaptativo: 50 en GPU o 25 en CPU para responder en < 60 segundos
            limite_frames = 50 if YOLO_DEVICE == 0 else 25
            
            # Ejecutar en threadpool desacoplado para no bloquear el loop HTTP ni provocar timeout
            resultado_video = await run_in_threadpool(
                procesar_video_con_modelos,
                video_bytes=file_bytes,
                model=model,
                seg_model=seg_model,
                depth_calc_func=calcular_mapa_profundidad_cm,
                class_names=CLASS_NAMES,
                rejilla_keywords=REJILLA_KEYWORDS,
                resolver_capacidad_sku=lambda sku: resolver_capacidad_sku(sku, numero_serie=numero_serie),
                resolver_paso_profundidad=lambda sku: resolver_paso_profundidad(sku, numero_serie=numero_serie),
                extraer_profundidad_bbox=extraer_profundidad_bbox,
                extraer_coordenadas_rejillas=extraer_coordenadas_rejillas,
                obtener_profundidad_rejilla_knn_pura=obtener_profundidad_rejilla_knn_pura,
                output_dir=CARPETA_VIDEOS,
                device=YOLO_DEVICE,
                max_frames=limite_frames,
            )
            resultado_video["build"] = BUILD
            return resultado_video

        # Procesar la imagen normalmente
        pil_img = ImageOps.exif_transpose(
            Image.open(io.BytesIO(file_bytes))
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

        cajas_det = resultado.obb if getattr(resultado, "obb", None) is not None else resultado.boxes

        if cajas_det is not None and len(cajas_det) > 0:
            ids_cls = cajas_det.cls.cpu().numpy().astype(int)
            confs = cajas_det.conf.cpu().numpy()
            xyxys = cajas_det.xyxy.cpu().numpy() 

            for indice in range(len(ids_cls)):
                clase_id = int(ids_cls[indice])
                nombre = CLASS_NAMES.get(
                    clase_id,
                    model.names.get(clase_id, f"Clase_{clase_id}"),
                )
                x1, y1, x2, y2 = map(float, xyxys[indice])
                productos.append({
                    "id_deteccion": f"det_{indice + 1}",
                    "nombre": nombre,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "x_centro": (x1 + x2) / 2.0,
                    "y_centro": (y1 + y2) / 2.0,
                    "confianza": float(confs[indice]),
                })

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

        print(f"[{BUILD}] cajas={0 if cajas_det is None else len(cajas_det)} "
              f"productos={len(productos)} rejillas={len(rejillas)} CLASS_NAMES={len(CLASS_NAMES)}")

        # 3. Obtención del mapa de profundidad en centímetros (cm)
        depth_cm = calcular_mapa_profundidad_cm(pil_img)
        
        # 4. Construcción del k-NN espacial para rejillas
        knn_data = extraer_coordenadas_rejillas(rejillas, ancho, alto)

        detalle_detecciones = []
        resumen_por_sku = {}

        # 5. Cálculo de faltantes en centímetros
        for prod in productos:
            nombre_sku = prod["nombre"]
            capacidad_max = resolver_capacidad_sku(nombre_sku)      
            paso_profundidad_cm = resolver_paso_profundidad(nombre_sku) 

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
                diferencia_cm = abs(profundidad_sku - profundidad_rejilla)
                faltantes = int(round(diferencia_cm / max(paso_profundidad_cm, 1e-6)))
                faltantes = int(np.clip(faltantes, 0, capacidad_max - 1))
                
                inventario_estimado = int(np.clip(capacidad_max - faltantes, 1, capacidad_max))
                estado = "estimado_exitoso"

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
        depth_vis_invertido = cv2.bitwise_not(depth_vis) 
        
        depth_colored = cv2.applyColorMap(depth_vis_invertido, cv2.COLORMAP_INFERNO)
        buffer_depth = io.BytesIO()
        Image.fromarray(cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)).save(buffer_depth, format="JPEG", quality=90)
        imagen_depth_base64 = base64.b64encode(buffer_depth.getvalue()).decode("utf-8")

        return {
            "status": "success",
            "tipo": "image",
            "build": BUILD,
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
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
