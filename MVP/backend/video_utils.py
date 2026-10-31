import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import uuid
import base64
from pathlib import Path
from typing import Dict, List, Tuple, Optional

def crear_panoramica_desde_video(video_bytes: bytes) -> Image.Image:
    """
    Toma un archivo de video en bytes, extrae los frames relevantes 
    y los une para crear una única imagen panorámica.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(video_bytes)
        temp_video_path = temp_video.name

    cap = cv2.VideoCapture(temp_video_path)
    frames = []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames > 0:
        step = max(1, total_frames // 15)
        for i in range(0, total_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            if len(frames) >= 15:
                break
    else:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            
        if len(frames) > 15:
            indices = np.linspace(0, len(frames)-1, 15, dtype=int)
            frames = [frames[i] for i in indices]

    cap.release()
    try:
        os.remove(temp_video_path)
    except Exception:
        pass

    if not frames:
        raise ValueError("No se pudieron extraer frames del video")

    if len(frames) == 1:
        return Image.fromarray(cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB))

    stitcher = cv2.Stitcher_create()
    status, stitched = stitcher.stitch(frames)

    if status == cv2.Stitcher_OK:
        img = cv2.cvtColor(stitched, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img)
    else:
        print(f"Advertencia: El stitching falló con código {status}. Usando el frame central como fallback.")
        mid = len(frames) // 2
        img = cv2.cvtColor(frames[mid], cv2.COLOR_BGR2RGB)
        return Image.fromarray(img)


def procesar_video_con_modelos(
    video_bytes: bytes,
    model,
    seg_model,
    depth_calc_func,
    class_names: Dict[int, str],
    rejilla_keywords: Tuple[str, ...],
    resolver_capacidad_sku,
    resolver_paso_profundidad,
    extraer_profundidad_bbox,
    extraer_coordenadas_rejillas,
    obtener_profundidad_rejilla_knn_pura,
    output_dir: Path,
    device=0,
    target_fps: float = 10.0,
    max_frames: int = 35,
) -> Dict:
    """
    Procesa un video completo generando:
    1. Video con bounding boxes de YOLO (Detección de productos)
    2. Video con polígonos/máscaras de YOLO (Segmentación de rejillas)
    3. Video con mapa de calor de profundidad (Depth Crafter / Depth Anything)
    Consolida las detecciones y estimaciones de stock a lo largo de los frames.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    session_id = uuid.uuid4().hex[:10]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_in:
        temp_in.write(video_bytes)
        input_video_path = temp_in.name

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        try:
            os.remove(input_video_path)
        except Exception:
            pass
        raise ValueError("No se pudo abrir el archivo de video para lectura.")

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if orig_fps <= 0 or np.isnan(orig_fps):
        orig_fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Control de sampling para asegurar rendimiento óptimo en tiempo real
    if total_frames > max_frames:
        step = max(1, int(round(total_frames / max_frames)))
    elif orig_fps > target_fps:
        step = max(1, int(round(orig_fps / target_fps)))
    else:
        step = 1

    out_fps = max(5.0, round(orig_fps / step, 1))

    # Nombres de archivos de salida
    det_filename = f"video_det_{session_id}.mp4"
    seg_filename = f"video_seg_{session_id}.mp4"
    depth_filename = f"video_depth_{session_id}.mp4"

    det_path = output_dir / det_filename
    seg_path = output_dir / seg_filename
    depth_path = output_dir / depth_filename

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer_det = cv2.VideoWriter(str(det_path), fourcc, out_fps, (width, height))
    writer_seg = cv2.VideoWriter(str(seg_path), fourcc, out_fps, (width, height))
    writer_depth = cv2.VideoWriter(str(depth_path), fourcc, out_fps, (width, height))

    best_frame_details = []
    best_frame_score = -1.0
    best_frame_det_bgr = None
    best_frame_seg_bgr = None
    best_frame_depth_bgr = None
    best_frame_plano_bgr = None

    sku_max_visibles: Dict[str, int] = {}
    sku_best_estimados: Dict[str, int] = {}
    frame_counter = 0
    processed_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_counter % step != 0:
                frame_counter += 1
                continue
            frame_counter += 1
            processed_count += 1

            # Convertir frame BGR a PIL Image RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(frame_rgb)

            # 1. Detección YOLO
            resultado_det = model(pil_frame, conf=0.25, device=device)[0]
            frame_det_bgr = resultado_det.plot(boxes=True, masks=False)
            writer_det.write(frame_det_bgr)

            # 2. Segmentación YOLO
            resultado_seg = seg_model(
                pil_frame,
                conf=0.10,
                device=device,
                retina_masks=True
            )[0]
            frame_seg_bgr = resultado_seg.plot(
                boxes=False,
                masks=True,
                labels=False,
                probs=False
            )
            writer_seg.write(frame_seg_bgr)

            # 3. Estimación de Profundidad (Depth Crafter / Depth Anything V2)
            depth_cm = depth_calc_func(pil_frame)
            depth_vis = cv2.normalize(depth_cm, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_vis_inv = cv2.bitwise_not(depth_vis)
            frame_depth_bgr = cv2.applyColorMap(depth_vis_inv, cv2.COLORMAP_INFERNO)
            writer_depth.write(frame_depth_bgr)

            # Extraer cajas de productos en este frame
            cajas_det = resultado_det.obb if getattr(resultado_det, "obb", None) is not None else resultado_det.boxes
            productos_frame = []
            if cajas_det is not None and len(cajas_det) > 0:
                ids_cls = cajas_det.cls.cpu().numpy().astype(int)
                confs = cajas_det.conf.cpu().numpy()
                xyxys = cajas_det.xyxy.cpu().numpy()

                for idx in range(len(ids_cls)):
                    clase_id = int(ids_cls[idx])
                    nombre = class_names.get(
                        clase_id,
                        model.names.get(clase_id, f"Clase_{clase_id}")
                    )
                    x1, y1, x2, y2 = map(float, xyxys[idx])
                    productos_frame.append({
                        "id_deteccion": f"f{processed_count}_det_{idx + 1}",
                        "nombre": nombre,
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "x_centro": (x1 + x2) / 2.0,
                        "y_centro": (y1 + y2) / 2.0,
                        "confianza": float(confs[idx]),
                    })

            # Extraer rejillas en este frame
            rejillas_frame = []
            if resultado_seg.boxes is not None:
                masks = resultado_seg.masks.data.cpu().numpy() if resultado_seg.masks is not None else None
                for idx, box in enumerate(resultado_seg.boxes):
                    clase_id = int(box.cls[0].item())
                    nombre = seg_model.names.get(clase_id, f"Clase_{clase_id}")
                    es_rejilla = any(k in nombre.lower() for k in rejilla_keywords) or len(seg_model.names) == 1
                    if es_rejilla:
                        x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                        mask = masks[idx] if masks is not None and idx < len(masks) else None
                        if mask is not None and mask.shape != (height, width):
                            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

                        rejillas_frame.append({
                            "id_deteccion": f"f{processed_count}_seg_{idx + 1}",
                            "nombre": nombre,
                            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                            "x_centro": (x1 + x2) / 2.0,
                            "y_centro": (y1 + y2) / 2.0,
                            "confianza": float(box.conf[0].item()),
                            "mask": mask
                        })

            # Calcular estimación métrica con el segundo mapa de calor (polígonos de rejilla) y k=3
            from depth_utils import construir_imagen_poligonos_rejillas, calibrar_mapa_profundidad_plano
            pil_frame_poligonos, poligonos_frame = construir_imagen_poligonos_rejillas(pil_frame, rejillas_frame)
            depth_plano_raw = depth_calc_func(pil_frame_poligonos)
            depth_plano_cm = calibrar_mapa_profundidad_plano(
                depth_plano_raw=depth_plano_raw,
                depth_cm_orig=depth_cm,
                rejillas=rejillas_frame,
                poligonos=poligonos_frame,
            )
            plano_vis = cv2.normalize(depth_plano_cm, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            plano_vis_inv = cv2.bitwise_not(plano_vis)
            frame_plano_bgr = cv2.applyColorMap(plano_vis_inv, cv2.COLORMAP_INFERNO)

            knn_data = extraer_coordenadas_rejillas(rejillas_frame, width, height)
            detalles_frame = []
            sku_counts_this_frame: Dict[str, int] = {}
            sku_stock_this_frame: Dict[str, int] = {}

            for prod in productos_frame:
                nombre_sku = prod["nombre"]
                capacidad_max = resolver_capacidad_sku(nombre_sku)
                paso_profundidad_cm = resolver_paso_profundidad(nombre_sku)

                profundidad_sku = extraer_profundidad_bbox(depth_cm, prod)
                profundidad_rejilla = obtener_profundidad_rejilla_knn_pura(
                    depth_cm=depth_plano_cm,
                    producto=prod,
                    knn_data=knn_data,
                    k=3
                )

                if profundidad_sku is None or profundidad_rejilla is None:
                    faltantes = 0
                    inventario_estimado = 1
                    diferencia_cm = 0.0
                    estado = "sin_profundidad"
                    profundidad_texto = "N/A"
                else:
                    diferencia_cm = abs(profundidad_sku - profundidad_rejilla)
                    faltantes = int(round(diferencia_cm / max(paso_profundidad_cm, 1e-6)))
                    faltantes = int(np.clip(faltantes, 0, capacidad_max - 1))
                    inventario_estimado = int(np.clip(capacidad_max - faltantes, 1, capacidad_max))
                    estado = "estimado_exitoso"
                    profundidad_texto = f"{round(profundidad_sku, 1)}cm / {round(profundidad_rejilla, 1)}cm"

                detalles_frame.append({
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

                sku_counts_this_frame[nombre_sku] = sku_counts_this_frame.get(nombre_sku, 0) + 1
                sku_stock_this_frame[nombre_sku] = sku_stock_this_frame.get(nombre_sku, 0) + inventario_estimado

            # Actualizar máximos detectados a lo largo del video por SKU
            for sku, count in sku_counts_this_frame.items():
                if count > sku_max_visibles.get(sku, 0):
                    sku_max_visibles[sku] = count

            for sku, stock in sku_stock_this_frame.items():
                if stock > sku_best_estimados.get(sku, 0):
                    sku_best_estimados[sku] = stock

            # Calificar frame para representatividad del reporte
            score_frame = len(productos_frame) * 10.0 + sum(p["confianza"] for p in productos_frame)
            if score_frame > best_frame_score:
                best_frame_score = score_frame
                best_frame_details = detalles_frame
                best_frame_det_bgr = frame_det_bgr.copy()
                best_frame_seg_bgr = frame_seg_bgr.copy()
                best_frame_depth_bgr = frame_depth_bgr.copy()
                best_frame_plano_bgr = frame_plano_bgr.copy()

    finally:
        cap.release()
        writer_det.release()
        writer_seg.release()
        writer_depth.release()
        try:
            os.remove(input_video_path)
        except Exception:
            pass

    # Codificar videos a base64
    video_det_b64 = base64.b64encode(det_path.read_bytes()).decode("utf-8")
    video_seg_b64 = base64.b64encode(seg_path.read_bytes()).decode("utf-8")
    video_depth_b64 = base64.b64encode(depth_path.read_bytes()).decode("utf-8")

    # Si hay un frame representativo, generar imágenes resumen base64
    img_det_b64 = ""
    img_seg_b64 = ""
    img_depth_b64 = ""
    img_plano_b64 = ""
    if best_frame_det_bgr is not None:
        buf = cv2.imencode(".jpg", best_frame_det_bgr)[1]
        img_det_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    if best_frame_seg_bgr is not None:
        buf = cv2.imencode(".jpg", best_frame_seg_bgr)[1]
        img_seg_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    if best_frame_depth_bgr is not None:
        buf = cv2.imencode(".jpg", best_frame_depth_bgr)[1]
        img_depth_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    if best_frame_plano_bgr is not None:
        buf = cv2.imencode(".jpg", best_frame_plano_bgr)[1]
        img_plano_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

    # Resumen acumulado de inventario
    resumen_skus = []
    for sku, visibles in sku_max_visibles.items():
        stock_estimado = sku_best_estimados.get(sku, visibles)
        resumen_skus.append({
            "etiqueta": sku,
            "visibles": visibles,
            "stock_estimado": stock_estimado
        })

    return {
        "status": "success",
        "tipo": "video",
        "video_det_url": f"/videos/{det_filename}",
        "video_seg_url": f"/videos/{seg_filename}",
        "video_depth_url": f"/videos/{depth_filename}",
        "video_det_base64": video_det_b64,
        "video_seg_base64": video_seg_b64,
        "video_depth_base64": video_depth_b64,
        "imagen_base64": img_det_b64,
        "imagen_seg_base64": img_seg_b64,
        "imagen_depth_base64": img_depth_b64,
        "imagen_plano_base64": img_plano_b64,
        "detecciones": resumen_skus,
        "detalle_detecciones": best_frame_details,
        "total_skus_detectados": sum(sku_max_visibles.values()),
        "total_frames_procesados": processed_count,
    }

