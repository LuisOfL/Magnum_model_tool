from typing import Dict, List, Optional
import numpy as np
import torch.nn.functional as F
from PIL import Image
from scipy.spatial import cKDTree
import cv2
from models import depth_estimator


def calcular_mapa_profundidad_cm(pil_img: Image.Image) -> np.ndarray:
    salida = depth_estimator(pil_img)
    predicted = salida.get("predicted_depth")

    if predicted is None:
        depth = np.asarray(salida["depth"], dtype=np.float32)
        depth_img = Image.fromarray(depth).resize(pil_img.size, Image.Resampling.BICUBIC)
        depth_m = np.asarray(depth_img, dtype=np.float32)
        return depth_m * 100.0

    if predicted.ndim == 2:
        predicted = predicted.unsqueeze(0).unsqueeze(0)
    elif predicted.ndim == 3:
        predicted = predicted.unsqueeze(1)

    predicted = F.interpolate(
        predicted.float(),
        size=(pil_img.height, pil_img.width),
        mode="bicubic",
        align_corners=False,
    )[0, 0]

    depth_tensor = predicted.detach().cpu().numpy().astype(np.float32)
    
    # 1. Extraer la imagen en escala de grises para usarla como Guía Estructural
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray_guide = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    # 2. Normalizar la profundidad nativa de Depth Anything V2 a [0, 1]
    d_min, d_max = depth_tensor.min(), depth_tensor.max()
    if d_max - d_min > 1e-6:
        depth_norm = (depth_tensor - d_min) / (d_max - d_min)
        
        # 3. Aplicar Filtro Guiado de Kaiming He de forma 100% nativa sobre el modelo
        try:
            depth_guiado = cv2.ximgproc.guidedFilter(
                guide=gray_guide, 
                src=depth_norm, 
                radius=4,    # Reducido de 8 a 4 para detectar barrotes más finos
                eps=1e-4     # Reducido de 1e-3 a 1e-4 para forzar los bordes
            )
        except AttributeError:
            d_uint8 = (depth_norm * 255.0).astype(np.uint8)
            # También ajustamos el filtro bilateral de respaldo para bordes finos
            depth_guiado = cv2.bilateralFilter(d_uint8, d=5, sigmaColor=50, sigmaSpace=50).astype(np.float32) / 255.0

        depth_tensor = depth_guiado * (d_max - d_min) + d_min

    # 4. Mapeo métrico físico final basado enteramente en la salida guiada del modelo
    v_min, v_max = np.percentile(depth_tensor, [2, 98])
    if v_max - v_min > 1e-6:
        depth_normalizada = (depth_tensor - v_min) / (v_max - v_min)
        depth_cm = (1.0 - depth_normalizada) * 25.0 + 5.0 
    else:
        depth_cm = np.zeros_like(depth_tensor) + 15.0

    return depth_cm


def extraer_profundidad_bbox(depth_cm: np.ndarray, item: Dict, factor_recorte: float = 0.50) -> Optional[float]:
    if not item:
        return None
    
    alto, ancho = depth_cm.shape
    cx = (item["x1"] + item["x2"]) / 2.0
    cy_modificado = item["y1"] + (item["y2"] - item["y1"]) * 0.25
    
    bw = (item["x2"] - item["x1"]) * factor_recorte
    bh = (item["y2"] - item["y1"]) * factor_recorte

    x1 = max(0, int(cx - bw / 2))
    y1 = max(0, int(cy_modificado - bh / 2))
    x2 = min(ancho, int(cx + bw / 2))
    y2 = min(alto, int(cy_modificado + bh / 2))

    region = depth_cm[y1:y2, x1:x2]
    validos = region[np.isfinite(region)]
    
    if validos.size == 0:
        return None
        
    sin_rejilla = validos[validos > 7.0]
    if sin_rejilla.size > (validos.size * 0.15): 
        validos = sin_rejilla

    return float(np.percentile(validos, 90))


def extraer_coordenadas_rejillas(rejillas: List[Dict], ancho: int, alto: int) -> Optional[tuple[cKDTree, np.ndarray]]:
    coords_list = []

    for r in rejillas:
        if "mask" in r and r["mask"] is not None:
            mask = r["mask"]
            ys, xs = np.where(mask > 0)
            if len(xs) > 0:
                coords = np.column_stack((xs, ys))
                coords_list.append(coords)
        else:
            x1, y1 = max(0, int(r["x1"])), max(0, int(r["y1"]))
            x2, y2 = min(ancho, int(r["x2"])), min(alto, int(r["y2"]))
            
            grid_y, grid_x = np.mgrid[y1:y2:3, x1:x2:3]
            if grid_y.size > 0:
                coords = np.column_stack((grid_x.ravel(), grid_y.ravel()))
                coords_list.append(coords)

    if not coords_list:
        return None

    todas_coords = np.vstack(coords_list)
    tree = cKDTree(todas_coords)
    return tree, todas_coords


def construir_imagen_poligonos_rejillas(
    pil_img: Image.Image,
    rejillas: List[Dict],
) -> tuple[Image.Image, List[np.ndarray]]:
    """
    Toma la imagen original y las coordenadas/máscaras detectadas por el segundo modelo
    de YOLO (segmentación de rejillas) y crea una imagen con polígonos sólidos dibujados
    sobre cada canastilla, bien delineados y delimitados por las rejillas detectadas.
    """
    img_rgb = np.array(pil_img).copy()
    alto, ancho = img_rgb.shape[:2]

    if not rejillas:
        return Image.fromarray(img_rgb), []

    vert_segments = []
    horiz_segments = []
    mask_verticales_total = np.zeros((alto, ancho), dtype=np.uint8)
    mask_top_horiz_total = np.zeros((alto, ancho), dtype=np.uint8)

    y_mid_ref = alto * 0.55

    for r in rejillas:
        x1, y1, x2, y2 = float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])
        bw = x2 - x1
        bh = y2 - y1
        mask = r.get("mask")

        if mask is not None:
            ys, xs = np.where(mask > 0)
        else:
            ys, xs = np.array([]), np.array([])

        # Segmentos verticales de rejilla (divisores de canastillas)
        if bh > bw * 1.8 and bh > alto * 0.06:
            if len(xs) >= 10:
                span_y = float(ys.max() - ys.min())
                if span_y > alto * 0.08:
                    m, b = np.polyfit(ys.astype(np.float64), xs.astype(np.float64), 1)
                    m = float(np.clip(m, -0.22, 0.22))
                    b = float(np.median(xs - m * ys))
                else:
                    m = 0.0
                    b = float(np.median(xs))
                x_mid = m * y_mid_ref + b
                vert_segments.append({
                    "m": m,
                    "b": b,
                    "x_mid": x_mid,
                    "xs": xs,
                    "ys": ys,
                    "y1": float(ys.min()),
                    "y2": float(ys.max()),
                    "span_y": span_y,
                    "x1": x1,
                    "x2": x2,
                    "mask": mask,
                })
            else:
                x_mid = (x1 + x2) / 2.0
                vert_segments.append({
                    "m": 0.0,
                    "b": x_mid,
                    "x_mid": x_mid,
                    "xs": np.array([x1, x2]),
                    "ys": np.array([y1, y2]),
                    "y1": y1,
                    "y2": y2,
                    "span_y": y2 - y1,
                    "x1": x1,
                    "x2": x2,
                    "mask": None,
                })
        # Segmentos horizontales de rejilla (barras transversales)
        elif bw > bh * 1.5 and bw > ancho * 0.04:
            horiz_segments.append({
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "y_centro": (y1 + y2) / 2.0,
                "xs": xs,
                "ys": ys,
                "mask": mask,
            })

    # Agrupar segmentos verticales pertenecientes al mismo divisor de canastilla
    vert_segments.sort(key=lambda s: s["x_mid"])
    clusters = []
    cluster_dist_thresh = ancho * 0.065

    for seg in vert_segments:
        if not clusters or abs(seg["x_mid"] - np.mean([s["x_mid"] for s in clusters[-1]])) > cluster_dist_thresh:
            clusters.append([seg])
        else:
            clusters[-1].append(seg)

    valid_clusters = [
        c for c in clusters
        if (max(s["y2"] for s in c) - min(s["y1"] for s in c)) > alto * 0.12
        or sum(s["y2"] - s["y1"] for s in c) > alto * 0.10
    ]
    if len(valid_clusters) >= 2:
        clusters = valid_clusters

    divisores = []
    for c in clusters:
        all_xs = np.concatenate([s["xs"] for s in c if len(s["xs"]) > 0])
        all_ys = np.concatenate([s["ys"] for s in c if len(s["ys"]) > 0])
        for s in c:
            if s["mask"] is not None:
                mask_verticales_total = np.maximum(mask_verticales_total, (s["mask"] > 0).astype(np.uint8))

        long_segs = [s for s in c if s["span_y"] > alto * 0.12]
        if long_segs:
            weights = np.array([s["span_y"] for s in long_segs], dtype=np.float64)
            m = float(np.sum(np.array([s["m"] for s in long_segs]) * weights) / np.sum(weights))
        elif len(all_xs) >= 10 and (all_ys.max() - all_ys.min()) > alto * 0.12:
            m, _ = np.polyfit(all_ys.astype(np.float64), all_xs.astype(np.float64), 1)
            m = float(np.clip(m, -0.22, 0.22))
        else:
            m = 0.0

        if len(all_xs) >= 10:
            res_all = all_xs - m * all_ys
            b = float(np.median(res_all))
            b_izq = float(np.percentile(res_all, 20))
            b_der = float(np.percentile(res_all, 80))
        else:
            b = float(np.mean([s["x_mid"] for s in c]))
            b_izq = b - 3.0
            b_der = b + 3.0

        divisores.append({
            "m": m,
            "b": b,
            "m_izq": m,
            "b_izq": b_izq,
            "m_der": m,
            "b_der": b_der,
            "x_mid": m * y_mid_ref + b,
            "y_min": float(min(s["y1"] for s in c)),
            "y_max": float(max(s["y2"] for s in c)),
        })

    # Si las barras horizontales muestran una canastilla en un extremo donde faltó el divisor exterior
    if horiz_segments and divisores:
        min_x_horiz = min(h["x1"] for h in horiz_segments)
        max_x_horiz = max(h["x2"] for h in horiz_segments)
        if divisores[0]["x_mid"] - min_x_horiz > ancho * 0.12:
            m_ext = divisores[0]["m"]
            b_ext = min_x_horiz - m_ext * y_mid_ref
            divisores.insert(0, {
                "m": m_ext, "b": b_ext,
                "m_izq": m_ext, "b_izq": b_ext,
                "m_der": m_ext, "b_der": b_ext,
                "x_mid": min_x_horiz,
                "y_min": divisores[0]["y_min"],
                "y_max": divisores[0]["y_max"],
            })
        if max_x_horiz - divisores[-1]["x_mid"] > ancho * 0.12:
            m_ext = divisores[-1]["m"]
            b_ext = max_x_horiz - m_ext * y_mid_ref
            divisores.append({
                "m": m_ext, "b": b_ext,
                "m_izq": m_ext, "b_izq": b_ext,
                "m_der": m_ext, "b_der": b_ext,
                "x_mid": max_x_horiz,
                "y_min": divisores[-1]["y_min"],
                "y_max": divisores[-1]["y_max"],
            })

    # Identificar las barras horizontales superiores que delimitan el inicio de las canastillas
    y_min_divisores = float(np.percentile([d["y_min"] for d in divisores], 25)) if divisores else alto * 0.15
    top_horiz = [
        h for h in horiz_segments
        if h["y_centro"] < alto * 0.28 and h["y_centro"] <= y_min_divisores + alto * 0.10
    ]
    for h in top_horiz:
        if h["mask"] is not None:
            mask_top_horiz_total = np.maximum(mask_top_horiz_total, (h["mask"] > 0).astype(np.uint8))

    if top_horiz:
        min_top_y = min(h["y_centro"] for h in top_horiz)
        top_bars_principales = [h for h in top_horiz if h["y_centro"] <= min_top_y + alto * 0.06]
        y_top_global = float(np.median([h["y2"] for h in top_bars_principales]))
    elif divisores:
        y_top_global = y_min_divisores
    else:
        y_top_global = alto * 0.15

    poligonos = []
    for i in range(len(divisores) - 1):
        d_izq = divisores[i]
        d_der = divisores[i + 1]

        ancho_canastilla = d_der["x_mid"] - d_izq["x_mid"]
        if ancho_canastilla < ancho * 0.08:
            continue

        bars_en_canastilla = [
            h for h in top_horiz
            if h["x2"] > d_izq["x_mid"] + ancho_canastilla * 0.15
            and h["x1"] < d_der["x_mid"] - ancho_canastilla * 0.15
        ]
        if bars_en_canastilla:
            bar_top = min(bars_en_canastilla, key=lambda h: h["y_centro"])
            if len(bar_top["xs"]) >= 10 and (bar_top["xs"].max() - bar_top["xs"].min()) > ancho * 0.05:
                mt, bt = np.polyfit(bar_top["xs"].astype(np.float64), bar_top["ys"].astype(np.float64), 1)
                mt = float(np.clip(mt, -0.15, 0.15))
                res_t = bar_top["ys"] - (mt * bar_top["xs"] + bt)
                bt_borde = bt + float(np.percentile(res_t, 75))
                x_approx_izq = d_izq["m_der"] * bar_top["y2"] + d_izq["b_der"]
                x_approx_der = d_der["m_izq"] * bar_top["y2"] + d_der["b_izq"]
                y_top_izq = mt * x_approx_izq + bt_borde
                y_top_der = mt * x_approx_der + bt_borde
            else:
                y_top_izq = bar_top["y2"]
                y_top_der = bar_top["y2"]
        else:
            y_top_izq = max(y_top_global, min(d_izq["y_min"], alto * 0.25))
            y_top_der = max(y_top_global, min(d_der["y_min"], alto * 0.25))

        y_bot_izq = float(alto) if d_izq["y_max"] >= alto * 0.96 else max(d_izq["y_max"], d_der["y_max"])
        y_bot_der = float(alto) if d_der["y_max"] >= alto * 0.96 else max(d_der["y_max"], d_izq["y_max"])
        if d_izq["y_max"] > alto * 0.70 and d_der["y_max"] > alto * 0.70:
            y_bot_izq = float(alto) if d_izq["y_max"] >= alto * 0.96 else d_izq["y_max"]
            y_bot_der = float(alto) if d_der["y_max"] >= alto * 0.96 else d_der["y_max"]

        x_tl = np.clip(d_izq["m_der"] * y_top_izq + d_izq["b_der"], 0, ancho - 1)
        x_tr = np.clip(d_der["m_izq"] * y_top_der + d_der["b_izq"], 0, ancho - 1)
        x_br = np.clip(d_der["m_izq"] * y_bot_der + d_der["b_izq"], 0, ancho - 1)
        x_bl = np.clip(d_izq["m_der"] * y_bot_izq + d_izq["b_der"], 0, ancho - 1)

        if x_tr - x_tl > 15 and x_br - x_bl > 15:
            pts = np.array([
                [int(round(x_tl)), int(round(y_top_izq))],
                [int(round(x_tr)), int(round(y_top_der))],
                [int(round(x_br)), int(round(y_bot_der))],
                [int(round(x_bl)), int(round(y_bot_izq))],
            ], dtype=np.int32)
            poligonos.append(pts)

    # Respaldo si no se pudieron formar polígonos por falta de detecciones
    if not poligonos and rejillas:
        xs_all = [r["x1"] for r in rejillas] + [r["x2"] for r in rejillas]
        ys_all = [r["y1"] for r in rejillas] + [r["y2"] for r in rejillas]
        x_min_r, x_max_r = int(max(0, min(xs_all))), int(min(ancho - 1, max(xs_all)))
        y_min_r, y_max_r = int(max(0, min(ys_all))), int(min(alto - 1, max(ys_all)))
        if x_max_r - x_min_r > 40 and y_max_r - y_min_r > 40:
            w_col = (x_max_r - x_min_r) // 4
            for i in range(4):
                cx1 = x_min_r + i * w_col + 4
                cx2 = x_min_r + (i + 1) * w_col - 4
                pts = np.array([[cx1, y_min_r], [cx2, y_min_r], [cx2, y_max_r], [cx1, y_max_r]], dtype=np.int32)
                poligonos.append(pts)

    color_poligono = (205, 210, 218)
    img_con_poligonos = img_rgb.copy()
    for pts in poligonos:
        cv2.fillPoly(img_con_poligonos, [pts], color_poligono)

    # Mantener intactos los barrotes de rejilla detectados por YOLO segmentación que delimitan los polígonos
    mask_protegida = np.maximum(mask_verticales_total, mask_top_horiz_total)
    if np.any(mask_protegida > 0):
        kernel = np.ones((3, 3), np.uint8)
        mask_dilatada = cv2.dilate(mask_protegida, kernel, iterations=1) > 0
        img_con_poligonos[mask_dilatada] = img_rgb[mask_dilatada]

    return Image.fromarray(img_con_poligonos), poligonos


def obtener_profundidad_rejilla_knn_pura(
    depth_cm: np.ndarray, 
    producto: Dict, 
    knn_data: Optional[tuple[cKDTree, np.ndarray]], 
    k: int = 3
) -> Optional[float]:
    """
    Obtiene la profundidad de referencia de la rejilla buscando los k=3 píxeles
    más cercanos al producto sobre el segundo mapa de calor (generado aplicando
    Depth Anything a la imagen con los polígonos delimitados por YOLO segmentación).
    """
    if knn_data is None:
        return None

    tree, todas_coords = knn_data
    centro_paleta = np.array([producto["x_centro"], producto["y_centro"]])

    k_actual = min(k, len(todas_coords))
    _, indices = tree.query(centro_paleta, k=k_actual)

    if k_actual == 1:
        indices = [indices]

    pixeles_cercanos = todas_coords[indices]

    profundidades = [
        depth_cm[py, px] 
        for px, py in pixeles_cercanos 
        if 0 <= py < depth_cm.shape[0] and 0 <= px < depth_cm.shape[1] and np.isfinite(depth_cm[py, px])
    ]

    if profundidades:
        array_prof = np.array(profundidades, dtype=np.float32)
        return float(np.mean(array_prof))

    return None
