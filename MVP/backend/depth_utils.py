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
    
    # 1. Extraer la imagen original en escala de grises para usarla como Guía Estructural
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


def obtener_profundidad_rejilla_knn_pura(
    depth_cm: np.ndarray, 
    producto: Dict, 
    knn_data: Optional[tuple[cKDTree, np.ndarray]], 
    k: int = 30
) -> Optional[float]:
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
        array_prof = np.array(profundidades)
        # Usamos el percentil 15 para enfocarnos en la parte frontal detectada naturalmente
        return float(np.percentile(array_prof, 15))

    return None