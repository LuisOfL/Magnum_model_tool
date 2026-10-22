from typing import Dict, List, Optional
import cv2
import numpy as np
import torch.nn.functional as F
from PIL import Image
from scipy.spatial import cKDTree
from models import depth_estimator


def aplicar_filtro_guiado(
    guia_rgb: np.ndarray, 
    depth: np.ndarray, 
    radius: int = 6, 
    eps: float = 1e-3
) -> np.ndarray:
    """
    Aplica el Filtro Guiado (Guided Image Filter de Kaiming He) usando la imagen RGB 
    como guía para transferir los bordes nítidos de las varillas y rejillas metálicas
    al mapa de profundidad continuo generado por Depth Anything.
    
    Implementación nativa y vectorizada en O(N) con cv2.boxFilter.
    """
    if guia_rgb.ndim == 3:
        guia_gray = cv2.cvtColor(guia_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    else:
        guia_gray = guia_rgb.astype(np.float32) / 255.0

    p = depth.astype(np.float32)
    ksize = (2 * radius + 1, 2 * radius + 1)

    # Medias locales y correlaciones
    mean_I = cv2.boxFilter(guia_gray, cv2.CV_32F, ksize)
    mean_p = cv2.boxFilter(p, cv2.CV_32F, ksize)
    corr_I = cv2.boxFilter(guia_gray * guia_gray, cv2.CV_32F, ksize)
    corr_Ip = cv2.boxFilter(guia_gray * p, cv2.CV_32F, ksize)

    # Varianza y Covarianza local
    var_I = corr_I - (mean_I * mean_I)
    cov_Ip = corr_Ip - (mean_I * mean_p)

    # Coeficientes lineales a y b
    a = cov_Ip / (var_I + eps)
    b = mean_p - (a * mean_I)

    # Promedio de coeficientes
    mean_a = cv2.boxFilter(a, cv2.CV_32F, ksize)
    mean_b = cv2.boxFilter(b, cv2.CV_32F, ksize)

    # Salida refinada con bordes transferidos
    q = mean_a * guia_gray + mean_b
    return q


def calcular_mapa_profundidad_cm(pil_img: Image.Image, usar_filtro_guiado: bool = True) -> np.ndarray:
    salida = depth_estimator(pil_img)
    predicted = salida.get("predicted_depth")

    if predicted is None:
        depth = np.asarray(salida["depth"], dtype=np.float32)
        depth_img = Image.fromarray(depth).resize(pil_img.size, Image.Resampling.BICUBIC)
        depth_tensor = np.asarray(depth_img, dtype=np.float32)
    else:
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

    # CORRECCIÓN DE ESCALA: Normalización [0, 1] basada en percentiles
    v_min, v_max = np.percentile(depth_tensor, [2, 98])
    if v_max - v_min > 1e-6:
        depth_normalizada = np.clip((depth_tensor - v_min) / (v_max - v_min), 0.0, 1.0)
    else:
        depth_normalizada = np.zeros_like(depth_tensor) + 0.5

    # FILTRADO GUIADO: Rescatar bordes finos de las varillas usando la imagen RGB como guía
    if usar_filtro_guiado:
        rgb_np = np.array(pil_img)
        depth_normalizada = aplicar_filtro_guiado(
            guia_rgb=rgb_np,
            depth=depth_normalizada,
            radius=6,
            eps=1e-3
        )
        depth_normalizada = np.clip(depth_normalizada, 0.0, 1.0)

    # Invertir si el modelo da profundidad inversa y escalar a cm (máximo 30 cm de fondo)
    # 5.0 cm = frente, 30.0 cm = fondo del congelador
    depth_cm = (1.0 - depth_normalizada) * 25.0 + 5.0

    return depth_cm


def extraer_profundidad_bbox(depth_cm: np.ndarray, item: Dict, factor_recorte: float = 0.50) -> Optional[float]:
    """Obtiene la mediana de la profundidad (en cm) en la región central del Bounding Box."""
    if not item:
        return None
    
    alto, ancho = depth_cm.shape
    cx = (item["x1"] + item["x2"]) / 2.0
    cy = (item["y1"] + item["y2"]) / 2.0
    bw = (item["x2"] - item["x1"]) * factor_recorte
    bh = (item["y2"] - item["y1"]) * factor_recorte

    x1 = max(0, int(cx - bw / 2))
    y1 = max(0, int(cy - bh / 2))
    x2 = min(ancho, int(cx + bw / 2))
    y2 = min(alto, int(cy + bh / 2))

    region = depth_cm[y1:y2, x1:x2]
    validos = region[np.isfinite(region)]
    return float(np.median(validos)) if validos.size > 0 else None


def extraer_coordenadas_rejillas(rejillas: List[Dict], ancho: int, alto: int) -> Optional[tuple[cKDTree, np.ndarray]]:
    """Extrae la lista de coordenadas (x, y) de todas las rejillas detectadas y genera el cKDTree."""
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
    k: int = 20
) -> Optional[float]:
    """Busca los K píxeles de rejilla más cercanos y retorna el promedio de sus profundidades en cm."""
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
        # Usamos percentil 25 en lugar de promedio simple para ser robustos
        # frente a píxeles en el límite de la varilla que toquen el fondo del congelador
        return float(np.percentile(profundidades, 25))

    return None


def calcular_profundidad_referencia_frontal(depth_cm: np.ndarray, rejillas: List[Dict] = None) -> float:
    """
    Calcula la profundidad métrica (en cm) del plano frontal de referencia del congelador.
    Utiliza el marco frontal sólido o el borde superior de las canastillas (típicamente 5.5 a 6.5 cm).
    """
    if rejillas:
        alturas_superiores = []
        for r in rejillas:
            y1 = max(0, int(r["y1"]))
            y2 = min(depth_cm.shape[0], int(r["y1"] + 35))
            x1 = max(0, int(r["x1"]))
            x2 = min(depth_cm.shape[1], int(r["x2"]))
            if y2 > y1 and x2 > x1:
                franja = depth_cm[y1:y2, x1:x2]
                validos = franja[np.isfinite(franja)]
                if validos.size > 0:
                    alturas_superiores.append(float(np.percentile(validos, 15)))
        if alturas_superiores:
            return float(np.median(alturas_superiores))

    validos_global = depth_cm[np.isfinite(depth_cm)]
    if validos_global.size > 0:
        return float(np.percentile(validos_global, 3))
    
    return 6.0


def obtener_profundidad_referencia_local(
    depth_cm: np.ndarray, 
    prod: Dict, 
    rejillas: List[Dict] = None
) -> float:
    """
    Calcula la profundidad métrica de referencia (nivel 'lleno' al tope) LOCAL
    para un Bounding Box específico.
    
    Compensa la perspectiva tridimensional del congelador:
    - Las cajas del frente (cerca de la cámara) se comparan con el nivel frontal (~7-9 cm).
    - Las cajas del fondo (más lejos de la cámara) se comparan con el nivel del fondo (~13-16 cm),
      evitando que aparezcan como 'vacías' por la inclinación natural de la cámara.
    """
    cx = prod.get("x_centro", (prod.get("x1", 0) + prod.get("x2", 0)) / 2.0)
    cy = prod.get("y_centro", (prod.get("y1", 0) + prod.get("y2", 0)) / 2.0)
    alto_img, ancho_img = depth_cm.shape

    # 1. Si el producto cae en una canastilla/rejilla segmentada por YOLO
    if rejillas:
        for r in rejillas:
            rx1, ry1, rx2, ry2 = r["x1"], r["y1"], r["x2"], r["y2"]
            if (rx1 - 20 <= cx <= rx2 + 20) and (ry1 - 20 <= cy <= ry2 + 20):
                # Borde superior/frontal de esta canastilla específica
                y_top = max(0, int(ry1))
                y_bottom = min(alto_img, int(ry1 + 45))
                x_left = max(0, int(rx1))
                x_right = min(ancho_img, int(rx2))
                
                franja_canastilla = depth_cm[y_top:y_bottom, x_left:x_right]
                validos_can = franja_canastilla[np.isfinite(franja_canastilla)]
                if validos_can.size > 0:
                    return float(np.percentile(validos_can, 12))

    # 2. Compensación por franja horizontal de perspectiva en el eje Y
    cy_int = int(np.clip(cy, 0, alto_img - 1))
    bw = int(abs(prod.get("x2", 100) - prod.get("x1", 0)))
    bh = int(abs(prod.get("y2", 100) - prod.get("y1", 0)))
    delta_y = max(30, bh // 2)

    y_min = max(0, cy_int - delta_y)
    y_max = min(alto_img, cy_int + delta_y)
    franja_h = depth_cm[y_min:y_max, :]
    validos_h = franja_h[np.isfinite(franja_h)]
    if validos_h.size > 0:
        # El 8vo percentil captura el marco o superficie frontal en esa misma cota de profundidad
        return float(np.percentile(validos_h, 8))

    # 3. Fallback al borde frontal general
    validos_global = depth_cm[np.isfinite(depth_cm)]
    if validos_global.size > 0:
        return float(np.percentile(validos_global, 3))

    return 6.5


def agrupar_en_carriles(productos: List[Dict], rejillas: List[Dict] = None) -> List[List[Dict]]:
    """
    Agrupa los productos detectados en 'carriles' o 'filas' tridimensionales (de adelante hacia atrás).
    Dos productos pertenecen al mismo carril si:
    1. Tienen el mismo SKU.
    2. Pertenecen a la misma canastilla (si existen máscaras de rejilla).
    3. Están alineados horizontalmente en el eje X (uno detrás de otro en profundidad).
    """
    if not productos:
        return []

    carriles = []
    prods_ordenados = sorted(productos, key=lambda p: p["x_centro"])

    for prod in prods_ordenados:
        asignado = False
        
        # Identificar canastilla contenedora
        id_canastilla = None
        if rejillas:
            for i, r in enumerate(rejillas):
                if (r["x1"] <= prod["x_centro"] <= r["x2"] and 
                    r["y1"] <= prod["y_centro"] <= r["y2"]):
                    id_canastilla = i
                    break

        for carril in carriles:
            if prod["nombre"] == carril[0]["nombre"]:
                # Misma canastilla si está disponible
                carril_canastilla = carril[0].get("_canastilla_id")
                if id_canastilla is not None and carril_canastilla is not None:
                    if id_canastilla != carril_canastilla:
                        continue

                # Alineación horizontal en X (mismo carril)
                ref = carril[0]
                ancho_ref = ref["x2"] - ref["x1"]
                ancho_prod = prod["x2"] - prod["x1"]
                umbral_x = max(ancho_ref, ancho_prod) * 0.65
                
                if abs(prod["x_centro"] - ref["x_centro"]) <= umbral_x:
                    prod["_canastilla_id"] = id_canastilla
                    carril.append(prod)
                    asignado = True
                    break

        if not asignado:
            prod["_canastilla_id"] = id_canastilla
            carriles.append([prod])

    return carriles