from pathlib import Path
from typing import Dict, Tuple

BASE_DIR = Path(__file__).parent.resolve()

MODEL_PATH = BASE_DIR / "best.pt"
SEG_MODEL_PATH = BASE_DIR / "segmentation.pt"
CLASSES_PATH = BASE_DIR / "classes.txt"

# Capacidad máxima por SKU (Todos en 12)
REGLAS_CAPACIDAD_TEXTO: Dict[str, int] = {
    "mordisko": 12,
    "mordizco": 12,
    "paleta": 12,
    "magnum": 12,
    "solero": 12,
    "cornetto": 12,
    "pint": 12,
    "DEFAULT": 12,
}

# PASO DE PROFUNDIDAD EN CENTÍMETROS (2.0 cm por unidad)
PASO_PROFUNDIDAD_POR_SKU: Dict[str, float] = {
    "mordisko": 2.0,
    "mordizco": 2.0,
    "paleta": 2.0,
    "magnum": 2.0,
    "solero": 2.0,
    "cornetto": 2.0,
    "pint": 2.0,
    "DEFAULT": 2.0,
}

REJILLA_KEYWORDS: Tuple[str, ...] = (
    "rejilla", "canastilla", "separador", "carril", "parrilla", "estante"
)