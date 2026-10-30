from pathlib import Path
from typing import Dict, Tuple

BASE_DIR = Path(__file__).parent.resolve()

MODEL_PATH = BASE_DIR / "best.pt"
SEG_MODEL_PATH = BASE_DIR / "segmentation.pt"
CLASSES_PATH = BASE_DIR / "classes.txt"

# Configuraciones de capacidad dependiendo del número de serie
REGLAS_CAPACIDAD_POR_SERIE: Dict[str, Dict[str, int]] = {
    "DEFAULT": {
        "mordisko": 12, "mordizco": 12, "paleta": 12, "magnum": 12,
        "solero": 12, "cornetto": 12, "pint": 12, "DEFAULT": 12,
    },
    "A001": {
        "mordisko": 15, "mordizco": 15, "paleta": 15, "magnum": 15,
        "solero": 15, "cornetto": 15, "pint": 15, "DEFAULT": 15,
    },
    "B002": {
        "mordisko": 13, "mordizco": 13, "paleta": 13, "magnum": 13,
        "solero": 13, "cornetto": 13, "pint": 13, "DEFAULT": 13,
    }
}

# PASO DE PROFUNDIDAD EN CENTÍMETROS dependiendo del número de serie
PASO_PROFUNDIDAD_POR_SERIE: Dict[str, Dict[str, float]] = {
    "DEFAULT": {
        "mordisko": 2.0, "mordizco": 2.0, "paleta": 2.0, "magnum": 2.0,
        "solero": 2.0, "cornetto": 2.0, "pint": 2.0, "DEFAULT": 2.0,
    },
    "SERIE_A001": {
        "mordisko": 1.8, "mordizco": 1.8, "paleta": 1.8, "magnum": 2.5,
        "solero": 2.5, "cornetto": 2.5, "pint": 2.5, "DEFAULT": 2.0,
    },
    "SERIE_B002": {
        "mordisko": 2.2, "mordizco": 2.2, "paleta": 2.0, "magnum": 1.8,
        "solero": 1.8, "cornetto": 2.2, "pint": 2.2, "DEFAULT": 2.0,
    }
}

REJILLA_KEYWORDS: Tuple[str, ...] = (
    "rejilla", "canastilla", "separador", "carril", "parrilla", "estante"
)