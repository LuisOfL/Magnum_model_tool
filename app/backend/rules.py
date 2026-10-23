import re
from config import REGLAS_CAPACIDAD_TEXTO, PASO_PROFUNDIDAD_POR_SKU

def normalizar_texto(texto: str) -> str:
    """
    Limpia una cadena de texto: minúsculas, sin caracteres especiales y espacios normalizados.
    """
    if not texto:
        return ""
    
    texto = texto.lower()
    texto = re.sub(r'[^a-z0-9\s]', '', texto)
    return " ".join(texto.split())

def resolver_capacidad_texto(sku_detectado: str) -> int:
    """
    Busca la marca en el SKU normalizado y devuelve su capacidad de texto 
    importada desde config.py.
    """
    sku_limpio = normalizar_texto(sku_detectado)
    
    for marca, capacidad in REGLAS_CAPACIDAD_TEXTO.items():
        if marca in sku_limpio:
            return capacidad
            
    return REGLAS_CAPACIDAD_TEXTO.get("DEFAULT", 12)

# Alias para compatibilidad con llamadas en main.py
resolver_capacidad_sku = resolver_capacidad_texto

def resolver_paso_profundidad(sku_detectado: str) -> float:
    """
    Busca la marca en el SKU normalizado y devuelve su paso de profundidad
    importado desde config.py.
    """
    sku_limpio = normalizar_texto(sku_detectado)
    
    for marca, paso in PASO_PROFUNDIDAD_POR_SKU.items():
        if marca in sku_limpio:
            return paso
            
    return PASO_PROFUNDIDAD_POR_SKU.get("DEFAULT", 2.0)