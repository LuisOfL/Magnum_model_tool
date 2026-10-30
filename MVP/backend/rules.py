import re
from config import REGLAS_CAPACIDAD_POR_SERIE, PASO_PROFUNDIDAD_POR_SERIE

def normalizar_texto(texto: str) -> str:
    if not texto:
        return ""
    texto = texto.lower()
    texto = re.sub(r'[^a-z0-9\s]', '', texto)
    return " ".join(texto.split())

def resolver_capacidad_texto(sku_detectado: str, numero_serie: str = "DEFAULT") -> int:
    sku_limpio = normalizar_texto(sku_detectado)
    
    # Obtener las reglas para la serie específica, o usar DEFAULT si no existe
    reglas = REGLAS_CAPACIDAD_POR_SERIE.get(numero_serie, REGLAS_CAPACIDAD_POR_SERIE["DEFAULT"])
    
    for marca, capacidad in reglas.items():
        if marca in sku_limpio:
            return capacidad
            
    return reglas.get("DEFAULT", 12)

resolver_capacidad_sku = resolver_capacidad_texto

def resolver_paso_profundidad(sku_detectado: str, numero_serie: str = "DEFAULT") -> float:
    sku_limpio = normalizar_texto(sku_detectado)
    
    reglas = PASO_PROFUNDIDAD_POR_SERIE.get(numero_serie, PASO_PROFUNDIDAD_POR_SERIE["DEFAULT"])
    
    for marca, paso in reglas.items():
        if marca in sku_limpio:
            return paso
            
    return reglas.get("DEFAULT", 2.0)