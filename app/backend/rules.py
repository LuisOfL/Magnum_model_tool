import re

# Constantes por defecto
CAPACIDAD_DEFAULT = 25
PROFUNDIDAD_DEFAULT = 0.10

# Diccionarios de configuración
REGLAS_CAPACIDAD_TEXTO = {
    "magnum": 30,
    "solero": 25,
    "cornetto": 20
}

PASO_PROFUNDIDAD_POR_SKU = {
    "magnum": 0.15,
    "solero": 0.12,
    "cornetto": 0.10
}

def normalizar_texto(texto: str) -> str:
    """
    Limpia una cadena de texto: minúsculas, sin caracteres especiales y espacios normalizados.
    """
    if not texto:
        return ""
    
    texto = texto.lower()
    # Elimina cualquier cosa que no sea letra, número o espacio
    texto = re.sub(r'[^a-z0-9\s]', '', texto)
    # Elimina espacios múltiples
    return " ".join(texto.split())

def resolver_capacidad_texto(sku_detectado: str) -> int:
    """
    Busca la marca en el SKU normalizado y devuelve su capacidad de texto.
    """
    sku_limpio = normalizar_texto(sku_detectado)
    
    for marca, capacidad in REGLAS_CAPACIDAD_TEXTO.items():
        if marca in sku_limpio:  # Usamos 'in' por si el input es "helado magnum almendras"
            return capacidad
            
    return CAPACIDAD_DEFAULT

# Alias para compatibilidad con llamadas en main.py
resolver_capacidad_sku = resolver_capacidad_texto

def resolver_paso_profundidad(sku_detectado: str) -> float:
    """
    Busca la marca en el SKU normalizado y devuelve su paso de profundidad.
    """
    sku_limpio = normalizar_texto(sku_detectado)
    
    for marca, paso in PASO_PROFUNDIDAD_POR_SKU.items():
        if marca in sku_limpio:
            return paso
            
    return PROFUNDIDAD_DEFAULT