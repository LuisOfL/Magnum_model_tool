import hashlib
import os
import shutil

# ==============================================================================
# RUTAS AUTOMÁTICAS (Toman la ubicación de este script 'train/' como base)
# ==============================================================================
# BASE_DIR será: C:\Users\luigu\OneDrive\Escritorio\Magnum_app\train
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CARPETA_3K = os.path.join(BASE_DIR, "imagenes_tiendas")
CARPETA_YOLO = os.path.join(BASE_DIR, "images")
CARPETA_RESPALDO = os.path.join(BASE_DIR, "temp")

# Si quieres BORRAR directamente sin mover a 'temp', cambia a: BORRAR_DIRECTO = True
BORRAR_DIRECTO = False

EXTENSIONES_IMAGEN = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff")


def calcular_hash(ruta_archivo, chunk_size=8192):
  """Calcula el hash SHA-256 para comparar el contenido real de la imagen."""
  hasher = hashlib.sha256()
  with open(ruta_archivo, "rb") as f:
    while chunk := f.read(chunk_size):
      hasher.update(chunk)
  return hasher.hexdigest()


def filtrar_imagenes_unicas():
  print(f"Ruta 3K detectada  : {CARPETA_3K}")
  print(f"Ruta YOLO detectada: {CARPETA_YOLO}\n")

  if not os.path.exists(CARPETA_3K) or not os.path.exists(CARPETA_YOLO):
    print("Error: No se encontraron las carpetas en la ruta indicada.")
    return

  if not BORRAR_DIRECTO and CARPETA_RESPALDO:
    os.makedirs(CARPETA_RESPALDO, exist_ok=True)

  print("1. Indexando imágenes de la carpeta YOLO...")
  nombres_yolo = set()
  hashes_yolo = set()

  archivos_yolo = os.listdir(CARPETA_YOLO)
  for archivo in archivos_yolo:
    if archivo.lower().endswith(EXTENSIONES_IMAGEN):
      nombre_base = os.path.splitext(archivo)[0].lower()
      nombres_yolo.add(nombre_base)

      ruta_img = os.path.join(CARPETA_YOLO, archivo)
      hashes_yolo.add(calcular_hash(ruta_img))

  print(
      f"   -> Se registraron {len(nombres_yolo)} imágenes en la carpeta YOLO."
  )

  print("\n2. Filtrando la carpeta de 3k imágenes...")
  archivos_3k = [
      f for f in os.listdir(CARPETA_3K) if f.lower().endswith(EXTENSIONES_IMAGEN)
  ]
  total_inicial = len(archivos_3k)
  eliminadas = 0

  for archivo in archivos_3k:
    ruta_img_3k = os.path.join(CARPETA_3K, archivo)
    nombre_base_3k = os.path.splitext(archivo)[0].lower()

    # Verifica coincidencia por nombre O por hash de contenido
    coincide_nombre = nombre_base_3k in nombres_yolo
    coincide_hash = False

    if not coincide_nombre:
      hash_3k = calcular_hash(ruta_img_3k)
      coincide_hash = hash_3k in hashes_yolo

    if coincide_nombre or coincide_hash:
      if BORRAR_DIRECTO:
        os.remove(ruta_img_3k)
      else:
        shutil.move(ruta_img_3k, os.path.join(CARPETA_RESPALDO, archivo))
      eliminadas += 1

  restantes = total_inicial - eliminadas

  print("\n================ RESUMEN ================")
  print(f"• Total de imágenes iniciales en carpeta 3k: {total_inicial}")
  print(f"• Imágenes removidas (estaban en YOLO)    : {eliminadas}")
  print(f"• Imágenes restantes en carpeta 3k       : {restantes}")
  print(
      f"• Carpeta YOLO permanece intacta con      : {len(os.listdir(CARPETA_YOLO))} archivos."
  )


if __name__ == "__main__":
  filtrar_imagenes_unicas()