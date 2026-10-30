# Databricks notebook source
# ==============================================================================
# SMART FREEZER AUDIT - DATABRICKS ENTERPRISE SINGLE-CELL LAUNCHER
# ==============================================================================
# Este script está optimizado para entornos Databricks corporativos:
# 1. Cero riesgo de seguridad/firewall: No requiere túneles externos (ngrok) ni tokens públicos.
# 2. Utiliza de forma nativa el 'Databricks Driver Proxy' sobre HTTPS seguro.
# 3. Detecta automáticamente la ruta del repositorio en /Workspace/ o local.
# 4. Libera el puerto si se vuelve a ejecutar la celda para evitar 'Address already in use'.
# 5. Levanta el servidor FastAPI en un hilo daemon desacoplado y renderiza el frontend HTML.
# ==============================================================================

# COMMAND ----------

import os
import sys
import time
import json
import urllib.request
import subprocess
import threading
from pathlib import Path

# --- 1. VERIFICACIÓN E INSTALACIÓN DE DEPENDENCIAS ---
def asegurar_dependencias():
    paquetes = {
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
        "python_multipart": "python-multipart",
        "ultralytics": "ultralytics",
        "transformers": "transformers",
        "timm": "timm",
        "cv2": "opencv-python-headless",
        "PIL": "pillow"
    }
    a_instalar = []
    for mod, pkg in paquetes.items():
        try:
            if mod == "python_multipart":
                try:
                    import multipart
                except ImportError:
                    import python_multipart
            else:
                __import__(mod)
        except ImportError:
            a_instalar.append(pkg)
    
    if a_instalar:
        print(f"📦 [Databricks] Instalando paquetes necesarios: {a_instalar}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *a_instalar])
        print("✅ [Databricks] Paquetes listos.")

asegurar_dependencias()

# --- 2. LOCALIZACIÓN DINÁMICA DE CARPETAS ---
def encontrar_directorio_mvp() -> Path:
    posibles = [
        Path.cwd(),
        Path.cwd() / "MVP",
        Path(os.getcwd()),
        Path(os.getcwd()) / "MVP",
    ]
    
    # Si estamos en Databricks Workspace/Repos
    try:
        nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
        if nb_path:
            p_nb = Path("/Workspace") / nb_path.lstrip("/")
            posibles.extend([p_nb.parent, p_nb.parent / "MVP", p_nb.parent.parent / "MVP"])
    except Exception:
        pass

    for p in posibles:
        if (p / "backend" / "main.py").exists():
            return p.resolve()
        if (p / "main.py").exists() and (p / "models.py").exists():
            return p.parent.resolve()
            
    # Búsqueda recursiva hacia arriba
    curr = Path.cwd().resolve()
    for _ in range(4):
        if (curr / "MVP" / "backend" / "main.py").exists():
            return (curr / "MVP").resolve()
        if (curr / "backend" / "main.py").exists():
            return curr.resolve()
        curr = curr.parent

    raise FileNotFoundError("❌ No se encontró la carpeta 'backend/main.py'. Asegúrate de estar en el repositorio.")

MVP_DIR = encontrar_directorio_mvp()
BACKEND_DIR = MVP_DIR / "backend"
print(f"📁 [Databricks] Directorio del proyecto detectado: {MVP_DIR}")

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(MVP_DIR) not in sys.path:
    sys.path.insert(0, str(MVP_DIR))

# --- 3. SELECCIÓN DE PUERTO SEGURO (SIN MATAR EL KERNEL) ---
import socket

def buscar_puerto_libre(puerto_base=8000) -> int:
    # Detener servidor previo de Python si la celda se reejecuta
    if "_FREEZER_SERVER_INSTANCE" in globals() and globals()["_FREEZER_SERVER_INSTANCE"]:
        try:
            print("🔄 Deteniendo servidor anterior en memoria...")
            globals()["_FREEZER_SERVER_INSTANCE"].should_exit = True
            time.sleep(1.0)
        except Exception:
            pass

    for p in range(puerto_base, puerto_base + 30):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("0.0.0.0", p))
                return p
        except Exception:
            continue
    return puerto_base

PUERTO = buscar_puerto_libre(8000)
print(f"🔌 [Databricks] Puerto asignado: {PUERTO}")

# --- 4. CARGA DE FASTAPI Y MAIN ---
import uvicorn
from main import app

# --- 5. INICIO DE SERVIDOR EN SEGUNDO PLANO ---
config = uvicorn.Config(
    app=app,
    host="0.0.0.0",
    port=PUERTO,
    log_level="warning",
    access_log=False
)
server = uvicorn.Server(config)
globals()["_FREEZER_SERVER_INSTANCE"] = server

thread = threading.Thread(target=server.run, daemon=True)
thread.start()

# Esperar a que el endpoint /health responda
servidor_listo = False
for _ in range(40):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PUERTO}/health", timeout=1) as resp:
            if resp.status == 200:
                servidor_listo = True
                break
    except Exception:
        time.sleep(0.5)

if not servidor_listo:
    print("⚠️ Advertencia: El servidor tardó en responder /health, pero continuará iniciando.")

# --- 6. OBTENCIÓN DE URL DEL DATABRICKS DRIVER PROXY ---
def resolver_driver_proxy_url(puerto: int) -> str:
    try:
        ctx_str = dbutils.notebook.entry_point.getDbutils().notebook().getContext().toJson()
        ctx = json.loads(ctx_str)
        tags = ctx.get("tags", {})
        extra = ctx.get("extraContext", {})
        
        host = tags.get("browserHostName") or extra.get("api_url", "").replace("https://", "").replace("http://", "").strip("/")
        org_id = tags.get("orgId") or ctx.get("orgId")
        cluster_id = tags.get("clusterId") or ctx.get("clusterId")
        
        if host and org_id and cluster_id:
            # Nota importante: la diagonal final '/' es fundamental para que el navegador resuelva
            # las peticiones fetch('predict') relativas sin desviar la ruta base del proxy.
            return f"https://{host}/driver-proxy/o/{org_id}/{cluster_id}/{puerto}/"
    except Exception:
        pass
    
    return f"http://127.0.0.1:{puerto}/"

PROXY_URL = resolver_driver_proxy_url(PUERTO)

print("\n" + "=" * 70)
print(f"🚀 SMART FREEZER AUDIT ESTÁ ACTIVO EN DATABRICKS")
print(f"🔗 URL DIRECTA DEL DRIVER PROXY: {PROXY_URL}")
print("=" * 70 + "\n")

# --- 7. RENDERIZADO DEL FRONTEND EN DATABRICKS ---
html_embed = f"""
<div style="font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0B0F19; padding: 16px 20px; border-radius: 12px; margin-bottom: 12px; border: 1px solid #374151; display: flex; justify-content: space-between; align-items: center;">
    <div>
        <h3 style="color: #38BDF8; margin: 0 0 4px 0; font-size: 17px; font-weight: 600;">🍦 Smart Freezer Audit - Databricks Enterprise</h3>
        <p style="color: #9CA3AF; margin: 0; font-size: 13px;">Servidor FastAPI y modelos ejecutándose en el Driver Node (Puerto {PUERTO})</p>
    </div>
    <div style="display: flex; gap: 10px;">
        <a href="{PROXY_URL}" target="_blank" style="background: #38BDF8; color: #0B0F19; font-weight: bold; padding: 9px 18px; border-radius: 8px; text-decoration: none; font-size: 13px; display: inline-flex; align-items: center; gap: 6px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.5);">
            Abrir en Nueva Pestaña ↗
        </a>
    </div>
</div>
<iframe src="{PROXY_URL}" width="100%" height="860" style="border: 1px solid #374151; border-radius: 12px; background: #0B0F19;" allow="clipboard-read; clipboard-write"></iframe>
"""

try:
    displayHTML(html_embed)
except NameError:
    print("ℹ️ displayHTML no está disponible fuera de Databricks. Abre la aplicación en tu navegador:", PROXY_URL)
