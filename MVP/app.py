import flet as ft
import requests
import mimetypes
import os
import base64
import tempfile
import time

API_URL = "http://127.0.0.1:8000/predict"

def main(page: ft.Page):
    page.title = "Smart Freezer Audit - IA"
    page.theme_mode = ft.ThemeMode.DARK
    page.bgcolor = "#0B0F19"
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.padding = 20
    page.scroll = ft.ScrollMode.AUTO

    page.appbar = ft.AppBar(
        leading=ft.Icon(ft.Icons.ICECREAM, color="#38BDF8", size=28),
        title=ft.Text("Smart Freezer Audit - Detección, Segmentación y Profundidad", weight=ft.FontWeight.W_600, color=ft.Colors.WHITE),
        center_title=True,
        bgcolor="#111827",
        elevation=2,
    )

    alert_banner = ft.Container(
        content=ft.Text("", size=15, weight=ft.FontWeight.W_500),
        padding=15,
        border_radius=10,
        visible=False,
        alignment=ft.alignment.center,
        border=ft.border.all(1, ft.Colors.TRANSPARENT)
    )

    loading_ring = ft.Container(
        content=ft.Row(
            controls=[
                ft.ProgressRing(width=20, height=20, stroke_width=3, color="#38BDF8"),
                ft.Text("Procesando imagen/video con los modelos de IA...", size=16, color="#9CA3AF", weight=ft.FontWeight.W_400)
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            spacing=15
        ),
        visible=False,
        margin=ft.margin.symmetric(vertical=15)
    )

    # Campo de texto para el número de serie
    input_serie = ft.TextField(
        label="Número de Serie del Congelador",
        hint_text="Ej: SERIE_A001 (Dejar en blanco para DEFAULT)",
        prefix_icon=ft.Icons.QR_CODE,
        width=1100,
        border_color="#38BDF8",
        color=ft.Colors.WHITE,
        bgcolor="#1F2937",
        focused_border_color="#34D399"
    )

    # Controles para medios (Imágenes o Videos dinámicos)
    img_result_det = ft.Image(width=330, height=260, fit=ft.ImageFit.CONTAIN, border_radius=ft.border_radius.all(10))
    img_result_seg = ft.Image(width=330, height=260, fit=ft.ImageFit.CONTAIN, border_radius=ft.border_radius.all(10))
    img_result_depth = ft.Image(width=330, height=260, fit=ft.ImageFit.CONTAIN, border_radius=ft.border_radius.all(10))

    media_box_det = ft.Container(content=img_result_det, width=330, height=260, alignment=ft.alignment.center)
    media_box_seg = ft.Container(content=img_result_seg, width=330, height=260, alignment=ft.alignment.center)
    media_box_depth = ft.Container(content=img_result_depth, width=330, height=260, alignment=ft.alignment.center)

    label_det = ft.Text("Detección (Boxes)", size=15, weight=ft.FontWeight.W_600, color="#38BDF8")
    card_det = ft.Container(
        content=ft.Column(
            controls=[
                label_det,
                media_box_det,
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=10,
        ),
        bgcolor="#1F2937",
        border_radius=15,
        padding=12,
        border=ft.border.all(1, "#374151"),
        shadow=ft.BoxShadow(blur_radius=15, color="#000000"),
    )

    label_seg = ft.Text("Segmentación (Polígonos)", size=15, weight=ft.FontWeight.W_600, color="#34D399")
    card_seg = ft.Container(
        content=ft.Column(
            controls=[
                label_seg,
                media_box_seg,
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=10,
        ),
        bgcolor="#1F2937",
        border_radius=15,
        padding=12,
        border=ft.border.all(1, "#374151"),
        shadow=ft.BoxShadow(blur_radius=15, color="#000000"),
    )

    label_depth = ft.Text("Mapa de Profundidad", size=15, weight=ft.FontWeight.W_600, color="#F59E0B")
    card_depth = ft.Container(
        content=ft.Column(
            controls=[
                label_depth,
                media_box_depth,
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=10,
        ),
        bgcolor="#1F2937",
        border_radius=15,
        padding=12,
        border=ft.border.all(1, "#374151"),
        shadow=ft.BoxShadow(blur_radius=15, color="#000000"),
    )

    images_row = ft.Row(
        controls=[card_det, card_seg, card_depth],
        alignment=ft.MainAxisAlignment.CENTER,
        spacing=15,
        visible=False,
        wrap=True,
    )

    # Tablas
    table_results = ft.DataTable(
        columns=[
            ft.DataColumn(label=ft.Text("Producto Identificado", weight=ft.FontWeight.BOLD, color="#E5E7EB")),
            ft.DataColumn(label=ft.Text("Detecciones YOLO", weight=ft.FontWeight.BOLD, color="#E5E7EB"), numeric=True),
            ft.DataColumn(label=ft.Text("Stock Estimado Total", weight=ft.FontWeight.BOLD, color="#E5E7EB"), numeric=True),
        ],
        rows=[],
        width=1000,
        heading_row_color="#374151",
        border_radius=10,
        divider_thickness=0.5,
    )

    table_details = ft.DataTable(
        columns=[
            ft.DataColumn(label=ft.Text("ID Det.", weight=ft.FontWeight.BOLD, color="#E5E7EB")),
            ft.DataColumn(label=ft.Text("SKU Detectado", weight=ft.FontWeight.BOLD, color="#E5E7EB")),
            ft.DataColumn(label=ft.Text("Confianza", weight=ft.FontWeight.BOLD, color="#E5E7EB"), numeric=True),
            ft.DataColumn(label=ft.Text("Capacidad Máx.", weight=ft.FontWeight.BOLD, color="#E5E7EB"), numeric=True),
            ft.DataColumn(label=ft.Text("Faltantes", weight=ft.FontWeight.BOLD, color="#E5E7EB"), numeric=True),
            ft.DataColumn(label=ft.Text("Stock Estimado", weight=ft.FontWeight.BOLD, color="#E5E7EB"), numeric=True),
            ft.DataColumn(label=ft.Text("Prof. SKU / Rejilla", weight=ft.FontWeight.BOLD, color="#E5E7EB")),
        ],
        rows=[],
        width=1000,
        heading_row_color="#374151",
        border_radius=10,
        divider_thickness=0.5,
    )
    
    table_card = ft.Container(
        content=ft.Column(
            controls=[
                ft.Text("Resumen General de Inventario", size=20, weight=ft.FontWeight.W_600, color="#38BDF8"),
                ft.Divider(color="#374151"),
                ft.Container(
                    content=ft.Column([table_results], scroll=ft.ScrollMode.AUTO),
                    height=180,
                ),
                ft.Container(height=15),
                ft.Text("Desglose Detallado por Objeto Detectado", size=18, weight=ft.FontWeight.W_600, color="#F59E0B"),
                ft.Text("Muestra la estimación individual generada para cada Bounding Box de YOLO", size=13, color="#9CA3AF"),
                ft.Divider(color="#374151"),
                ft.Container(
                    content=ft.Column([table_details], scroll=ft.ScrollMode.AUTO),
                    height=320,
                )
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER
        ),
        bgcolor="#1F2937",
        border_radius=15,
        border=ft.border.all(1, "#374151"),
        padding=25,
        shadow=ft.BoxShadow(blur_radius=20, spread_radius=2, color="#000000"),
        visible=False,
        width=1100
    )

    def on_file_selected(e: ft.FilePickerResultEvent):
        if not e.files or len(e.files) == 0:
            return

        file_path = e.files[0].path
        images_row.visible = False
        table_card.visible = False
        alert_banner.visible = False
        loading_ring.visible = True
        page.update()

        serie_val = input_serie.value.strip()
        if not serie_val:
            serie_val = "DEFAULT"

        try:
            # Reconocer el mime_type para que el backend detecte si es video o imagen
            mime_type, _ = mimetypes.guess_type(file_path)
            if not mime_type:
                mime_type = "application/octet-stream"

            with open(file_path, "rb") as f:
                response = requests.post(
                    API_URL,
                    files={"file": (e.files[0].name, f, mime_type)},
                    data={"numero_serie": serie_val},
                    timeout=180, # Timeout extendido para procesar videos grandes
                )

            if response.status_code == 200:
                data = response.json()
                es_video = data.get("tipo") == "video" or "video_det_base64" in data

                if es_video:
                    label_det.value = "Video: Detección (Boxes)"
                    label_seg.value = "Video: Segmentación (Polígonos)"
                    label_depth.value = "Video: Mapa de Profundidad"

                    # Guardar videos temporales locales para reproducción fluida con Flet
                    temp_dir = tempfile.gettempdir()
                    ts = int(time.time())
                    v_det_path = os.path.join(temp_dir, f"audit_det_{ts}.mp4")
                    v_seg_path = os.path.join(temp_dir, f"audit_seg_{ts}.mp4")
                    v_depth_path = os.path.join(temp_dir, f"audit_depth_{ts}.mp4")

                    if data.get("video_det_base64"):
                        with open(v_det_path, "wb") as f_det:
                            f_det.write(base64.b64decode(data["video_det_base64"]))
                    if data.get("video_seg_base64"):
                        with open(v_seg_path, "wb") as f_seg:
                            f_seg.write(base64.b64decode(data["video_seg_base64"]))
                    if data.get("video_depth_base64"):
                        with open(v_depth_path, "wb") as f_depth:
                            f_depth.write(base64.b64decode(data["video_depth_base64"]))

                    media_box_det.content = ft.Video(
                        playlist=[ft.VideoMedia(v_det_path)],
                        autoplay=True,
                        show_controls=True,
                        width=330,
                        height=260,
                        fit=ft.ImageFit.CONTAIN,
                        muted=True,
                    )
                    media_box_seg.content = ft.Video(
                        playlist=[ft.VideoMedia(v_seg_path)],
                        autoplay=True,
                        show_controls=True,
                        width=330,
                        height=260,
                        fit=ft.ImageFit.CONTAIN,
                        muted=True,
                    )
                    media_box_depth.content = ft.Video(
                        playlist=[ft.VideoMedia(v_depth_path)],
                        autoplay=True,
                        show_controls=True,
                        width=330,
                        height=260,
                        fit=ft.ImageFit.CONTAIN,
                        muted=True,
                    )
                else:
                    label_det.value = "Detección (Boxes)"
                    label_seg.value = "Segmentación (Polígonos)"
                    label_depth.value = "Mapa de Profundidad"

                    img_result_det.src_base64 = data["imagen_base64"]
                    img_result_seg.src_base64 = data.get("imagen_seg_base64", "")
                    img_result_depth.src_base64 = data.get("imagen_depth_base64", "")

                    media_box_det.content = img_result_det
                    media_box_seg.content = img_result_seg
                    media_box_depth.content = img_result_depth

                images_row.visible = True

                table_results.rows.clear()
                table_details.rows.clear()

                detecciones = data.get("detecciones", [])
                detalles = data.get("detalle_detecciones", [])

                if detecciones:
                    total_piezas = 0
                    total_visibles = 0
                    for item in detecciones:
                        stock = item.get("stock_estimado", 0)
                        visibles = item.get("visibles", 0)
                        total_piezas += stock
                        total_visibles += visibles

                        table_results.rows.append(
                            ft.DataRow(
                                cells=[
                                    ft.DataCell(ft.Text(item.get("etiqueta", "Desconocido"), color="#D1D5DB")),
                                    ft.DataCell(ft.Text(str(visibles), weight=ft.FontWeight.BOLD, color="#38BDF8")),
                                    ft.DataCell(ft.Text(str(stock), weight=ft.FontWeight.BOLD, color="#34D399")),
                                ]
                            )
                        )

                if detalles:
                    for det in detalles:
                        prof_sku = det.get("profundidad_sku")
                        prof_rej = det.get("profundidad_rejilla")
                        txt_prof = f"{prof_sku} / {prof_rej}" if prof_sku and prof_rej else "N/A"

                        table_details.rows.append(
                            ft.DataRow(
                                cells=[
                                    ft.DataCell(ft.Text(det.get("id_deteccion", "-"), color="#9CA3AF")),
                                    ft.DataCell(ft.Text(det.get("sku", "Desconocido"), color="#D1D5DB")),
                                    ft.DataCell(ft.Text(f"{det.get('confianza', 0):.2f}", color="#38BDF8")),
                                    ft.DataCell(ft.Text(str(det.get("capacidad_maxima", 0)), color="#9CA3AF")),
                                    ft.DataCell(ft.Text(str(det.get("faltantes_calculados", 0)), color="#EF4444")),
                                    ft.DataCell(ft.Text(str(det.get("inventario_estimado", 0)), weight=ft.FontWeight.BOLD, color="#34D399")),
                                    ft.DataCell(ft.Text(txt_prof, size=11, color="#9CA3AF")),
                                ]
                            )
                        )

                    table_card.visible = True
                    tipo_str = "Video" if es_video else "Foto"
                    alert_banner.content.value = f"✅ Auditoría completada ({tipo_str} - {serie_val}): {total_visibles} detectados por YOLO / {total_piezas} piezas estimadas."
                    alert_banner.bgcolor = "#064E3B"
                    alert_banner.border = ft.border.all(1, "#059669")
                    alert_banner.content.color = "#6EE7B7"
                else:
                    alert_banner.content.value = "⚠️ Congelador detectado vacío o productos irreconocibles."
                    alert_banner.bgcolor = "#451A03"
                    alert_banner.border = ft.border.all(1, "#D97706")
                    alert_banner.content.color = "#FCD34D"
            else:
                detail = response.json().get("detail", "Error desconocido")
                alert_banner.content.value = f"❌ Error en el backend: {detail}"
                alert_banner.bgcolor = "#450A0A"
                alert_banner.border = ft.border.all(1, "#DC2626")
                alert_banner.content.color = "#FCA5A5"

        except Exception as err:
            alert_banner.content.value = f"❌ Error en la solicitud: {err}"
            alert_banner.bgcolor = "#450A0A"
            alert_banner.border = ft.border.all(1, "#DC2626")
            alert_banner.content.color = "#FCA5A5"

        finally:
            alert_banner.visible = True
            loading_ring.visible = False
            page.update()

    file_picker = ft.FilePicker(on_result=on_file_selected)
    page.overlay.append(file_picker)

    upload_zone = ft.Container(
        content=ft.Column(
            controls=[
                ft.Icon(ft.Icons.CLOUD_UPLOAD_OUTLINED, size=55, color="#38BDF8"),
                ft.Text("Haz clic aquí para subir una foto o video", size=18, weight=ft.FontWeight.W_500, color="#E5E7EB"),
                ft.Text("Soporta JPG, PNG, MP4, AVI, MOV", size=13, color="#6B7280"),
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        ),
        width=1100,
        height=180,
        bgcolor="#1F2937",
        border_radius=15,
        border=ft.border.all(2, "#374151"),
        ink=True,
        # Cambiado a ANY o personalizado para admitir video
        on_click=lambda _: file_picker.pick_files(
            allow_multiple=False,
            allowed_extensions=["png", "jpg", "jpeg", "mp4", "avi", "mov", "webm"]
        ),
        shadow=ft.BoxShadow(blur_radius=15, color="#000000"),
    )

    page.add(
        ft.Column(
            controls=[
                ft.Container(height=15),
                input_serie,
                upload_zone,
                loading_ring,
                alert_banner,
                images_row,
                table_card,
                ft.Container(height=30),
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=25,
        )
    )

if __name__ == "__main__":
    ft.app(target=main)
