import cv2
import numpy as np
from PIL import Image


def crear_panoramica_desde_video(
    video_path: str,
    max_frames: int = 20
):
    cap = cv2.VideoCapture(video_path)

    frames_scored = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        score = cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()

        frames_scored.append(
            (score, frame)
        )

    cap.release()

    if len(frames_scored) < 2:
        raise RuntimeError(
            "No hay suficientes frames para generar panorámica"
        )

    frames_scored.sort(
        key=lambda x: x[0],
        reverse=True
    )

    mejores_frames = [
        frame
        for _, frame in frames_scored[:max_frames]
    ]

    stitcher = cv2.Stitcher_create(
        cv2.Stitcher_PANORAMA
    )

    status, panorama = stitcher.stitch(
        mejores_frames
    )

    if status != cv2.Stitcher_OK:
        raise RuntimeError(
            f"Error creando panorámica. Status={status}"
        )

    panorama_rgb = cv2.cvtColor(
        panorama,
        cv2.COLOR_BGR2RGB
    )

    return Image.fromarray(
        panorama_rgb
    )