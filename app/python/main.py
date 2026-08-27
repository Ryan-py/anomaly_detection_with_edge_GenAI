import io
import threading
import uuid
from pathlib import Path

import cv2
from fastapi import File, Form, HTTPException, Response, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image

from arduino.app_bricks.web_ui import WebUI
from arduino.app_peripherals.camera import (
    Camera,
    CameraConfigError,
    CameraOpenError,
    CameraReadError,
    V4LCamera,
)
from arduino.app_utils import App, Bridge, Logger
from arduino.app_utils.image import compress_to_jpeg

import database
import inference_engine
import report_generator

logger = Logger("EdgeAnomalyDashboard")

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "assets" / "static"
CAPTURES_DIR = STATIC_DIR / "captures"
CAPTURES_DIR.mkdir(parents=True, exist_ok=True)

database.init_db()

ui = WebUI()

_camera_lock = threading.Lock()
_active_camera = None
_active_camera_device = None


def _get_or_create_camera(device: int):
    global _active_camera, _active_camera_device
    with _camera_lock:
        if _active_camera is not None and _active_camera_device == device:
            return _active_camera
        if _active_camera is not None:
            try:
                _active_camera.stop()
            except Exception as e:
                logger.warning(f"Error stopping previous camera: {e}")
        # fps=30 matches this camera's real hardware rate (confirmed via profiling —
        # it silently negotiates 30fps regardless of a lower request). Note: the
        # dominant per-frame cost (~85ms, independent of resolution) is this webcam's
        # own auto-exposure lengthening its shutter time under the current ambient
        # lighting, not decode/resize/encode (all under 5ms combined) — better
        # lighting on the scene is what actually raises the achievable frame rate.
        camera = Camera(device, resolution=(640, 480), fps=30)
        camera.start()
        try:
            # V4L2 UVC controls persist on the physical device across app restarts
            # (only the software session restarts, not the camera's own state) —
            # this restores normal auto-exposure in case a prior run left the
            # device in manual mode with a fixed dark exposure value.
            camera._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # V4L2: 3 = auto
        except Exception as e:
            logger.warning(f"Could not reset auto-exposure: {e}")
        _active_camera = camera
        _active_camera_device = device
        return _active_camera


def _stop_active_camera() -> None:
    global _active_camera, _active_camera_device
    with _camera_lock:
        if _active_camera is not None:
            try:
                _active_camera.stop()
            except Exception as e:
                logger.warning(f"Error stopping camera: {e}")
            _active_camera = None
            _active_camera_device = None


def _run_and_store_inspection(instance_id: int, pil_image: Image.Image) -> dict:
    instance = database.get_instance(instance_id)
    if instance is None:
        raise HTTPException(status_code=404, detail="instance not found")

    uid = uuid.uuid4().hex[:12]
    raw_rel = f"captures/insp_{uid}_raw.jpg"
    heatmap_rel = f"captures/insp_{uid}_heatmap.png"
    pil_image.convert("RGB").save(STATIC_DIR / raw_rel, "JPEG", quality=90)

    threshold = database.get_critical_threshold()
    try:
        result = inference_engine.run_inspection(pil_image, str(STATIC_DIR / heatmap_rel), threshold)
    except inference_engine.ModelLoadError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except inference_engine.InferenceError as e:
        raise HTTPException(status_code=500, detail=str(e))

    row = database.insert_inspection(
        instance_id,
        result["error_percentage"],
        raw_rel,
        heatmap_rel,
        result["is_critical"],
    )

    try:
        Bridge.call("set_matrix_alert", 1 if result["is_critical"] else 0)
    except Exception as e:
        logger.warning(f"LED matrix alert not updated: {e}")

    return {
        **row,
        "raw_image_url": f"/static/{raw_rel}",
        "heatmap_url": f"/static/{heatmap_rel}",
    }


def api_list_components():
    return database.list_components()


def api_list_instances(component_id: int | None = None):
    return database.list_instances(component_id)


def api_create_instance(body: dict):
    try:
        return database.create_instance(body["component_id"], body["label"])
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


def api_list_cameras():
    return {"devices": V4LCamera.list_devices()}


def api_inspect_camera(body: dict):
    instance_id = body["instance_id"]
    device = body.get("device", 0)

    try:
        camera = _get_or_create_camera(device)
        frame = camera.capture()
        if frame is None:
            raise HTTPException(status_code=503, detail="camera returned no frame")
    except (CameraOpenError, CameraReadError, CameraConfigError) as e:
        raise HTTPException(status_code=503, detail=f"camera not available: {e}")
    finally:
        # Turn the camera off as soon as the shot is taken, rather than leaving
        # it running for the duration of inference or indefinitely afterward.
        _stop_active_camera()

    pil_image = Image.fromarray(frame[:, :, ::-1])  # BGR -> RGB
    return _run_and_store_inspection(instance_id, pil_image)


def api_camera_stop():
    _stop_active_camera()
    return {"stopped": True}


def api_camera_preview(device: int = 0):
    try:
        camera = _get_or_create_camera(device)
    except (CameraOpenError, CameraConfigError) as e:
        raise HTTPException(status_code=503, detail=f"camera not available: {e}")

    def generate_frames():
        try:
            while True:
                frame = camera.capture()
                if frame is None:
                    continue
                jpeg = compress_to_jpeg(frame, quality=55)
                if jpeg is None:
                    continue
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
        except Exception as e:
            logger.warning(f"Camera preview stream ended: {e}")

    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store"},
    )


async def api_inspect_upload(instance_id: int = Form(...), file: UploadFile = File(...)):
    data = await file.read()
    try:
        pil_image = Image.open(io.BytesIO(data))
    except Exception:
        raise HTTPException(status_code=400, detail="invalid image file")
    return _run_and_store_inspection(instance_id, pil_image)


def api_instance_timeseries(instance_id: int):
    instance = database.get_instance(instance_id)
    if instance is None:
        raise HTTPException(status_code=404, detail="instance not found")
    return {"instance": instance, "series": database.get_timeseries(instance_id)}


def api_compare(instance_a: int, instance_b: int):
    return database.get_comparison_series(instance_a, instance_b)


def api_get_inspection(inspection_id: int):
    inspection = database.get_inspection(inspection_id)
    if inspection is None:
        raise HTTPException(status_code=404, detail="inspection not found")
    return inspection


def api_download_report(inspection_id: int):
    inspection = database.get_inspection(inspection_id)
    if inspection is None:
        raise HTTPException(status_code=404, detail="inspection not found")

    inspection = {
        **inspection,
        "raw_image_abs_path": str(STATIC_DIR / inspection["raw_image_path"]),
        "heatmap_abs_path": str(STATIC_DIR / inspection["heatmap_path"]),
    }
    buffer = report_generator.generate_inspection_report(inspection)
    return Response(
        content=buffer.getvalue(),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="inspection_{inspection_id}.pdf"'},
    )


def api_status():
    status = inference_engine.model_status()
    return {
        "model_loaded": status["loaded"],
        "model_error": status["error"],
        "cameras": V4LCamera.list_devices(),
        "critical_threshold": database.get_critical_threshold(),
    }


def api_get_settings():
    return {"critical_threshold": database.get_critical_threshold()}


def api_update_settings(body: dict):
    value = body.get("critical_threshold")
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not (0 < value <= 100):
        raise HTTPException(status_code=400, detail="critical_threshold must be a number between 0 and 100")
    database.set_critical_threshold(float(value))
    return {"critical_threshold": float(value)}


def _delete_inspection_files(row: dict) -> None:
    for key in ("raw_image_path", "heatmap_path"):
        try:
            (STATIC_DIR / row[key]).unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Failed to delete file for inspection {row['id']} ({key}): {e}")


def api_delete_inspection(inspection_id: int):
    row = database.delete_inspection(inspection_id)
    if row is None:
        raise HTTPException(status_code=404, detail="inspection not found")
    _delete_inspection_files(row)
    return {"deleted": True, "id": inspection_id}


def api_clear_instance_inspections(instance_id: int):
    if database.get_instance(instance_id) is None:
        raise HTTPException(status_code=404, detail="instance not found")
    rows = database.delete_inspections_for_instance(instance_id)
    for row in rows:
        _delete_inspection_files(row)
    return {"deleted_count": len(rows)}


ui.expose_api("GET", "/api/components", api_list_components)
ui.expose_api("GET", "/api/instances", api_list_instances)
ui.expose_api("POST", "/api/instances", api_create_instance)
ui.expose_api("GET", "/api/cameras", api_list_cameras)
ui.expose_api("GET", "/api/camera/preview", api_camera_preview)
ui.expose_api("POST", "/api/camera/stop", api_camera_stop)
ui.expose_api("POST", "/api/inspect/camera", api_inspect_camera)
ui.expose_api("POST", "/api/inspect/upload", api_inspect_upload)
ui.expose_api("GET", "/api/instances/{instance_id}/timeseries", api_instance_timeseries)
ui.expose_api("GET", "/api/compare", api_compare)
ui.expose_api("GET", "/api/inspections/{inspection_id}", api_get_inspection)
ui.expose_api("DELETE", "/api/inspections/{inspection_id}", api_delete_inspection)
ui.expose_api("DELETE", "/api/instances/{instance_id}/inspections", api_clear_instance_inspections)
ui.expose_api("GET", "/api/reports/download/{inspection_id}", api_download_report)
ui.expose_api("GET", "/api/status", api_status)
ui.expose_api("GET", "/api/settings", api_get_settings)
ui.expose_api("PUT", "/api/settings", api_update_settings)

try:
    inference_engine.warm_up()
except Exception as e:
    logger.warning(f"Model warm-up failed (will retry lazily on first inspection): {e}")

App.run()
