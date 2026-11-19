"""
FastAPI backend for Box Dimensioning Mobile PWA.
Accepts video upload from mobile, runs the pipeline, returns JSON + 3D wireframe.
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import tempfile, os, sys, json
from pathlib import Path

# Make sure parent project is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

app = FastAPI(title="Box Dimensioning API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve PWA static files (HTML, JS, icons, manifest)
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_pwa():
    """Serve the main PWA shell."""
    index = Path(__file__).parent / "static" / "index.html"
    return FileResponse(index)


@app.get("/manifest.json")
async def manifest():
    return JSONResponse({
        "name": "Box Dimensioner",
        "short_name": "BoxDim",
        "description": "Measure any box with your phone camera",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#0f172a",
        "theme_color": "#6366f1",
        "orientation": "portrait",
        "icons": [
            {"src": "/static/icon-192.png", "sizes": "192x192", "type": "image/png"},
            {"src": "/static/icon-512.png", "sizes": "512x512", "type": "image/png"},
        ]
    })


@app.post("/api/measure")
async def measure_box(
    video: UploadFile = File(...),
    phone_make: str = Form(default=""),
    phone_model: str = Form(default=""),
    scale_ref: str = Form(default="none"),
):
    """
    Receive a video file from mobile and return measurements.
    """
    # Save upload to temp file
    suffix = Path(video.filename).suffix if video.filename else ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await video.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        from box_dimensioning.pipeline.custom_video_processor import process_custom_mobile_video

        result = process_custom_mobile_video(
            video_path=tmp_path,
            phone_make=phone_make or None,
            phone_model=phone_model or None,
            scale_reference=scale_ref if scale_ref != "none" else None,
            use_deep_detector=True,
        )

        return JSONResponse({
            "success": True,
            "length_cm": result.get("length_cm"),
            "width_cm": result.get("width_cm"),
            "height_cm": result.get("height_cm"),
            "volume_cm3": result.get("volume_cm3"),
            "volumetric_weight_kg": result.get("volumetric_weight_kg"),
            "confidence": result.get("confidence", 0),
            "method": result.get("method", "unknown"),
            "scale_source": result.get("scale_source", "unknown"),
            "frames_used": result.get("frames_used", 0),
            "wireframe_data": result.get("wireframe_data"),  # for 3D render on client
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "2.0"}
