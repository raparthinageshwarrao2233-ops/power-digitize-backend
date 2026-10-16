"""
POWER DIGITIZE — FastAPI Backend
Main application entry point
"""

import os
import json
import re
import time
import uuid
import shutil
from pathlib import Path
from typing import Optional, Dict, List

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Header, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response

from engine.pipeline import DigitizingPipeline
from engine.dst_encoder import DSTEncoder
from engine.stitch_renderer import StitchRenderer
from engine.text_engine import TextEmbroideryEngine
from engine.monogram_engine import MonogramEngine
from engine.validator import DSTValidator
from engine.thread_calculator import ThreadCalculator
from engine.qa_engine import QAEngine
from engine.refinement_engine import RefinementEngine
from engine.stitch_generators import Stitch, StitchBlock
from engine.photo_stitch_engine import PhotoStitchEngine
import pyembroidery

app = FastAPI(
    title="POWER DIGITIZE API",
    description="AI-Powered Embroidery Digitizing & DST Generation",
    version="1.0.0",
)

# ─── Optional API Key Auth (via X-API-Key header) ───
# Set env var to enable auth; leave empty/undefined for local-dev no-auth mode
REQUIRED_API_KEY: Optional[str] = os.getenv("POWER_DIGITIZE_API_KEY") or None

# ─── In-memory Rate Limiter (safe, no extra deps) ───
# Per-IP sliding window counters (60s window)
RATE_LIMIT_WINDOW = 60.0
RATE_LIMIT_MAX = 300              # per IP per minute (health checks + normal use)
UPLOAD_RATE_LIMIT_MAX = 60        # uploads per IP per minute (disk-heavy)
HEAVY_RATE_LIMIT_MAX = 120        # digitize/optimize/validate/refine calls per IP per minute (CPU-heavy)

_rate_counters: Dict[str, list] = {}

def _check_rate_limit(client_id: str, max_hits: int, window: float = RATE_LIMIT_WINDOW):
    """Check rate limit for a client identifier. Raises HTTP 429 if over quota."""
    now = time.time()
    if client_id not in _rate_counters:
        _rate_counters[client_id] = []
    # Prune expired hits
    hits = [t for t in _rate_counters[client_id] if now - t < window]
    if len(hits) >= max_hits:
        _rate_counters[client_id] = hits
        retry_after = int(window - (now - hits[0])) + 1
        raise HTTPException(status_code=429, detail=f"Rate limit exceeded. Retry after {retry_after}s.")
    hits.append(now)
    _rate_counters[client_id] = hits

async def _client_ip(request: Request) -> str:
    """Use the socket peer address; forwarded headers are untrusted by default."""
    client = request.client
    return client.host if client else "unknown"

async def require_auth_and_rate(
    request: Request,
    max_hits: int = RATE_LIMIT_MAX,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
) -> str:
    """Combined dependency: verify API key (if configured) + check rate limit. Returns client_id."""
    if REQUIRED_API_KEY is not None and x_api_key != REQUIRED_API_KEY:
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header.")
    client_id = await _client_ip(request)
    _check_rate_limit(client_id, max_hits)
    return client_id


# ─── Typed parameterized rate-limit dependencies (FastAPI Depends-safe) ───
# NOTE: Depends(lambda ...) with untyped params breaks FastAPI DI. Use these named helpers:
async def _rate_dep(request: Request, max_hits: int) -> str:
    return await require_auth_and_rate(request, max_hits=max_hits)

def rate_limit(max_hits: int):
    """Factory: create a FastAPI-compatible dependency with the given rate-limit cap."""
    async def _dep(request: Request, x_api_key: Optional[str] = Header(None, alias="X-API-Key")) -> str:
        if REQUIRED_API_KEY is not None and x_api_key != REQUIRED_API_KEY:
            raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header.")
        client_id = await _client_ip(request)
        _check_rate_limit(client_id, max_hits)
        return client_id
    return _dep

# Pre-built deps for the three tiers
RATE_GENERAL  = rate_limit(RATE_LIMIT_MAX)
RATE_UPLOAD   = rate_limit(UPLOAD_RATE_LIMIT_MAX)
RATE_HEAVY    = rate_limit(HEAVY_RATE_LIMIT_MAX)

# CORS for frontend
allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Storage directories
UPLOAD_DIR = Path("storage/uploads")
PROCESSED_DIR = Path("storage/processed")
DST_DIR = Path("storage/dst")
PREVIEW_DIR = Path("storage/previews")

for d in [UPLOAD_DIR, PROCESSED_DIR, DST_DIR, PREVIEW_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Pipeline instance
pipeline = DigitizingPipeline()
text_engine = TextEmbroideryEngine()
monogram_engine = MonogramEngine()
validator = DSTValidator()
thread_calc = ThreadCalculator()
qa_engine = QAEngine()
refinement_engine = RefinementEngine()


def validate_design_id(design_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]{4,36}", design_id):
        raise HTTPException(status_code=400, detail="Invalid design ID format.")
    return design_id


async def read_limited_upload(file: UploadFile, max_bytes: int) -> bytes:
    chunks = []
    total = 0
    while chunk := await file.read(min(1024 * 1024, max_bytes - total + 1)):
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(status_code=413, detail="Uploaded file exceeds the size limit.")
        chunks.append(chunk)
    return b"".join(chunks)


# ─── Health Check ───
@app.get("/api/health")
async def health():
    status = "ok"
    details = {}
    
    try:
        import pyembroidery
        details["pyembroidery"] = "ok"
    except ImportError:
        status = "error"
        details["pyembroidery"] = "missing"
        
    try:
        import cv2
        details["opencv"] = "ok"
    except ImportError:
        status = "error"
        details["opencv"] = "missing"
        
    try:
        test_file = UPLOAD_DIR / ".health_test"
        test_file.touch()
        test_file.unlink()
        details["storage"] = "writable"
    except Exception as e:
        status = "error"
        details["storage"] = f"error: {str(e)}"
        
    if status == "error":
        raise HTTPException(status_code=503, detail={"status": status, "checks": details})
        
    return {"status": status, "service": "POWER DIGITIZE", "checks": details}


# ─── Upload & Analyze ───
@app.post("/api/designs/upload")
async def upload_design(
    file: UploadFile = File(...),
    _client: str = Depends(RATE_UPLOAD),
    # Upload endpoint uses RATE_UPLOAD tier (6 uploads/min per IP, 11 endpoints found)
):
    """Upload an image and start the digitizing pipeline."""

    # Validate MIME type
    allowed_types = ["image/png", "image/jpeg", "image/webp"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Please upload PNG, JPG, or WEBP. (SVG not supported yet)",
        )

    # Validate file size (max 20MB)
    contents = await read_limited_upload(file, 20 * 1024 * 1024)
    try:
        from io import BytesIO
        from PIL import Image
        with Image.open(BytesIO(contents)) as image:
            if image.format not in {"PNG", "JPEG", "WEBP"}:
                raise HTTPException(status_code=400, detail="File content must be PNG, JPG, or WEBP.")
            if image.width * image.height > 40_000_000:
                raise HTTPException(status_code=413, detail="Image dimensions exceed the 40-megapixel limit.")
            image.verify()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid or corrupt image file.") from exc

    # Save uploaded file
    design_id = str(uuid.uuid4())[:12]
    ext = Path(file.filename or "image.png").suffix.lower()
    if ext not in [".png", ".jpg", ".jpeg", ".webp"]:
        ext = ".png"

    upload_path = UPLOAD_DIR / f"{design_id}{ext}"
    with open(upload_path, "wb") as f:
        f.write(contents)

    return {
        "design_id": design_id,
        "filename": file.filename,
        "size": len(contents),
        "status": "uploaded",
    }


# ─── Photo Digitizing Pipeline (PhotoStitch – for faces/portraits/icons) ───
@app.post("/api/designs/{design_id}/photo-digitize")
async def photo_digitize_design(
    design_id: str,
    design_width: float  = Query(default=80.0,  gt=1, le=500, allow_inf_nan=False),
    design_height: float = Query(default=80.0,  gt=1, le=500, allow_inf_nan=False),
    color_count: int     = Query(default=8,      ge=2, le=16),
    remove_bg: bool      = Query(default=True),
    detail_preservation: str = Query(default="high"),
    _client: str = Depends(RATE_HEAVY),
):
    """Convert ANY photo (face, portrait, icon, logo) to embroidery using tone-density PhotoStitch.

    Uses industry-standard techniques:
    - Grayscale tone → stitch density mapping (dark=dense, light=sparse)
    - Per-region variable-density tatami fill
    - CIELAB Delta-E thread color matching
    - Canny-edge detail pass for fine features
    """
    validate_design_id(design_id)

    upload_path = None
    for ext in [".png", ".jpg", ".jpeg", ".webp"]:
        p = UPLOAD_DIR / f"{design_id}{ext}"
        if p.exists():
            upload_path = p
            break

    if not upload_path:
        raise HTTPException(status_code=404, detail="Design not found. Upload first.")

    try:
        engine = PhotoStitchEngine()
        result = engine.process(
            image_path=str(upload_path),
            target_width_mm=design_width,
            target_height_mm=design_height,
            color_count=color_count,
            remove_bg=remove_bg,
            detail_preservation=detail_preservation,
        )

        stitch_data = result["stitch_blocks"]

        if not stitch_data:
            raise HTTPException(status_code=422, detail="No stitchable regions found in image.")

        # DST
        dst_path = DST_DIR / f"{design_id}.dst"
        DSTEncoder.encode(stitch_data, str(dst_path))

        # Previews
        preview_path      = PREVIEW_DIR / f"{design_id}_preview.png"
        flat_preview_path = PREVIEW_DIR / f"{design_id}_flat.png"
        StitchRenderer.render_preview(stitch_data, str(preview_path),      width=800, height=800, render_mode="thread")
        StitchRenderer.render_preview(stitch_data, str(flat_preview_path), width=800, height=800, render_mode="flat")

        return {
            "design_id":    design_id,
            "status":       "completed",
            "mode":         "photo_stitch",
            "dst_path":     f"/api/designs/{design_id}/export/dst",
            "preview_path": f"/api/designs/{design_id}/preview",
            "stats":        result["stats"],
            "colors":       result["colors"],
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Photo digitizing failed: {str(e)}")


# ─── Full Digitizing Pipeline ───
@app.post("/api/designs/{design_id}/digitize")
async def digitize_design(
    design_id: str,
    hoop_width: float = Query(default=100.0, gt=1, le=500, allow_inf_nan=False),
    hoop_height: float = Query(default=100.0, gt=1, le=500, allow_inf_nan=False),
    design_width: float = Query(default=80.0, gt=1, le=500, allow_inf_nan=False),
    design_height: float = Query(default=52.0, gt=1, le=500, allow_inf_nan=False),
    density: float = Query(default=0.4, gt=0, le=5, allow_inf_nan=False),
    color_count: int = Query(default=8, ge=2, le=16),
    stitch_length: float = Query(default=3.0, gt=0, le=12, allow_inf_nan=False),
    pull_compensation: float = Query(default=0.3, ge=0, le=5, allow_inf_nan=False),
    underlay: str = Query(default="auto"),
    remove_bg: bool = Query(default=True),
    repeat_count: int = Query(default=1, ge=1, le=5),
    run_mode: str = Query(default="single"),
    speed_spm: int = Query(default=800, ge=200, le=1500),
    machine_wattage: float = Query(default=100.0, ge=10, le=5000),
    electricity_tariff: float = Query(default=8.0, ge=0.1, le=100),
    detail_preservation: str = Query(default="high"),
    _client: str = Depends(RATE_HEAVY),
):
    """Run the full digitizing pipeline on an uploaded image with native micro-detail preservation."""
    validate_design_id(design_id)

    # Find uploaded file
    upload_path = None
    for ext in [".png", ".jpg", ".jpeg", ".webp", ".svg"]:
        p = UPLOAD_DIR / f"{design_id}{ext}"
        if p.exists():
            upload_path = p
            break

    if not upload_path:
        raise HTTPException(status_code=404, detail="Design not found. Upload first.")

    try:
        # Run pipeline
        result = pipeline.process(
            image_path=str(upload_path),
            design_id=design_id,
            target_width_mm=design_width,
            target_height_mm=design_height,
            hoop_width_mm=hoop_width,
            hoop_height_mm=hoop_height,
            density=density,
            color_count=color_count,
            stitch_length=stitch_length,
            pull_compensation=pull_compensation,
            underlay_mode=underlay,
            remove_bg=remove_bg,
            repeat_count=repeat_count,
            run_mode=run_mode,
            machine_speed_spm=speed_spm,
            machine_wattage=machine_wattage,
            electricity_tariff=electricity_tariff,
            detail_preservation=detail_preservation,
        )

        # Generate DST file
        dst_path = DST_DIR / f"{design_id}.dst"
        DSTEncoder.encode(result["stitch_data"], str(dst_path))

        # Generate preview images (both 3D thread and 2D flat)
        preview_path = PREVIEW_DIR / f"{design_id}_preview.png"
        flat_preview_path = PREVIEW_DIR / f"{design_id}_flat.png"
        StitchRenderer.render_preview(
            result["stitch_data"], str(preview_path), width=800, height=800, show_jumps=False, render_mode="thread"
        )
        StitchRenderer.render_preview(
            result["stitch_data"], str(flat_preview_path), width=800, height=800, show_jumps=False, render_mode="flat"
        )

        actual_stats = validator.analyze_dst(str(dst_path))
        quality = validator.validate(str(dst_path))
        roundtrip = validator.validate_roundtrip(str(dst_path), design_width, design_height)

        # Collect actual color segments
        color_segments = [c["hex"] for c in result.get("colors", []) if "hex" in c]
        if not color_segments:
            color_segments = [obj.get("color") for obj in result.get("objects", []) if obj.get("color")]

        # Combine actual stats with breakdown and power info
        merged_stats = {
            **actual_stats,
            "color_segments": color_segments,
            "breakdown": result["stats"].get("breakdown", {}),
            "timing": result["stats"].get("timing", {}),
            "power": result["stats"].get("power", {}),
            "cost": result["stats"].get("cost", {}),
            "fidelity": result["stats"].get("fidelity", {}),
        }

        return {
            "design_id": design_id,
            "status": "completed",
            "dst_path": f"/api/designs/{design_id}/export/dst",
            "preview_path": f"/api/designs/{design_id}/preview",
            "stats": merged_stats,
            "objects": result["objects"],
            "hierarchy": result.get("hierarchy", {}),
            "line_art_stats": result.get("line_art_stats", {}),
            "colors": result["colors"],
            "analysis": result["analysis"],
            "quality": quality,
            "roundtrip": roundtrip,
            "stitch_data_summary": {
                "total_stitches": actual_stats["stitch_count"],
                "total_objects": len(result["objects"]),
                "total_colors": actual_stats["color_count"],
                "protected_details": result.get("line_art_stats", {}).get("protected_details", 0),
            },
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Digitizing failed. Your original artwork is safe. Error: {str(e)}",
        )


@app.get("/api/designs/{design_id}/roundtrip")
async def get_design_roundtrip(
    design_id: str,
    requested_width: float = Query(default=80.0),
    requested_height: float = Query(default=52.0),
    _client: str = Depends(RATE_GENERAL),
):
    """Run DST round-trip test and physical dimension validation."""
    validate_design_id(design_id)
    dst_path = DST_DIR / f"{design_id}.dst"
    if not dst_path.exists():
        raise HTTPException(status_code=404, detail="DST not found.")

    return validator.validate_roundtrip(str(dst_path), requested_width, requested_height)


@app.get("/api/designs/{design_id}/production-report")
async def get_production_report(
    design_id: str,
    customer_name: str = Query(default="Valued Customer"),
    design_name: str = Query(default="Custom Embroidery"),
    spm: int = Query(default=800),
    wattage: float = Query(default=100.0),
    tariff: float = Query(default=8.0),
    _client: str = Depends(RATE_GENERAL),
):
    """Generate full production worksheet report."""
    validate_design_id(design_id)
    dst_path = DST_DIR / f"{design_id}.dst"
    if not dst_path.exists():
        raise HTTPException(status_code=404, detail="DST not found.")

    stats = validator.analyze_dst(str(dst_path))
    thread_data = thread_calc.calculate(str(dst_path))
    validation = validator.validate(str(dst_path))

    operating_hours = stats.get("estimated_time", 1) / 60.0
    energy_kwh = round((wattage * operating_hours) / 1000.0, 4)
    power_cost = round(energy_kwh * tariff, 2)
    thread_cost = thread_data["total_cost"]
    machine_cost = round(operating_hours * 25.0, 2)
    total_cost = round(thread_cost + power_cost + machine_cost, 2)

    return {
        "design_id": design_id,
        "design_name": design_name,
        "customer": customer_name,
        "physical_size": f"{stats['width']} mm × {stats['height']} mm",
        "stitch_count": stats["stitch_count"],
        "color_count": stats["color_count"],
        "jump_count": stats["jump_count"],
        "trim_count": stats["trim_count"],
        "total_thread_meters": thread_data["total_thread_m"],
        "estimated_runtime_minutes": stats["estimated_time"],
        "machine_profile": {
            "speed_spm": spm,
            "wattage_w": wattage,
            "electricity_tariff_inr": tariff,
            "energy_kwh": energy_kwh,
            "estimated_electricity_cost_inr": power_cost,
        },
        "cost_breakdown": {
            "thread_cost": thread_cost,
            "electricity_cost": power_cost,
            "machine_overhead_cost": machine_cost,
            "total_estimated_production_cost": total_cost,
            "currency": "INR",
        },
        "color_sequence": thread_data["colors"],
        "quality_status": validation["overall"],
        "compatible_hoops": validation["compatible_hoops"],
        "disclaimer": "Physical output may vary depending on machine, fabric, thread, stabilizer and machine calibration.",
    }


# ─── Separate Background Removal Preview Endpoint ───
@app.post("/api/designs/{design_id}/remove-bg")
async def remove_background_preview(
    design_id: str,
    _client: str = Depends(RATE_HEAVY),
):
    """Run background removal on uploaded artwork and return clean PNG preview URL."""
    validate_design_id(design_id)

    upload_path = None
    for ext in [".png", ".jpg", ".jpeg", ".webp"]:
        p = UPLOAD_DIR / f"{design_id}{ext}"
        if p.exists():
            upload_path = p
            break

    if not upload_path:
        raise HTTPException(status_code=404, detail="Design not found.")

    clean_path = PROCESSED_DIR / f"{design_id}_clean.png"
    from engine.background_remover import BackgroundRemover
    bg = BackgroundRemover()
    info = bg.remove(str(upload_path), str(clean_path))

    return {
        "design_id": design_id,
        "status": "bg_removed",
        "clean_image_url": f"/api/designs/{design_id}/clean-image",
        "info": info,
    }


@app.get("/api/designs/{design_id}/clean-image")
async def get_clean_image(design_id: str):
    """Serve the background-removed or original uploaded PNG image."""
    validate_design_id(design_id)
    clean_path = PROCESSED_DIR / f"{design_id}_clean.png"
    if clean_path.exists():
        return FileResponse(str(clean_path), media_type="image/png")
    
    for ext in [".png", ".jpg", ".jpeg", ".webp", ".svg"]:
        p = UPLOAD_DIR / f"{design_id}{ext}"
        if p.exists():
            media_type = "image/svg+xml" if ext == ".svg" else f"image/{ext.replace('.', '')}"
            return FileResponse(str(p), media_type=media_type)
    
    raise HTTPException(status_code=404, detail="Original or clean artwork not found.")


# ─── Get Design Stats ───
@app.get("/api/designs/{design_id}/stats")
async def get_design_stats(design_id: str, _client: str = Depends(RATE_GENERAL)):
    """Get stitch statistics for a digitized design."""
    validate_design_id(design_id)
    dst_path = DST_DIR / f"{design_id}.dst"
    if not dst_path.exists():
        raise HTTPException(status_code=404, detail="DST not found. Digitize first.")

    stats = validator.analyze_dst(str(dst_path))
    return stats


# ─── Get Preview ───
@app.get("/api/designs/{design_id}/preview")
async def get_preview(
    design_id: str,
    mode: str = Query(default="thread", pattern="^(flat|thread)$"),
):
    """Get the stitch preview image."""
    validate_design_id(design_id)
    preview_path = PREVIEW_DIR / f"{design_id}_preview.png"
    flat_preview_path = PREVIEW_DIR / f"{design_id}_flat.png"
    
    if mode == "thread" and preview_path.exists():
        return FileResponse(str(preview_path), media_type="image/png", headers={"Cache-Control": "no-store"})
    
    if mode == "flat" and flat_preview_path.exists():
        return FileResponse(str(flat_preview_path), media_type="image/png", headers={"Cache-Control": "no-store"})

    dst_file = DST_DIR / f"{design_id}.dst"
    if not dst_file.exists():
        raise HTTPException(status_code=404, detail="Preview not found.")

    # Fallback to render from DST if pre-rendered preview is missing
    metadata_path = DST_DIR / f"{design_id}.colors.json"
    colors = ["#111111", "#cf2028", "#aab3bd", "#ffffff"]
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as metadata_file:
            colors = json.load(metadata_file).get("segments", colors)

    pattern = pyembroidery.read(str(dst_file))
    if pattern is None:
        raise HTTPException(status_code=404, detail="Preview not found.")

    blocks = []
    color_index = 0
    block = StitchBlock(color=colors[0] if colors else "#111111", stitch_type="fill", name="Thread")
    for x, y, command in pattern.stitches:
        command &= 0xFF
        if command == pyembroidery.COLOR_CHANGE:
            if block.stitches:
                blocks.append(block)
            color_index += 1
            color = colors[min(color_index, len(colors) - 1)] if colors else "#111111"
            block = StitchBlock(color=color, stitch_type="fill", name="Thread")
        elif command in (pyembroidery.STITCH, pyembroidery.JUMP, pyembroidery.TRIM):
            flags = Stitch.JUMP if command == pyembroidery.JUMP else Stitch.TRIM if command == pyembroidery.TRIM else Stitch.NORMAL
            block.add(x / 10.0, y / 10.0, flags)
    if block.stitches:
        blocks.append(block)

    target_path = flat_preview_path if mode == "flat" else preview_path
    StitchRenderer.render_preview(blocks, str(target_path), width=800, height=800, show_jumps=False, render_mode=mode)
    return FileResponse(str(target_path), media_type="image/png", headers={"Cache-Control": "no-store"})


# ─── Export DST (with Machine Scale & Offset Calibration) ───

def cleanup_design_files(design_id: str):
    """Zero-retention privacy cleanup:
    Immediately after the DST download stream finishes, all temporary design files
    (artwork, preview images, DST binary, and QA data) are wiped from the disk.
    Zero database retention.
    """
    try:
        # Delete uploaded images
        for ext in [".png", ".jpg", ".jpeg", ".webp", ".svg"]:
            p = UPLOAD_DIR / f"{design_id}{ext}"
            if p.exists():
                try: p.unlink()
                except Exception: pass
        # Delete DST files
        for p in DST_DIR.glob(f"{design_id}*"):
            try: p.unlink()
            except Exception: pass
        # Delete preview images
        for p in PREVIEW_DIR.glob(f"{design_id}*"):
            try: p.unlink()
            except Exception: pass
        # Delete any storage records
        for p in (BASE_DIR / "storage").glob(f"**/*{design_id}*"):
            try:
                if p.is_file(): p.unlink()
            except Exception: pass
        print(f"[ZeroRetention] Cleaned all files for {design_id} immediately after download.")
    except Exception as e:
        print(f"[ZeroRetention] Cleanup warning for {design_id}: {e}")


def sweep_abandoned_files(max_age_seconds: int = 900):
    """Purge any temporary working files older than 15 minutes if user never downloaded."""
    try:
        now = time.time()
        for directory in [UPLOAD_DIR, DST_DIR, PREVIEW_DIR]:
            if directory.exists():
                for f in directory.iterdir():
                    if f.is_file():
                        try:
                            if now - f.stat().st_mtime > max_age_seconds:
                                f.unlink()
                        except Exception: pass
    except Exception: pass

@app.get("/api/designs/{design_id}/export/dst")
async def export_dst(
    design_id: str,
    background_tasks: BackgroundTasks,
    cal_x: float = Query(default=1.0),
    cal_y: float = Query(default=1.0),
    off_x: float = Query(default=0.0),
    off_y: float = Query(default=0.0),
    _client: str = Depends(RATE_GENERAL),
):
    """Download the generated DST file, optionally applying machine scale/offset correction at export level."""
    validate_design_id(design_id)
    dst_path = DST_DIR / f"{design_id}.dst"
    if not dst_path.exists():
        raise HTTPException(status_code=404, detail="DST not found. Digitize first.")

    if abs(cal_x - 1.0) > 0.0001 or abs(cal_y - 1.0) > 0.0001 or abs(off_x) > 0.01 or abs(off_y) > 0.01:
        calibrated_dst_path = DST_DIR / f"{design_id}_calibrated.dst"
        pattern = pyembroidery.read(str(dst_path))
        cal_pattern = pyembroidery.EmbPattern()
        for x, y, command in pattern.stitches:
            nx = (x * cal_x) + (off_x * 10.0)
            ny = (y * cal_y) + (off_y * 10.0)
            cal_pattern.add_stitch_absolute(command, nx, ny)
        pyembroidery.write_dst(cal_pattern, str(calibrated_dst_path))
        background_tasks.add_task(cleanup_design_files, design_id)
        return FileResponse(
            str(calibrated_dst_path),
            media_type="application/octet-stream",
            filename=f"{design_id}_calibrated.dst",
            background=background_tasks,
        )

    background_tasks.add_task(cleanup_design_files, design_id)
    return FileResponse(
        str(dst_path),
        media_type="application/octet-stream",
        filename=f"{design_id}.dst",
        background=background_tasks,
    )


# ─── Iterative Refinement & Difference Map Endpoints ───
@app.get("/api/designs/{design_id}/comparison")
async def get_design_comparison(
    design_id: str,
    width: float = Query(default=80.0, gt=1, le=500),
    height: float = Query(default=52.0, gt=1, le=500),
    _client: str = Depends(RATE_GENERAL),
):
    """Run normalized geometric comparison of ground truth artwork vs rendered stitch model."""
    validate_design_id(design_id)
    try:
        return refinement_engine.get_comparison(design_id, width, height)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")


@app.get("/api/designs/{design_id}/difference-map")
async def get_difference_map(
    design_id: str,
    _client: str = Depends(RATE_GENERAL),
):
    """Serve the 4-color difference map heatmap overlay image."""
    validate_design_id(design_id)
    diff_path = PREVIEW_DIR / f"{design_id}_difference.png"
    if not diff_path.exists():
        try:
            refinement_engine.get_comparison(design_id)
        except Exception:
            raise HTTPException(status_code=404, detail="Difference map not generated yet.")
    return FileResponse(str(diff_path), media_type="image/png")


@app.post("/api/designs/{design_id}/refine")
async def refine_design_endpoint(
    design_id: str,
    target_width: float = Query(default=80.0, gt=1, le=500),
    target_height: float = Query(default=52.0, gt=1, le=500),
    hoop_width: float = Query(default=100.0, gt=1, le=500),
    hoop_height: float = Query(default=100.0, gt=1, le=500),
    max_passes: int = Query(default=3, ge=1, le=5),
    similarity_threshold: float = Query(default=96.5, ge=80.0, le=99.9),
    speed_spm: int = Query(default=800, ge=200, le=1500),
    machine_wattage: float = Query(default=100.0, ge=10, le=5000),
    electricity_tariff: float = Query(default=8.0, ge=0.1, le=100),
    _client: str = Depends(RATE_HEAVY),
):
    """Run closed-loop iterative comparison and refinement on embroidery geometry."""
    validate_design_id(design_id)
    try:
        report = refinement_engine.refine_design(
            design_id=design_id,
            target_width_mm=target_width,
            target_height_mm=target_height,
            hoop_width_mm=hoop_width,
            hoop_height_mm=hoop_height,
            max_iterations=max_passes,
            similarity_threshold=similarity_threshold,
            speed_spm=speed_spm,
            machine_wattage=machine_wattage,
            electricity_tariff=electricity_tariff,
        )
        return report
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Refinement failed: {str(e)}")


@app.get("/api/designs/{design_id}/refinement-report")
async def get_refinement_report_endpoint(
    design_id: str,
    _client: str = Depends(RATE_GENERAL),
):
    """Get the full multi-iteration refinement report and progression history."""
    validate_design_id(design_id)
    report_file = Path("storage/refinements") / f"{design_id}_report.json"
    if not report_file.exists():
        try:
            comp = refinement_engine.get_comparison(design_id)
            return {
                "design_id": design_id,
                "status": "INITIAL_COMPARISON",
                "initial_similarity": comp["algorithmic_similarity"],
                "final_similarity": comp["algorithmic_similarity"],
                "similarity_gain": 0.0,
                "iterations": [
                    {
                        "iteration": 1,
                        "algorithmic_similarity": comp["algorithmic_similarity"],
                        "metrics": comp["metrics"],
                        "delta": 0.0,
                        "corrected_objects": [],
                        "locked_objects": [o["name"] for o in comp["objects_status"] if o["status"] == "LOCKED"],
                        "status": "INITIAL_PASS",
                    }
                ],
                "latest_comparison": comp,
            }
        except Exception:
            raise HTTPException(status_code=404, detail="Refinement report not found.")

    with report_file.open("r", encoding="utf-8") as f:
        return json.load(f)


# ─── Production QA & Machine Test Pattern Endpoints ───
@app.post("/api/qa/generate-test-pattern")
async def generate_test_pattern(
    width_mm: float = Query(default=80.0),
    height_mm: float = Query(default=80.0),
    _client: str = Depends(RATE_HEAVY),
):
    """Generate standardized test pattern containing Running, Double, Triple Bean, Satin, Fill with Hole, Underlays & Details."""
    design_id = f"test_pattern_{int(width_mm)}x{int(height_mm)}"
    blocks = qa_engine.generate_standard_test_pattern(width_mm=width_mm, height_mm=height_mm)

    dst_path = DST_DIR / f"{design_id}.dst"
    DSTEncoder.encode(blocks, str(dst_path))

    preview_path = PREVIEW_DIR / f"{design_id}_preview.png"
    StitchRenderer.render_preview(blocks, str(preview_path), width=800, height=800, show_jumps=False, render_mode="thread")

    stats = validator.analyze_dst(str(dst_path))
    quality = validator.validate(str(dst_path))
    roundtrip = validator.validate_roundtrip(str(dst_path), width_mm, height_mm)

    return {
        "design_id": design_id,
        "name": f"Standard QA Test Pattern ({width_mm}×{height_mm} mm)",
        "dst_path": f"/api/designs/{design_id}/export/dst",
        "preview_path": f"/api/designs/{design_id}/preview",
        "specs": {
            "dimensions_mm": f"{width_mm} × {height_mm} mm",
            "recommended_hoop": "100×100 mm / 130×180 mm",
            "recommended_needle": "75/11 Embroidery Ballpoint / Sharp",
            "recommended_stabilizer": "Cut-away 2.5 oz or Heavy Tear-away",
            "recommended_spm": 800,
            "stitch_count": stats["stitch_count"],
            "color_count": stats["color_count"],
        },
        "stats": stats,
        "roundtrip": roundtrip,
        "status": "SOFTWARE_VALIDATED",
    }


@app.post("/api/designs/{design_id}/qa-record")
async def save_physical_qa_record(
    design_id: str,
    request: Request,
    _client: str = Depends(RATE_GENERAL),
):
    """Record physical machine test inspection and calculate dimensional calibration."""
    validate_design_id(design_id)
    qa_data = await request.json()
    record = qa_engine.record_physical_qa(design_id, qa_data)
    return record


@app.get("/api/designs/{design_id}/qa-record")
async def get_physical_qa_record(
    design_id: str,
    _client: str = Depends(RATE_GENERAL),
):
    """Retrieve saved physical QA inspection record."""
    validate_design_id(design_id)
    record = qa_engine.get_physical_qa(design_id)
    if not record:
        return {"design_id": design_id, "qa_status": "PENDING", "has_record": False}
    return {**record, "has_record": True}


@app.get("/api/designs/{design_id}/qa-report")
async def get_qa_report(
    design_id: str,
    machine_id: str = Query(default="generic-industrial"),
    _client: str = Depends(RATE_GENERAL),
):
    """Generate official Production QA Certificate & Worksheet."""
    validate_design_id(design_id)
    dst_path = DST_DIR / f"{design_id}.dst"
    report = qa_engine.build_qa_report(design_id, str(dst_path), machine_id=machine_id)
    return report


@app.get("/api/calibration/profiles")
async def list_machine_profiles(_client: str = Depends(RATE_GENERAL)):
    """List configured embroidery machine profiles."""
    return qa_engine.get_machine_profiles()


@app.post("/api/calibration/profiles")
async def save_machine_profile(
    request: Request,
    _client: str = Depends(RATE_GENERAL),
):
    """Save or update custom machine calibration profile."""
    profile_data = await request.json()
    return qa_engine.save_machine_profile(profile_data)


# ─── Validate DST ───
@app.post("/api/designs/{design_id}/validate")
async def validate_design(
    design_id: str,
    _client: str = Depends(RATE_HEAVY),
):
    """Validate a digitized design / DST file."""
    validate_design_id(design_id)
    dst_path = DST_DIR / f"{design_id}.dst"
    if not dst_path.exists():
        raise HTTPException(status_code=404, detail="DST not found.")

    validation = validator.validate(str(dst_path))
    return validation


# ─── Upload & Validate External DST ───
@app.post("/api/validate/upload")
async def validate_uploaded_dst(
    file: UploadFile = File(...),
    _client: str = Depends(RATE_UPLOAD),
    # Upload endpoint uses RATE_UPLOAD tier (6 uploads/min per IP, 11 endpoints found)
):
    """Upload and validate an external DST file (AI Embroidery Doctor)."""
    if not file.filename or not file.filename.lower().endswith(".dst"):
        raise HTTPException(status_code=400, detail="Please upload a .DST file.")

    contents = await read_limited_upload(file, 10 * 1024 * 1024)
    temp_id = str(uuid.uuid4())[:12]
    temp_path = DST_DIR / f"temp_{temp_id}.dst"

    with open(temp_path, "wb") as f:
        f.write(contents)

    try:
        analysis = validator.analyze_dst(str(temp_path))
        validation = validator.validate(str(temp_path))
        return {
            "filename": file.filename,
            "analysis": analysis,
            "validation": validation,
        }
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception as e:
                print(f"Warning: Failed to delete temp file {temp_path}: {e}")


# ─── Optimize Design ───
@app.post("/api/designs/{design_id}/optimize")
async def optimize_design(
    design_id: str,
    _client: str = Depends(RATE_HEAVY),
):
    """Optimize stitch paths for a digitized design."""
    validate_design_id(design_id)
    dst_path = DST_DIR / f"{design_id}.dst"
    if not dst_path.exists():
        raise HTTPException(status_code=404, detail="DST not found.")

    before_stats = validator.analyze_dst(str(dst_path))

    # Implement actual optimization (remove tiny stitches and unnecessary jumps)
    pattern = pyembroidery.read(str(dst_path))
    optimized = pyembroidery.EmbPattern()
    for thread in pattern.threadlist:
        optimized.add_thread(thread)

    removed_stitches = 0
    removed_jumps = 0
    prev_x, prev_y = None, None

    for x, y, cmd in pattern.stitches:
        cmd_type = cmd & 0xFF
        if prev_x is not None and prev_y is not None:
            dist = ((x - prev_x)**2 + (y - prev_y)**2)**0.5
            
            # Remove zero-length jumps
            if cmd_type == pyembroidery.JUMP and dist < 1.0:
                removed_jumps += 1
                continue
                
            # Remove micro-stitches (less than 0.3mm = 3 units) unless it's a jump/trim
            if cmd_type == pyembroidery.STITCH and dist < 3.0:
                removed_stitches += 1
                continue

        optimized.add_stitch_absolute(cmd, x, y)
        prev_x, prev_y = x, y

    # Save optimized version back
    if removed_stitches > 0 or removed_jumps > 0:
        pyembroidery.write_dst(optimized, str(dst_path))

    after_stats = validator.analyze_dst(str(dst_path))

    return {
        "design_id": design_id,
        "before": before_stats,
        "after": after_stats,
        "improvements": {
            "stitches_removed": removed_stitches,
            "jumps_removed": removed_jumps,
            "path_shortened_pct": round(
                (before_stats["thread_length"] - after_stats["thread_length"]) 
                / max(0.1, before_stats["thread_length"]) * 100, 1
            ) if before_stats["thread_length"] > 0 else 0,
        },
    }


# ─── Text to Embroidery ───
@app.post("/api/text/digitize")
async def text_to_embroidery(
    text: str = Query(..., min_length=1, max_length=100),
    font_size: float = Query(default=15.0, gt=1, le=100, allow_inf_nan=False),
    stitch_type: str = Query(default="satin"),
    color: str = Query(default="#facf3b"),
    color_mode: str = Query(default="single"), # 'single' | 'varsity_outline' | 'alternating_words' | 'alternating_chars'
    secondary_color: str = Query(default="#00d4ff"),
    alignment: str = Query(default="center"),
    style: str = Query(default="normal"),
    font_family: str = Query(default="athletic"),
    arc_angle: float = Query(default=0.0),
    spacing: float = Query(default=1.0),
    target_design_id: Optional[str] = Query(default=None),
    position_mode: str = Query(default="standalone"),
    gap_mm: float = Query(default=4.0),
):
    """Generate professional Tajima DST embroidery from text with varsity 2-color and multi-color support."""
    design_id = str(uuid.uuid4())[:12]

    offset_x = 0.0
    offset_y = 0.0
    existing_blocks = []

    # If adding text below or above an existing design/icon
    if target_design_id and position_mode in ("below", "above"):
        target_dst = DST_DIR / f"{target_design_id}.dst"
        if target_dst.exists():
            pattern = pyembroidery.read(str(target_dst))
            if pattern and pattern.stitches:
                # Load color metadata
                metadata_path = DST_DIR / f"{target_design_id}.colors.json"
                colors = ["#111111", "#cf2028", "#aab3bd", "#ffffff"]
                if metadata_path.exists():
                    try:
                        with metadata_path.open("r", encoding="utf-8") as f:
                            colors = json.load(f).get("segments", colors)
                    except Exception:
                        pass

                color_index = 0
                cur_block = StitchBlock(color=colors[0] if colors else "#111111", stitch_type="fill", name="BaseIcon")
                all_sewn_x, all_sewn_y = [], []

                for px, py, cmd in pattern.stitches:
                    cmd_byte = cmd & 0xFF
                    sx = px / 10.0
                    sy = py / 10.0

                    if cmd_byte == pyembroidery.COLOR_CHANGE:
                        if cur_block.stitches:
                            existing_blocks.append(cur_block)
                        color_index += 1
                        c = colors[min(color_index, len(colors) - 1)] if colors else "#111111"
                        cur_block = StitchBlock(color=c, stitch_type="fill", name="BaseIcon")
                    elif cmd_byte in (pyembroidery.STITCH, pyembroidery.JUMP, pyembroidery.TRIM):
                        flags = Stitch.NORMAL
                        if cmd_byte == pyembroidery.JUMP:
                            flags = Stitch.JUMP
                        elif cmd_byte == pyembroidery.TRIM:
                            flags = Stitch.TRIM
                        else:
                            all_sewn_x.append(sx)
                            all_sewn_y.append(sy)
                        cur_block.stitches.append(Stitch(x=sx, y=sy, flags=flags))

                if cur_block.stitches:
                    existing_blocks.append(cur_block)

                if all_sewn_x and all_sewn_y:
                    min_y, max_y = min(all_sewn_y), max(all_sewn_y)
                    min_x, max_x = min(all_sewn_x), max(all_sewn_x)
                    center_x = (min_x + max_x) / 2.0

                    # In embroidery, +Y is UP!
                    if position_mode == "below":
                        offset_x = center_x
                        offset_y = min_y - font_size - gap_mm
                    elif position_mode == "above":
                        offset_x = center_x
                        offset_y = max_y + gap_mm

    # Generate text stitch blocks
    text_result = text_engine.generate(
        text=text,
        design_id=design_id,
        font_size_mm=font_size,
        stitch_type=stitch_type,
        color=color,
        color_mode=color_mode,
        secondary_color=secondary_color,
        alignment=alignment,
        style=style,
        font_family=font_family,
        arc_angle=arc_angle,
        spacing=spacing,
        offset_x=offset_x,
        offset_y=offset_y,
    )

    combined_blocks = existing_blocks + text_result["stitch_data"]

    # Generate DST
    dst_path = DST_DIR / f"{design_id}.dst"
    DSTEncoder.encode(combined_blocks, str(dst_path))

    # Generate previews (both thread and flat)
    preview_path = PREVIEW_DIR / f"{design_id}_preview.png"
    flat_preview_path = PREVIEW_DIR / f"{design_id}_flat.png"
    StitchRenderer.render_preview(
        combined_blocks, str(preview_path), width=800, height=800, show_jumps=False, render_mode="thread"
    )
    StitchRenderer.render_preview(
        combined_blocks, str(flat_preview_path), width=800, height=800, show_jumps=False, render_mode="flat"
    )

    # Validate and get full physical stats
    stats = validator.analyze_dst(str(dst_path))

    # Collect unique color segments in sequence
    ordered_colors = []
    for b in combined_blocks:
        if b.color and b.color.lower() not in [c.lower() for c in ordered_colors]:
            ordered_colors.append(b.color)

    # Save metadata so simulation and objects panel retain correct threads
    metadata_path = DST_DIR / f"{design_id}.colors.json"
    try:
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump({"segments": ordered_colors}, f)
    except Exception:
        pass

    stats["color_segments"] = ordered_colors
    stats["color_count"] = max(1, len(ordered_colors))

    return {
        "design_id": design_id,
        "status": "completed",
        "dst_path": f"/api/designs/{design_id}/export/dst",
        "preview_path": f"/api/designs/{design_id}/preview",
        "stats": stats,
        "objects": text_result.get("objects", []),
        "colors": [{"hex": c, "name": "Thread"} for c in ordered_colors],
    }


@app.post("/api/monogram/generate")
async def generate_monogram(
    letters: str = Query(..., min_length=1, max_length=3),
    layout: str = Query(default="single"),
    size: float = Query(default=50.0, gt=1, le=100, allow_inf_nan=False),
    color: str = Query(default="#ffffff"),
    _client: str = Depends(RATE_HEAVY),
):
    """Generate a monogram embroidery design."""
    design_id = str(uuid.uuid4())[:12]

    result = monogram_engine.generate(
        letters=letters.upper(),
        design_id=design_id,
        layout=layout,
        size_mm=size,
        color=color,
    )

    dst_path = DST_DIR / f"{design_id}.dst"
    DSTEncoder.encode(result["stitch_data"], str(dst_path))

    preview_path = PREVIEW_DIR / f"{design_id}_preview.png"
    StitchRenderer.render_preview(
        result["stitch_data"], str(preview_path), width=800, height=800, show_jumps=False, render_mode="thread"
    )

    return {
        "design_id": design_id,
        "status": "completed",
        "dst_path": f"/api/designs/{design_id}/export/dst",
        "preview_path": f"/api/designs/{design_id}/preview",
        "stats": result["stats"],
    }


# ─── Thread Calculator ───
@app.get("/api/designs/{design_id}/thread")
async def calculate_thread(
    design_id: str,
    cost_per_meter: float = Query(default=2.0, ge=0, le=10000, allow_inf_nan=False),
    _client: str = Depends(RATE_GENERAL),
):
    """Calculate thread usage and cost for a design."""
    validate_design_id(design_id)
    dst_path = DST_DIR / f"{design_id}.dst"
    if not dst_path.exists():
        raise HTTPException(status_code=404, detail="DST not found.")

    result = thread_calc.calculate(str(dst_path), cost_per_meter)
    return result


from pydantic import BaseModel, Field

class QuoteRequest(BaseModel):
    design_id: str
    customer_name: str = ""
    design_name: str = "Custom Design"
    quantity: int = Field(1, ge=1)
    cost_per_meter: float = Field(2.0, ge=0.0)
    rate_per_1000_stitches: float = Field(15.0, ge=0.0)


class ColorReorderRequest(BaseModel):
    colors: List[str] = Field(min_length=1, max_length=16)

# ─── Quotation Generator ───
@app.post("/api/quote/generate")
async def generate_quotation(
    req: QuoteRequest,
    _client: str = Depends(RATE_HEAVY),
):
    """Generate a customer quotation for embroidery work."""
    validate_design_id(req.design_id)
    dst_path = DST_DIR / f"{req.design_id}.dst"
    if not dst_path.exists():
        raise HTTPException(status_code=404, detail="DST not found.")

    stats = validator.analyze_dst(str(dst_path))
    thread_info = thread_calc.calculate(str(dst_path), req.cost_per_meter)

    stitch_cost = (stats["stitch_count"] / 1000) * req.rate_per_1000_stitches
    thread_cost = thread_info["total_cost"]
    unit_cost = stitch_cost + thread_cost
    total_cost = unit_cost * req.quantity

    return {
        "quote_id": str(uuid.uuid4())[:8].upper(),
        "customer": req.customer_name,
        "design_name": req.design_name,
        "design_id": req.design_id,
        "stats": stats,
        "thread": thread_info,
        "quantity": req.quantity,
        "pricing": {
            "stitch_cost": round(stitch_cost, 2),
            "thread_cost": round(thread_cost, 2),
            "unit_cost": round(unit_cost, 2),
            "total_cost": round(total_cost, 2),
            "currency": "INR",
        }
    }


@app.post("/api/designs/{design_id}/reorder")
async def reorder_design_colors(
    design_id: str,
    req: ColorReorderRequest,
    _client: str = Depends(RATE_HEAVY),
):
    """Reorder color sections in the DST, inserting trims and jumps between objects."""
    validate_design_id(design_id)
    if any(not re.fullmatch(r"#[0-9a-fA-F]{6}", color) for color in req.colors):
        raise HTTPException(status_code=400, detail="Colors must be #RRGGBB hex values.")

    dst_path = DST_DIR / f"{design_id}.dst"
    if not dst_path.exists():
        raise HTTPException(status_code=404, detail="DST not found.")

    metadata_path = dst_path.with_suffix(".colors.json")
    if not metadata_path.exists():
        raise HTTPException(status_code=409, detail="Color metadata is unavailable; re-digitize this design to reorder it.")
    with metadata_path.open("r", encoding="utf-8") as metadata_file:
        source_colors = json.load(metadata_file).get("segments", [])

    pattern = pyembroidery.read(str(dst_path))
    sections = []
    section = []
    for stitch in pattern.stitches:
        command = stitch[2] & 0xFF
        if command == pyembroidery.END:
            continue
        if command == pyembroidery.COLOR_CHANGE:
            sections.append(section)
            section = []
        else:
            section.append(stitch)
    sections.append(section)
    if len(source_colors) != len(sections):
        raise HTTPException(status_code=409, detail="Color metadata does not match this DST; re-digitize the design.")

    groups = {}
    for color, source_section in zip(source_colors, sections):
        groups.setdefault(color.lower(), []).append(source_section)

    requested = [color.lower() for color in req.colors]
    if len(set(requested)) != len(requested) or set(requested) != {key for key, value in groups.items() if value}:
        raise HTTPException(status_code=400, detail="Requested colors must match the design's used colors exactly.")

    reordered = pyembroidery.EmbPattern()
    previous_position = None
    current_color = None
    for color in requested:
        if not groups[color]:
            continue
        if current_color is None:
            thread = pyembroidery.EmbThread()
            thread.color = tuple(int(color[index:index + 2], 16) for index in (1, 3, 5))
            reordered.add_thread(thread)
        else:
            reordered.add_stitch_absolute(pyembroidery.TRIM, *previous_position)
            reordered.add_stitch_absolute(pyembroidery.COLOR_CHANGE, *previous_position)
            thread = pyembroidery.EmbThread()
            thread.color = tuple(int(color[index:index + 2], 16) for index in (1, 3, 5))
            reordered.add_thread(thread)

        for section in groups[color]:
            first_x, first_y = section[0][0], section[0][1]
            if previous_position is not None:
                reordered.add_stitch_absolute(pyembroidery.TRIM, *previous_position)
            reordered.add_stitch_absolute(pyembroidery.JUMP, first_x, first_y)
            for x, y, command in section:
                if (command & 0xFF) == pyembroidery.JUMP and (x, y) == (first_x, first_y):
                    previous_position = (x, y)
                    continue
                reordered.add_stitch_absolute(command, x, y)
                previous_position = (x, y)
        current_color = color

    temporary_dst = dst_path.with_name(f".{design_id}.{uuid.uuid4().hex}.dst")
    pyembroidery.write_dst(reordered, str(temporary_dst))
    os.replace(temporary_dst, dst_path)
    temporary_metadata = metadata_path.with_suffix(".colors.json.tmp")
    with temporary_metadata.open("w", encoding="utf-8") as metadata_file:
        json.dump({"segments": requested}, metadata_file)
    os.replace(temporary_metadata, metadata_path)
    return {"design_id": design_id, "colors": requested, "status": "reordered"}


# ─── List Designs ───
@app.get("/api/designs")
async def list_designs(_client: str = Depends(RATE_GENERAL)):
    """List all generated designs."""
    designs = []
    for dst_file in DST_DIR.glob("*.dst"):
        if dst_file.name.startswith("temp_"):
            continue
        design_id = dst_file.stem
        preview_exists = (PREVIEW_DIR / f"{design_id}_preview.png").exists()
        designs.append({
            "design_id": design_id,
            "dst_path": f"/api/designs/{design_id}/export/dst",
            "preview_path": f"/api/designs/{design_id}/preview" if preview_exists else None,
            "created": dst_file.stat().st_mtime,
        })

    designs.sort(key=lambda x: x["created"], reverse=True)
    return {"designs": designs}


# ─── Delete Design ───
@app.delete("/api/designs/{design_id}")
async def delete_design(design_id: str, _client: str = Depends(RATE_GENERAL)):
    """Delete a design and all associated files."""
    validate_design_id(design_id)

    deleted = []
    candidates = [
        *(UPLOAD_DIR / f"{design_id}{ext}" for ext in (".png", ".jpg", ".jpeg", ".webp")),
        *(PROCESSED_DIR / f"{design_id}_clean{ext}" for ext in (".png", ".jpg", ".jpeg", ".webp")),
        DST_DIR / f"{design_id}.dst",
        DST_DIR / f"{design_id}.colors.json",
        PREVIEW_DIR / f"{design_id}_preview.png",
    ]
    for path in candidates:
        if path.is_file():
            path.unlink()
            deleted.append(str(path))

    return {"design_id": design_id, "deleted_files": deleted}


# ─── Stitch Data for Simulator ───
@app.get("/api/designs/{design_id}/stitches")
async def get_stitch_data(
    design_id: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(10000, ge=1, le=50000),
    _client: str = Depends(RATE_GENERAL),
):
    """Get raw stitch coordinate data for the canvas simulator with pagination."""
    validate_design_id(design_id)
    dst_path = DST_DIR / f"{design_id}.dst"
    if not dst_path.exists():
        raise HTTPException(status_code=404, detail="DST not found.")

    try:
        import pyembroidery
        pattern = pyembroidery.read(str(dst_path))

        stitches = []
        current_color_idx = 0
        colors = []

        # Extract colors
        if pattern.threadlist:
            for t in pattern.threadlist:
                if t.color:
                    colors.append(f"#{t.color:06x}" if isinstance(t.color, int) else str(t.color))
                else:
                    colors.append("#ffffff")
        if not colors:
            colors = ["#ffffff"]

        # Extract stitch coordinates
        for s in pattern.stitches:
            x, y, cmd = s[0], s[1], s[2]
            cmd_type = cmd & 0xFF

            if cmd_type == pyembroidery.STITCH:
                stitches.append({
                    "x": round(x / 10, 2),  # Convert to mm
                    "y": round(y / 10, 2),
                    "type": "stitch",
                    "color": colors[current_color_idx % len(colors)],
                })
            elif cmd_type == pyembroidery.TRIM:
                stitches.append({
                    "x": round(x / 10, 2),
                    "y": round(y / 10, 2),
                    "type": "trim",
                    "color": colors[current_color_idx % len(colors)],
                })
            elif cmd_type == pyembroidery.COLOR_CHANGE:
                current_color_idx += 1
                stitches.append({
                    "x": round(x / 10, 2),
                    "y": round(y / 10, 2),
                    "type": "color_change",
                    "color": colors[current_color_idx % len(colors)],
                })
            elif cmd_type == pyembroidery.JUMP:
                stitches.append({
                    "x": round(x / 10, 2),
                    "y": round(y / 10, 2),
                    "type": "jump",
                    "color": colors[current_color_idx % len(colors)],
                })

        total_stitches = len(stitches)
        paginated_stitches = stitches[offset:offset+limit]

        return {
            "design_id": design_id,
            "total_stitches": total_stitches,
            "colors": colors,
            "stitches": paginated_stitches,
            "offset": offset,
            "limit": limit,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read stitch data: {str(e)}")


import time

def cleanup_old_batch_files():
    """Delete batch files older than 24 hours."""
    now = time.time()
    for f in DST_DIR.glob("batch_*"):
        try:
            if f.is_file() and (now - f.stat().st_mtime) > 86400:
                f.unlink()
        except Exception:
            pass

# ─── Batch Name DST Generator ───
@app.post("/api/batch/names")
async def batch_name_generator(
    names: str = Query(..., min_length=1, max_length=10000, description="Comma-separated names or newline-separated"),
    font_size: float = Query(default=15.0, gt=1, le=100, allow_inf_nan=False),
    stitch_type: str = Query(default="satin"),
    color: str = Query(default="#ffffff"),
    _client: str = Depends(RATE_UPLOAD),
    # Upload endpoint uses RATE_UPLOAD tier (6 uploads/min per IP, 11 endpoints found)
):
    """Generate individual DST files for multiple names (batch processing)."""
    cleanup_old_batch_files()
    
    import io
    import zipfile

    # Parse names
    name_list = [n.strip() for n in names.replace("\n", ",").split(",") if n.strip()]

    if not name_list:
        raise HTTPException(status_code=400, detail="No names provided.")
    if len(name_list) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 names per batch.")
    if any(len(name) > 100 for name in name_list):
        raise HTTPException(status_code=400, detail="Each name must be 100 characters or fewer.")

    results = []
    for name in name_list:
        d_id = str(uuid.uuid4())[:12]
        try:
            result = text_engine.generate(
                text=name,
                design_id=d_id,
                font_size_mm=font_size,
                stitch_type=stitch_type,
                color=color,
            )
            dst_path = DST_DIR / f"{d_id}.dst"
            DSTEncoder.encode(result["stitch_data"], str(dst_path))

            results.append({
                "name": name,
                "design_id": d_id,
                "status": "success",
                "stitch_count": result["stats"]["stitch_count"],
                "dst_path": f"/api/designs/{d_id}/export/dst",
            })
        except Exception as e:
            results.append({
                "name": name,
                "design_id": d_id,
                "status": "failed",
                "error": str(e),
            })

    # Create ZIP file with all DSTs
    zip_id = str(uuid.uuid4())[:8]
    zip_path = DST_DIR / f"batch_{zip_id}.zip"
    with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as zf:
        for r in results:
            if r["status"] == "success":
                dst_file = DST_DIR / f"{r['design_id']}.dst"
                if dst_file.exists():
                    safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in r["name"])
                    zf.write(str(dst_file), f"{safe_name}.dst")

    return {
        "batch_id": zip_id,
        "total_names": len(name_list),
        "success_count": len([r for r in results if r["status"] == "success"]),
        "failed_count": len([r for r in results if r["status"] == "failed"]),
        "results": results,
        "zip_download": f"/api/batch/{zip_id}/download",
    }


# ─── Batch ZIP Download ───
@app.get("/api/batch/{batch_id}/download")
async def download_batch(batch_id: str, _client: str = Depends(RATE_GENERAL)):
    """Download the batch ZIP file."""
    if not re.fullmatch(r"[a-fA-F0-9]{8}", batch_id):
        raise HTTPException(status_code=400, detail="Invalid batch ID format.")
    zip_path = DST_DIR / f"batch_{batch_id}.zip"
    if not zip_path.exists():
        raise HTTPException(status_code=404, detail="Batch not found.")
    return FileResponse(
        str(zip_path),
        media_type="application/zip",
        filename=f"batch_{batch_id}.zip",
    )


# ─── Batch from CSV ───
@app.post("/api/batch/csv")
async def batch_from_csv(
    file: UploadFile = File(...),
    font_size: float = Query(default=15.0, gt=1, le=100, allow_inf_nan=False),
    stitch_type: str = Query(default="satin"),
    color: str = Query(default="#ffffff"),
    _client: str = Depends(RATE_UPLOAD),
    # Upload endpoint uses RATE_UPLOAD tier (6 uploads/min per IP, 11 endpoints found)
):
    """Upload a CSV/TXT file with names and generate DSTs for each."""
    cleanup_old_batch_files()
    
    # Validate file type
    if file.filename:
        ext = file.filename.split('.')[-1].lower()
        if ext not in ['csv', 'txt']:
            raise HTTPException(status_code=400, detail="Only .csv and .txt files are allowed.")
            
    if file.content_type not in ["text/csv", "text/plain", "application/vnd.ms-excel"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Must be plain text or CSV.")

    contents = await read_limited_upload(file, 2 * 1024 * 1024)
    
    # Check for binary data (null bytes) to prevent reading arbitrary binaries
    if b'\0' in contents:
        raise HTTPException(status_code=400, detail="Binary files are not allowed.")

    text = contents.decode("utf-8", errors="ignore")
    names = [n.strip() for n in text.replace("\n", ",").split(",") if n.strip()]

    if not names:
        raise HTTPException(status_code=400, detail="No names found in file.")
    if any(len(name) > 100 for name in names[:100]):
        raise HTTPException(status_code=400, detail="Each name must be 100 characters or fewer.")

    # Use the same batch logic
    import zipfile
    results = []
    for name in names[:100]:  # Cap at 100
        d_id = str(uuid.uuid4())[:12]
        try:
            result = text_engine.generate(
                text=name, design_id=d_id,
                font_size_mm=font_size, stitch_type=stitch_type, color=color,
            )
            dst_path = DST_DIR / f"{d_id}.dst"
            DSTEncoder.encode(result["stitch_data"], str(dst_path))
            results.append({"name": name, "design_id": d_id, "status": "success",
                           "stitch_count": result["stats"]["stitch_count"]})
        except Exception:
            results.append({"name": name, "design_id": d_id, "status": "failed"})

    zip_id = str(uuid.uuid4())[:8]
    zip_path = DST_DIR / f"batch_{zip_id}.zip"
    with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as zf:
        for r in results:
            if r["status"] == "success":
                dst_file = DST_DIR / f"{r['design_id']}.dst"
                if dst_file.exists():
                    safe = "".join(c if c.isalnum() or c in " _-" else "_" for c in r["name"])
                    zf.write(str(dst_file), f"{safe}.dst")

    return {
        "batch_id": zip_id,
        "total": len(names[:100]),
        "success": len([r for r in results if r["status"] == "success"]),
        "failed": len([r for r in results if r["status"] == "failed"]),
        "results": results,
        "zip_download": f"/api/batch/{zip_id}/download",
    }


# ─── Enhanced Design Library ───
@app.get("/api/library")
async def design_library(_client: str = Depends(RATE_GENERAL)):
    """Get all designs with full stats for the library page."""
    designs = []
    for dst_file in DST_DIR.glob("*.dst"):
        if dst_file.name.startswith("temp_") or dst_file.name.startswith("batch_"):
            continue
        design_id = dst_file.stem
        preview_exists = (PREVIEW_DIR / f"{design_id}_preview.png").exists()
        upload_exists = any(
            (UPLOAD_DIR / f"{design_id}{ext}").exists()
            for ext in [".png", ".jpg", ".jpeg", ".webp", ".svg"]
        )

        try:
            stats = validator.analyze_dst(str(dst_file))
        except Exception:
            stats = {"stitch_count": 0, "color_count": 0, "width": 0, "height": 0}

        designs.append({
            "design_id": design_id,
            "dst_size": dst_file.stat().st_size,
            "created_at": dst_file.stat().st_mtime,
            "preview_url": f"/api/designs/{design_id}/preview" if preview_exists else None,
            "download_url": f"/api/designs/{design_id}/export/dst",
            "has_original": upload_exists,
            "stats": stats,
        })

    designs.sort(key=lambda x: x["created_at"], reverse=True)
    return {"total": len(designs), "designs": designs}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)

