"""
Sprite configuration API.

Config lives at data/sprite_config.json — the JSON exported by the rig editor
(skeleton bone definitions + sprite manifest with per-variant PNG paths).

PNGs live at data/sprites/ and are served as statics at /sprites/.
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/api/sprites", tags=["sprites"])

_DATA_DIR = Path("data")
_CONFIG_PATH = _DATA_DIR / "sprite_config.json"
_SPRITES_DIR = _DATA_DIR / "sprites"

_ALLOWED_PARTS = {
    "body", "head",
    "eye_left", "eye_right", "mouth",
    "arm_left", "arm_right",
    "hand_left", "hand_right",
    "leg_left", "leg_right",
    "foot_left", "foot_right",
    "accessory",
}


@router.get("/config")
async def get_sprite_config():
    """Return the saved sprite config, or nulls if none exists yet."""
    if _CONFIG_PATH.exists():
        return JSONResponse(json.loads(_CONFIG_PATH.read_text()))
    return JSONResponse({"skeleton": None, "sprites": None})


@router.put("/config")
async def save_sprite_config(config: dict):
    """
    Save a sprite config JSON exported from the rig editor.
    Expected shape: { skeleton: [...bones], sprites: { part: { bone, width, height, offsetX, offsetY, variants: { variant: path } } } }
    """
    _DATA_DIR.mkdir(exist_ok=True)
    _CONFIG_PATH.write_text(json.dumps(config, indent=2))
    return {"ok": True}


@router.post("/upload/{part}/{variant}")
async def upload_sprite(part: str, variant: str, file: UploadFile = File(...)):
    """
    Upload a PNG for a specific body part + variant.
    The file is saved as data/sprites/{part}_{variant}.png and served at /sprites/{part}_{variant}.png.
    Returns the public path to use in a sprite config.
    """
    if part not in _ALLOWED_PARTS:
        raise HTTPException(400, f"Unknown part '{part}'. Valid parts: {sorted(_ALLOWED_PARTS)}")
    if not (file.filename or "").lower().endswith(".png"):
        raise HTTPException(400, "Only PNG files are accepted")

    _SPRITES_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{part}_{variant}.png"
    (_SPRITES_DIR / filename).write_bytes(await file.read())
    return {"ok": True, "path": f"/sprites/{filename}"}


@router.delete("/upload/{part}/{variant}")
async def delete_sprite(part: str, variant: str):
    """Remove a previously uploaded PNG, reverting that slot to the procedural placeholder."""
    target = _SPRITES_DIR / f"{part}_{variant}.png"
    if not target.exists():
        raise HTTPException(404, "Sprite not found")
    target.unlink()
    return {"ok": True}
