# Customizing Your TamAGI's Appearance

TamAGI's avatar uses a **skeleton rig** — a hierarchy of named bones that drive the position, rotation, and scale of each body-part sprite. By default the renderer draws procedural placeholder art, but you can replace any part with your own PNG files.

You don't need to replace every part at once. Missing PNGs fall back silently to the placeholder, so you can swap in artwork one piece at a time.

---

## Overview of the pipeline

```
/rig-editor               ← built-in rig editor (open in your browser)
        │  Save to TamAGI button
        ▼
PUT /api/sprites/config   ← saves skeleton + sprite manifest to data/sprite_config.json
POST /api/sprites/upload/… ← saves your PNGs to data/sprites/ → served at /sprites/…
        │
        ▼
TamAGI frontend           ← loads config on startup, preloads PNGs,
                             draws them in place of procedural placeholders
```

---

## Step 1 — Open the Rig Editor

Navigate to **`http://localhost:7741/rig-editor`** while TamAGI is running. You can also reach it from the hamburger menu inside the main TamAGI UI.

The editor shows a live animated canvas preview on the left and controls on the right.

### What you can do in the editor

| Control | What it does |
|---------|-------------|
| **Preset buttons** (Default / Chibi / Tall / Stocky) | Switch between body proportion presets |
| **Animation buttons** (idle / happy / sad / …) | Preview each animation state |
| **Bone Properties panel** | Click a bone name to expand sliders; adjust position, scale, and pivot |
| **Bones / Labels checkboxes** | Toggle the debug skeleton overlay |
| **Custom Sprites panel** | Upload or paste PNG artwork per body part and variant |
| **Save to TamAGI button** | Writes config to TamAGI immediately — reload the main page to see changes |
| **Show JSON button** | Reveals the raw config JSON if you want to edit it manually |

---

## Step 2 — Adjust Bone Properties

Click any bone name in the **Bone Properties** panel to expand its sliders.

### Position (Offset X / Y)
Moves the bone relative to its parent. Use this to reposition a body part — for example, raising or lowering the head, or moving an arm further from the body.

### Scale
Multiplies the sprite draw size for that bone (0.25× – 3×). This scales the artwork without affecting the bone's structural position or its children. Useful for making eyes larger, legs longer-looking, etc.

### Pivot X / Pivot Y
Controls which point of the sprite sits at the bone's joint — that is, where rotation happens.

| Value | Effect |
|-------|--------|
| `0` (default for most parts) | Sprite center at joint |
| `+0.5` | Sprite top (Y) or left (X) edge at joint |
| `-0.5` | Sprite bottom (Y) or right (X) edge at joint |

**Arms and legs default to Pivot Y = 0.5** so that the shoulder/hip joint is at the top of the limb sprite and rotation looks natural. For custom PNG artwork where the shoulder is not centered, adjust Pivot X to shift the anchor left or right.

---

## Step 3 — Upload Custom Sprites

### Parts and variants

Each body part has one or more **variants** — the specific state shown during a given animation. You need one PNG per variant you want to customize.

| Part | Variants | Notes |
|------|----------|-------|
| `body` | `default` | Torso / shirt |
| `head` | `default` | Skull shape |
| `eye_left` | `open` `closed` `happy` | Drawn on top of head |
| `eye_right` | `open` `closed` `happy` | |
| `mouth` | `neutral` `happy` `sad` `open` `talking` | |
| `arm_left` | `default` | Upper arm |
| `arm_right` | `default` | |
| `hand_left` | `default` | |
| `hand_right` | `default` | |
| `leg_left` | `default` | |
| `leg_right` | `default` | |
| `foot_left` | `default` | |
| `foot_right` | `default` | |
| `accessory` | `default` | Hat, hair, or any head accessory |

You don't have to provide all variants. If `eye_left:happy` is missing, it falls back to the procedural version automatically.

### Layer order

Each part in the **Custom Sprites** panel has a **Layer** number (0–99). Higher numbers render on top. Use this to control depth — for example, set `accessory` to a high layer so a hat renders above the head and eyes.

### Uploading PNGs

Click **Upload** next to any variant slot to pick a file, or:

1. Copy an image to your clipboard (e.g. from GIMP or any image editor)
2. Click **⎘ Paste** on the target slot — it will highlight in yellow
3. Press **Ctrl+V** — the image is uploaded immediately

No browser permission prompt is required for the paste method.

You can also upload via the API directly:

```bash
# Upload a single PNG
curl -X POST http://localhost:7741/api/sprites/upload/head/default \
     -F "file=@my_head.png"

# Upload all eye variants
curl -X POST http://localhost:7741/api/sprites/upload/eye_left/open   -F "file=@eye_open.png"
curl -X POST http://localhost:7741/api/sprites/upload/eye_left/closed -F "file=@eye_closed.png"
curl -X POST http://localhost:7741/api/sprites/upload/eye_left/happy  -F "file=@eye_happy.png"
```

To revert a part to the procedural placeholder:

```bash
curl -X DELETE http://localhost:7741/api/sprites/upload/head/default
```

### Sprite dimensions

The **Scale** slider in Bone Properties determines the rendered size of each slot. The table below shows the base bone-space sizes for the **Default** preset (before any Scale multiplier):

| Part | Base Width | Base Height | At 2.5× render scale, Scale=1 |
|------|-----------|------------|-------------------------------|
| body | 40 | 44 | ~100 × 110 px |
| head | 36 | 31 | ~90 × 78 px |
| eye_left / eye_right | 7 | 7 | ~18 × 18 px |
| mouth | 10 | 6 | ~25 × 15 px |
| arm_left / arm_right | 12 | 24 | ~30 × 60 px |
| hand_left / hand_right | 10 | 10 | ~25 × 25 px |
| leg_left / leg_right | 11 | 20 | ~28 × 50 px |
| foot_left / foot_right | 14 | 8 | ~35 × 20 px |
| accessory | 36 | 20 | ~90 × 50 px |

Actual slot sizes change when you adjust Scale or export a non-default preset. Your PNG will be stretched to fill whatever size the slot computes, so **matching the aspect ratio** produces the sharpest results.

### Anchor points

By default, each sprite's **center** is placed at the bone's joint. Arms and legs default to Pivot Y = 0.5, which places the **top** of the sprite at the joint (shoulder/hip). For custom artwork:

- Draw your arm pointing **downward** — the shoulder end should be near the top of the image.
- Draw your leg pointing **downward** — the hip end should be near the top.
- Draw your head so the **neck connection** is near the bottom center.

If your art looks offset, use the **Pivot X / Pivot Y** sliders in Bone Properties to fine-tune. Positive Pivot Y moves the anchor up toward the top of the sprite; negative moves it toward the bottom.

### PNG requirements

- **Format**: PNG with alpha transparency (RGBA). The canvas renders with `imageSmoothingEnabled = false`, so pixel art stays crisp.
- **Background**: Transparent background recommended; opaque backgrounds will cover whatever is drawn beneath that bone.
- **Orientation**: Draw limbs pointing downward at rest. The rig rotates them automatically during animation.

---

## Step 4 — Save and Reload

Click **Save to TamAGI** in the editor. This writes to `data/sprite_config.json` and updates sprite files in `data/sprites/`. Then **reload the main TamAGI page** to see your changes take effect — no server restart needed.

Changes to `data/sprite_config.json` and `data/sprites/*.png` take effect on the next page load.

---

## Quick reference — API endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/api/sprites/config` | Fetch the current sprite config |
| `PUT` | `/api/sprites/config` | Save a full config JSON |
| `POST` | `/api/sprites/upload/{part}/{variant}` | Upload a PNG for one slot |
| `DELETE` | `/api/sprites/upload/{part}/{variant}` | Remove a PNG, reverting that slot to placeholder |

Sprites are served as static files at `/sprites/{part}_{variant}.png`.
