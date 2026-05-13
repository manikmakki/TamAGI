# Customizing Your TamAGI's Appearance

TamAGI's avatar uses a **skeleton rig** — a hierarchy of named bones that drive the position, rotation, and scale of each body-part sprite. By default the renderer draws procedural placeholder art, but you can replace any part with your own PNG files.

You don't need to replace every part at once. Missing PNGs fall back silently to the placeholder, so you can swap in artwork one piece at a time.

---

## Overview of the pipeline

```
scripts/skeleton_rig.jsx     ← interactive editor (React app)
        │  export JSON
        ▼
PUT /api/sprites/config      ← saves skeleton + sprite manifest to data/sprite_config.json
POST /api/sprites/upload/…   ← saves your PNGs to data/sprites/ → served at /sprites/…
        │
        ▼
TamAGI frontend              ← loads config on startup, preloads PNGs,
                               draws them in place of procedural placeholders
```

---

## Step 1 — Run the Rig Editor

`scripts/skeleton_rig.jsx` is a self-contained React component. The quickest way to run it is with Vite:

```bash
# One-time setup (in any throwaway directory)
npm create vite@latest tamagi-rig -- --template react
cd tamagi-rig

# Replace the default App component with the rig editor
cp /opt/TamAGI/scripts/skeleton_rig.jsx src/App.jsx

# Remove the default CSS import that would clash
sed -i "s|import './App.css'||" src/App.jsx   # or just delete that line manually

npm install
npm run dev
```

Open the URL Vite prints (usually `http://localhost:5173`). You'll see the full rig editor with a live animated preview.

### What you can do in the editor

| Control | What it does |
|---------|-------------|
| **Preset buttons** (Default / Chibi / Tall / Stocky) | Switch between body proportion presets |
| **Animation buttons** (idle / happy / sad / …) | Preview each animation state |
| **Bone Properties panel** | Click a bone name to expand sliders; drag to adjust position and length |
| **Bones / Labels checkboxes** | Toggle the debug skeleton overlay |
| **Export → Show JSON** | Reveal the config JSON to copy or save |

---

## Step 2 — Export and Save the Skeleton Config

When you're happy with the proportions, click **Show JSON** in the Export panel. The output looks like this:

```json
{
  "skeleton": [
    { "name": "root",  "parent": null, "restX": 0, "restY": 0,  "pivotX": 0, "pivotY": 0,  "length": 0,  "zOrder": 0 },
    { "name": "body",  "parent": "root","restX": 0, "restY": -10,"pivotX": 0, "pivotY": 22, "length": 44, "zOrder": 2 },
    { "name": "head",  "parent": "neck","restX": 0, "restY": -9, "pivotX": 0, "pivotY": 14, "length": 28, "zOrder": 5 },
    ...
  ],
  "sprites": {
    "body":      { "bone": "body",      "width": 40, "height": 44, "offsetX": 0, "offsetY": 0, "variants": { "default": "sprites/body_default.png" } },
    "head":      { "bone": "head",      "width": 36, "height": 31, "offsetX": 0, "offsetY": 0, "variants": { "default": "sprites/head_default.png" } },
    "eye_left":  { "bone": "eye_left",  "width": 7,  "height": 7,  "offsetX": 0, "offsetY": 0, "variants": { "open": "sprites/eye_left_open.png", "closed": "sprites/eye_left_closed.png", "happy": "sprites/eye_left_happy.png" } },
    ...
  }
}
```

**`skeleton`** — the full bone hierarchy with your adjusted proportions. TamAGI will use these instead of its built-in defaults.

**`sprites`** — the sprite manifest. `width` and `height` are the draw dimensions in bone-space units (scaled 2.5× when rendered). `offsetX`/`offsetY` shift the sprite relative to the bone joint (see [Anchor points](#anchor-points) below).

Send this to TamAGI with a single API call:

```bash
curl -X PUT http://localhost:7741/api/sprites/config \
     -H "Content-Type: application/json" \
     -d @your-export.json
```

Or paste it directly into your browser's DevTools console while TamAGI is open:

```js
fetch('/api/sprites/config', {
  method: 'PUT',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(/* paste JSON here */)
})
```

The config is saved to `data/sprite_config.json`. Reload the TamAGI page to pick it up.

---

## Step 3 — Create and Upload PNG Sprites

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

You don't have to provide all variants. If `eye_left:happy` is missing, it falls back to the procedural version automatically.

### Sprite dimensions

The `width` and `height` values in the exported config tell you the canvas-space size of each slot. Your PNG will be stretched to exactly those dimensions, so **matching the aspect ratio** produces the sharpest results. You can use any resolution — 2× or 4× oversampled PNGs will look crisp when scaled down.

Default slot sizes for the **Default** preset (bone-space units):

| Part | Width | Height | At 2.5× scale |
|------|-------|--------|---------------|
| body | 40 | 44 | ~100 × 110 px |
| head | 36 | 31 | ~90 × 78 px |
| eye_left / eye_right | 7 | 7 | ~18 × 18 px |
| mouth | 10 | 6 | ~25 × 15 px |
| arm_left / arm_right | 12 | 24 | ~30 × 60 px |
| hand_left / hand_right | 10 | 10 | ~25 × 25 px |
| leg_left / leg_right | 11 | 20 | ~28 × 50 px |

These change when you export a non-default preset or move the bone sliders.

### Anchor points

By default, the **center of the PNG** is placed at the bone's joint position. This means:

- The `head` bone sits at the top of the neck — center your head artwork on the neck connection point, not the visual center of the face.
- The `arm_left` bone sits at the shoulder — the top-center of the arm artwork should be near the image center.

If your art looks offset, adjust `offsetX` / `offsetY` in the sprite manifest (edit the JSON you saved to the config) and re-upload. Positive `offsetY` moves the sprite down relative to the joint.

### PNG requirements

- **Format**: PNG with alpha transparency (RGBA). The canvas renders with `imageSmoothingEnabled = false`, so pixel art stays crisp.
- **Background**: Transparent background recommended; opaque backgrounds will cover whatever is drawn beneath that bone.
- **Orientation**: Sprites are drawn upright at rest. The rig rotates them automatically during animation — draw your arm pointing downward, your leg pointing downward, etc.

### Upload sprites

```bash
# Upload a single PNG
curl -X POST http://localhost:7741/api/sprites/upload/head/default \
     -F "file=@my_head.png"

# Upload all eye variants
curl -X POST http://localhost:7741/api/sprites/upload/eye_left/open   -F "file=@eye_open.png"
curl -X POST http://localhost:7741/api/sprites/upload/eye_left/closed -F "file=@eye_closed.png"
curl -X POST http://localhost:7741/api/sprites/upload/eye_left/happy  -F "file=@eye_happy.png"
```

Files are saved to `data/sprites/{part}_{variant}.png` and immediately available at `/sprites/{part}_{variant}.png`. Reload the page to see them.

To revert a part to the procedural placeholder:

```bash
curl -X DELETE http://localhost:7741/api/sprites/upload/head/default
```

---

## Reload and iterate

After uploading sprites or saving a new config, **reload the TamAGI page**. The frontend fetches the config and preloads all PNGs during startup. Once loaded, the avatar renders your artwork for any part that has a matching file, and procedural placeholders for the rest.

Changes to `data/sprite_config.json` and `data/sprites/*.png` take effect on the next page load — no server restart needed.

---

## Quick reference — API endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/api/sprites/config` | Fetch the current sprite config (returns `{"skeleton":null,"sprites":null}` if none saved) |
| `PUT` | `/api/sprites/config` | Save a full config JSON (from the rig editor export) |
| `POST` | `/api/sprites/upload/{part}/{variant}` | Upload a PNG for one slot |
| `DELETE` | `/api/sprites/upload/{part}/{variant}` | Remove a PNG, reverting that slot to placeholder |

Sprites are served as static files at `/sprites/{part}_{variant}.png`.
