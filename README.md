# fast-edit

**fast-edit** is a lightweight, Lightroom-inspired RAW and image editor built around a true linear-light float32 develop pipeline, data-driven JSON filter presets, per-photo sidecar state, and multi-photo project/batch management.

The repository includes two implementations that share the exact same [filters/](filters/) preset library and sidecar/project formats:
1. **Python / CustomTkinter (`fast-edit-v2.py`)** — easy one-command setup via virtual environment.
2. **Native C11 / GTK3 + OpenMP ([`fast-edit-c/`](fast-edit-c/))** — high-performance native GUI and CLI with headless rendering, batch export, and self-tests.

---

## Features

- **10-Stage Linear Float32 Develop Pipeline**:
  1. **RAW Decode** — decodes Sony `.ARW` and other RAW formats (plus standard `.jpg`, `.png`, `.tif`, `.webp`, `.bmp`) into linear float32 RGB.
  2. **Camera Calibration** — rotates/scales RGB primaries globally for cohesive color shifts without hard hue masking.
  3. **White Balance** — temperature and tint channel gains in linear light.
  4. **Exposure** — linear gain with a filmic highlight roll-off shoulder.
  5. **Tone & Curve** — shadows/highlights recovery, contrast, blacks/whites, and parametric tone curve with matte (`fade`) black point.
  6. **8-Band HSL Color Mixer** — soft overlapping hue bands (`red`, `orange`, `yellow`, `green`, `aqua`, `blue`, `purple`, `magenta`).
  7. **3-Way Color Grading** — shadows, midtones, and highlights split-toning wheels.
  8. **Vibrance & Saturation** — skin-tone and high-saturation aware vibrance.
  9. **Presence** — texture, clarity, and dehaze (local contrast).
  10. **Optical & Analog Effects** — threshold bloom, luminance-mapped film grain, soft vignette, chromatic aberration, scanlines, VCR timestamp overlays, lens distortion/vignette correction, and retro pixel-art palettes/dithering.
- **Data-Driven Filter Library**: 60+ presets across 11 categories in [`filters/`](filters/) (Sage & Muted, Film, B&W, Clean & Airy, Moody & Dark, Vintage, Golden & Warm, Cool & Clean, Lo-Fi & Damage, Pixel Art).
- **Per-Photo Non-Destructive Edits**: Every edited photo automatically saves its active filter and slider offsets to `<photo>.fastedit.json` next to the image.
- **Project & Batch Workflow**: Load folders or curated photo lists, save/load `.feproj` project files (with embedded per-photo edit states), apply a preset across selected photos while preserving individual slider tweaks, sync full settings across a shoot, or batch-export at full resolution.

---

## Quick Start (Python Version)

### Prerequisites
- Python 3.9+ with `venv` and `tkinter` installed (e.g. `sudo apt install python3-venv python3-tk` on Ubuntu/Debian).

### Run
From the repository root:

```bash
make run
```

What `make run` does:
1. Creates a `.venv` virtual environment (on first run).
2. Installs dependencies (`numpy`, `scipy`, `pillow`, `opencv-python-headless`, `exifread`, `rawpy`, `customtkinter`).
3. Launches `fast-edit-v2.py`.

To remove the virtual environment:
```bash
make clean
```

---

## Quick Start (Native C / GTK3 Version)

See [`fast-edit-c/README.md`](fast-edit-c/README.md) for full details on the C implementation and CLI options.

### Prerequisites (Ubuntu / Debian)
```bash
sudo apt install build-essential pkg-config libgtk-3-dev libcairo2-dev libraw-dev
```
*(Optional: `libopencv-dev` and `libonnxruntime-dev` are auto-detected via `pkg-config` if installed.)*

### Build & Run
```bash
cd fast-edit-c
make run
```

### Headless CLI Usage
The C binary (`fast-edit-c/build/fast-edit`) also runs headlessly without a display:
```bash
# List all available filter categories and keys
./fast-edit-c/build/fast-edit --list

# Render a single photo with a filter preset to PNG
./fast-edit-c/build/fast-edit --render input.ARW output.png 01-sage-muted/light-sage

# Render a contact sheet of every filter preset into a directory
./fast-edit-c/build/fast-edit --sheet input.ARW /tmp/sheet

# Run numerical pipeline & project self-tests
./fast-edit-c/build/fast-edit --selftest
```

---

## Editor Usage Guide

1. **Opening Photos & Folders**
   - **Open Photo**: Load a single RAW (`.ARW`, `.CR2`, `.NEF`, `.DNG`, etc.) or standard image file.
   - **Open Folder**: Populate the bottom film-strip with all supported images in a directory for rapid culling and editing.
2. **Applying Filters & Fine-Tuning**
   - Browse filter categories on the right panel and click any preset to apply it.
   - Use the left-hand tabs (**Fine-Tune**, **Grain**, **Effects**, **Lens**) to adjust sliders.
   - **Slider Offsets**: Sliders represent *offsets* relative to the active filter preset (`0` = the filter's default authored look). Switching filters preserves or rebases cleanly, and your adjustments are debounced during dragging for smooth real-time previews.
3. **Automatic Sidecar Saving**
   - Modifying any filter or slider automatically writes `<filename>.fastedit.json` next to your image. Merely viewing an unedited photo does not clutter your folder with sidecar files.
4. **Projects & Batch Operations**
   - **Add to Project / Project Manager**: Collect images from one or more folders into a project.
   - **Apply Filter to Selected**: Changes the base filter on all selected project photos while keeping each photo's custom exposure/slider offsets intact.
   - **Sync Edits to Selected**: Copies both the active filter and all slider offsets from the current photo to all selected photos.
   - **Save / Load Project (`.feproj`)**: Saves the image list and embeds each photo's edit state into a portable JSON project file.
   - **Batch Export**: Exports all selected project photos at full resolution.

---

## Creating Custom Filters

Filters live in [`filters/<category-folder>/<filter-name>.json`](filters/) and are discovered automatically at startup—no code changes or recompilation required.

Each category folder contains a `category.json`:
```json
{
  "label": "Sage & Muted",
  "color": "#638C75"
}
```

And one or more `<filter-name>.json` files:
```json
{
  "label": "Light Sage",
  "description": "Airy muted botanical green.",
  "settings": {
    "temperature": 0.05,
    "exposure": 0.12,
    "contrast": -0.05,
    "highlights": -0.12,
    "shadows": 0.06,
    "fade": 0.025,
    "mixer": {
      "green": [-12.0, -0.22, 0.08]
    },
    "grade_highlight": [44.0, 0.04],
    "grain_amount": 0.018,
    "vignette_amount": 0.08
  },
  "effects": {
    "chromatic_aberration": 0,
    "scanlines": 0.0,
    "vcr_overlay": false
  }
}
```

---

## Repository Structure

```text
.
├── Makefile               # Python venv setup & launcher (`make run`)
├── fast-edit-v2.py        # Python / CustomTkinter Lightroom-style editor
├── fast-edit.py           # Legacy v1 prototype
├── filters/               # Shared JSON filter categories and presets
└── fast-edit-c/           # Native C11 / GTK3 / OpenMP port & CLI
    ├── Makefile
    ├── src/               # C pipeline, GTK3 GUI, project/sidecar state, lens & vision
    └── vendor/            # Bundled single-file libraries (cJSON, stb_image)
```
