# fast-edit-c (Native C11 / GTK3 Port)

**fast-edit-c** is the native C11 and OpenMP implementation of **fast-edit**. It reproduces the Python editor's float32 linear-light develop pipeline with native multithreaded performance, a GTK3 dark-themed desktop interface, and a headless CLI for batch rendering and automated self-testing.

---

## Dependencies

`fast-edit-c`'s [`Makefile`](Makefile) uses `pkg-config` to detect libraries automatically:

- **Compiler**: GCC or Clang with C11 and OpenMP support (`build-essential`)
- **GUI**: `gtk+-3.0` and `cairo` (`libgtk-3-dev`, `libcairo2-dev`) — builds headless CLI-only if GTK3 is absent
- **RAW Decoding**: `libraw` (`libraw-dev`) — falls back to `dcraw` if `libraw` headers are unavailable
- **Optional**: `opencv4` (`libopencv-dev`) and `onnxruntime` (`libonnxruntime-dev`)

On Ubuntu/Debian:
```bash
sudo apt install build-essential pkg-config libgtk-3-dev libcairo2-dev libraw-dev
```

Check detected dependencies at any time with:
```bash
make deps
```

---

## Building & Running

```bash
# Build ./build/fast-edit
make

# Build and launch the GTK3 GUI editor
make run

# Run numerical pipeline and project/sidecar self-tests
make selftest

# Render every filter preset on an input image into an output directory
make sheet IN=path/to/photo.ARW OUTDIR=/tmp/sheet

# Clean build artifacts
make clean
```

### Custom Sysroot (Rootless Builds)
If development headers are unpacked in a local sysroot directory rather than installed system-wide:
```bash
make SYSROOT=/path/to/sysroot
```

---

## Command-Line Interface (Headless Mode)

The compiled binary `./build/fast-edit` supports both interactive GUI mode and headless CLI operations:

```text
usage: ./build/fast-edit [options]
  (no options)              open the GTK3 editor
  --list                    list all filter categories and keys
  --render IN OUT KEY       render one filter preset to OUT (PNG)
  --decode IN OUT           write the decoded base image (no filter)
  --sheet  IN OUTDIR        render every filter preset into OUTDIR
  --selftest [ref.json]     run numeric pipeline and project self-checks
  --max-edge N              limit preview/render long edge to N pixels (default: full)
  --dither                  apply TPDF dither on 8-bit output
```

### Examples

1. **List all filters**:
   ```bash
   ./build/fast-edit --list
   ```
2. **Render a single preset at full resolution**:
   ```bash
   ./build/fast-edit --render DSC0001.ARW out.png 02-film/portra-400
   ```
3. **Render a fast 1200px preview with TPDF dithering**:
   ```bash
   ./build/fast-edit --max-edge 1200 --dither --render DSC0001.ARW preview.png 05-moody-dark/midnight-rain
   ```

---

## Architecture Overview (`src/`)

- [`main.c`](src/main.c) — CLI argument parsing, headless commands (`--list`, `--render`, `--sheet`), and `--selftest` suite.
- [`gui.c`](src/gui.c) — GTK3 3-column interface (fine-tune slider tabs, Cairo viewport preview, filter browser, folder/project film-strip).
- [`fe_pipeline.c`](src/fe_pipeline.c) / [`fe_math.c`](src/fe_math.c) — OpenMP-accelerated float32 linear-light develop pipeline (calibration, WB, exposure, PCHIP tone curves, 8-band HSL mixer, 3-way color grading, presence, grain, bloom, vignette, pixel-art quantization).
- [`fe_filters.c`](src/fe_filters.c) / [`fe_fields.c`](src/fe_fields.c) — JSON filter loader (`../filters/`) and named setting field reflection/offset application.
- [`fe_state.c`](src/fe_state.c) — Per-photo non-destructive sidecar reading/writing (`<photo>.fastedit.json`).
- [`fe_project.c`](src/fe_project.c) — Multi-photo project management (`.feproj`), multithreaded thumbnail generation, batch filter application, state syncing, and multithreaded batch export.
- [`fe_lens.c`](src/fe_lens.c) / [`fe_lens_auto.c`](src/fe_lens_auto.c) — Radial distortion (`k1`, `k2`), transverse chromatic aberration (`tca_r`, `tca_b`), vignette correction, and automatic edge-based lens analysis.
- [`fe_io.c`](src/fe_io.c) / [`fe_overlay.c`](src/fe_overlay.c) — LibRaw / `stb_image` decoding, EXIF date parsing, and VCR / scanline / chromatic aberration effects.
