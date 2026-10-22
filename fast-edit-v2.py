#!/usr/bin/env python3
"""
fast-edit v2 — a small Lightroom-style RAW editor for Sony ARW files.

WHY THIS FILE EXISTS (v1 → v2)
==============================
v1 (fast-edit.py) applied "presets" as a chain of blunt image operations:
a hard hue mask (`h > 20 & h < 90`), global HSV saturation multipliers and
per-channel additions.  That produced visible banding, dead greens and a
"filtered JPEG" feeling instead of the soft, integrated look Lightroom
presets such as *Light Sage* / *Deep Sage* produce.

v2 re-implements the app around a real develop pipeline that mirrors the
order Lightroom applies its adjustments in:

    1.  Decode RAW          → linear float32 (not 8-bit!)
    2.  Calibration         → rotate/scale the RGB *primaries*
                              (changes the colour "DNA" of EVERY pixel,
                              which is why LR looks integrated, not masked)
    3.  White balance       → temperature / tint as channel gains
    4.  Exposure            → linear gain + filmic highlight shoulder
    5.  Tone (Basic panel)  → shadows / highlights recovery, parametric
                              tone curve with matte ("fade") black point
    6.  Color Mixer (HSL)   → 8 soft, overlapping hue bands — no hard masks
    7.  Color Grading       → three-way tint wheels (shadows / mids / highs)
    8.  Vibrance / Saturation → vibrance protects skin & saturated colours
    9.  Presence            → texture / clarity / dehaze (local contrast)
    10. Effects             → threshold bloom, luminance-aware grain, vignette

The "sage" presets therefore don't *select* green pixels and crush them;
they re-interpret colour globally (calibration), then gently steer the
colour bands (mixer) and tint the tonal ranges apart (grading).  That is
exactly the recipe behind Lightroom's muted-green looks:

    • green primary rotated toward gold  → foliage loses its "digital neon"
    • blues desaturated globally         → calm, cohesive palette
    • lifted blacks + soft highlights    → film-like, "satiny" tonality
    • cool-green shadows vs warm highs   → the dreamy / otherworldly split
    • light grain + wide soft vignette   → texture that glues it together

Everything below is commented with the *reason* for the math, not just
what the math does, so the project can be read top-to-bottom.

FILTERS AS DATA
===============
Looks are no longer hard-coded in Python.  Every filter lives in
./filters/<category>/<filter-name>.json and is loaded at startup:

    filters/
      01-sage-muted/
        category.json      {"label": "Sage & Muted", "color": "#638C75"}
        light-sage.json    {"label", "description", "settings", "effects"}

A filter's "settings" block is a partial DevelopSettings (omitted keys keep
their defaults, like LR presets that only move a few sliders) and an optional
"effects" block declares analog damage (chromatic aberration / scanlines /
VCR overlay) applied after develop.  Drop a new JSON into a folder and it
appears in the GUI's category accordion — no code changes needed.
"""

from __future__ import annotations

import colorsys
import copy
import json
import os
from dataclasses import dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from scipy.interpolate import PchipInterpolator

import customtkinter as ctk
from tkinter import filedialog
from PIL import Image

import rawpy
import exifread

# =============================================================================
# RESOLUTION REFERENCE
#    Every pixel-sized effect in this app (grain clumps, bloom radius, texture
#    detail) is authored assuming the image's long edge is PREVIEW_LONG_EDGE.
#    develop() rescales those parameters for any other resolution, so a
#    full-res export looks exactly like the preview you approved on screen
#    instead of coming out finer-grained and tighter-glowed.
# =============================================================================
PREVIEW_LONG_EDGE = 1800
# While a slider is being dragged we develop an even smaller draft and only
# pay for the full preview once the slider is released — the same trick
# Lightroom's "Preview quality" setting uses.  This affects ONLY the transient
# on-screen image during the drag; the released preview and every export are
# always full quality.
DRAFT_LONG_EDGE = 850

# =============================================================================
# 1. LOW-LEVEL COLOUR MATH
#    All functions operate on float32 arrays shaped (H, W, 3) with values in
#    [0, 1] unless stated otherwise.  Working in float avoids the rounding
#    damage v1 suffered by hopping through uint8 after every step.
# =============================================================================

# Rec. 709 luma weights — the same ones video/broadcast uses for "brightness".
LUMA_W = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def luma(img: np.ndarray) -> np.ndarray:
    """Perceptual brightness, shape (H, W)."""
    return img @ LUMA_W


def srgb_to_linear(img: np.ndarray) -> np.ndarray:
    """Undo the sRGB gamma.  Light maths (exposure, WB) must happen in linear
    light, otherwise bright pixels get treated like dark ones."""
    return np.where(img <= 0.04045, img / 12.92, ((img + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(img: np.ndarray) -> np.ndarray:
    """Re-apply the sRGB gamma so the image looks correct on screen."""
    img = np.maximum(img, 0.0)
    return np.where(img <= 0.0031308, img * 12.92, 1.055 * img ** (1.0 / 2.4) - 0.055).astype(np.float32)


def soft_clip(x: np.ndarray, shoulder: float = 0.72) -> np.ndarray:
    """Filmic highlight rolloff.

    Plain clipping (x -> min(x, 1)) turns bright skies into flat white paper.
    Real film compresses highlights gracefully; a tanh shoulder does the same:
    values below `shoulder` are untouched, values above bend towards 1.0
    without ever reaching it abruptly.  This is the single biggest reason
    digital edits look "digital".
    """
    x = np.maximum(x, 0.0)
    over = x - shoulder
    return np.where(x <= shoulder, x, shoulder + (1.0 - shoulder) * np.tanh(over / (1.0 - shoulder)))


def rgb_to_hue(img: np.ndarray) -> np.ndarray:
    """Hue in degrees [0, 360), shape (H, W).  Matches the HSV wheel Lightroom's
    colour labels sit on (red 0°, yellow 60°, green 120°, aqua 180°, blue 240°).
    Thin wrapper over rgb_to_hsv_view so there is only one implementation."""
    return rgb_to_hsv_view(img)[0]


def rgb_to_sat(img: np.ndarray) -> np.ndarray:
    """HSV-style saturation (chroma relative to brightness), shape (H, W).
    Thin wrapper over rgb_to_hsv_view so there is only one implementation."""
    return rgb_to_hsv_view(img)[1]


def rgb_to_hsv_view(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hue, saturation and value in ONE pass.

    hue/sat/val each need the same per-pixel max & min, and at 24 MP those
    reductions dominate the runtime — computing them together instead of
    calling rgb_to_hue/rgb_to_sat separately cuts a quarter of the work.
    """
    maxc = img.max(axis=-1)
    minc = img.min(axis=-1)
    chroma = maxc - minc
    safe = np.where(chroma > 1e-8, chroma, 1.0)
    sat = np.where(maxc > 1e-8, chroma / np.maximum(maxc, 1e-8), 0.0).astype(np.float32)

    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    hue = np.zeros_like(maxc)
    rm = (chroma > 1e-8) & (maxc == r)
    gm = (chroma > 1e-8) & (maxc == g) & ~rm
    bm = (chroma > 1e-8) & (maxc == b) & ~rm & ~gm
    hue[rm] = ((g - b)[rm] / safe[rm]) % 6
    hue[gm] = ((b - r)[gm] / safe[gm]) + 2
    hue[bm] = ((r - g)[bm] / safe[bm]) + 4
    return (hue * 60.0) % 360.0, sat, maxc


def hsv_to_rgb(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Vectorised inverse of the two functions above.  h in degrees, s/v in [0, 1]."""
    h6 = (h % 360.0) / 60.0
    i = np.floor(h6).astype(np.uint8) % 6
    f = h6 - np.floor(h6)
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    # Pick each sector's components with boolean masks.  (np.choose would be
    # the obvious tool but benchmarks ~2x slower on multi-megapixel frames.)
    m0, m1, m2 = i == 0, i == 1, i == 2
    m3, m4, m5 = i == 3, i == 4, i == 5
    r = v * m0 + q * m1 + p * m2 + p * m3 + t * m4 + v * m5
    g = t * m0 + v * m1 + v * m2 + q * m3 + p * m4 + p * m5
    b = p * m0 + p * m1 + t * m2 + v * m3 + v * m4 + q * m5
    return np.stack([r, g, b], axis=-1).astype(np.float32)


def smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    """S-curve falloff between two edges.  Used everywhere instead of hard
    thresholds — hard thresholds are what create banding and masks."""
    t = np.clip((x - edge0) / (edge1 - edge0 + 1e-9), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def cheap_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Large-radius Gaussian blur that stays fast on 24 MP images.

    cv2.GaussianBlur slows down linearly with sigma; shrinking the image first,
    blurring, then scaling back up is visually identical for big radii and
    ~100x faster.  Works for both (H, W) and (H, W, 3) arrays.
    """
    squeezed = img.ndim == 2
    src = img[..., None] if squeezed else img
    h, w = src.shape[:2]
    step = max(1.0, sigma / 6.0)
    sw, sh = max(2, int(w / step)), max(2, int(h / step))
    small = cv2.resize(src, (sw, sh), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), max(0.5, sigma / step))
    out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return out[..., 0] if squeezed else out


# =============================================================================
# 2. ADJUSTMENT STAGES
#    Each function is one "panel" of the Lightroom develop module.  They are
#    deliberately separate so presets can be read as a list of slider values.
# =============================================================================


def calibration_matrix(red_h: float = 0.0, green_h: float = 0.0, blue_h: float = 0.0,
                       red_s: float = 1.0, green_s: float = 1.0, blue_s: float = 1.0) -> np.ndarray:
    """Build a 3x3 colour matrix that mimics Lightroom's Calibration panel.

    WHY THIS IS THE SECRET SAUCE
    ----------------------------
    The HSL/Mixer only touches pixels that *already look* green.  The
    Calibration panel instead redefines what "red / green / blue" *mean* for
    every pixel in the frame — including the mixed colours inside a leaf, a
    face or the sky.  Because the shift happens at the "sensor" level, the
    result reads as one cohesive interpretation of the scene rather than a
    patchwork of local edits.  That is what makes LR presets feel both
    natural (no seams) and otherworldly (the whole palette moved together).

    MATH
    ----
    • Take each pure primary (red 0°, green 120°, blue 240° on the hue wheel)
      and rotate it by its *_h slider, exactly like turning Adobe's hue dial.
    • Scale the rotated primary by *_s (blue saturation +20 in Adobe == a
      richer, more "Fuji" palette; -12 == calmer, filmic colours).
    • Put the three new primaries in the columns of a matrix; normalise the
      rows to sum to 1 so neutral grey stays neutral grey (no colour cast
      from a pure contrast change).
    """
    primaries = []
    for base_hue, shift, sat in ((0.0, red_h, red_s),
                                 (120.0, green_h, green_s),
                                 (240.0, blue_h, blue_s)):
        rgb = colorsys.hsv_to_rgb(((base_hue + shift) % 360.0) / 360.0, 1.0, 1.0)
        primaries.append(np.array(rgb, dtype=np.float32) * sat)
    m = np.stack(primaries, axis=1)                # columns = new primaries
    m /= m.sum(axis=1, keepdims=True)              # grey in → grey out
    return m


def apply_calibration(img: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Apply the primary matrix to a linear-light image."""
    return img @ m.T


def white_balance(img: np.ndarray, temperature: float, tint: float) -> np.ndarray:
    """White balance as channel gains, applied in LINEAR light.

    temperature > 0 → warmer (R up, B down), like moving LR's Temp slider right.
    tint        > 0 → greener (LR's Tint slider is green↔magenta).
    The gains are re-normalised against luma so a WB change doesn't also
    change overall brightness (Lightroom behaves the same way).
    """
    if abs(temperature) < 1e-6 and abs(tint) < 1e-6:
        return img
    gains = np.array([
        1.0 + 0.30 * temperature - 0.08 * tint,
        1.0 + 0.16 * tint,
        1.0 - 0.30 * temperature - 0.08 * tint,
    ], dtype=np.float32)
    gains /= gains @ LUMA_W                          # keep exposure constant
    return img * gains


def range_adjust(img: np.ndarray, shadows: float = 0.0, highlights: float = 0.0) -> np.ndarray:
    """Lightroom's Shadows / Highlights sliders.

    Lifts or compresses only the dark (shadows) or bright (highlights) parts
    of the image, with a smooth luminance mask so the transition into the
    midtones is invisible.  Only the luminance is changed; the RGB channels
    are scaled together so hue and saturation survive the move.
    """
    if abs(shadows) < 1e-6 and abs(highlights) < 1e-6:
        return img
    L = luma(img)
    # Near-black protection: like Lightroom, the Shadows slider lifts actual
    # shadows but leaves true black untouched — otherwise dividing by a tiny
    # luminance would amplify sensor noise 100x in crushed areas.
    shadow_mask = (1.0 - smoothstep(0.0, 0.62, L)) * smoothstep(0.0, 0.08, L)
    high_mask = smoothstep(0.38, 1.0, L)
    delta = shadows * shadow_mask + highlights * high_mask
    factor = (L + delta) / np.maximum(L, 1e-4)
    return np.clip(img * factor[..., None], 0.0, 1.0)


def tone_curve_lut(contrast: float = 0.0, blacks: float = 0.0, whites: float = 0.0,
                   fade: float = 0.0, points: int = 1024) -> np.ndarray:
    """Parametric tone curve, baked into a LUT.

    Builds the same kind of curve as LR's Tone Curve panel:
      • contrast — pivot-based S-curve around mid-grey (0.45 display)
      • blacks   — moves the very bottom of the curve (crush or lift)
      • whites   — moves the very top (recover or boost)
      • fade     — lifts ONLY the black point: the "matte / satiny" look that
                   every sage, film and muted preset relies on.  No pixel ever
                   reaches pure black, which softens the whole image.

    PCHIP (monotone cubic) interpolation keeps the curve smooth without the
    overshoot a plain cubic spline would add — no accidental halo bumps.
    """
    xs = np.array([0.0, 0.08, 0.25, 0.5, 0.75, 0.92, 1.0], dtype=np.float32)
    pivot = 0.45
    ys = pivot + (xs - pivot) * (1.0 + contrast)      # S-curve around pivot
    ys = ys + blacks * (1.0 - xs) ** 1.6              # shadow end
    ys = ys - whites * xs ** 1.6                      # highlight end
    ys = ys + fade * (1.0 - xs) ** 3.0                # matte black lift
    ys = np.clip(ys, 0.0, 1.0)
    curve = PchipInterpolator(xs, ys)
    return np.clip(curve(np.linspace(0.0, 1.0, points)), 0.0, 1.0).astype(np.float32)


def apply_lut(img: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Apply a [0,1]→[0,1] lookup curve to every channel (C-speed via np.interp)."""
    x = np.linspace(0.0, 1.0, lut.size, dtype=np.float32)
    return np.interp(img, x, lut).astype(np.float32)


# Lightroom's eight colour-mixer bands and their hue centres (degrees).
MIXER_BANDS: Dict[str, float] = {
    "red": 0.0, "orange": 30.0, "yellow": 60.0, "green": 120.0,
    "aqua": 180.0, "blue": 240.0, "purple": 285.0, "magenta": 330.0,
}


def color_mixer(img: np.ndarray, adjustments: Dict[str, Tuple[float, float, float]]) -> np.ndarray:
    """Lightroom's HSL / Color Mixer panel.

    adjustments maps a band name → (hue shift°, saturation Δ, luminance Δ),
    where 0.0 = no change on the last two, e.g. {"green": (-18, -0.45, +0.18)}
    moves greens 18° toward yellow, cuts their saturation by 45% and
    brightens them 18%.

    WHY SOFT BANDS MATTER
    ---------------------
    v1 masked green with a hard `h > 20 & h < 90` test: pixels at hue 19°
    were untouched while 21° got crushed → banding and "cut-out" colours.
    Here every pixel receives a smooth weight per band (full inside ±35°,
    fading to zero at ±75°) and the weighted adjustments are summed.  Bands
    overlap on purpose — Lightroom's sliders overlap too, and that overlap is
    why a green→yellow shift also warms the yellow edge of the same leaf
    instead of stopping at an arbitrary line.
    """
    if not adjustments:
        return img
    hue, sat, val = rgb_to_hsv_view(img)

    d_hue = np.zeros_like(hue)      # accumulated hue shifts (degrees)
    s_delta = np.zeros_like(hue)    # accumulated saturation deltas (0 = untouched)
    v_delta = np.zeros_like(hue)    # accumulated luminance deltas

    for band, (shift, sat_delta_band, val_delta_band) in adjustments.items():
        centre = MIXER_BANDS[band]
        distance = np.abs((hue - centre + 180.0) % 360.0 - 180.0)   # wrap-aware
        weight = smoothstep(75.0, 35.0, distance)                  # 1 inside ±35°, 0 at ±75°
        # Accumulate, don't stack: a pixel where two bands overlap gets the
        # *sum* of their edits.  Since 0.0 means "no change", a neutral band
        # contributes exactly zero no matter how wide its weight is.
        d_hue += weight * shift
        s_delta += weight * sat_delta_band
        v_delta += weight * val_delta_band

    new_sat = np.clip(sat * (1.0 + s_delta), 0.0, 1.0)
    new_val = np.clip(val * (1.0 + v_delta), 0.0, 1.0)
    return hsv_to_rgb(hue + d_hue, new_sat, new_val)


def color_grading(img: np.ndarray,
                  shadow: Tuple[float, float] = (0.0, 0.0),
                  midtone: Tuple[float, float] = (0.0, 0.0),
                  highlight: Tuple[float, float] = (0.0, 0.0),
                  blending: float = 0.45) -> np.ndarray:
    """Lightroom's three-way Color Grading wheels (formerly Split Toning).

    Each argument is (hue°, saturation 0..1).  The tonal ranges form a clean
    partition of the luminance axis: shadows fade out, highlights fade in,
    midtones fill whatever is left, so the three weights always sum to 1.

    The tint is added as CHROMA ONLY — the tint colour's own brightness is
    subtracted before adding, so grading changes *colour* without changing
    *exposure*.  This is the trick behind "natural" filmic grades: the
    shadows go cool-green and the highlights warm-cream, yet the histogram
    barely moves.

    The sage looks get their "otherworldly" quality almost entirely from this
    panel: a gentle cool-green wash in the shadows against creamy highlights
    creates colour separation that never exists in-camera.
    """
    if all(abs(s) < 1e-6 for _, s in (shadow, midtone, highlight)):
        return img
    L = luma(img)
    width = 0.5 + blending * 0.5
    shadow_w = 1.0 - smoothstep(0.0, width, L)
    high_w = smoothstep(1.0 - width, 1.0, L)
    mid_w = np.clip(1.0 - shadow_w - high_w, 0.0, 1.0)

    tint = np.zeros_like(img)
    for (hue, strength), weight in ((shadow, shadow_w), (midtone, mid_w), (highlight, high_w)):
        if strength < 1e-6:
            continue
        wheel = hsv_to_rgb(np.full_like(L, hue), np.full_like(L, strength), np.ones_like(L))
        wheel -= luma(wheel)[:, :, None]             # strip brightness, keep colour
        tint += wheel * weight[:, :, None]
    return np.clip(img + tint, 0.0, 1.0)


def saturation_vibrance(img: np.ndarray, saturation: float = 0.0, vibrance: float = 0.0) -> np.ndarray:
    """Global Saturation + smart Vibrance.

    • saturation scales every colour equally (harsh, kills skin tones fast).
    • vibrance acts mostly on the *muted* colours: its weight is (1 - sat)²,
      so already-vivid pixels barely move while dull ones come alive.  It is
      additionally halved on skin hues (10°–50°), which is exactly how
      Lightroom avoids orange faces when you push Vibrance to +60.
    """
    if abs(saturation) < 1e-6 and abs(vibrance) < 1e-6:
        return img
    hue, sat, val = rgb_to_hsv_view(img)      # one pass feeds skin mask + rebuild
    s = sat * (1.0 + saturation)
    if abs(vibrance) > 1e-6:
        muted_weight = (1.0 - sat) ** 2
        skin = smoothstep(5.0, 12.0, hue) * (1.0 - smoothstep(45.0, 55.0, hue))
        muted_weight *= 1.0 - 0.5 * skin
        s = s * (1.0 + vibrance * muted_weight)
    return hsv_to_rgb(hue, np.clip(s, 0.0, 1.0), val)


def presence(img: np.ndarray, texture: float = 0.0, clarity: float = 0.0,
             dehaze: float = 0.0, pixel_scale: float = 1.0) -> np.ndarray:
    """Lightroom's Presence trio, implemented as multi-scale local contrast.

    • texture — small-radius detail (fabric, leaves, skin pores).
    • clarity — large-radius "midtone contrast"; adds punch and structure
                without touching exposure.  Positive = crisp, negative = the
                dreamy diffusion that pairs beautifully with sage.
    • dehaze  — subtracts the low-frequency glare veil; negative values ADD
                haze, which gives soft atmospheric depth.
    `pixel_scale` converts radii authored for the preview resolution to the
    current one, so texture stays the same *relative* size in a full-res
    export (clarity/dehaze already scale themselves by image dimensions).
    All work on RGB with modest amounts to avoid halos around edges.
    """
    if abs(texture) > 1e-3:
        img = img + texture * 0.7 * (img - cv2.GaussianBlur(img, (0, 0), 2.0 * pixel_scale))
    if abs(clarity) > 1e-3:
        img = img + clarity * 0.6 * (img - cheap_blur(img, sigma=max(img.shape[:2]) / 45.0))
    if abs(dehaze) > 1e-3:
        glare = cheap_blur(img, sigma=max(img.shape[:2]) / 14.0)
        img = img + dehaze * 0.8 * (img - glare)
    return np.clip(img, 0.0, 1.0)


def threshold_bloom(img: np.ndarray, amount: float = 0.0, radius: float = 25.0,
                    threshold: float = 0.72) -> np.ndarray:
    """Highlight-only glow (v1 blurred the whole image, which just made mud).

    Only pixels brighter than `threshold` contribute to the glow, so bright
    windows, specular hits and skies bleed softly into their surroundings
    the way halation does on film — while shadows stay clean and readable.
    """
    if amount < 1e-4:
        return img
    highlights = np.maximum(img - threshold, 0.0)
    glow = cheap_blur(highlights, sigma=radius)
    return np.clip(img + glow * amount, 0.0, 1.0)


# -----------------------------------------------------------------------------
# FILM GRAIN — modelled on darktable's grain module (which RawTherapee's film
# grain tool is a direct port of; src/iop/grain.c, Henrik Andersson, GPLv3).
# Two ideas from that implementation are what separate *film* grain from the
# "dirty lens / sensor noise" look this app used to produce:
#
#   1. POWER SPECTRUM.  darktable sums 3 octaves of simplex noise with
#      amplitudes fitted to scans of real film ("parametrization of octaves to
#      match power spectrum of real grain scans"): {0.234, 0.785, 1.215} at
#      frequencies {0.491, 0.944, 1.728}.  Note the amplitudes GROW with
#      frequency — the finest octave dominates, so grain energy sits at the
#      clump scale.  Gaussian-blurred white noise (what this app used before)
#      piles energy at LOW frequencies instead: that reads as a haze over the
#      whole frame, i.e. "someone vaped in front of the sensor".
#
#   2. PAPER RESPONSE.  Instead of adding noise linearly, the noise is pushed
#      through a logistic "paper curve" D(exposure): a displacement through a
#      sigmoid is self-limiting — grain is strongest where the curve is steep
#      (midtones) and vanishes where the emulsion saturates (true black and
#      blown highlights).  The mid-tone bias parameter shifts that band,
#      exactly like darktable's "mid-tone bias" control.
#
# Grain is applied to LUMINANCE only and the RGB channels are rescaled
# together, so hue and saturation survive (darktable works on Lab L).  A tiny
# correlated chroma component ("dye clouds") is added afterwards, masked by
# the local saturation so neutral areas stay neutral — coloured noise on grey
# is precisely the digital-chroma-noise tell we are removing.
# -----------------------------------------------------------------------------
GRAIN_OCTAVE_FREQ = (0.4910, 0.9441, 1.7280)   # darktable: matched to grain scans
GRAIN_OCTAVE_AMP = (0.2340, 0.7850, 1.2150)   # fine octaves dominate on purpose
GRAIN_OCTAVE_NORM = 1.465                     # sqrt(sum of squares of amplitudes)
GRAIN_STRENGTH_SCALE = 0.6                    # filter "amount" → density σ at midtones


def _value_noise_octaves(h: int, w: int, cell_px: float, rng: np.random.Generator) -> np.ndarray:
    """Multi-octave value noise standing in for darktable's simplex noise.

    Each octave is white noise generated on a coarse lattice and cubicly
    upsampled — spectrally that is band-limited noise with a sharp cutoff at
    the lattice frequency, the same role the simplex lattice plays, but fully
    vectorised.  `cell_px` is the lattice spacing of the finest reference
    octave in output pixels, so grain size scales with resolution like every
    other pixel-sized effect in this app.
    """
    total = np.zeros((h, w), np.float32)
    for freq, amp in zip(GRAIN_OCTAVE_FREQ, GRAIN_OCTAVE_AMP):
        cell = max(0.75, cell_px / freq)
        nh, nw = max(2, int(h / cell) + 2), max(2, int(w / cell) + 2)
        lattice = rng.standard_normal((nh, nw)).astype(np.float32)
        total += cv2.resize(lattice, (w, h), interpolation=cv2.INTER_CUBIC) * amp
    return total / GRAIN_OCTAVE_NORM


def paper_curve(density: np.ndarray, bias: float = 100.0, gamma: float = 1.0) -> np.ndarray:
    """darktable's paper response: logistic exposure→density curve.

        D(e) = (1+2δ)/(1+exp(4γ(0.5-e)/(1+2δ))) - δ

    Centred at 0.5 with slope γ there.  δ comes from the mid-tone bias: the
    higher the bias, the smaller δ, the steeper the curve around the centre —
    and the more strictly grain is confined to the midtones.
    """
    delta = 2.0 * float(np.exp((bias / 100.0) * np.log(1e-4)))   # 2e-4 at bias 100
    z = np.exp(np.clip(4.0 * gamma * (0.5 - density) / (1.0 + 2.0 * delta), -60.0, 60.0))
    return ((1.0 + 2.0 * delta) / (1.0 + z) - delta).astype(np.float32)


def paper_curve_inverse(density: np.ndarray, bias: float = 100.0, gamma: float = 1.0) -> np.ndarray:
    """Inverse of paper_curve (darktable's paper_resp_inverse)."""
    delta = 2.0 * float(np.exp((bias / 100.0) * np.log(1e-4)))
    d = np.clip(density, 1e-4, 1.0 - 1e-4)
    return (-np.log((1.0 + 2.0 * delta) / (d + delta) - 1.0)
            * (1.0 + 2.0 * delta) / (4.0 * gamma) + 0.5).astype(np.float32)


def film_grain(img: np.ndarray, amount: float = 0.0, size: float = 1.2,
               color: float = 0.02, bias: float = 100.0,
               pixel_scale: float = 1.0) -> np.ndarray:
    """Film grain via the darktable paper-response model (see block comment).

    amount — grain strength (σ of the density displacement at midtones)
    size   — coarseness in "grain diameters" (1.0 ≈ fine 100-speed, 2+ ≈ 3200)
    color  — dye-cloud chroma whisper, only where the image already has colour
    bias   — mid-tone bias: lower spreads grain into shadows/highlights,
             higher confines it to the midtones (darktable's control of same)
    """
    if amount < 1e-4:
        return img
    h, w = img.shape[:2]
    rng = np.random.default_rng(20261021)   # seeded: renders are reproducible
    cell_px = max(1.0, size * pixel_scale * 2.0)

    # -- luminance grain through the paper curve --------------------------------
    noise = _value_noise_octaves(h, w, cell_px, rng)
    L = luma(img)
    exposure = paper_curve_inverse(L, bias)
    new_L = paper_curve(exposure + noise * (amount * GRAIN_STRENGTH_SCALE), bias)
    # Rescale RGB together with L: hue/saturation survive, only "density" moves.
    # The ratio is ~1 wherever the curve saturated (blacks, highlights), and the
    # clip guards the tiny-L division without lifting true black.
    factor = np.clip(new_L / np.maximum(L, 1e-3), 0.2, 5.0)
    out = np.clip(img * factor[..., None], 0.0, 1.0)

    # -- dye clouds (chroma), masked by existing saturation ---------------------
    if color > 1e-4:
        clouds = _value_noise_octaves(h, w, cell_px * 1.6, rng)
        chroma = np.max(out, axis=-1) - np.min(out, axis=-1)
        # Only pixels that already carry dye may get coloured grain — neutral
        # areas would otherwise show RGB speckle, i.e. digital chroma noise.
        mask = smoothstep(0.01, 0.08, chroma)
        out = np.clip(out + (clouds * mask * (color * amount * 0.8))[..., None], 0.0, 1.0)
    return out


def vignette(img: np.ndarray, amount: float = 0.0, start: float = 0.35) -> np.ndarray:
    """Smooth elliptical darkening towards the corners.

    A cosine-style falloff via smoothstep — a hard or linear vignette reads as
    a cheap frame; a wide, gentle falloff subconsciously focuses the eye.
    """
    if amount < 1e-4:
        return img
    h, w = img.shape[:2]
    yy = (np.arange(h) - (h - 1) / 2.0) / (h / 2.0)
    xx = (np.arange(w) - (w - 1) / 2.0) / (w / 2.0)
    dist = np.sqrt(yy[:, None] ** 2 + xx[None, :] ** 2) / np.sqrt(2.0)
    mask = 1.0 - amount * smoothstep(start, 1.0, dist)
    return np.clip(img * mask[:, :, None], 0.0, 1.0)


# =============================================================================
# 3. THE DEVELOP SETTINGS (the "preset" is now just data — like an XMP file)
# =============================================================================


@dataclass
class DevelopSettings:
    """Every knob of the pipeline, mirroring Lightroom's develop module.

    Ranges are chosen so 0.0 == "no change"; presets only move what they need.
    """

    # -- Profile / White Balance -------------------------------------------------
    temperature: float = 0.0        # -1 cool … +1 warm
    tint: float = 0.0               # -1 magenta … +1 green

    # -- Calibration (primary colour DNA) ----------------------------------------
    cal_red_hue: float = 0.0        # degrees, + towards orange
    cal_green_hue: float = 0.0      # degrees, + towards aqua, - towards gold
    cal_blue_hue: float = 0.0       # degrees, + towards purple, - towards teal
    cal_red_sat: float = 1.0
    cal_green_sat: float = 1.0
    cal_blue_sat: float = 1.0

    # -- Basic tonality ----------------------------------------------------------
    exposure: float = 0.0           # stops
    contrast: float = 0.0           # -0.6 flat … +0.6 punchy
    highlights: float = 0.0         # -1 recover … +1 boost
    shadows: float = 0.0            # -1 crush … +1 lift
    blacks: float = 0.0             # curve bottom
    whites: float = 0.0             # curve top
    fade: float = 0.0               # matte black lift (the "film" slider)

    # -- Color Mixer: band → (hue°, sat Δ, lum Δ) --------------------------------
    mixer: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)

    # -- Color Grading wheels: (hue°, saturation 0..1) ---------------------------
    grade_shadow: Tuple[float, float] = (0.0, 0.0)
    grade_midtone: Tuple[float, float] = (0.0, 0.0)
    grade_highlight: Tuple[float, float] = (0.0, 0.0)
    grade_blending: float = 0.45

    # -- Colour intensity --------------------------------------------------------
    saturation: float = 0.0         # -1 mono … +1 extreme
    vibrance: float = 0.0           # smart saturation

    # -- Presence ----------------------------------------------------------------
    texture: float = 0.0
    clarity: float = 0.0
    dehaze: float = 0.0

    # -- Effects -----------------------------------------------------------------
    grain_amount: float = 0.0
    grain_size: float = 1.2
    grain_color: float = 0.02         # dye-cloud chroma whisper (0 = mono grain)
    grain_bias: float = 100.0         # mid-tone bias: lower = grain spreads wider
    vignette_amount: float = 0.0
    bloom_amount: float = 0.0
    bloom_radius: float = 25.0
    bloom_threshold: float = 0.72      # luminance above which pixels start to glow

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DevelopSettings":
        """Build settings from a JSON object (a filter file's "settings" block).

        Only the keys a filter actually needs are listed — everything else
        keeps its default, exactly like Lightroom presets that only move a
        handful of sliders.  An unknown key is a hard error: silently ignoring
        a typo like "exposre" would produce a look nobody can reproduce.
        """
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"unknown setting(s): {', '.join(unknown)}")
        return cls(**data)


# =============================================================================
# 4. FILTERS (JSON files in ./filters/<category>/<name>.json)
#     A filter is *just data*: label, description, develop settings and an
#     optional analog-damage effect chain.  Adding a new look means dropping
#     a new JSON into a category folder — no code changes, no restart logic.
# =============================================================================

# Files with these names are read as category metadata, not as filters.
CATEGORY_META_NAME = "category.json"


@dataclass
class Filter:
    """One entry of the filter browser, loaded from a JSON file."""

    key: str                       # unique id: "<category-dir>/<file-stem>"
    category: str                  # category dir name, e.g. "01-sage-muted"
    category_label: str            # display name ("Sage & Muted")
    label: str                     # button text ("Light Sage")
    description: str               # shown in the status bar on selection
    colour: str                    # button colour (category colour)
    settings: DevelopSettings      # the develop recipe
    effects: Dict[str, Any]        # analog damage chain, {} for none


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path.name}: {exc}") from exc


def load_filters(root: Path) -> Dict[str, List[Filter]]:
    """Scan ./filters and return {category_dir_name: [Filter, ...]}.

    Layout:
        filters/
          01-sage-muted/
            category.json          {"label": "Sage & Muted", "color": "#638C75"}
            light-sage.json        {"label", "description", "settings", "effects"}

    The numeric folder prefix controls ordering without extra config; the
    display label strips it.  Each filter file may contain:
        label       display name (default: file name title-cased)
        description one-line explanation shown on selection
        settings    partial DevelopSettings — omitted keys keep defaults
        effects     {"chromatic_aberration": px, "scanlines": 0.2,
                     "vcr_overlay": true} — applied AFTER develop

    Bad files don't take the app down: they're skipped with a message on
    stdout, and everything else keeps working.
    """
    categories: Dict[str, List[Filter]] = {}
    if not root.is_dir():
        print(f"filter folder not found: {root}")
        return categories

    for cat_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        # Category presentation metadata (label/colour), with sensible defaults.
        label = cat_dir.name
        colour = "#4A6FA5"
        if label[:2].isdigit() and label[2:3] == "-":
            label = label[3:]                      # strip "01-" ordering prefix
        label = label.replace("-", " ").title()
        meta_path = cat_dir / CATEGORY_META_NAME
        if meta_path.is_file():
            try:
                meta = _read_json(meta_path)
                label = str(meta.get("label", label))
                colour = str(meta.get("color", colour))
            except ValueError as exc:
                print(f"warning: {exc}")

        filters: List[Filter] = []
        for path in sorted(cat_dir.glob("*.json")):
            if path.name == CATEGORY_META_NAME:
                continue
            try:
                doc = _read_json(path)
                settings = DevelopSettings.from_dict(doc.get("settings", {}))
                effects = doc.get("effects") or {}
                if not isinstance(effects, dict):
                    raise ValueError('"effects" must be an object')
                filters.append(Filter(
                    key=f"{cat_dir.name}/{path.stem}",
                    category=cat_dir.name,
                    category_label=label,
                    label=doc.get("label", path.stem.replace("-", " ").title()),
                    description=doc.get("description", ""),
                    colour=colour,
                    settings=settings,
                    effects=effects,
                ))
            except ValueError as exc:
                print(f"warning: skipping filter {path}: {exc}")
        if filters:
            categories[cat_dir.name] = filters
    return categories


# =============================================================================
# 5. THE PIPELINE
# =============================================================================


def develop(linear_rgb: np.ndarray, s: DevelopSettings) -> np.ndarray:
    """Run one image through the full develop pipeline.

    `linear_rgb` is a float32 RGB array already in LINEAR light (decode the
    RAW, then srgb_to_linear if the source was gamma-encoded).  Returns a
    display-ready float RGB image in [0, 1].

    The order below is not arbitrary — it mirrors Lightroom's develop module
    so the sliders behave the way an LR user expects:
    calibration first (it redefines colour), then white balance, exposure and
    tone, then colour work (mixer, grading), then presence and effects.
    """
    # 2 — Calibration: recolour the primaries before anything else sees them.
    img = apply_calibration(linear_rgb, calibration_matrix(
        s.cal_red_hue, s.cal_green_hue, s.cal_blue_hue,
        s.cal_red_sat, s.cal_green_sat, s.cal_blue_sat))

    # 3 — White balance, in linear light where it physically belongs.
    img = white_balance(img, s.temperature, s.tint)

    # 4 — Exposure + filmic shoulder, then back to display gamma.
    img = soft_clip(img * (2.0 ** s.exposure), shoulder=0.72)
    img = linear_to_srgb(img)

    # 5 — Basic tonality: recover ranges, then bend the curve.
    img = range_adjust(img, shadows=s.shadows, highlights=s.highlights)
    img = apply_lut(img, tone_curve_lut(contrast=s.contrast, blacks=s.blacks,
                                        whites=s.whites, fade=s.fade))

    # 6 — Colour work, display space.
    # ORDER MATTERS: global saturation/vibrance runs BEFORE grading, the way
    # Lightroom applies Saturation first and lets the grading wheels sit on
    # top.  This is what lets a "Black & White Sepia" filter work: the image
    # is fully desaturated (saturation = -1) and the sepia tint is then added
    # by the shadow/highlight wheels — if grading ran first, a -1 saturation
    # would wipe the tint right back out.
    img = color_mixer(img, s.mixer)
    img = saturation_vibrance(img, saturation=s.saturation, vibrance=s.vibrance)
    img = color_grading(img, shadow=s.grade_shadow, midtone=s.grade_midtone,
                        highlight=s.grade_highlight, blending=s.grade_blending)

    # 7 — Presence and effects last: they texture the *finished* look.
    # Pixel-sized effects (grain clumps, bloom radius, texture detail) are
    # authored for PREVIEW_LONG_EDGE; scale them so a full-res export matches
    # the preview instead of coming out with 3x finer grain.
    px = max(img.shape[:2]) / PREVIEW_LONG_EDGE
    img = presence(img, texture=s.texture, clarity=s.clarity, dehaze=s.dehaze, pixel_scale=px)
    img = threshold_bloom(img, amount=s.bloom_amount, radius=s.bloom_radius * px,
                          threshold=s.bloom_threshold)
    img = vignette(img, amount=s.vignette_amount)
    img = film_grain(img, amount=s.grain_amount, size=max(0.8, s.grain_size),
                     color=s.grain_color, bias=s.grain_bias, pixel_scale=px)
    return np.clip(img, 0.0, 1.0)


def auto_exposure(linear_rgb: np.ndarray) -> float:
    """Solve the exposure (stops) that lights this photo correctly.

    Surveyed the Python-RAW-editor field for this one (GitHub: photo-pipe,
    photo-edit-app); both boil down to the same two-part recipe this uses:

      1. GEOMETRIC-MEAN KEY.  Take the log-average of linear luminance and
         move it to middle grey (0.18).  The geometric mean is the standard
         "perceived brightness" statistic (Reinhard's global tone-mapping key
         factor) because averaging in log space stops a handful of blown
         specular pixels from dragging the estimate — which is exactly what a
         arithmetic mean would do.

      2. HIGHLIGHT ANCHOR.  photo-pipe additionally caps the solve so the
         brightest highlights (99.9th percentile) may reach at most 1.6x
         white.  Without the cap, a backlit scene gets "corrected" straight
         into the shoulder and the highlights die.  Aiming slightly above
         white lets the filmic shoulder roll them off gracefully instead of
         clipping them flat.

    WB and calibration are exposure-neutral by construction (see their
    gain-normalisation and the "calibration preserves gray" test), so solving
    against the raw source is consistent with where develop() applies
    exposure — this value can be handed straight to the Exposure slider.
    Returns stops; clamping to the slider's range happens in the GUI.
    """
    L = luma(linear_rgb).ravel()
    L = L[L > 1e-4]                        # ignore true-black pixels entirely
    if L.size == 0:
        return 0.0
    log_avg = float(np.exp(np.mean(np.log(L))))
    if log_avg <= 1e-9:
        return 0.0
    stops = float(np.log2(0.18 / log_avg))                      # 1 — key solve
    anchor = float(np.percentile(L, 99.9))                      # 2 — highlight cap
    if anchor > 1e-6:
        stops = min(stops, float(np.log2(1.6 / anchor)))
    return stops


# =============================================================================
# 6. LO-FI EFFECTS (VHS / JDM overlay chain, applied AFTER develop)
# =============================================================================


def chromatic_aberration(img: np.ndarray, shift: float = 4.0) -> np.ndarray:
    """Fake lateral CA: pull the red and blue channels apart horizontally.

    Real lateral CA grows towards the frame edges; a constant offset is a
    cheap but convincing stand-in for a tired lens.  Runs in FLOAT — v1
    round-tripped through 8-bit here, which banded the bloom glow — and
    mirrors the border instead of rolling it, so no seam appears at the edge.
    """
    s = int(round(shift))
    if s <= 0:
        return img
    h, w = img.shape[:2]
    out = img.copy()
    red_pad = cv2.copyMakeBorder(img[..., 0], 0, 0, s, s, cv2.BORDER_REFLECT)
    blue_pad = cv2.copyMakeBorder(img[..., 2], 0, 0, s, s, cv2.BORDER_REFLECT)
    out[..., 0] = red_pad[:, 2 * s:2 * s + w]     # red slides left
    out[..., 2] = blue_pad[:, 0:w]                # blue slides right
    return out


def scanlines(img: np.ndarray, intensity: float = 0.18) -> np.ndarray:
    """Tape interlace lines: every third row slightly darkened (float-safe)."""
    mask = np.ones(img.shape[0], dtype=np.float32)
    mask[::3] = 1.0 - intensity
    return img * mask[:, None, None]


def quantize_u8(img: np.ndarray, dither: bool = False) -> np.ndarray:
    """float [0, 1] → uint8.

    dither=False — plain truncation (the original path, kept byte-identical).

    dither=True  — add triangular-PDF noise (±1 LSB, zero mean) before
    rounding.  Idea taken from shravankumar147/photo-edit-app, the one Python
    editor in the survey that dithers: without it, 8-bit quantisation carves
    smooth gradients (fade/matte looks are gradient-heavy by nature) into
    visible staircase bands.  Spreading each value over ±1 code level turns
    those bands into noise below the display's just-noticeable difference.
    The SAME noise is added to R, G and B — monochrome dither reads as
    texture, per-channel dither would read as the chroma artifacts we just
    removed from the grain.  Seeded, so exports stay reproducible.
    """
    x = np.clip(img, 0.0, 1.0) * 255.0
    if not dither:
        return x.astype(np.uint8)
    rng = np.random.default_rng(20261022)
    # Difference of two i.i.d. uniforms is triangular on (-1, 1): the classic
    # TPDF with peak-to-peak 2 code levels for a 1-bit-depth reduction.
    noise = rng.random(x.shape[:2], np.float32) - rng.random(x.shape[:2], np.float32)
    x += noise[..., None] if x.ndim == 3 else noise
    np.rint(x, out=x)
    np.clip(x, 0.0, 255.0, out=x)
    return x.astype(np.uint8)


def analog_damage_u8(img: np.ndarray, effects: Dict[str, Any], date_str: str,
                     dither: bool = False) -> np.ndarray:
    """Extra analog wear declared by a filter's "effects" block, AFTER develop.

    The effects dict comes straight from the filter JSON, so lo-fi looks are
    composable from data:
        {"chromatic_aberration": 7.0, "scanlines": 0.2, "vcr_overlay": true}
    Returns the final 8-bit image.  Everything stays in float until the very
    last step, so bloom, grain and scanlines never pick up quantisation
    banding the way v1's per-step uint8 conversions did.  CA shift scales
    with resolution like the other pixel-sized effects.

    `dither=True` is for exports only: the preview gets LANCZOS-downscaled
    after this, which re-quantises anyway and would wash the dither out — but
    a saved PNG/JPEG is viewed at 100%, where bands would show.
    """
    if effects:
        px = max(img.shape[:2]) / PREVIEW_LONG_EDGE
        ca = float(effects.get("chromatic_aberration", 0.0))
        if ca > 0.0:
            img = chromatic_aberration(img, shift=ca * px)
        sl = float(effects.get("scanlines", 0.0))
        if sl > 0.0:
            img = scanlines(img, intensity=sl)
        u8 = quantize_u8(img, dither=dither)
        if effects.get("vcr_overlay"):
            return vcr_overlay(u8, date_str)
        return u8
    return quantize_u8(img, dither=dither)


def read_exif_date(file_path: str) -> str:
    """VCR-style timestamp from EXIF, or a plausible fallback."""
    try:
        with open(file_path, "rb") as f:
            tags = exifread.process_file(f, details=False, stop_tag="EXIF DateTimeOriginal")
            if "EXIF DateTimeOriginal" in tags:
                dt = datetime.strptime(str(tags["EXIF DateTimeOriginal"]), "%Y:%m:%d %H:%M:%S")
                return dt.strftime("%b. %d %Y").upper() + "  " + dt.strftime("%p %I:%M")
    except Exception:
        pass
    return "OCT. 20 2026  PM 11:35"


def vcr_overlay(img_u8: np.ndarray, date_str: str) -> np.ndarray:
    """PLAY ▶ / date stamp, drawn at a resolution-dependent scale."""
    h, w = img_u8.shape[:2]
    scale = max(0.5, h / 900.0)
    thickness = max(1, int(scale * 1.5))
    font = cv2.FONT_HERSHEY_SIMPLEX

    def draw(text, x, y):
        # Three warm passes instead of the old single white pass: a burnt-
        # ember halo bleeding down-right, a smoldering orange body and a hot
        # amber core.  The stamp should glow like phosphor burned into tape
        # (vintage camcorder), not sit on the frame like cool LCD text.
        halo = (74, 16, 4)      # burnt ember, deep in the frame
        body = (196, 84, 26)    # smoldering orange
        core = (255, 176, 96)   # hot amber filament
        cv2.putText(img_u8, text, (x + 3, y + 3), font, scale, halo, thickness * 4, cv2.LINE_AA)
        cv2.putText(img_u8, text, (x + 1, y + 1), font, scale, body, thickness * 3, cv2.LINE_AA)
        cv2.putText(img_u8, text, (x, y), font, scale, core, thickness, cv2.LINE_AA)

    draw("PLAY  > ", int(w * 0.05), int(h * 0.10))
    draw(date_str, int(w * 0.62), int(h * 0.90))
    return img_u8


# =============================================================================
# 7. GUI
# =============================================================================


ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class RawEditorApp(ctk.CTk):
    """Main window: a filter browser on the left, live preview on the right.

    The interaction model copies Lightroom: pick a filter (a stack of develop
    settings loaded from JSON), then fine-tune with the sliders, which simply
    offset the filter's values.  Every render starts again from the *original*
    image so adjustments never stack or drift.

    The browser shows one accordion per category folder under ./filters —
    click a category header to expand/collapse its filters.
    """

    def __init__(self):
        super().__init__()
        self.title("Sage RAW Studio v2")
        self.geometry("1340x860")
        self.minsize(960, 620)

        # State ------------------------------------------------------------------
        self.source_linear: np.ndarray | None = None   # working-resolution linear RGB
        self.full_linear: np.ndarray | None = None     # full-res linear RGB (lazy)
        self.full_is_loaded = False
        self.file_path = ""
        self.camera_date = ""
        # Filters live next to this script so the app works from any cwd.
        filters_root = Path(__file__).resolve().parent / "filters"
        self.filter_categories = load_filters(filters_root)
        self.current_filter: Filter | None = None
        self.base_settings: DevelopSettings = DevelopSettings()
        self.filter_buttons: Dict[str, ctk.CTkButton] = {}   # key → button
        self.category_headers: Dict[str, ctk.CTkButton] = {}  # dir → header
        self.category_panels: Dict[str, List[ctk.CTkFrame]] = {}  # dir → rows
        self.sliders: Dict[str, ctk.CTkSlider] = {}          # attr → offset slider
        self._slider_rest: Dict[str, float] = {}             # attr → resting value
        self._render_job: str | None = None            # debounce handle
        self._draft_preview = False                    # dragging a slider?

        self._build_ui()

        # Start on "Original" if it exists, else the very first filter.
        for filters in self.filter_categories.values():
            self.select_filter(filters[0].key)
            break

    # -- UI construction -------------------------------------------------------

    def _build_ui(self):
        # 3 columns: left = tabbed fine-tune, centre = preview, right = filters.
        self.grid_columnconfigure(0, weight=0)   # fixed-width control sidebar
        self.grid_columnconfigure(1, weight=1)   # preview grows
        self.grid_columnconfigure(2, weight=0)   # fixed-width filter sidebar
        self.grid_rowconfigure(0, weight=1)

        sidebar = ctk.CTkFrame(self, width=300, corner_radius=0)
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_columnconfigure(0, weight=1)

        row = 0
        ctk.CTkLabel(sidebar, text="SAGE RAW STUDIO",
                     font=ctk.CTkFont(size=22, weight="bold")).grid(row=row, column=0, pady=(18, 0))
        row += 1
        ctk.CTkLabel(sidebar, text="v2 — Lightroom-style pipeline",
                     text_color="gray").grid(row=row, column=0, pady=(0, 8))
        row += 1

        ctk.CTkButton(sidebar, text="Load Image", command=self.load_image).grid(
            row=row, column=0, padx=18, pady=(14, 10))
        row += 1

        # -- Fine-tune controls, split into tabs ---------------------------------
        # Every slider is an OFFSET on the active filter's value (0 = the filter
        # exactly as authored), so tabs stay cheap: they only decide which
        # offsets are visible.  Three groups: the core tone/colour sliders, the
        # grain model (strength / coarseness / dye-cloud colour / mid-tone
        # bias), and the "look" effects (bloom, vignette, texture, clarity).
        tabs = ctk.CTkTabview(sidebar, segmented_button_selected_color="#4A6FA5",
                              segmented_button_selected_hover_color="#3E5F8A")
        tabs.grid(row=row, column=0, padx=10, pady=(6, 4), sticky="nsew")
        sidebar.grid_rowconfigure(row, weight=1)   # tabs absorb spare height
        row += 1
        tabs.add("Fine-tune")
        tabs.add("Grain")
        tabs.add("Effects")

        # (attribute, label, from, to) — offsets applied on top of the filter.
        self.sliders = {}
        self._add_slider(tabs.tab("Fine-tune"), "exposure",    "Exposure",      -2.0, 2.0)
        self._add_slider(tabs.tab("Fine-tune"), "contrast",    "Contrast",      -0.5, 0.5)
        self._add_slider(tabs.tab("Fine-tune"), "temperature", "Temp",          -0.4, 0.4)
        self._add_slider(tabs.tab("Fine-tune"), "tint",        "Tint",          -0.4, 0.4)
        self._add_slider(tabs.tab("Fine-tune"), "highlights",  "Highlights",    -0.5, 0.5)
        self._add_slider(tabs.tab("Fine-tune"), "shadows",     "Shadows",       -0.5, 0.5)
        self._add_slider(tabs.tab("Fine-tune"), "fade",        "Fade (matte)",  -0.1, 0.2)
        self._add_slider(tabs.tab("Fine-tune"), "vibrance",    "Vibrance",      -0.5, 0.5)

        # Auto exposure — the one feature adopted wholesale from the surveyed
        # Python editors (photo-pipe's log-average key solve with a highlight
        # anchor, photo-edit-app's auto button).  Solves for THIS photo, then
        # writes the answer into the Exposure offset slider above.
        ctk.CTkButton(tabs.tab("Fine-tune"), text="Auto Exposure", height=28,
                      fg_color="#4A6FA5", hover_color="#3E5F8A",
                      command=self.run_auto_exposure).pack(fill="x", padx=8, pady=(8, 4))

        self._add_slider(tabs.tab("Grain"), "grain_amount", "Amount",     -0.05, 0.15)
        self._add_slider(tabs.tab("Grain"), "grain_size",   "Coarseness", -0.5, 1.5)
        self._add_slider(tabs.tab("Grain"), "grain_color",  "Dye colour", -0.02, 0.10)
        self._add_slider(tabs.tab("Grain"), "grain_bias",   "Mid bias",   -60.0, 80.0)

        self._add_slider(tabs.tab("Effects"), "bloom_amount",   "Bloom",    -0.2, 0.4)
        self._add_slider(tabs.tab("Effects"), "vignette_amount", "Vignette", -0.2, 0.3)
        self._add_slider(tabs.tab("Effects"), "texture",        "Texture",  -0.3, 0.3)
        self._add_slider(tabs.tab("Effects"), "clarity",        "Clarity",  -0.5, 0.5)

        ctk.CTkButton(sidebar, text="Reset Sliders", fg_color="gray",
                      command=self.reset_sliders).grid(row=row, column=0, padx=18, pady=(4, 2))
        row += 1

        self.fullres_var = ctk.StringVar(value="0")
        ctk.CTkCheckBox(sidebar, text="Full-res on export (slow)",
                        variable=self.fullres_var).grid(row=row, column=0, padx=20, pady=6, sticky="w")
        row += 1

        ctk.CTkButton(sidebar, text="Save Image", fg_color="#196F3D", hover_color="#145A32",
                      command=self.save_image).grid(row=row, column=0, padx=18, pady=(10, 4))
        row += 1

        self.status_label = ctk.CTkLabel(sidebar, text="Load an ARW to start",
                                         text_color="gray", wraplength=240)
        self.status_label.grid(row=row, column=0, padx=18, pady=(4, 12))
        row += 1
        # (no spare-row weight here: the browser pane above already expands)

        # Preview ---------------------------------------------------------------
        image_frame = ctk.CTkFrame(self)
        image_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        image_frame.grid_rowconfigure(0, weight=1)
        image_frame.grid_columnconfigure(0, weight=1)
        self.image_label = ctk.CTkLabel(image_frame, text="No image loaded")
        self.image_label.grid(row=0, column=0, sticky="nsew")

        self._build_filter_browser()

    def _add_slider(self, parent, attr: str, label: str, lo: float, hi: float):
        """One labelled offset slider inside a tab.

        All sliders are centred on 0 (the filter exactly as authored) and drive
        live preview: dragging updates a DRAFT render, releasing pays for a
        full-quality one.
        """
        ctk.CTkLabel(parent, text=label, anchor="w").pack(fill="x", padx=8, pady=(6, 0))
        sld = ctk.CTkSlider(parent, from_=lo, to=hi, number_of_steps=100,
                            command=lambda _v: self._on_slider_move())
        sld.set(0.0)
        sld.bind("<ButtonRelease-1>", lambda _e: self._on_slider_release())
        sld.pack(fill="x", padx=8, pady=(0, 2))
        # CTkSlider snaps set(0.0) to the nearest step, and 0 is not exactly on
        # the grid for most (lo, hi) pairs (it lands on -0.001, 0.2, ...).  That
        # tiny offset would bias every render away from the filter as authored,
        # so record the actual resting value and subtract it back out later.
        self._slider_rest[attr] = sld.get()
        self.sliders[attr] = sld

    def _build_filter_browser(self):
        """Right-hand sidebar: one accordion per category of JSON filters.

        It lives on its own so the ~50 filters don't crowd the fine-tune
        controls, and it scrolls independently.  Each category header
        expands/collapses a panel of filter buttons (grid_remove reclaims the
        space cleanly inside the scrollable frame).
        """
        panel_bar = ctk.CTkFrame(self, width=260, corner_radius=0)
        panel_bar.grid(row=0, column=2, sticky="nsew")
        panel_bar.grid_columnconfigure(0, weight=1)
        panel_bar.grid_rowconfigure(1, weight=1)   # browser absorbs spare height

        ctk.CTkLabel(panel_bar, text="FILTERS", anchor="w",
                     font=ctk.CTkFont(size=16, weight="bold")).grid(
            row=0, column=0, padx=14, pady=(16, 4), sticky="ew")

        browser = ctk.CTkScrollableFrame(panel_bar)
        browser.grid(row=1, column=0, padx=6, sticky="nsew")
        browser.grid_columnconfigure(0, weight=1)

        first_category = True
        b_row = 0
        for cat_dir, filters in self.filter_categories.items():
            colour = filters[0].colour
            header = ctk.CTkButton(
                browser, height=30, anchor="w", fg_color=colour,
                text=self._category_text(cat_dir, expanded=first_category),
                command=lambda d=cat_dir: self.toggle_category(d))
            header.grid(row=b_row, column=0, sticky="ew", padx=2, pady=(5, 0))
            self.category_headers[cat_dir] = header
            b_row += 1

            panel = ctk.CTkFrame(browser, fg_color="transparent")
            panel.grid(row=b_row, column=0, sticky="ew", padx=2)
            panel.grid_columnconfigure(0, weight=1)
            self.category_panels[cat_dir] = panel
            b_row += 1

            for i, flt in enumerate(filters):
                btn = ctk.CTkButton(
                    panel, height=26, anchor="w", fg_color=colour, text=flt.label,
                    command=lambda k=flt.key: self.select_filter(k))
                btn.grid(row=i, column=0, sticky="ew", padx=(10, 0), pady=2)
                self.filter_buttons[flt.key] = btn
            if not first_category:
                panel.grid_remove()
            first_category = False

    # -- State helpers ---------------------------------------------------------

    def set_status(self, text: str):
        self.status_label.configure(text=text)
        self.update_idletasks()

    # Attributes that can never go negative no matter the slider offset, and
    # the floor for each (grain_size has a floor so the noise lattice stays
    # above one pixel; grain_bias keeps the paper curve well-conditioned).
    _NON_NEGATIVE = {
        "fade": 0.0, "grain_amount": 0.0, "grain_size": 0.6,
        "grain_color": 0.0, "grain_bias": 20.0,
        "bloom_amount": 0.0, "vignette_amount": 0.0,
    }

    def _settings_with_slider_offsets(self) -> DevelopSettings:
        """Filter values + the user's slider offsets, as a fresh copy.

        Generic loop over whatever sliders exist, so adding a tab never means
        editing this method.  Offset-only sliders keep 0 = "the filter as
        authored"; the _NON_NEGATIVE entries clamp at a safe floor.
        """
        s = copy.deepcopy(self.base_settings)
        for attr, sld in self.sliders.items():
            # Subtract the resting value so "no user input" is exactly the
            # filter as authored, even though the widget snaps to a step grid.
            value = getattr(s, attr) + (sld.get() - self._slider_rest[attr])
            if attr in self._NON_NEGATIVE:
                value = max(self._NON_NEGATIVE[attr], value)
            setattr(s, attr, value)
        return s

    def _on_slider_move(self):
        """Slider dragged: request a DRAFT-quality render (fast, slightly
        soft) so the preview keeps up with the mouse."""
        self._draft_preview = True
        self.schedule_render()

    def _on_slider_release(self):
        """Slider released: pay for the full-quality render once."""
        self._draft_preview = False
        self.schedule_render()

    def schedule_render(self):
        """Debounce: sliders fire on every pixel of travel; re-render 90 ms
        after the last movement so dragging stays smooth."""
        if self.source_linear is None:
            return
        if self._render_job is not None:
            self.after_cancel(self._render_job)
        self._render_job = self.after(60 if self._draft_preview else 90,
                                      self.render_preview)

    # -- Actions ---------------------------------------------------------------

    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("RAW / images", "*.ARW *.arw *.jpg *.jpeg *.png"), ("All files", "*.*")])
        if not path:
            return
        self.file_path = path
        self.set_status("Decoding RAW…")
        try:
            self.source_linear = self._decode(path, max_edge=PREVIEW_LONG_EDGE)
            self.full_linear = None
            self.full_is_loaded = False
            self.camera_date = read_exif_date(path)
            if self.current_filter is not None:
                self.select_filter(self.current_filter.key)   # re-baseline sliders
            self.set_status(os.path.basename(path))
        except Exception as exc:                      # surface decode errors, don't crash
            self.set_status(f"Error: {exc}")

    def _decode(self, path: str, max_edge: int | None) -> np.ndarray:
        """Decode ARW (or a regular image) to LINEAR float RGB at `max_edge`.

        rawpy hands us a gamma-encoded sRGB image; converting to linear once
        at load time means the whole pipeline can do physically sensible math.
        """
        if path.lower().endswith((".arw", ".raw")):
            with rawpy.imread(path) as raw:
                half = max_edge is not None          # preview pass uses half-size demosaic
                rgb16 = raw.postprocess(use_camera_wb=True, half_size=half, output_bps=16)
                img = rgb16.astype(np.float32) / 65535.0
        else:
            bgr = cv2.imread(path, cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"cannot read {path}")
            img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        linear = srgb_to_linear(img)
        if max_edge is not None:
            linear = self._downscale(linear, max_edge)
        return np.ascontiguousarray(linear, dtype=np.float32)

    @staticmethod
    def _downscale(img: np.ndarray, max_edge: int) -> np.ndarray:
        h, w = img.shape[:2]
        if max(h, w) <= max_edge:
            return img
        scale = max_edge / float(max(h, w))
        return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    def select_filter(self, key: str):
        """Make `key` the active filter.

        Loads the filter's develop settings as the new slider baseline,
        outlines its button, resets the slider offsets and re-renders (if an
        image is loaded).  Also callable before any image exists — it just
        sets the state so the first render already uses the right filter.
        """
        filters = [f for flt in self.filter_categories.values() for f in flt if f.key == key]
        if not filters:
            return
        flt = filters[0]
        self.current_filter = flt
        self.base_settings = flt.settings
        for f_key, btn in self.filter_buttons.items():
            btn.configure(border_width=2 if f_key == key else 0,
                          border_color="#FFFFFF")
        self._draft_preview = False       # never leave the preview stuck in draft
        self.reset_sliders(keep_status=True)
        self.set_status(flt.description or flt.label)

    def _category_text(self, cat_dir: str, expanded: bool) -> str:
        """Accordion header text: arrow + category label + filter count."""
        filters = self.filter_categories.get(cat_dir, [])
        label = filters[0].category_label if filters else cat_dir
        return f"{'▾' if expanded else '▸'} {label}  ({len(filters)})"

    def toggle_category(self, cat_dir: str):
        """Expand / collapse one category's filter list (the accordion)."""
        panel = self.category_panels.get(cat_dir)
        header = self.category_headers.get(cat_dir)
        if panel is None or header is None:
            return
        expanded = panel.winfo_ismapped()
        if expanded:
            panel.grid_remove()
        else:
            panel.grid()
        header.configure(text=self._category_text(cat_dir, expanded=not expanded))

    def reset_sliders(self, keep_status: bool = False):
        # Restore each slider to its resting value (its snapped stand-in for
        # "no offset"), not literal 0.0 — CTkSlider would snap 0.0 again and
        # leave the same tiny bias the resting value exists to cancel.
        for attr, sld in self.sliders.items():
            sld.set(self._slider_rest[attr])
        if not keep_status:
            self.set_status("Sliders reset")
        self.schedule_render()

    def run_auto_exposure(self):
        """Auto-exposure: solve stops for this photo, apply as the offset.

        auto_exposure() returns the ABSOLUTE exposure the image needs, but the
        slider holds an offset from the active filter — so subtract the
        filter's own exposure to land the total where the solve wants it.
        The clamp keeps the answer inside the slider's -2..+2 span (the
        feature is a starting point, not an autopilot).
        """
        if self.source_linear is None:
            self.set_status("Load an image first")
            return
        target = auto_exposure(self.source_linear)
        offset = max(-2.0, min(2.0, target - self.base_settings.exposure))
        self.sliders["exposure"].set(self._slider_rest["exposure"] + offset)
        self.set_status(f"Auto exposure: {self.base_settings.exposure + offset:+.2f} stops")
        self.schedule_render()

    def render_preview(self):
        """Re-develop the working image with the current settings and show it."""
        self._render_job = None
        if self.source_linear is None:
            return
        self.set_status("Developing…")
        settings = self._settings_with_slider_offsets()

        source = self.source_linear
        if self._draft_preview:
            source = self._downscale(source, DRAFT_LONG_EDGE)
        out = develop(source, settings)
        u8 = analog_damage_u8(out, self.current_effects(), self.camera_date)
        self.display(u8)

    def current_effects(self) -> Dict[str, Any]:
        """The active filter's analog-damage chain ({} when nothing selected)."""
        return self.current_filter.effects if self.current_filter else {}

    def display(self, rgb_u8: np.ndarray):
        """Fit the preview into the panel without distorting the aspect ratio."""
        fw = self.image_label.winfo_width() or 900
        fh = self.image_label.winfo_height() or 600
        if fw <= 10:
            fw = 900
        if fh <= 10:
            fh = 600

        pil = Image.fromarray(rgb_u8)
        pil.thumbnail((fw - 20, fh - 20), Image.Resampling.LANCZOS)
        ctk_img = ctk.CTkImage(light_image=pil, dark_image=pil, size=pil.size)
        self.image_label.configure(image=ctk_img, text="")
        self.image_label.image = ctk_img                  # keep a reference (GC guard)

    def save_image(self):
        if self.source_linear is None:
            self.set_status("Nothing to save")
            return
        # PNG is offered first: at this point of the pipeline quality is the
        # only thing left to trade, and JPEG recompression of grain + bloom
        # is visible.  JPEG stays for sharing.
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG (lossless)", "*.png"), ("JPEG", "*.jpg")])
        if not path:
            return
        settings = self._settings_with_slider_offsets()

        if self.fullres_var.get() == "1":
            self.set_status("Decoding full-res (can take a few seconds)…")
            self.update_idletasks()
            if not self.full_is_loaded:
                self.full_linear = self._decode(self.file_path, max_edge=None)
                self.full_is_loaded = True
            source = self.full_linear
        else:
            source = self.source_linear

        out = develop(source, settings)
        # dither=True on export only — see quantize_u8: saves must not band
        # at 100% zoom, while the preview's post-LANCZOS re-quantise would
        # wash it away anyway.
        u8 = analog_damage_u8(out, self.current_effects(), self.camera_date, dither=True)
        bgr = cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)
        if path.lower().endswith(".png"):
            cv2.imwrite(path, bgr)                       # lossless, no quality knob
        else:
            cv2.imwrite(path, bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 98])
        self.set_status(f"Saved {os.path.basename(path)}")


if __name__ == "__main__":
    app = RawEditorApp()
    app.mainloop()
