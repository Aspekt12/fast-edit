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
from PIL import Image, ImageDraw

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

    # -- Lens correction / Sharpening / Pixel Art --------------------------------
    lens_distortion: float = 0.0
    lens_tca_r: float = 0.0
    lens_tca_b: float = 0.0
    lens_vignette_corr: float = 0.0
    lens_k1: float = 0.0
    lens_k2: float = 0.0
    lens_k3: float = 0.0
    lens_ca_r: float = 0.0
    lens_ca_b: float = 0.0
    lens_keystone_h: float = 0.0
    lens_keystone_v: float = 0.0
    lens_rotate: float = 0.0
    lens_enable: int = 0
    sharpening_amount: float = 0.0
    sharpening_radius: float = 0.5
    sharpening_detail: float = 0.0
    sharpening_masking: float = 0.0
    nr_luminance: float = 0.0
    nr_color: float = 0.0
    curve_highlights: float = 0.0
    curve_lights: float = 0.0
    curve_darks: float = 0.0
    curve_shadows: float = 0.0
    crop_x: float = 0.0
    crop_y: float = 0.0
    crop_w: float = 0.0
    crop_h: float = 0.0
    straighten_angle: float = 0.0
    pixel_size: float = 0.0

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


def lens_correct(img: np.ndarray, s: DevelopSettings) -> np.ndarray:
    """Brown-Conrady distortion, TCA, perspective keystone and vignetting correction."""
    h, w = img.shape[:2]
    if h < 2 or w < 2:
        return img

    # 1. Perspective keystone
    if abs(s.lens_keystone_h) > 1e-6 or abs(s.lens_keystone_v) > 1e-6:
        nx = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
        ny = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
        scale_h = 1.0 + float(s.lens_keystone_h) * ny * 0.5
        scale_v = 1.0 + float(s.lens_keystone_v) * nx * 0.5
        map_x = ((nx / scale_h * 0.5 + 0.5) * w).astype(np.float32)
        map_y = ((ny / scale_v * 0.5 + 0.5) * h).astype(np.float32)
        img = cv2.remap(img, np.broadcast_to(map_x, (h, w)).copy(),
                        np.broadcast_to(map_y, (h, w)).copy(),
                        interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # 2. Radial distortion
    k1 = s.lens_k1 if abs(s.lens_k1) > 1e-9 else -s.lens_distortion * 0.2
    k2, k3 = s.lens_k2, s.lens_k3
    if abs(k1) > 1e-6 or abs(k2) > 1e-6 or abs(k3) > 1e-6:
        cx, cy = w * 0.5, h * 0.5
        maxr = float(np.hypot(cx, cy))
        dx = (np.arange(w, dtype=np.float32)[None, :] - cx) / maxr
        dy = (np.arange(h, dtype=np.float32)[:, None] - cy) / maxr
        r2 = dx * dx + dy * dy
        factor = 1.0 + float(k1) * r2 + float(k2) * (r2 ** 2) + float(k3) * (r2 ** 3)
        map_x = (dx / factor * maxr + cx).astype(np.float32)
        map_y = (dy / factor * maxr + cy).astype(np.float32)
        img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT)

    # 3. Transverse chromatic aberration (TCA)
    tca_r = s.lens_tca_r if abs(s.lens_tca_r) > 1e-6 else s.lens_ca_r
    tca_b = s.lens_tca_b if abs(s.lens_tca_b) > 1e-6 else s.lens_ca_b
    if abs(tca_r) > 1e-6 or abs(tca_b) > 1e-6:
        cx, cy = w * 0.5, h * 0.5
        maxr = float(np.hypot(cx, cy))
        dx = (np.arange(w, dtype=np.float32)[None, :] - cx) / maxr
        dy = (np.arange(h, dtype=np.float32)[:, None] - cy) / maxr
        r2 = dx * dx + dy * dy
        out = img.copy()
        if abs(tca_r) > 1e-6:
            sr = 1.0 + float(tca_r) * r2
            out[..., 0] = cv2.remap(img[..., 0], (dx * sr * maxr + cx).astype(np.float32),
                                    (dy * sr * maxr + cy).astype(np.float32),
                                    interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        if abs(tca_b) > 1e-6:
            sb = 1.0 + float(tca_b) * r2
            out[..., 2] = cv2.remap(img[..., 2], (dx * sb * maxr + cx).astype(np.float32),
                                    (dy * sb * maxr + cy).astype(np.float32),
                                    interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        img = out

    # 4. Lens vignetting correction
    if abs(s.lens_vignette_corr) > 1e-6:
        cx, cy = w * 0.5, h * 0.5
        maxr = float(np.hypot(cx, cy))
        dx = (np.arange(w, dtype=np.float32)[None, :] - cx) / maxr
        dy = (np.arange(h, dtype=np.float32)[:, None] - cy) / maxr
        r2 = dx * dx + dy * dy
        corr = np.clip(1.0 + float(s.lens_vignette_corr) * (r2 + 0.5 * r2 * r2), 0.2, 4.0)
        img = np.clip(img * corr[..., None], 0.0, 1.0)

    return img


def auto_lens_analyze(linear_rgb: np.ndarray) -> Tuple[float, float, float, float]:
    """Estimate (lens_distortion, lens_tca_r, lens_tca_b, lens_vignette_corr)."""
    h, w = linear_rgb.shape[:2]
    cx, cy = w * 0.5, h * 0.5
    maxr = float(np.hypot(cx, cy))
    if maxr < 1.0:
        return 0.0, 0.0, 0.0, 0.0
    L = luma(linear_rgb)
    dx = (np.arange(w, dtype=np.float32)[None, :] - cx) / maxr
    dy = (np.arange(h, dtype=np.float32)[:, None] - cy) / maxr
    r = np.hypot(dx, dy)
    inner = L[(r < 0.35) & (L > 0.01)]
    outer = L[(r > 0.70) & (r <= 1.0) & (L > 0.01)]
    vig = 0.0
    if inner.size > 50 and outer.size > 50:
        ratio = float(np.mean(outer)) / max(float(np.mean(inner)), 1e-4)
        if ratio < 0.92:
            vig = float(np.clip((1.0 - ratio) * 0.8, -0.5, 0.5))
    return 0.0, 0.0, 0.0, vig


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

    # Pixelation for pixel art looks (matching fe_pipeline.c).
    if s.pixel_size > 1.0:
        psize = int(np.clip(s.pixel_size, 2, 32))
        h, w = img.shape[:2]
        nw, nh = max(1, w // psize), max(1, h // psize)
        small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        small = np.floor(small * 4.0 + 0.5) / 4.0
        img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    # 7 — Presence, sharpening, lens correction and effects last.
    px = max(img.shape[:2]) / PREVIEW_LONG_EDGE
    img = presence(img, texture=s.texture, clarity=s.clarity, dehaze=s.dehaze, pixel_scale=px)
    if s.sharpening_amount > 1e-4:
        sr = max(0.5, float(s.sharpening_radius))
        blur = cv2.GaussianBlur(img, (0, 0), sr)
        img = np.clip(img + float(s.sharpening_amount) * 0.5 * (img - blur), 0.0, 1.0)
    if (s.lens_enable or s.lens_distortion != 0 or s.lens_tca_r != 0 or s.lens_tca_b != 0
            or s.lens_vignette_corr != 0 or s.lens_keystone_h != 0 or s.lens_keystone_v != 0
            or s.lens_k1 != 0 or s.lens_k2 != 0 or s.lens_k3 != 0):
        img = lens_correct(img, s)
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
# 6b. PER-PHOTO SIDECAR (<photo>.fastedit.json) & PROJECT (.feproj) PERSISTENCE
#     Matches fast-edit-c's fe_state.c and fe_project.c byte-for-byte in schema,
#     so edits and projects are 100% interoperable between Python and C.
# =============================================================================

RAW_EXTENSIONS = {".arw", ".raw", ".cr2", ".nef", ".dng", ".raf", ".orf", ".rw2"}
IMAGE_EXTENSIONS = RAW_EXTENSIONS | {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

NON_NEGATIVE_FLOORS: Dict[str, float] = {
    "fade": 0.0, "grain_amount": 0.0, "grain_size": 0.6,
    "grain_color": 0.0, "grain_bias": 20.0,
    "bloom_amount": 0.0, "vignette_amount": 0.0,
    "sharpening_amount": 0.0, "pixel_size": 0.0,
}


def sidecar_path(image_path: str) -> Path:
    return Path(f"{image_path}.fastedit.json")


def load_photo_state(image_path: str) -> Tuple[str, Dict[str, float]] | None:
    """Read <image_path>.fastedit.json. Returns (filter_key, offsets) or None."""
    if not image_path:
        return None
    p = sidecar_path(image_path)
    if not p.is_file():
        return None
    try:
        doc = json.loads(p.read_text())
        if not isinstance(doc, dict):
            return None
        filter_key = str(doc.get("filter") or "")
        offsets_raw = doc.get("offsets")
        offsets: Dict[str, float] = {}
        if isinstance(offsets_raw, dict):
            for k, v in offsets_raw.items():
                if isinstance(v, (int, float)) and abs(float(v)) >= 1e-9:
                    offsets[str(k)] = float(v)
        else:
            for k, v in doc.items():
                if k not in ("version", "filter") and isinstance(v, (int, float)):
                    if abs(float(v)) >= 1e-9:
                        offsets[str(k)] = float(v)
        return filter_key, offsets
    except Exception:
        return None


def save_photo_state(image_path: str, filter_key: str, offsets: Dict[str, float]) -> bool:
    """Write <image_path>.fastedit.json atomically via a .tmp file + rename."""
    if not image_path:
        return False
    p = sidecar_path(image_path)
    tmp = Path(f"{p}.tmp")
    clean_offsets = {k: float(v) for k, v in offsets.items() if abs(float(v)) >= 1e-9}
    payload = {
        "version": 1,
        "filter": filter_key or "00-basics/original",
        "offsets": clean_offsets,
    }
    try:
        tmp.write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(tmp, p)
        return True
    except OSError:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        return False


def remove_photo_state(image_path: str) -> bool:
    if not image_path:
        return False
    try:
        sidecar_path(image_path).unlink(missing_ok=True)
        return True
    except OSError:
        return False


def apply_offsets_to_settings(base: DevelopSettings, offsets: Dict[str, float]) -> DevelopSettings:
    s = copy.deepcopy(base)
    for attr, delta in offsets.items():
        if hasattr(s, attr):
            val = float(getattr(s, attr)) + float(delta)
            if attr in NON_NEGATIVE_FLOORS:
                val = max(NON_NEGATIVE_FLOORS[attr], val)
            setattr(s, attr, val)
    return s


# =============================================================================
# 7. GUI — Themes, Customization & Studio Interface
# =============================================================================


THEMES: List[Dict[str, Any]] = [
    {
        "key": "gruvbox-dark", "label": "Gruvbox Dark", "is_dark": True,
        "bg_main": "#1d2021", "bg_sidebar": "#282828", "bg_surface": "#3c3836",
        "bg_hover": "#504945", "canvas_bg": "#141617", "fg_primary": "#ebdbb2",
        "fg_muted": "#a89984", "accent": "#fabd2f", "accent_fg": "#1d2021",
        "border": "#504945", "danger": "#cc241d", "success": "#689d6a",
    },
    {
        "key": "gruvbox-light", "label": "Gruvbox Light", "is_dark": False,
        "bg_main": "#f9f5d7", "bg_sidebar": "#fbf1c7", "bg_surface": "#ebdbb2",
        "bg_hover": "#d5c4a1", "canvas_bg": "#e0d3af", "fg_primary": "#3c3836",
        "fg_muted": "#7c6f64", "accent": "#d65d0e", "accent_fg": "#fbf1c7",
        "border": "#d5c4a1", "danger": "#9d0006", "success": "#427b58",
    },
    {
        "key": "studio-dark", "label": "Studio Dark (Lightroom)", "is_dark": True,
        "bg_main": "#121316", "bg_sidebar": "#1a1b1f", "bg_surface": "#26282e",
        "bg_hover": "#32353d", "canvas_bg": "#0b0c0e", "fg_primary": "#eceef2",
        "fg_muted": "#8e939e", "accent": "#ffd479", "accent_fg": "#121316",
        "border": "#2f323a", "danger": "#9e3b3b", "success": "#237a4b",
    },
    {
        "key": "sage-atelier", "label": "Sage Atelier", "is_dark": True,
        "bg_main": "#131815", "bg_sidebar": "#1b221e", "bg_surface": "#26312b",
        "bg_hover": "#324038", "canvas_bg": "#0d110f", "fg_primary": "#e2ebe5",
        "fg_muted": "#8fa396", "accent": "#7fa98e", "accent_fg": "#111714",
        "border": "#2e3b33", "danger": "#9e4444", "success": "#3d855c",
    },
    {
        "key": "catppuccin-mocha", "label": "Catppuccin Mocha", "is_dark": True,
        "bg_main": "#181825", "bg_sidebar": "#1e1e2e", "bg_surface": "#313244",
        "bg_hover": "#45475a", "canvas_bg": "#11111b", "fg_primary": "#cdd6f4",
        "fg_muted": "#a6adc8", "accent": "#cba6f7", "accent_fg": "#11111b",
        "border": "#45475a", "danger": "#f38ba8", "success": "#a6e3a1",
    },
    {
        "key": "nord", "label": "Nord Polar", "is_dark": True,
        "bg_main": "#242933", "bg_sidebar": "#2e3440", "bg_surface": "#3b4252",
        "bg_hover": "#434c5e", "canvas_bg": "#1d212a", "fg_primary": "#eceff4",
        "fg_muted": "#9aa5b8", "accent": "#88c0d0", "accent_fg": "#242933",
        "border": "#4c566a", "danger": "#bf616a", "success": "#a3be8c",
    },
    {
        "key": "tokyo-night", "label": "Tokyo Night", "is_dark": True,
        "bg_main": "#16161e", "bg_sidebar": "#1a1b26", "bg_surface": "#24283b",
        "bg_hover": "#2f3549", "canvas_bg": "#101016", "fg_primary": "#c0caf5",
        "fg_muted": "#7982a9", "accent": "#7aa2f7", "accent_fg": "#16161e",
        "border": "#3b4261", "danger": "#f7768e", "success": "#73daca",
    },
    {
        "key": "dracula", "label": "Dracula", "is_dark": True,
        "bg_main": "#1e1f29", "bg_sidebar": "#282a36", "bg_surface": "#383a4c",
        "bg_hover": "#44475a", "canvas_bg": "#15161d", "fg_primary": "#f8f8f2",
        "fg_muted": "#9499b3", "accent": "#bd93f9", "accent_fg": "#1e1f29",
        "border": "#44475a", "danger": "#ff5555", "success": "#50fa7b",
    },
    {
        "key": "rose-pine", "label": "Rosé Pine", "is_dark": True,
        "bg_main": "#191724", "bg_sidebar": "#1f1d2e", "bg_surface": "#26233a",
        "bg_hover": "#393552", "canvas_bg": "#12101a", "fg_primary": "#e0def4",
        "fg_muted": "#908caa", "accent": "#f6c177", "accent_fg": "#191724",
        "border": "#403d52", "danger": "#eb6f92", "success": "#31748f",
    },
    {
        "key": "solarized-dark", "label": "Solarized Dark", "is_dark": True,
        "bg_main": "#00212b", "bg_sidebar": "#002b36", "bg_surface": "#073642",
        "bg_hover": "#0f4654", "canvas_bg": "#001920", "fg_primary": "#eee8d5",
        "fg_muted": "#839496", "accent": "#b58900", "accent_fg": "#00212b",
        "border": "#164e5c", "danger": "#dc322f", "success": "#2aa198",
    },
]
THEME_BY_KEY: Dict[str, Dict[str, Any]] = {t["key"]: t for t in THEMES}
THEME_BY_LABEL: Dict[str, Dict[str, Any]] = {t["label"]: t for t in THEMES}

ACCENT_OPTIONS: List[Tuple[str, str, str | None]] = [
    ("theme",   "Theme Default",    None),
    ("yellow",  "Gruvbox Yellow",   "#fabd2f"),
    ("orange",  "Gruvbox Orange",   "#fe8019"),
    ("aqua",    "Gruvbox Aqua",     "#8ec07c"),
    ("gold",    "Warm Studio Gold", "#ffd479"),
    ("sage",    "Botanical Sage",   "#7fa98e"),
    ("frost",   "Nord Frost Cyan",  "#88c0d0"),
    ("mauve",   "Catppuccin Mauve", "#cba6f7"),
    ("blue",    "Tokyo Neon Blue",  "#7aa2f7"),
    ("rose",    "Rosé Gold",        "#ebbcba"),
    ("crimson", "Warm Coral",       "#fb4934"),
]

BACKDROP_OPTIONS: List[Tuple[str, str, str | None]] = [
    ("theme",     "Theme Default",      None),
    ("black",     "Pitch Black",        "#050505"),
    ("charcoal",  "Darkroom Charcoal",  "#101113"),
    ("gruvbox",   "Gruvbox Hard Dark",  "#1d2021"),
    ("midgrey",   "18% Neutral Grey",   "#4a4a4a"),
    ("warmpaper", "Warm Gallery Paper", "#ebdbb2"),
    ("white",     "Pure Gallery White", "#f7f7f7"),
]

FRAME_STYLES = [
    "Soft Studio Drop Shadow",
    "Gallery White Matboard",
    "Thin Accent Hairline",
    "Flat (No Border)",
]
DENSITY_LEVELS = ["Compact", "Comfortable (Default)", "Spacious"]
FILTER_STYLES = ["Modern Cards (Category Color Indicator)", "Classic Full-Color Buttons"]
GRID_MODES = ["None", "Rule of Thirds (3×3)", "Phi Golden Ratio", "Center Crosshair & Diagonals"]


def _ui_prefs_path() -> Path:
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        d = Path(xdg) / "fast-edit"
    else:
        d = Path.home() / ".config" / "fast-edit"
    try:
        d.mkdir(parents=True, exist_ok=True)
        return d / "ui.json"
    except Exception:
        return Path(".fastedit-ui.json")


def _hex_to_rgb(h: str) -> Tuple[int, int, int]:
    s = h.lstrip("#")
    if len(s) >= 6:
        return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
    return 20, 22, 24


def _shade_hex(h: str, factor: float) -> str:
    r, g, b = _hex_to_rgb(h)
    r = max(0, min(255, int(r * factor)))
    g = max(0, min(255, int(g * factor)))
    b = max(0, min(255, int(b * factor)))
    return f"#{r:02x}{g:02x}{b:02x}"


ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class RawEditorApp(ctk.CTk):
    """Main window: studio header bar on top, fine-tune tabs on the left, live
    preview in the centre, filter browser on the right, and a Folder / Project
    film-strip at the bottom.

    Every photo keeps its own edit in <photo>.fastedit.json:
      * Switching photos flushes pending edits (only if modified) and restores
        the next photo's saved filter + slider offsets (or resets to Original
        when the photo has never been edited).
      * Merely clicking a photo to view it never writes a sidecar.
    """

    def __init__(self):
        super().__init__()
        self.title("Sage RAW Studio v2")
        self.geometry("1480x940")
        self.minsize(1060, 700)

        # UI Preferences (shared with C edition in ~/.config/fast-edit/ui.json) --
        self.ui_prefs: Dict[str, Any] = {
            "theme": "gruvbox-dark",
            "accent": "theme",
            "backdrop": "theme",
            "frame_style": 0,
            "density": 1,
            "filter_style": 0,
            "show_histogram": True,
            "grid_mode": 0,
            "show_hud": True,
        }
        self._load_ui_prefs()

        # State ------------------------------------------------------------------
        self.source_linear: np.ndarray | None = None   # working-resolution linear RGB
        self.full_linear: np.ndarray | None = None     # full-res linear RGB (lazy)
        self.full_is_loaded = False
        self.file_path = ""
        self.camera_date = ""
        filters_root = Path(__file__).resolve().parent / "filters"
        self.filter_categories = load_filters(filters_root)
        self.all_filters: Dict[str, Filter] = {
            f.key: f for flts in self.filter_categories.values() for f in flts
        }
        self.current_filter: Filter | None = None
        self.base_settings: DevelopSettings = DevelopSettings()
        self.filter_buttons: Dict[str, ctk.CTkButton] = {}
        self.filter_cat_colours: Dict[str, str] = {}
        self.category_headers: Dict[str, ctk.CTkButton] = {}
        self.category_panels: Dict[str, ctk.CTkFrame] = {}
        self.sliders: Dict[str, ctk.CTkSlider] = {}
        self.slider_val_labels: Dict[str, ctk.CTkLabel] = {}
        self.slider_name_labels: List[ctk.CTkLabel] = []
        self._slider_rest: Dict[str, float] = {}
        self._slider_bounds: Dict[str, Tuple[float, float]] = {}
        self._render_job: str | None = None
        self._save_job: str | None = None
        self._draft_preview = False
        self._suppress = 0
        self._dirty = False
        self._clipboard: Tuple[str, Dict[str, float]] | None = None
        self._last_u8: np.ndarray | None = None
        self._orig_u8: np.ndarray | None = None

        # Compare state (0 = off, 1 = split wipe, 2 = before only) ---------------
        self.compare_mode: int = 0
        self.split_ratio: float = 0.5
        self._dragging_split: bool = False

        # Folder & Project state -------------------------------------------------
        self.current_folder: str = ""
        self.folder_paths: List[str] = []
        self.project_items: List[Dict[str, Any]] = []  # [{"path": str, "selected": bool}]
        self.project_path: str = ""
        self._folder_cards: Dict[str, Dict[str, Any]] = {}
        self._project_cards: Dict[str, Dict[str, Any]] = {}

        # Themed widget references for live theme switching ----------------------
        self._themed_frames: List[Tuple[Any, str]] = []
        self._themed_buttons: List[Tuple[ctk.CTkButton, str]] = []
        self._themed_labels: List[Tuple[ctk.CTkLabel, str]] = []
        self._customize_win: ctk.CTkToplevel | None = None

        self._build_ui()
        self.apply_theme()

        # Start on "Original" under suppress so startup never marks dirty.
        self._suppress += 1
        for filters in self.filter_categories.values():
            self.select_filter(filters[0].key)
            break
        self._suppress -= 1
        self._update_slider_labels()

        self.bind("<KeyPress-backslash>", lambda _e: self.cycle_compare())
        self.bind("<KeyPress-bracketleft>", lambda _e: self.step_photo(-1))
        self.bind("<KeyPress-bracketright>", lambda _e: self.step_photo(1))

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # -- Theme & UI Preferences ------------------------------------------------

    def _load_ui_prefs(self):
        p = _ui_prefs_path()
        if not p.is_file():
            return
        try:
            doc = json.loads(p.read_text())
            if isinstance(doc.get("theme"), str) and doc["theme"] in THEME_BY_KEY:
                self.ui_prefs["theme"] = doc["theme"]
            if isinstance(doc.get("accent"), str):
                self.ui_prefs["accent"] = doc["accent"]
            if isinstance(doc.get("backdrop"), str):
                self.ui_prefs["backdrop"] = doc["backdrop"]
            for k, hi in (("frame_style", 3), ("density", 2), ("filter_style", 1), ("grid_mode", 3)):
                if isinstance(doc.get(k), int):
                    self.ui_prefs[k] = max(0, min(hi, int(doc[k])))
            for k in ("show_histogram", "show_hud"):
                if k in doc:
                    self.ui_prefs[k] = bool(doc[k])
        except Exception:
            pass

    def _save_ui_prefs(self):
        try:
            _ui_prefs_path().write_text(json.dumps(self.ui_prefs, indent=2) + "\n")
        except Exception:
            pass

    def active_theme(self) -> Dict[str, Any]:
        return THEME_BY_KEY.get(self.ui_prefs["theme"], THEMES[0])

    def active_accent(self) -> str:
        t = self.active_theme()
        key = self.ui_prefs.get("accent", "theme")
        for k, _lbl, hx in ACCENT_OPTIONS:
            if k == key and hx:
                return hx
        return t["accent"]

    def active_backdrop(self) -> str:
        t = self.active_theme()
        key = self.ui_prefs.get("backdrop", "theme")
        for k, _lbl, hx in BACKDROP_OPTIONS:
            if k == key and hx:
                return hx
        return t["canvas_bg"]

    # -- UI construction -------------------------------------------------------

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=0)   # left control sidebar
        self.grid_columnconfigure(1, weight=1)   # centre preview
        self.grid_columnconfigure(2, weight=0)   # right filter browser
        self.grid_rowconfigure(0, weight=0)      # top studio header bar
        self.grid_rowconfigure(1, weight=1)      # main workspace
        self.grid_rowconfigure(2, weight=0)      # bottom folder/project strip

        # -- Top Studio Header Bar ---------------------------------------------
        self.header_bar = ctk.CTkFrame(self, height=48, corner_radius=0)
        self.header_bar.grid(row=0, column=0, columnspan=3, sticky="ew")
        self._themed_frames.append((self.header_bar, "sidebar"))

        brand_box = ctk.CTkFrame(self.header_bar, fg_color="transparent")
        brand_box.pack(side="left", padx=(14, 10), pady=6)
        self.title_lbl = ctk.CTkLabel(
            brand_box, text="SAGE RAW STUDIO",
            font=ctk.CTkFont(size=17, weight="bold"))
        self.title_lbl.pack(side="left", padx=(0, 8))
        self._themed_labels.append((self.title_lbl, "accent"))

        self.sub_lbl = ctk.CTkLabel(
            brand_box, text="v2  •  RAW Color & Film Pipeline",
            font=ctk.CTkFont(size=12))
        self.sub_lbl.pack(side="left")
        self._themed_labels.append((self.sub_lbl, "muted"))

        btn_open = ctk.CTkButton(self.header_bar, text="Open Photo", width=105, height=30,
                                 command=self.load_image)
        btn_open.pack(side="left", padx=4, pady=6)
        self._themed_buttons.append((btn_open, "surface"))

        btn_folder = ctk.CTkButton(self.header_bar, text="Open Folder", width=105, height=30,
                                   command=self.browse_folder)
        btn_folder.pack(side="left", padx=4, pady=6)
        self._themed_buttons.append((btn_folder, "surface"))

        self.compare_btn = ctk.CTkButton(self.header_bar, text="Compare (\\)", width=112, height=30,
                                         command=self.cycle_compare)
        self.compare_btn.pack(side="left", padx=(10, 4), pady=6)

        self.hist_btn = ctk.CTkButton(self.header_bar, text="Histogram", width=92, height=30,
                                      command=self.toggle_histogram)
        self.hist_btn.pack(side="left", padx=4, pady=6)

        self.grid_btn = ctk.CTkButton(self.header_bar, text="Grid", width=96, height=30,
                                      command=self.cycle_grid)
        self.grid_btn.pack(side="left", padx=4, pady=6)

        # Right side of header bar: Export, Customize, Theme switcher
        self.export_top_btn = ctk.CTkButton(
            self.header_bar, text="Export Image", width=118, height=30,
            font=ctk.CTkFont(size=12, weight="bold"), command=self.save_image)
        self.export_top_btn.pack(side="right", padx=(6, 14), pady=6)
        self._themed_buttons.append((self.export_top_btn, "success"))

        self.customize_btn = ctk.CTkButton(
            self.header_bar, text="⚙ Customize", width=106, height=30,
            command=self.open_customize_dialog)
        self.customize_btn.pack(side="right", padx=4, pady=6)
        self._themed_buttons.append((self.customize_btn, "accent"))

        self.theme_menu = ctk.CTkOptionMenu(
            self.header_bar, width=165, height=30,
            values=[t["label"] for t in THEMES],
            command=self._on_theme_menu_selected)
        self.theme_menu.set(self.active_theme()["label"])
        self.theme_menu.pack(side="right", padx=6, pady=6)

        # -- Left Develop Sidebar ----------------------------------------------
        self.sidebar = ctk.CTkFrame(self, width=320, corner_radius=0)
        self.sidebar.grid(row=1, column=0, sticky="nsew")
        self.sidebar.grid_columnconfigure(0, weight=1)
        self._themed_frames.append((self.sidebar, "sidebar"))

        row = 0
        dev_hdr = ctk.CTkLabel(
            self.sidebar, text="DEVELOP CONTROLS", anchor="w",
            font=ctk.CTkFont(size=12, weight="bold"))
        dev_hdr.grid(row=row, column=0, padx=14, pady=(10, 2), sticky="ew")
        self._themed_labels.append((dev_hdr, "accent"))
        row += 1

        # -- Fine-tune controls, split into 4 tabs -------------------------------
        self.tabs = ctk.CTkTabview(self.sidebar, width=295)
        self.tabs.grid(row=row, column=0, padx=10, pady=(2, 4), sticky="nsew")
        self.sidebar.grid_rowconfigure(row, weight=1)
        row += 1
        self.tabs.add("Fine-tune")
        self.tabs.add("Grain")
        self.tabs.add("Effects")
        self.tabs.add("Lens")

        self.sliders = {}
        self._add_slider(self.tabs.tab("Fine-tune"), "exposure",    "Exposure",      -2.0, 2.0)
        self._add_slider(self.tabs.tab("Fine-tune"), "contrast",    "Contrast",      -0.5, 0.5)
        self._add_slider(self.tabs.tab("Fine-tune"), "temperature", "Temp",          -0.4, 0.4)
        self._add_slider(self.tabs.tab("Fine-tune"), "tint",        "Tint",          -0.4, 0.4)
        self._add_slider(self.tabs.tab("Fine-tune"), "highlights",  "Highlights",    -0.5, 0.5)
        self._add_slider(self.tabs.tab("Fine-tune"), "shadows",     "Shadows",       -0.5, 0.5)
        self._add_slider(self.tabs.tab("Fine-tune"), "fade",        "Fade (matte)",  -0.1, 0.2)
        self._add_slider(self.tabs.tab("Fine-tune"), "vibrance",    "Vibrance",      -0.5, 0.5)

        btn_auto_exp = ctk.CTkButton(
            self.tabs.tab("Fine-tune"), text="Auto Exposure", height=28,
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self.run_auto_exposure)
        btn_auto_exp.pack(fill="x", padx=8, pady=(8, 4))
        self._themed_buttons.append((btn_auto_exp, "accent"))

        self._add_slider(self.tabs.tab("Grain"), "grain_amount", "Amount",     -0.05, 0.15)
        self._add_slider(self.tabs.tab("Grain"), "grain_size",   "Coarseness", -0.5, 1.5)
        self._add_slider(self.tabs.tab("Grain"), "grain_color",  "Dye colour", -0.02, 0.10)
        self._add_slider(self.tabs.tab("Grain"), "grain_bias",   "Mid bias",   -60.0, 80.0)

        self._add_slider(self.tabs.tab("Effects"), "bloom_amount",    "Bloom",    -0.2, 0.4)
        self._add_slider(self.tabs.tab("Effects"), "vignette_amount", "Vignette", -0.2, 0.3)
        self._add_slider(self.tabs.tab("Effects"), "texture",         "Texture",  -0.3, 0.3)
        self._add_slider(self.tabs.tab("Effects"), "clarity",         "Clarity",  -0.5, 0.5)

        self._add_slider(self.tabs.tab("Lens"), "lens_distortion",    "Distortion",    -1.0, 1.0)
        self._add_slider(self.tabs.tab("Lens"), "lens_tca_r",         "TCA Red",       -0.3, 0.3)
        self._add_slider(self.tabs.tab("Lens"), "lens_tca_b",         "TCA Blue",      -0.3, 0.3)
        self._add_slider(self.tabs.tab("Lens"), "lens_vignette_corr", "Vignette Corr", -0.5, 0.5)
        self._add_slider(self.tabs.tab("Lens"), "lens_keystone_h",    "Keystone H",    -0.3, 0.3)
        self._add_slider(self.tabs.tab("Lens"), "lens_keystone_v",    "Keystone V",    -0.3, 0.3)
        self._add_slider(self.tabs.tab("Lens"), "sharpening_amount",  "Sharpening",     0.0, 1.0)
        self._add_slider(self.tabs.tab("Lens"), "pixel_size",         "Pixel Size",     0.0, 12.0)

        btn_auto_lens = ctk.CTkButton(
            self.tabs.tab("Lens"), text="Auto Lens Correction", height=28,
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self.run_auto_lens)
        btn_auto_lens.pack(fill="x", padx=8, pady=(8, 4))
        self._themed_buttons.append((btn_auto_lens, "accent"))

        reset_row = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        reset_row.grid(row=row, column=0, padx=12, pady=(4, 2), sticky="ew")
        reset_row.grid_columnconfigure((0, 1), weight=1)
        btn_rst_sld = ctk.CTkButton(reset_row, text="Reset Sliders", height=28,
                                    command=self.reset_sliders)
        btn_rst_sld.grid(row=0, column=0, padx=(0, 3), sticky="ew")
        self._themed_buttons.append((btn_rst_sld, "surface"))

        btn_rst_photo = ctk.CTkButton(reset_row, text="Reset Photo", height=28,
                                      command=self.reset_photo)
        btn_rst_photo.grid(row=0, column=1, padx=(3, 0), sticky="ew")
        self._themed_buttons.append((btn_rst_photo, "danger"))
        row += 1

        clip_row = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        clip_row.grid(row=row, column=0, padx=12, pady=(2, 2), sticky="ew")
        clip_row.grid_columnconfigure((0, 1), weight=1)
        btn_copy = ctk.CTkButton(clip_row, text="Copy Edits", height=28,
                                 command=self.copy_edits)
        btn_copy.grid(row=0, column=0, padx=(0, 3), sticky="ew")
        self._themed_buttons.append((btn_copy, "surface"))

        btn_paste = ctk.CTkButton(clip_row, text="Paste Edits", height=28,
                                  command=self.paste_edits)
        btn_paste.grid(row=0, column=1, padx=(3, 0), sticky="ew")
        self._themed_buttons.append((btn_paste, "surface"))
        row += 1

        self.fullres_var = ctk.StringVar(value="1")
        self.fullres_chk = ctk.CTkCheckBox(
            self.sidebar, text="Full-res on export (slow)", variable=self.fullres_var)
        self.fullres_chk.grid(row=row, column=0, padx=16, pady=4, sticky="w")
        row += 1

        self.status_label = ctk.CTkLabel(
            self.sidebar, text="Open an ARW or folder to start",
            wraplength=275, justify="left", anchor="w")
        self.status_label.grid(row=row, column=0, padx=14, pady=(2, 10), sticky="ew")
        self._themed_labels.append((self.status_label, "muted"))
        row += 1

        # -- Centre Preview Viewport --------------------------------------------
        self.image_frame = ctk.CTkFrame(self, corner_radius=8)
        self.image_frame.grid(row=1, column=1, sticky="nsew", padx=8, pady=8)
        self.image_frame.grid_rowconfigure(0, weight=1)
        self.image_frame.grid_columnconfigure(0, weight=1)
        self.image_label = ctk.CTkLabel(
            self.image_frame,
            text="SAGE RAW STUDIO\n\nOpen a RAW/image file or browse a folder to start.\n"
                 "Shortcuts:  [ / ] Step Photos   •   \\ Compare Before/After",
            font=ctk.CTkFont(size=14))
        self.image_label.grid(row=0, column=0, sticky="nsew")
        self.image_label.bind("<ButtonPress-1>", self._on_preview_press)
        self.image_label.bind("<B1-Motion>", self._on_preview_drag)
        self.image_label.bind("<ButtonRelease-1>", self._on_preview_release)

        self._build_filter_browser()
        self._build_bottom_panel()

    def _add_slider(self, parent, attr: str, label: str, lo: float, hi: float):
        hdr = ctk.CTkFrame(parent, fg_color="transparent")
        hdr.pack(fill="x", padx=8, pady=(4, 0))
        lbl = ctk.CTkLabel(hdr, text=label, anchor="w", font=ctk.CTkFont(size=12))
        lbl.pack(side="left")
        self.slider_name_labels.append(lbl)

        val_lbl = ctk.CTkLabel(
            hdr, text="+0.00", anchor="e",
            font=ctk.CTkFont(family="monospace", size=11, weight="bold"))
        val_lbl.pack(side="right")
        val_lbl.bind("<Button-1>", lambda _e, a=attr: self._reset_single_slider(a))
        self.slider_val_labels[attr] = val_lbl

        sld = ctk.CTkSlider(parent, from_=lo, to=hi, number_of_steps=120,
                            command=lambda _v: self._on_slider_move())
        sld.set(0.0)
        sld.bind("<ButtonRelease-1>", lambda _e: self._on_slider_release())
        sld.bind("<Double-Button-1>", lambda _e, a=attr: self._reset_single_slider(a))
        sld.bind("<Button-3>", lambda _e, a=attr: self._reset_single_slider(a))
        sld.pack(fill="x", padx=8, pady=(0, 2))
        self._slider_rest[attr] = sld.get()
        self._slider_bounds[attr] = (lo, hi)
        self.sliders[attr] = sld

    def _reset_single_slider(self, attr: str):
        if attr not in self.sliders:
            return
        self._suppress += 1
        self.sliders[attr].set(self._slider_rest[attr])
        self._suppress -= 1
        self._update_slider_labels()
        self._draft_preview = False
        self._schedule_save()
        self.schedule_render()
        self.set_status(f"Reset {attr} to preset default")

    def _update_slider_labels(self):
        t = self.active_theme()
        acc = self.active_accent()
        for attr, sld in self.sliders.items():
            vl = self.slider_val_labels.get(attr)
            if vl is None:
                continue
            delta = float(sld.get() - self._slider_rest[attr])
            lo, hi = self._slider_bounds[attr]
            if hi - lo >= 20.0:
                txt = f"{delta:+.1f}"
            else:
                txt = f"{delta:+.2f}"
            active = abs(delta) >= 1e-4
            vl.configure(text=txt, text_color=acc if active else t["fg_muted"])

    def _build_filter_browser(self):
        self.panel_bar = ctk.CTkFrame(self, width=275, corner_radius=0)
        self.panel_bar.grid(row=1, column=2, sticky="nsew")
        self.panel_bar.grid_columnconfigure(0, weight=1)
        self.panel_bar.grid_rowconfigure(2, weight=1)
        self._themed_frames.append((self.panel_bar, "sidebar"))

        flt_hdr = ctk.CTkLabel(
            self.panel_bar, text="FILTER PRESETS", anchor="w",
            font=ctk.CTkFont(size=12, weight="bold"))
        flt_hdr.grid(row=0, column=0, padx=14, pady=(10, 2), sticky="ew")
        self._themed_labels.append((flt_hdr, "accent"))

        self.search_var = ctk.StringVar()
        self.search_var.trace_add("write", lambda *_: self._on_search_changed())
        self.search_entry = ctk.CTkEntry(
            self.panel_bar, placeholder_text="Search filters…",
            textvariable=self.search_var)
        self.search_entry.grid(row=1, column=0, padx=10, pady=(2, 6), sticky="ew")

        self.browser = ctk.CTkScrollableFrame(self.panel_bar)
        self.browser.grid(row=2, column=0, padx=6, pady=(0, 6), sticky="nsew")
        self.browser.grid_columnconfigure(0, weight=1)

        first_category = True
        b_row = 0
        for cat_dir, filters in self.filter_categories.items():
            colour = filters[0].colour
            header = ctk.CTkButton(
                self.browser, height=30, anchor="w", fg_color=colour,
                font=ctk.CTkFont(size=12, weight="bold"),
                text=self._category_text(cat_dir, expanded=first_category),
                command=lambda d=cat_dir: self.toggle_category(d))
            header.grid(row=b_row, column=0, sticky="ew", padx=2, pady=(5, 0))
            self.category_headers[cat_dir] = header
            b_row += 1

            panel = ctk.CTkFrame(self.browser, fg_color="transparent")
            panel.grid(row=b_row, column=0, sticky="ew", padx=2)
            panel.grid_columnconfigure(0, weight=1)
            self.category_panels[cat_dir] = panel
            b_row += 1

            for i, flt in enumerate(filters):
                self.filter_cat_colours[flt.key] = colour
                btn = ctk.CTkButton(
                    panel, height=27, anchor="w", fg_color=colour,
                    text=f"●  {flt.label}",
                    command=lambda k=flt.key: self.select_filter(k))
                btn.grid(row=i, column=0, sticky="ew", padx=(8, 0), pady=2)
                self.filter_buttons[flt.key] = btn
            if not first_category:
                panel.grid_remove()
            first_category = False

    def _build_bottom_panel(self):
        self.bottom_frame = ctk.CTkFrame(self, height=215, corner_radius=8)
        self.bottom_frame.grid(row=2, column=0, columnspan=3, sticky="ew", padx=8, pady=(0, 8))
        self.bottom_frame.grid_columnconfigure(0, weight=1)
        self._themed_frames.append((self.bottom_frame, "sidebar"))

        top_bar = ctk.CTkFrame(self.bottom_frame, fg_color="transparent")
        top_bar.grid(row=0, column=0, sticky="ew", padx=8, pady=(6, 2))
        lbl_fp = ctk.CTkLabel(
            top_bar, text="FILMSTRIP  •  FOLDER & PROJECT",
            font=ctk.CTkFont(size=12, weight="bold"))
        lbl_fp.pack(side="left", padx=(4, 10))
        self._themed_labels.append((lbl_fp, "accent"))

        for txt, cmd, w in (
            ("Browse Folder", self.browse_folder, 115),
            ("Add Folder to Project", self.add_folder_to_project, 150),
            ("Add Photos to Project", self.add_photos_to_project, 150),
        ):
            b = ctk.CTkButton(top_bar, text=txt, width=w, height=26, command=cmd)
            b.pack(side="left", padx=3)
            self._themed_buttons.append((b, "surface"))

        for txt, d in (("Next Photo ▶", 1), ("◀ Prev Photo", -1)):
            b = ctk.CTkButton(top_bar, text=txt, width=105, height=26,
                              command=lambda step=d: self.step_photo(step))
            b.pack(side="right", padx=3)
            self._themed_buttons.append((b, "surface"))

        self.bottom_tabs = ctk.CTkTabview(self.bottom_frame, height=165)
        self.bottom_tabs.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 6))
        self.bottom_tabs.add("Folder")
        self.bottom_tabs.add("Project")

        # Folder strip
        f_tab = self.bottom_tabs.tab("Folder")
        f_tab.grid_columnconfigure(0, weight=1)
        self.folder_scroll = ctk.CTkScrollableFrame(f_tab, orientation="horizontal", height=110)
        self.folder_scroll.grid(row=0, column=0, sticky="ew")

        # Project strip + toolbar
        p_tab = self.bottom_tabs.tab("Project")
        p_tab.grid_columnconfigure(0, weight=1)
        p_tools = ctk.CTkFrame(p_tab, fg_color="transparent")
        p_tools.grid(row=0, column=0, sticky="ew", pady=(0, 4))

        b_apply = ctk.CTkButton(p_tools, text="Apply Filter to Selected", width=155, height=24,
                                command=self.apply_filter_to_selected)
        b_apply.pack(side="left", padx=2)
        self._themed_buttons.append((b_apply, "accent"))

        b_sync = ctk.CTkButton(p_tools, text="Sync Edits to Selected", width=150, height=24,
                               command=self.sync_edits_to_selected)
        b_sync.pack(side="left", padx=2)
        self._themed_buttons.append((b_sync, "surface"))

        b_rst_sel = ctk.CTkButton(p_tools, text="Reset Selected Edits", width=140, height=24,
                                  command=self.reset_selected_edits)
        b_rst_sel.pack(side="left", padx=2)
        self._themed_buttons.append((b_rst_sel, "danger"))

        for txt, mode, w in (
            ("Select All", "all", 75),
            ("None", "none", 55),
            ("Invert", "invert", 60),
            ("Remove", "remove", 65),
        ):
            b = ctk.CTkButton(p_tools, text=txt, width=w, height=24,
                              command=lambda m=mode: self._project_select(m))
            b.pack(side="left", padx=2)
            self._themed_buttons.append((b, "surface"))

        for txt, cmd in (("Save Project", self.save_project_dialog), ("Load Project", self.load_project_dialog)):
            b = ctk.CTkButton(p_tools, text=txt, width=95, height=24, command=cmd)
            b.pack(side="left", padx=2)
            self._themed_buttons.append((b, "surface"))

        b_exp_sel = ctk.CTkButton(p_tools, text="Export Selected", width=115, height=24,
                                  font=ctk.CTkFont(size=12, weight="bold"),
                                  command=self.export_selected_project)
        b_exp_sel.pack(side="left", padx=4)
        self._themed_buttons.append((b_exp_sel, "success"))

        self.project_scroll = ctk.CTkScrollableFrame(p_tab, orientation="horizontal", height=95)
        self.project_scroll.grid(row=1, column=0, sticky="ew")

    # -- State & Sidecar helpers -----------------------------------------------

    def set_status(self, text: str):
        self.status_label.configure(text=text)
        self.update_idletasks()

    _NON_NEGATIVE = NON_NEGATIVE_FLOORS

    def _current_offsets(self) -> Dict[str, float]:
        offsets: Dict[str, float] = {}
        for attr, sld in self.sliders.items():
            delta = float(sld.get() - self._slider_rest[attr])
            if abs(delta) >= 1e-9:
                offsets[attr] = delta
        return offsets

    def _settings_with_slider_offsets(self) -> DevelopSettings:
        return apply_offsets_to_settings(self.base_settings, self._current_offsets())

    def _schedule_save(self):
        if self._suppress > 0 or not self.file_path:
            return
        self._dirty = True
        if self._save_job is not None:
            self.after_cancel(self._save_job)
        self._save_job = self.after(350, self._save_state_now)

    def _save_state_now(self):
        if self._save_job is not None:
            self.after_cancel(self._save_job)
            self._save_job = None
        if not self._dirty or not self.file_path:
            self._dirty = False
            return
        flt_key = self.current_filter.key if self.current_filter else "00-basics/original"
        if save_photo_state(self.file_path, flt_key, self._current_offsets()):
            self._dirty = False
            self._refresh_strip_badges()

    def _format_badge(self, image_path: str) -> Tuple[str, bool]:
        state = load_photo_state(image_path)
        if state is None:
            return "Unedited", False
        flt_key, offsets = state
        flt = self.all_filters.get(flt_key)
        flt_label = flt.label if flt else (flt_key or "Original")
        is_orig = (not flt_key or flt_key == "00-basics/original")
        if is_orig and not offsets:
            return "Original", False
        if offsets:
            return f"● {flt_label} (+{len(offsets)})", True
        return f"● {flt_label}", True

    def _refresh_strip_badges(self):
        t = self.active_theme()
        acc = self.active_accent()
        for cards in (self._folder_cards, self._project_cards):
            for path, info in cards.items():
                text, edited = self._format_badge(path)
                info["badge"].configure(text=text, text_color=acc if edited else t["fg_muted"])

    def _highlight_active_photo(self):
        t = self.active_theme()
        acc = self.active_accent()
        for cards in (self._folder_cards, self._project_cards):
            for path, info in cards.items():
                active = (path == self.file_path)
                info["frame"].configure(
                    fg_color=t["bg_hover"] if active else t["bg_surface"],
                    border_width=2 if active else 1,
                    border_color=acc if active else t["border"],
                )

    def _on_slider_move(self):
        self._update_slider_labels()
        if self._suppress > 0:
            return
        self._draft_preview = True
        self._schedule_save()
        self.schedule_render()

    def _on_slider_release(self):
        self._update_slider_labels()
        if self._suppress > 0:
            return
        self._draft_preview = False
        self._schedule_save()
        self.schedule_render()

    def schedule_render(self):
        if self.source_linear is None:
            return
        if self._render_job is not None:
            self.after_cancel(self._render_job)
        self._render_job = self.after(60 if self._draft_preview else 90,
                                      self.render_preview)

    # -- Theme engine & Customization Dialog -----------------------------------

    def _on_theme_menu_selected(self, label: str):
        t = THEME_BY_LABEL.get(label)
        if t is None:
            return
        self.ui_prefs["theme"] = t["key"]
        self.apply_theme()
        self._save_ui_prefs()
        self.set_status(f"Theme switched to {t['label']}")

    def _update_header_toggles(self):
        t = self.active_theme()
        acc = self.active_accent()
        cmp_lbl = ("Compare: Split" if self.compare_mode == 1
                   else "Compare: Before" if self.compare_mode == 2
                   else "Compare (\\)")
        self.compare_btn.configure(
            text=cmp_lbl,
            fg_color=acc if self.compare_mode else t["bg_surface"],
            text_color=t["accent_fg"] if self.compare_mode else t["fg_primary"],
            hover_color=_shade_hex(acc if self.compare_mode else t["bg_hover"], 1.08),
        )
        hist_on = bool(self.ui_prefs.get("show_histogram", True))
        self.hist_btn.configure(
            fg_color=acc if hist_on else t["bg_surface"],
            text_color=t["accent_fg"] if hist_on else t["fg_primary"],
            hover_color=_shade_hex(acc if hist_on else t["bg_hover"], 1.08),
        )
        gm = int(self.ui_prefs.get("grid_mode", 0))
        glbl = ("Grid: Thirds" if gm == 1
                else "Grid: Golden" if gm == 2
                else "Grid: Cross" if gm == 3
                else "Grid")
        self.grid_btn.configure(
            text=glbl,
            fg_color=acc if gm else t["bg_surface"],
            text_color=t["accent_fg"] if gm else t["fg_primary"],
            hover_color=_shade_hex(acc if gm else t["bg_hover"], 1.08),
        )

    def apply_theme(self):
        t = self.active_theme()
        acc = self.active_accent()
        backdrop = self.active_backdrop()

        ctk.set_appearance_mode("Dark" if t["is_dark"] else "Light")
        self.configure(fg_color=t["bg_main"])

        for frame, role in self._themed_frames:
            if role == "sidebar":
                frame.configure(fg_color=t["bg_sidebar"])
            elif role == "surface":
                frame.configure(fg_color=t["bg_surface"])

        for lbl, role in self._themed_labels:
            if role == "accent":
                lbl.configure(text_color=acc)
            elif role == "muted":
                lbl.configure(text_color=t["fg_muted"])
            else:
                lbl.configure(text_color=t["fg_primary"])

        for slbl in self.slider_name_labels:
            slbl.configure(text_color=t["fg_primary"])

        for btn, role in self._themed_buttons:
            if role == "accent":
                btn.configure(fg_color=acc, hover_color=_shade_hex(acc, 1.12),
                              text_color=t["accent_fg"])
            elif role == "success":
                btn.configure(fg_color=t["success"], hover_color=_shade_hex(t["success"], 1.12),
                              text_color="#ffffff")
            elif role == "danger":
                btn.configure(fg_color=t["danger"], hover_color=_shade_hex(t["danger"], 1.12),
                              text_color="#ffffff")
            else:
                btn.configure(fg_color=t["bg_surface"], hover_color=t["bg_hover"],
                              text_color=t["fg_primary"])

        self.theme_menu.configure(
            fg_color=t["bg_surface"], button_color=t["bg_hover"],
            button_hover_color=acc, text_color=t["fg_primary"])
        self.theme_menu.set(t["label"])

        for tv in (self.tabs, self.bottom_tabs):
            tv.configure(
                fg_color=t["bg_sidebar"],
                segmented_button_fg_color=t["bg_main"],
                segmented_button_selected_color=acc,
                segmented_button_selected_hover_color=_shade_hex(acc, 1.1),
                segmented_button_unselected_color=t["bg_surface"],
                segmented_button_unselected_hover_color=t["bg_hover"],
                text_color=t["accent_fg"],
            )

        dens = int(self.ui_prefs.get("density", 1))
        sld_h = 14 if dens == 0 else (20 if dens == 2 else 16)
        for sld in self.sliders.values():
            sld.configure(
                fg_color=t["bg_surface"],
                progress_color=acc,
                button_color=acc,
                button_hover_color=_shade_hex(acc, 1.15),
                height=sld_h,
            )

        self.fullres_chk.configure(
            fg_color=acc, hover_color=_shade_hex(acc, 1.1),
            text_color=t["fg_primary"], border_color=t["border"])
        self.search_entry.configure(
            fg_color=t["bg_surface"], border_color=t["border"],
            text_color=t["fg_primary"], placeholder_text_color=t["fg_muted"])
        self.browser.configure(fg_color=t["bg_sidebar"])
        self.folder_scroll.configure(fg_color=t["bg_main"])
        self.project_scroll.configure(fg_color=t["bg_main"])

        self.image_frame.configure(fg_color=backdrop, border_width=1, border_color=t["border"])
        self.image_label.configure(text_color=t["fg_muted"])

        # Style filter category headers & filter buttons (Modern vs Classic)
        modern = int(self.ui_prefs.get("filter_style", 0)) == 0
        btn_h = 24 if dens == 0 else (31 if dens == 2 else 27)
        active_key = self.current_filter.key if self.current_filter else ""
        for cat_dir, hdr in self.category_headers.items():
            flts = self.filter_categories.get(cat_dir, [])
            c_hex = flts[0].colour if flts else t["bg_surface"]
            if modern:
                hdr.configure(
                    fg_color=t["bg_surface"], hover_color=t["bg_hover"],
                    text_color=t["fg_primary"], border_width=1, border_color=c_hex)
            else:
                hdr.configure(
                    fg_color=c_hex, hover_color=_shade_hex(c_hex, 1.15),
                    text_color="#ffffff", border_width=0)

        for f_key, btn in self.filter_buttons.items():
            c_hex = self.filter_cat_colours.get(f_key, t["bg_surface"])
            is_act = (f_key == active_key)
            if modern:
                btn.configure(
                    height=btn_h,
                    fg_color=t["bg_hover"] if is_act else t["bg_surface"],
                    hover_color=t["bg_hover"],
                    text_color=acc if is_act else t["fg_primary"],
                    border_width=2 if is_act else 1,
                    border_color=acc if is_act else c_hex,
                )
            else:
                btn.configure(
                    height=btn_h,
                    fg_color=c_hex,
                    hover_color=_shade_hex(c_hex, 1.15),
                    text_color="#ffffff",
                    border_width=2 if is_act else 0,
                    border_color=acc,
                )

        self._update_header_toggles()
        self._update_slider_labels()
        self._refresh_strip_badges()
        self._highlight_active_photo()
        if self._last_u8 is not None:
            self.display(self._last_u8)

    def cycle_compare(self):
        self.compare_mode = (self.compare_mode + 1) % 3
        self._update_header_toggles()
        if self._last_u8 is not None:
            self.display(self._last_u8)
        msg = ("Comparison: Before / After Split Wipe (drag line on preview)" if self.compare_mode == 1
               else "Comparison: Viewing Original (Before)" if self.compare_mode == 2
               else "Comparison: Off (Viewing Edited)")
        self.set_status(msg)

    def toggle_histogram(self):
        self.ui_prefs["show_histogram"] = not bool(self.ui_prefs.get("show_histogram", True))
        self._save_ui_prefs()
        self._update_header_toggles()
        if self._last_u8 is not None:
            self.display(self._last_u8)

    def cycle_grid(self):
        self.ui_prefs["grid_mode"] = (int(self.ui_prefs.get("grid_mode", 0)) + 1) % 4
        self._save_ui_prefs()
        self._update_header_toggles()
        if self._last_u8 is not None:
            self.display(self._last_u8)

    def _on_preview_press(self, event):
        if self.compare_mode == 1:
            self._dragging_split = True
            self._update_split_from_event(event)

    def _on_preview_drag(self, event):
        if self._dragging_split and self.compare_mode == 1:
            self._update_split_from_event(event)

    def _on_preview_release(self, _event):
        self._dragging_split = False

    def _update_split_from_event(self, event):
        w = max(1, self.image_label.winfo_width())
        self.split_ratio = max(0.05, min(0.95, float(event.x) / float(w)))
        if self._last_u8 is not None:
            self.display(self._last_u8)

    def open_customize_dialog(self):
        if self._customize_win is not None and self._customize_win.winfo_exists():
            self._customize_win.focus()
            return

        t = self.active_theme()
        acc = self.active_accent()
        win = ctk.CTkToplevel(self)
        win.title("Customize Studio Appearance & Themes")
        win.geometry("470x590")
        win.resizable(False, False)
        win.configure(fg_color=t["bg_sidebar"])
        self._customize_win = win

        pad = {"padx": 18, "pady": 5}
        ctk.CTkLabel(
            win, text="STUDIO APPEARANCE & THEMES",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=acc).pack(anchor="w", padx=18, pady=(14, 6))

        def _row(label_text: str, values: List[str], current: str, on_change):
            f = ctk.CTkFrame(win, fg_color="transparent")
            f.pack(fill="x", **pad)
            ctk.CTkLabel(f, text=label_text, text_color=t["fg_primary"]).pack(side="left")
            om = ctk.CTkOptionMenu(
                f, width=230, values=values, command=on_change,
                fg_color=t["bg_surface"], button_color=t["bg_hover"],
                text_color=t["fg_primary"])
            om.set(current)
            om.pack(side="right")
            return om

        cur_theme_lbl = self.active_theme()["label"]
        cur_acc_lbl = next((l for k, l, _ in ACCENT_OPTIONS if k == self.ui_prefs["accent"]), ACCENT_OPTIONS[0][1])
        cur_bg_lbl = next((l for k, l, _ in BACKDROP_OPTIONS if k == self.ui_prefs["backdrop"]), BACKDROP_OPTIONS[0][1])

        def _set_theme(v: str):
            self._on_theme_menu_selected(v)

        def _set_accent(v: str):
            for k, l, _ in ACCENT_OPTIONS:
                if l == v:
                    self.ui_prefs["accent"] = k
                    break
            self.apply_theme()
            self._save_ui_prefs()

        def _set_backdrop(v: str):
            for k, l, _ in BACKDROP_OPTIONS:
                if l == v:
                    self.ui_prefs["backdrop"] = k
                    break
            self.apply_theme()
            self._save_ui_prefs()

        def _set_idx(pref_key: str, lst: List[str], v: str):
            if v in lst:
                self.ui_prefs[pref_key] = lst.index(v)
                self.apply_theme()
                self._save_ui_prefs()

        om_theme = _row("Color Theme", [x["label"] for x in THEMES], cur_theme_lbl, _set_theme)
        om_acc = _row("Accent Color", [x[1] for x in ACCENT_OPTIONS], cur_acc_lbl, _set_accent)
        om_bg = _row("Canvas Backdrop", [x[1] for x in BACKDROP_OPTIONS], cur_bg_lbl, _set_backdrop)
        om_frame = _row("Photo Frame Style", FRAME_STYLES,
                        FRAME_STYLES[int(self.ui_prefs["frame_style"])],
                        lambda v: _set_idx("frame_style", FRAME_STYLES, v))
        om_dens = _row("UI Density", DENSITY_LEVELS,
                       DENSITY_LEVELS[int(self.ui_prefs["density"])],
                       lambda v: _set_idx("density", DENSITY_LEVELS, v))
        om_fstyle = _row("Filter Browser Style", FILTER_STYLES,
                         FILTER_STYLES[int(self.ui_prefs["filter_style"])],
                         lambda v: _set_idx("filter_style", FILTER_STYLES, v))
        om_grid = _row("Composition Grid", GRID_MODES,
                       GRID_MODES[int(self.ui_prefs["grid_mode"])],
                       lambda v: _set_idx("grid_mode", GRID_MODES, v))

        hist_var = ctk.BooleanVar(value=bool(self.ui_prefs.get("show_histogram", True)))
        hud_var = ctk.BooleanVar(value=bool(self.ui_prefs.get("show_hud", True)))

        def _tog_hist():
            self.ui_prefs["show_histogram"] = bool(hist_var.get())
            self.apply_theme()
            self._save_ui_prefs()

        def _tog_hud():
            self.ui_prefs["show_hud"] = bool(hud_var.get())
            self.apply_theme()
            self._save_ui_prefs()

        ctk.CTkCheckBox(
            win, text="Show Live RGB / Luminance Histogram on Preview",
            variable=hist_var, command=_tog_hist, text_color=t["fg_primary"]
        ).pack(anchor="w", padx=18, pady=(12, 4))

        ctk.CTkCheckBox(
            win, text="Show Translucent Photo Info Pill on Preview",
            variable=hud_var, command=_tog_hud, text_color=t["fg_primary"]
        ).pack(anchor="w", padx=18, pady=4)

        def _reset_defaults():
            self.ui_prefs.update({
                "theme": "gruvbox-dark",
                "accent": "theme",
                "backdrop": "theme",
                "frame_style": 0,
                "density": 1,
                "filter_style": 0,
                "show_histogram": True,
                "grid_mode": 0,
                "show_hud": True,
            })
            om_theme.set(THEMES[0]["label"])
            om_acc.set(ACCENT_OPTIONS[0][1])
            om_bg.set(BACKDROP_OPTIONS[0][1])
            om_frame.set(FRAME_STYLES[0])
            om_dens.set(DENSITY_LEVELS[1])
            om_fstyle.set(FILTER_STYLES[0])
            om_grid.set(GRID_MODES[0])
            hist_var.set(True)
            hud_var.set(True)
            self.apply_theme()
            self._save_ui_prefs()

        ctk.CTkButton(
            win, text="Reset Appearance to Defaults (Gruvbox Dark)",
            fg_color=t["bg_surface"], hover_color=t["bg_hover"], text_color=t["fg_primary"],
            command=_reset_defaults
        ).pack(fill="x", padx=18, pady=(16, 6))

        ctk.CTkButton(
            win, text="Done", fg_color=acc, text_color=t["accent_fg"],
            command=win.destroy
        ).pack(fill="x", padx=18, pady=(4, 14))

    # -- Photo loading & state restoration -------------------------------------

    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("RAW / images", "*.ARW *.arw *.CR2 *.NEF *.DNG *.RAF *.jpg *.jpeg *.png"),
                       ("All files", "*.*")])
        if not path:
            return
        self.open_photo(path)

    def open_photo(self, path: str):
        """Flush current photo's pending edits (only if modified), decode `path`,
        and restore its own <path>.fastedit.json edit state (or reset to Original)."""
        if not path:
            return
        self._save_state_now()
        self.set_status(f"Decoding {os.path.basename(path)}…")
        try:
            self.source_linear = self._decode(path, max_edge=PREVIEW_LONG_EDGE)
            self.full_linear = None
            self.full_is_loaded = False
            self._orig_u8 = None
            self.file_path = path
            self.camera_date = read_exif_date(path)
        except Exception as exc:
            self.set_status(f"Error: {exc}")
            return

        state = load_photo_state(path)
        self._suppress += 1
        try:
            if state is not None:
                flt_key, offsets = state
                if flt_key not in self.all_filters:
                    flt_key = "00-basics/original"
                self.select_filter(flt_key)
                for attr, sld in self.sliders.items():
                    lo, hi = self._slider_bounds[attr]
                    delta = max(lo, min(hi, float(offsets.get(attr, 0.0))))
                    sld.set(self._slider_rest[attr] + delta)
            else:
                self.select_filter("00-basics/original")
                for attr, sld in self.sliders.items():
                    sld.set(self._slider_rest[attr])
        finally:
            self._suppress -= 1
        self._update_slider_labels()

        self._dirty = False
        self._draft_preview = False

        if not self.folder_paths:
            parent = str(Path(path).resolve().parent)
            self._load_folder_paths(parent, auto_select_first=False)

        self._highlight_active_photo()
        self.schedule_render()
        flt_label = self.current_filter.label if self.current_filter else "Original"
        if state is not None:
            self.set_status(f"{os.path.basename(path)} — {flt_label} ({len(state[1])} adjustments restored)")
        else:
            self.set_status(f"{os.path.basename(path)} — Original (unedited)")

    def _decode(self, path: str, max_edge: int | None) -> np.ndarray:
        ext = Path(path).suffix.lower()
        if ext in RAW_EXTENSIONS:
            with rawpy.imread(path) as raw:
                half = max_edge is not None
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

    # -- Filter selection & search ---------------------------------------------

    def select_filter(self, key: str):
        flt = self.all_filters.get(key)
        if flt is None:
            return
        self.current_filter = flt
        self.base_settings = flt.settings
        t = self.active_theme()
        acc = self.active_accent()
        modern = int(self.ui_prefs.get("filter_style", 0)) == 0
        for f_key, btn in self.filter_buttons.items():
            c_hex = self.filter_cat_colours.get(f_key, t["bg_surface"])
            is_act = (f_key == key)
            if modern:
                btn.configure(
                    fg_color=t["bg_hover"] if is_act else t["bg_surface"],
                    text_color=acc if is_act else t["fg_primary"],
                    border_width=2 if is_act else 1,
                    border_color=acc if is_act else c_hex,
                )
            else:
                btn.configure(border_width=2 if is_act else 0, border_color=acc)
        self._draft_preview = False
        self._suppress += 1
        for attr, sld in self.sliders.items():
            sld.set(self._slider_rest[attr])
        self._suppress -= 1
        self._update_slider_labels()
        if self._suppress == 0:
            self._schedule_save()
            self.set_status(flt.description or flt.label)
            self.schedule_render()

    def _on_search_changed(self):
        q = self.search_var.get().strip().lower()
        for cat_dir, filters in self.filter_categories.items():
            any_visible = False
            panel = self.category_panels[cat_dir]
            for flt in filters:
                btn = self.filter_buttons[flt.key]
                match = (not q) or (q in flt.label.lower()) or (q in flt.key.lower())
                if match:
                    btn.grid()
                    any_visible = True
                else:
                    btn.grid_remove()
            header = self.category_headers[cat_dir]
            if not q:
                header.grid()
            elif any_visible:
                header.grid()
                panel.grid()
                header.configure(text=self._category_text(cat_dir, expanded=True))
            else:
                header.grid_remove()
                panel.grid_remove()

    def _category_text(self, cat_dir: str, expanded: bool) -> str:
        filters = self.filter_categories.get(cat_dir, [])
        label = filters[0].category_label if filters else cat_dir
        return f"{'▾' if expanded else '▸'} {label}  ({len(filters)})"

    def toggle_category(self, cat_dir: str):
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

    # -- Per-photo edit tools --------------------------------------------------

    def reset_sliders(self, keep_status: bool = False):
        self._suppress += 1
        for attr, sld in self.sliders.items():
            sld.set(self._slider_rest[attr])
        self._suppress -= 1
        self._update_slider_labels()
        self._draft_preview = False
        if self._suppress == 0:
            self._schedule_save()
            self._save_state_now()
            if not keep_status:
                self.set_status("Sliders reset to preset")
            self.schedule_render()

    def reset_photo(self):
        if self.source_linear is None or not self.file_path:
            self.set_status("Load an image first")
            return
        if self._save_job is not None:
            self.after_cancel(self._save_job)
            self._save_job = None
        self._dirty = False
        remove_photo_state(self.file_path)
        self._suppress += 1
        self.select_filter("00-basics/original")
        for attr, sld in self.sliders.items():
            sld.set(self._slider_rest[attr])
        self._suppress -= 1
        self._update_slider_labels()
        self._refresh_strip_badges()
        self.schedule_render()
        self.set_status(f"Reset {os.path.basename(self.file_path)} to Original")

    def copy_edits(self):
        if self.source_linear is None:
            self.set_status("Load an image first")
            return
        flt_key = self.current_filter.key if self.current_filter else "00-basics/original"
        offs = self._current_offsets()
        self._clipboard = (flt_key, offs)
        self.set_status(f"Copied edits ({flt_key}, {len(offs)} adjustments)")

    def paste_edits(self):
        if self.source_linear is None or not self.file_path:
            self.set_status("Load an image first")
            return
        if self._clipboard is None:
            self.set_status("Nothing copied yet — click Copy Edits first")
            return
        flt_key, offs = self._clipboard
        self._suppress += 1
        self.select_filter(flt_key)
        for attr, sld in self.sliders.items():
            lo, hi = self._slider_bounds[attr]
            delta = max(lo, min(hi, float(offs.get(attr, 0.0))))
            sld.set(self._slider_rest[attr] + delta)
        self._suppress -= 1
        self._update_slider_labels()
        self._schedule_save()
        self._save_state_now()
        self.schedule_render()
        self.set_status(f"Pasted edits onto {os.path.basename(self.file_path)}")

    def run_auto_exposure(self):
        if self.source_linear is None:
            self.set_status("Load an image first")
            return
        target = auto_exposure(self.source_linear)
        offset = max(-2.0, min(2.0, target - self.base_settings.exposure))
        self._suppress += 1
        self.sliders["exposure"].set(self._slider_rest["exposure"] + offset)
        self._suppress -= 1
        self._update_slider_labels()
        self._schedule_save()
        self.set_status(f"Auto exposure: {self.base_settings.exposure + offset:+.2f} stops")
        self.schedule_render()

    def run_auto_lens(self):
        if self.source_linear is None:
            self.set_status("Load an image first")
            return
        dist, tca_r, tca_b, vig = auto_lens_analyze(self.source_linear)
        self._suppress += 1
        for attr, val in (("lens_distortion", dist), ("lens_tca_r", tca_r),
                          ("lens_tca_b", tca_b), ("lens_vignette_corr", vig)):
            if abs(val) > 1e-4:
                lo, hi = self._slider_bounds[attr]
                off = max(lo, min(hi, val - getattr(self.base_settings, attr)))
                self.sliders[attr].set(self._slider_rest[attr] + off)
        self._suppress -= 1
        self._update_slider_labels()
        self._schedule_save()
        self.set_status("Auto lens correction applied")
        self.schedule_render()

    # -- Folder & Project film-strip methods -----------------------------------

    def browse_folder(self):
        folder = filedialog.askdirectory(title="Select Photo Folder")
        if not folder:
            return
        self.bottom_tabs.set("Folder")
        self._load_folder_paths(folder, auto_select_first=True)

    @staticmethod
    def _scan_folder_images(folder: str) -> List[str]:
        p = Path(folder)
        if not p.is_dir():
            return []
        files = [f for f in p.iterdir() if f.is_file() and not f.name.startswith(".")
                 and f.suffix.lower() in IMAGE_EXTENSIONS]
        files.sort(key=lambda x: (x.name.lower(), x.name))
        has_raw = any(f.suffix.lower() in RAW_EXTENSIONS for f in files)
        if has_raw:
            files = [f for f in files if f.suffix.lower() in RAW_EXTENSIONS]
        return [str(f) for f in files]

    def _load_folder_paths(self, folder: str, auto_select_first: bool):
        self._save_state_now()
        self.current_folder = folder
        self.folder_paths = self._scan_folder_images(folder)
        for child in self.folder_scroll.winfo_children():
            child.destroy()
        self._folder_cards.clear()

        t = self.active_theme()
        acc = self.active_accent()
        for i, path in enumerate(self.folder_paths):
            card = ctk.CTkFrame(self.folder_scroll, corner_radius=8, fg_color=t["bg_surface"])
            card.grid(row=0, column=i, padx=4, pady=4)
            btn = ctk.CTkButton(card, text=os.path.basename(path)[:18], width=125, height=46,
                                fg_color=t["bg_sidebar"], hover_color=t["bg_hover"],
                                text_color=t["fg_primary"],
                                command=lambda p=path: self.open_photo(p))
            btn.pack(padx=4, pady=(4, 2))
            badge_text, edited = self._format_badge(path)
            badge = ctk.CTkLabel(card, text=badge_text, font=ctk.CTkFont(size=11, weight="bold" if edited else "normal"),
                                 text_color=acc if edited else t["fg_muted"])
            badge.pack(padx=4, pady=(0, 4))
            self._folder_cards[path] = {"frame": card, "badge": badge}

        self._highlight_active_photo()
        if auto_select_first and self.folder_paths:
            self.open_photo(self.folder_paths[0])

    def step_photo(self, direction: int):
        active_tab = self.bottom_tabs.get()
        if active_tab == "Project" and self.project_items:
            paths = [it["path"] for it in self.project_items]
        elif self.folder_paths:
            paths = self.folder_paths
        else:
            paths = [it["path"] for it in self.project_items]
        if not paths:
            return
        try:
            idx = paths.index(self.file_path)
            next_idx = (idx + direction) % len(paths)
        except ValueError:
            next_idx = 0
        self.open_photo(paths[next_idx])

    def _rebuild_project_strip(self):
        for child in self.project_scroll.winfo_children():
            child.destroy()
        self._project_cards.clear()
        t = self.active_theme()
        acc = self.active_accent()
        for i, item in enumerate(self.project_items):
            path = item["path"]
            card = ctk.CTkFrame(self.project_scroll, corner_radius=8, fg_color=t["bg_surface"])
            card.grid(row=0, column=i, padx=4, pady=4)
            var = ctk.BooleanVar(value=bool(item.get("selected", True)))

            def _toggle(v=var, it=item):
                it["selected"] = bool(v.get())

            chk = ctk.CTkCheckBox(card, text=os.path.basename(path)[:16], variable=var,
                                  text_color=t["fg_primary"], fg_color=acc,
                                  command=_toggle, width=120)
            chk.pack(padx=4, pady=(4, 2))
            btn = ctk.CTkButton(card, text="Open / Edit", width=120, height=28,
                                fg_color=t["bg_sidebar"], hover_color=t["bg_hover"],
                                text_color=t["fg_primary"],
                                command=lambda p=path: self.open_photo(p))
            btn.pack(padx=4, pady=2)
            badge_text, edited = self._format_badge(path)
            badge = ctk.CTkLabel(card, text=badge_text, font=ctk.CTkFont(size=11, weight="bold" if edited else "normal"),
                                 text_color=acc if edited else t["fg_muted"])
            badge.pack(padx=4, pady=(0, 4))
            self._project_cards[path] = {"frame": card, "badge": badge, "var": var}
        self._highlight_active_photo()

    def add_photos_to_project(self):
        paths = filedialog.askopenfilenames(
            title="Add Photos to Project",
            filetypes=[("RAW / images", "*.ARW *.arw *.CR2 *.NEF *.DNG *.RAF *.jpg *.jpeg *.png"),
                       ("All files", "*.*")])
        if not paths:
            return
        existing = {it["path"] for it in self.project_items}
        added = 0
        for p in paths:
            if p not in existing:
                self.project_items.append({"path": p, "selected": True})
                existing.add(p)
                added += 1
        self._rebuild_project_strip()
        self.bottom_tabs.set("Project")
        if self.source_linear is None and self.project_items:
            self.open_photo(self.project_items[0]["path"])
        self.set_status(f"Added {added} photo(s) to Project ({len(self.project_items)} total)")

    def add_folder_to_project(self):
        folder = self.current_folder
        if not folder or not self.folder_paths:
            folder = filedialog.askdirectory(title="Add Folder to Project")
            if not folder:
                return
            paths = self._scan_folder_images(folder)
        else:
            paths = list(self.folder_paths)
        existing = {it["path"] for it in self.project_items}
        added = 0
        for p in paths:
            if p not in existing:
                self.project_items.append({"path": p, "selected": True})
                existing.add(p)
                added += 1
        self._rebuild_project_strip()
        self.bottom_tabs.set("Project")
        if self.source_linear is None and self.project_items:
            self.open_photo(self.project_items[0]["path"])
        self.set_status(f"Added {added} photo(s) from folder to Project ({len(self.project_items)} total)")

    def _project_select(self, mode: str):
        if mode == "all":
            for it in self.project_items:
                it["selected"] = True
        elif mode == "none":
            for it in self.project_items:
                it["selected"] = False
        elif mode == "invert":
            for it in self.project_items:
                it["selected"] = not it.get("selected", True)
        elif mode == "remove":
            self.project_items = [it for it in self.project_items if not it.get("selected", True)]
        self._rebuild_project_strip()

    def apply_filter_to_selected(self):
        if not self.project_items or not self.current_filter:
            self.set_status("Select project photos and a filter first")
            return
        self._save_state_now()
        flt_key = self.current_filter.key
        done = 0
        for it in self.project_items:
            if not it.get("selected", True):
                continue
            existing = load_photo_state(it["path"])
            offs = existing[1] if existing else {}
            if save_photo_state(it["path"], flt_key, offs):
                done += 1
        self._refresh_strip_badges()
        self.set_status(f"Applied “{self.current_filter.label}” to {done} photo(s) (slider tweaks preserved)")

    def sync_edits_to_selected(self):
        if not self.project_items:
            self.set_status("Project is empty")
            return
        flt_key = self.current_filter.key if self.current_filter else "00-basics/original"
        offs = self._current_offsets()
        done = 0
        for it in self.project_items:
            if not it.get("selected", True):
                continue
            if save_photo_state(it["path"], flt_key, offs):
                done += 1
        self._dirty = False
        self._refresh_strip_badges()
        self.set_status(f"Synced current filter + {len(offs)} adjustment(s) to {done} photo(s)")

    def reset_selected_edits(self):
        if not self.project_items:
            return
        done = 0
        for it in self.project_items:
            if not it.get("selected", True):
                continue
            if remove_photo_state(it["path"]):
                done += 1
        self._refresh_strip_badges()
        self.set_status(f"Reset edits on {done} selected photo(s)")

    def save_project_dialog(self):
        if not self.project_items:
            self.set_status("Project is empty")
            return
        self._save_state_now()
        path = filedialog.asksaveasfilename(
            defaultextension=".feproj",
            filetypes=[("Fast-Edit Project", "*.feproj")])
        if not path:
            return
        items_out = []
        for it in self.project_items:
            entry: Dict[str, Any] = {"path": it["path"], "selected": 1 if it.get("selected", True) else 0}
            st = load_photo_state(it["path"])
            if st is not None:
                entry["state"] = {"filter": st[0], "offsets": st[1]}
            items_out.append(entry)
        doc = {
            "version": 1,
            "export_dir": "./export",
            "export_prefix": "",
            "format": "png",
            "quality": 95,
            "dither": 1,
            "fullres": 1 if self.fullres_var.get() == "1" else 0,
            "filter": self.current_filter.key if self.current_filter else "00-basics/original",
            "items": items_out,
        }
        Path(path).write_text(json.dumps(doc, indent=2) + "\n")
        self.project_path = path
        self.set_status(f"Saved project {os.path.basename(path)} ({len(items_out)} photos)")

    def load_project_dialog(self):
        self._save_state_now()
        path = filedialog.askopenfilename(
            filetypes=[("Fast-Edit Project", "*.feproj"), ("All files", "*.*")])
        if not path:
            return
        try:
            doc = json.loads(Path(path).read_text())
            self.project_items.clear()
            for it in doc.get("items", []):
                p = it.get("path")
                if not p:
                    continue
                self.project_items.append({"path": p, "selected": bool(it.get("selected", 1))})
                st_obj = it.get("state")
                if isinstance(st_obj, dict) and load_photo_state(p) is None:
                    save_photo_state(p, str(st_obj.get("filter") or ""),
                                     dict(st_obj.get("offsets") or {}))
            self._rebuild_project_strip()
            self.bottom_tabs.set("Project")
            if self.project_items:
                self.open_photo(self.project_items[0]["path"])
            self.set_status(f"Loaded project {os.path.basename(path)} ({len(self.project_items)} photos)")
        except Exception as exc:
            self.set_status(f"Could not load project: {exc}")

    def export_selected_project(self):
        selected = [it["path"] for it in self.project_items if it.get("selected", True)]
        if not selected:
            self.set_status("No photos selected in Project")
            return
        out_dir = filedialog.askdirectory(title="Export Selected Photos To")
        if not out_dir:
            return
        self._save_state_now()
        fullres = self.fullres_var.get() == "1"
        default_flt = self.current_filter or self.all_filters.get("00-basics/original")
        done = 0
        for i, p in enumerate(selected, 1):
            self.set_status(f"Exporting {i}/{len(selected)}: {os.path.basename(p)}…")
            try:
                src = self._decode(p, max_edge=None if fullres else PREVIEW_LONG_EDGE)
                date_str = read_exif_date(p)
                flt = default_flt
                st = load_photo_state(p)
                if st is not None:
                    flt_key, offs = st
                    flt = self.all_filters.get(flt_key, default_flt)
                    settings = apply_offsets_to_settings(flt.settings, offs)
                else:
                    settings = copy.deepcopy(flt.settings)
                out = develop(src, settings)
                u8 = analog_damage_u8(out, flt.effects if flt else {}, date_str, dither=True)
                bgr = cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)
                out_path = os.path.join(out_dir, f"{Path(p).stem}.png")
                cv2.imwrite(out_path, bgr)
                done += 1
            except Exception as exc:
                print(f"export error on {p}: {exc}")
        self.set_status(f"Exported {done}/{len(selected)} photo(s) to {out_dir}")

    def _on_close(self):
        self._save_state_now()
        self.destroy()

    # -- Render & Single Export ------------------------------------------------

    def render_preview(self):
        self._render_job = None
        if self.source_linear is None:
            return
        settings = self._settings_with_slider_offsets()

        source = self.source_linear
        if self._draft_preview:
            source = self._downscale(source, DRAFT_LONG_EDGE)
        out = develop(source, settings)
        u8 = analog_damage_u8(out, self.current_effects(), self.camera_date)
        self._last_u8 = u8
        self.display(u8)

    def current_effects(self) -> Dict[str, Any]:
        return self.current_filter.effects if self.current_filter else {}

    def _ensure_orig_u8(self) -> np.ndarray | None:
        if self.source_linear is None:
            return None
        if self._orig_u8 is None:
            srgb = np.clip(linear_to_srgb(self.source_linear) * 255.0 + 0.5, 0, 255).astype(np.uint8)
            self._orig_u8 = srgb
        return self._orig_u8

    def display(self, rgb_u8: np.ndarray):
        fw = self.image_label.winfo_width() or 900
        fh = self.image_label.winfo_height() or 600
        if fw <= 10:
            fw = 900
        if fh <= 10:
            fh = 600

        frame_style = int(self.ui_prefs.get("frame_style", 0))
        pad = 54 if frame_style == 1 else 28

        pil_after = Image.fromarray(rgb_u8)
        pil_after.thumbnail((max(64, fw - pad), max(64, fh - pad)), Image.Resampling.LANCZOS)
        iw, ih = pil_after.size

        if self.compare_mode in (1, 2):
            orig = self._ensure_orig_u8()
            if orig is not None:
                pil_before = Image.fromarray(orig).resize((iw, ih), Image.Resampling.LANCZOS)
                if self.compare_mode == 2:
                    pil_img = pil_before
                else:
                    split_px = max(4, min(iw - 4, int(iw * self.split_ratio)))
                    pil_img = pil_after.copy()
                    pil_img.paste(pil_before.crop((0, 0, split_px, ih)), (0, 0))
            else:
                pil_img = pil_after
        else:
            pil_img = pil_after

        # Compose onto backdrop canvas so frame styles, grids & histograms sit cleanly
        bg_rgb = _hex_to_rgb(self.active_backdrop())
        acc_rgb = _hex_to_rgb(self.active_accent())
        cw = max(iw + pad, min(fw - 8, iw + pad + 40))
        ch = max(ih + pad, min(fh - 8, ih + pad + 40))
        canvas = Image.new("RGB", (cw, ch), bg_rgb)
        draw = ImageDraw.Draw(canvas, "RGBA")
        ox = (cw - iw) // 2
        oy = (ch - ih) // 2

        if frame_style == 0:  # Soft Drop Shadow
            draw.rectangle([ox + 5, oy + 6, ox + iw + 5, oy + ih + 6], fill=(0, 0, 0, 125))
        elif frame_style == 1:  # Gallery White Matboard
            m = 16
            draw.rectangle([ox - m + 5, oy - m + 6, ox + iw + m + 5, oy + ih + m + 6], fill=(0, 0, 0, 120))
            draw.rectangle([ox - m, oy - m, ox + iw + m, oy + ih + m], fill=(245, 242, 235, 255))
            draw.rectangle([ox - 1, oy - 1, ox + iw, oy + ih], outline=(60, 60, 60, 110), width=1)

        canvas.paste(pil_img, (ox, oy))
        draw = ImageDraw.Draw(canvas, "RGBA")

        if frame_style == 2:  # Thin Accent Hairline
            draw.rectangle([ox - 1, oy - 1, ox + iw, oy + ih], outline=(*acc_rgb, 230), width=2)

        # Split wipe line & labels
        if self.compare_mode == 1:
            sx = ox + max(4, min(iw - 4, int(iw * self.split_ratio)))
            draw.line([(sx, oy), (sx, oy + ih)], fill=(*acc_rgb, 245), width=2)
            cy = oy + ih // 2
            draw.ellipse([sx - 9, cy - 9, sx + 9, cy + 9], fill=(20, 20, 24, 220), outline=(*acc_rgb, 255), width=2)
            draw.rectangle([ox + 8, oy + 8, ox + 64, oy + 26], fill=(15, 16, 20, 195))
            draw.text((ox + 14, oy + 11), "BEFORE", fill=(235, 235, 235, 255))
            draw.rectangle([ox + iw - 58, oy + 8, ox + iw - 8, oy + 26], fill=(15, 16, 20, 195))
            draw.text((ox + iw - 51, oy + 11), "AFTER", fill=(*acc_rgb, 255))

        # Composition Grid
        gm = int(self.ui_prefs.get("grid_mode", 0))
        if gm == 1:
            for i in (1, 2):
                gx = ox + int(iw * i / 3.0)
                gy = oy + int(ih * i / 3.0)
                draw.line([(gx, oy), (gx, oy + ih)], fill=(255, 255, 255, 95), width=1)
                draw.line([(ox, gy), (ox + iw, gy)], fill=(255, 255, 255, 95), width=1)
        elif gm == 2:
            for r in (0.381966, 0.618034):
                gx = ox + int(iw * r)
                gy = oy + int(ih * r)
                draw.line([(gx, oy), (gx, oy + ih)], fill=(255, 255, 255, 95), width=1)
                draw.line([(ox, gy), (ox + iw, gy)], fill=(255, 255, 255, 95), width=1)
        elif gm == 3:
            draw.line([(ox + iw // 2, oy), (ox + iw // 2, oy + ih)], fill=(255, 255, 255, 95), width=1)
            draw.line([(ox, oy + ih // 2), (ox + iw, oy + ih // 2)], fill=(255, 255, 255, 95), width=1)
            draw.line([(ox, oy), (ox + iw, oy + ih)], fill=(255, 255, 255, 75), width=1)
            draw.line([(ox + iw, oy), (ox, oy + ih)], fill=(255, 255, 255, 75), width=1)

        # Live RGB Histogram card
        if self.ui_prefs.get("show_histogram", True):
            hw, hh = 176, 76
            hx, hy = cw - hw - 12, 12
            if hx > 12:
                draw.rounded_rectangle([hx, hy, hx + hw, hy + hh], radius=6,
                                       fill=(16, 18, 22, 205), outline=(255, 255, 255, 45))
                bins = 48
                sample = rgb_u8[::4, ::4]
                px, py, pw, ph = hx + 8, hy + 18, hw - 16, hh - 24
                ch_cols = [(250, 85, 85, 195), (90, 225, 120, 195), (95, 160, 250, 195)]
                hists = [np.histogram(sample[:, :, c], bins=bins, range=(0, 256))[0] for c in range(3)]
                peak = max(1, int(max(h[1:-1].max() for h in hists)))
                for c in range(3):
                    pts = []
                    for b in range(bins):
                        bx = px + int(b / (bins - 1) * pw)
                        v = min(1.0, float(hists[c][b]) / peak)
                        by = py + ph - int(v * ph)
                        pts.append((bx, by))
                    if len(pts) > 1:
                        draw.line(pts, fill=ch_cols[c], width=2)
                draw.text((hx + 8, hy + 4), "HISTOGRAM RGB", fill=(215, 215, 210, 210))

        # Translucent HUD Info Pill
        if self.ui_prefs.get("show_hud", True) and self.file_path:
            fn = os.path.basename(self.file_path)[:28]
            fl = self.current_filter.label if self.current_filter else "Original"
            sh, sw = self.source_linear.shape[:2] if self.source_linear is not None else (ih, iw)
            info_txt = f"{fn}  •  {sw}×{sh}  •  {fl}"
            tw = len(info_txt) * 6 + 20
            py = ch - 32
            draw.rounded_rectangle([12, py, 12 + tw, py + 22], radius=5,
                                   fill=(16, 18, 22, 200), outline=(255, 255, 255, 40))
            draw.text((20, py + 5), info_txt, fill=(235, 230, 215, 240))

        ctk_img = ctk.CTkImage(light_image=canvas, dark_image=canvas, size=canvas.size)
        self.image_label.configure(image=ctk_img, text="")
        self.image_label.image = ctk_img

    def save_image(self):
        if self.source_linear is None:
            self.set_status("Nothing to save")
            return
        self._save_state_now()
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
        u8 = analog_damage_u8(out, self.current_effects(), self.camera_date, dither=True)
        bgr = cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)
        if path.lower().endswith(".png"):
            cv2.imwrite(path, bgr)
        else:
            cv2.imwrite(path, bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 98])
        self.set_status(f"Saved {os.path.basename(path)}")


if __name__ == "__main__":
    app = RawEditorApp()
    app.mainloop()
