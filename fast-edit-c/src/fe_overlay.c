#include "fe_overlay.h"
#include "fe_math.h"

#include <cairo.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Mirror the border (edge pixel duplicated) the way cv2's BORDER_REFLECT does. */
static int reflect_sym(int i, int n) {
    if (n <= 1) return 0;
    if (i < 0) return -i - 1;
    if (i >= n) return 2 * n - 1 - i;
    return i;
}

void fe_chromatic_aberration(FeImage *img, double shift) {
    int s = (int)lround(shift);
    if (s <= 0) return;
    int w = img->w, h = img->h;
    /* Read from a COPY: the red channel takes the pixel s columns to the
       RIGHT and the blue channel the one s columns to the LEFT, so writing
       into the same buffer would smear the shift across the whole frame.
       (The Python app does the same thing with img.copy().) */
    size_t floats = (size_t)w * (size_t)h * 3;
    float *src = (float *)malloc(floats * sizeof(float));
    if (!src) return;
    memcpy(src, img->data, floats * sizeof(float));
    float *p = img->data;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int idx = (y * w + x) * 3;
            /* red slides left  (takes the pixel s columns to the right) */
            p[idx + 0] = src[((size_t)y * w + reflect_sym(x + s, w)) * 3 + 0];
            /* blue slides right (takes the pixel s columns to the left) */
            p[idx + 2] = src[((size_t)y * w + reflect_sym(x - s, w)) * 3 + 2];
        }
    }
    free(src);
}

void fe_scanlines(FeImage *img, double intensity) {
    int w = img->w, h = img->h;
    float *p = img->data;
    for (int y = 0; y < h; y += 3) {
        float m = 1.0f - (float)intensity;
        for (int x = 0; x < w; x++) {
            int idx = (y * w + x) * 3;
            p[idx] *= m; p[idx + 1] *= m; p[idx + 2] *= m;
        }
    }
}

/* ------------------------------------------------------------------ */
/* The ember stamp                                                      */
/* ------------------------------------------------------------------ */

typedef struct {
    const char *text;
    int x, y;
    double r, g, b;
    int thickness;
} StampPass;

void fe_vcr_overlay(unsigned char *rgb, int w, int h, const char *date_str) {
    if (w <= 0 || h <= 0) return;

    /* The stamp scales with the frame, exactly like the Python app. */
    double scale = (double)h / 900.0;
    if (scale < 0.5) scale = 0.5;
    int thickness = (int)(scale * 1.5);
    if (thickness < 1) thickness = 1;

    /* Draw on a transparent scratch surface, then composite it over the
       frame — that keeps the anti-aliased edges soft. */
    cairo_surface_t *surface = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, w, h);
    cairo_t *cr = cairo_create(surface);

    cairo_select_font_face(cr, "sans-serif", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_NORMAL);
    cairo_set_font_size(cr, 22.0 * scale);
    cairo_set_line_join(cr, CAIRO_LINE_JOIN_ROUND);
    cairo_set_line_cap(cr, CAIRO_LINE_CAP_ROUND);

    StampPass passes[3] = {
        /* halo — burnt ember bleeding down-right */
        {"", 0, 0, 74.0/255.0, 16.0/255.0, 4.0/255.0, thickness * 4},
        /* body — smouldering orange */
        {"", 0, 0, 196.0/255.0, 84.0/255.0, 26.0/255.0, thickness * 3},
        /* core — hot amber filament */
        {"", 0, 0, 255.0/255.0, 176.0/255.0, 96.0/255.0, thickness},
    };
    const char *texts[2] = {"PLAY  > ", date_str};
    int xs[2] = {(int)(w * 0.05), (int)(w * 0.62)};
    int ys[2] = {(int)(h * 0.10), (int)(h * 0.90)};
    const int offs[3][2] = {{3, 3}, {1, 1}, {0, 0}};

    for (int t = 0; t < 2; t++) {
        for (int pass = 0; pass < 3; pass++) {
            cairo_new_path(cr);
            cairo_move_to(cr, xs[t] + offs[pass][0], ys[t] + offs[pass][1]);
            cairo_text_path(cr, texts[t]);
            cairo_set_source_rgb(cr, passes[pass].r, passes[pass].g, passes[pass].b);
            cairo_fill_preserve(cr);
            cairo_set_line_width(cr, (double)passes[pass].thickness);
            cairo_stroke(cr);
        }
    }

    cairo_destroy(cr);
    cairo_surface_flush(surface);

    unsigned char *src = cairo_image_surface_get_data(surface);
    int stride = cairo_image_surface_get_stride(surface);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            unsigned char *sp = src + (size_t)y * stride + (size_t)x * 4;
            unsigned alpha = sp[3];
            if (alpha == 0) continue;
            int idx = (y * w + x) * 3;
            /* cairo stores premultiplied BGRA on little-endian machines */
            float sa = (float)alpha / 255.0f;
            float inv = 1.0f - sa;
            rgb[idx + 0] = (unsigned char)fminf(255.0f, sp[2] + rgb[idx + 0] * inv);
            rgb[idx + 1] = (unsigned char)fminf(255.0f, sp[1] + rgb[idx + 1] * inv);
            rgb[idx + 2] = (unsigned char)fminf(255.0f, sp[0] + rgb[idx + 2] * inv);
        }
    }
    cairo_surface_destroy(surface);
}
