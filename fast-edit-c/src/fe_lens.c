#include "fe_lens.h"
#include "fe_math.h"

#include <math.h>
#include <string.h>
#include <stdlib.h>

static inline float bilerp(const float *data, int w, int h, float x, float y, int c) {
    if (w <= 1 || h <= 1) return 0.0f;
    /* Clamp to valid range to avoid artifacts */
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x > w - 1.001f) x = w - 1.001f;
    if (y > h - 1.001f) y = h - 1.001f;
    int x0 = (int)floorf(x), y0 = (int)floorf(y);
    int x1 = x0 + 1, y1 = y0 + 1;
    if (x1 >= w) x1 = w - 1;
    if (y1 >= h) y1 = h - 1;
    float fx = x - x0, fy = y - y0;
    float v00 = data[(y0 * w + x0) * 3 + c];
    float v10 = data[(y0 * w + x1) * 3 + c];
    float v01 = data[(y1 * w + x0) * 3 + c];
    float v11 = data[(y1 * w + x1) * 3 + c];
    float v0 = v00 * (1 - fx) + v10 * fx;
    float v1 = v01 * (1 - fx) + v11 * fx;
    return v0 * (1 - fy) + v1 * fy;
}

/* Brown-Conrady distortion model - industry standard */
void fe_lens_distortion_correct(FeImage *img, double k1, double k2, double k3) {
    if (fabs(k1) < 1e-9 && fabs(k2) < 1e-9 && fabs(k3) < 1e-9) return;
    
    int w = img->w, h = img->h;
    float *src = (float *)malloc((size_t)w * h * 3 * sizeof(float));
    memcpy(src, img->data, (size_t)w * h * 3 * sizeof(float));
    
    float cx = w * 0.5f, cy = h * 0.5f;
    float maxr = sqrtf(cx * cx + cy * cy);
    
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float dx = (x - cx) / maxr;
            float dy = (y - cy) / maxr;
            float r2 = dx * dx + dy * dy;
            float r4 = r2 * r2;
            float r6 = r4 * r2;
            /* Undistortion: apply inverse of distortion */
            float factor = 1.0f + (float)k1 * r2 + (float)k2 * r4 + (float)k3 * r6;
            /* For correction, we need to map from undistorted to distorted */
            float fx = dx / factor * maxr + cx;
            float fy = dy / factor * maxr + cy;
            
            int idx = (y * w + x) * 3;
            if (fx >= 0 && fx < w - 1 && fy >= 0 && fy < h - 1) {
                img->data[idx + 0] = bilerp(src, w, h, fx, fy, 0);
                img->data[idx + 1] = bilerp(src, w, h, fx, fy, 1);
                img->data[idx + 2] = bilerp(src, w, h, fx, fy, 2);
            } else {
                img->data[idx + 0] = img->data[idx + 1] = img->data[idx + 2] = 0;
            }
        }
    }
    free(src);
}

/* Transverse Chromatic Aberration correction - correct color fringing */
void fe_lens_tca_correct(FeImage *img, double r_shift, double b_shift) {
    if (fabs(r_shift) < 1e-9 && fabs(b_shift) < 1e-9) return;
    
    int w = img->w, h = img->h;
    float *src = (float *)malloc((size_t)w * h * 3 * sizeof(float));
    memcpy(src, img->data, (size_t)w * h * 3 * sizeof(float));
    
    float cx = w * 0.5f, cy = h * 0.5f;
    float maxr = sqrtf(cx * cx + cy * cy);
    
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float dx = (x - cx) / maxr;
            float dy = (y - cy) / maxr;
            float r = sqrtf(dx * dx + dy * dy);
            float scale_r = 1.0f + (float)r_shift * r * r;
            float scale_b = 1.0f + (float)b_shift * r * r;
            
            float fx_r = dx * scale_r * maxr + cx;
            float fy_r = dy * scale_r * maxr + cy;
            float fx_b = dx * scale_b * maxr + cx;
            float fy_b = dy * scale_b * maxr + cy;
            
            int idx = (y * w + x) * 3;
            /* Use clamping/sampling from source with bounds checking */
            img->data[idx + 0] = bilerp(src, w, h, fx_r, fy_r, 0);
            img->data[idx + 1] = bilerp(src, w, h, x, y, 1); /* green channel unchanged */
            img->data[idx + 2] = bilerp(src, w, h, fx_b, fy_b, 2);
        }
    }
    free(src);
}

/* Vignetting correction */
void fe_lens_vignette_correct(FeImage *img, double amount) {
    if (fabs(amount) < 1e-9) return;
    
    int w = img->w, h = img->h;
    float cx = w * 0.5f, cy = h * 0.5f;
    float maxr = sqrtf(cx * cx + cy * cy);
    
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float dx = (x - cx) / maxr;
            float dy = (y - cy) / maxr;
            float r2 = dx * dx + dy * dy;
            float r4 = r2 * r2;
            /* Correction factor - more correction at edges */
            float corr = 1.0f + (float)amount * (r2 + r4 * 0.5f);
            if (corr < 0.2f) corr = 0.2f;
            if (corr > 4.0f) corr = 4.0f;
            int idx = (y * w + x) * 3;
            img->data[idx + 0] *= corr;
            img->data[idx + 1] *= corr;
            img->data[idx + 2] *= corr;
        }
    }
}

/* Simple perspective/keystone correction */
void fe_lens_perspective_correct(FeImage *img, double keystone_h, double keystone_v) {
    if (fabs(keystone_h) < 1e-9 && fabs(keystone_v) < 1e-9) return;
    
    int w = img->w, h = img->h;
    float *src = (float *)malloc((size_t)w * h * 3 * sizeof(float));
    memcpy(src, img->data, (size_t)w * h * 3 * sizeof(float));
    
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            /* Normalize coordinates */
            float nx = (x / (float)w - 0.5f) * 2.0f; /* -1 to 1 */
            float ny = (y / (float)h - 0.5f) * 2.0f;
            /* Apply keystone */
            float scale_h = 1.0f + (float)keystone_h * ny * 0.5f;
            float scale_v = 1.0f + (float)keystone_v * nx * 0.5f;
            float sx = nx / scale_h * 0.5f + 0.5f;
            float sy = ny / scale_v * 0.5f + 0.5f;
            float fx = sx * w;
            float fy = sy * h;
            
            int idx = (y * w + x) * 3;
            if (fx >= 0 && fx < w - 1 && fy >= 0 && fy < h - 1) {
                img->data[idx + 0] = bilerp(src, w, h, fx, fy, 0);
                img->data[idx + 1] = bilerp(src, w, h, fx, fy, 1);
                img->data[idx + 2] = bilerp(src, w, h, fx, fy, 2);
            } else {
                img->data[idx + 0] = img->data[idx + 1] = img->data[idx + 2] = 0;
            }
        }
    }
    free(src);
}

void fe_lens_auto_correct(FeImage *img, const FeSettings *s) {
    if (!s || !img) return;
    /* Apply corrections in order: perspective, distortion, TCA, vignetting */
    if (fabs(s->lens_keystone_h) > 1e-9 || fabs(s->lens_keystone_v) > 1e-9)
        fe_lens_perspective_correct(img, s->lens_keystone_h, s->lens_keystone_v);
    /* Distortion: prefer explicit k1/k2/k3, else derive from slider.
     * Positive slider value = correct barrel distortion (wide angle) -> k1 < 0. */
    {
        double k1 = s->lens_k1, k2 = s->lens_k2, k3 = s->lens_k3;
        if (fabs(k1) < 1e-12 && fabs(k2) < 1e-12 && fabs(k3) < 1e-12 && fabs(s->lens_distortion) > 1e-12) {
            k1 = -s->lens_distortion * 0.2;
            k2 = s->lens_k2;
            k3 = s->lens_k3;
        }
        if (fabs(k1) > 1e-12 || fabs(k2) > 1e-12 || fabs(k3) > 1e-12)
            fe_lens_distortion_correct(img, k1, k2, k3);
    }
    if (fabs(s->lens_tca_r) > 1e-9 || fabs(s->lens_tca_b) > 1e-9)
        fe_lens_tca_correct(img, s->lens_tca_r, s->lens_tca_b);
    else if (fabs(s->lens_ca_r) > 1e-9 || fabs(s->lens_ca_b) > 1e-9)
        fe_lens_tca_correct(img, s->lens_ca_r, s->lens_ca_b);
    if (fabs(s->lens_vignette_corr) > 1e-9)
        fe_lens_vignette_correct(img, s->lens_vignette_corr);
}
