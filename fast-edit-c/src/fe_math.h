#ifndef FE_MATH_H
#define FE_MATH_H

#include "fe_defs.h"

static inline float fe_clampf(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

/* ---- colour primitives ------------------------------------------------- */
void fe_luma(const FeImage *img, float *out_hw);
void fe_srgb_to_linear(FeImage *img);
void fe_linear_to_srgb(FeImage *img);
void fe_soft_clip(FeImage *img, float shoulder);
void fe_rgb_to_hsv(const FeImage *img, float *h, float *s, float *v);
void fe_hsv_to_rgb_img(const float *hue, const float *s, const float *v, int w, int h, FeImage *out);
float fe_smoothstep(float e0, float e1, float x);
void fe_apply_lut(FeImage *img, const float *lut, int lutn);

/* ---- resampling (OpenCV-compatible conventions) ----------------------- */
/* INTER_AREA: exact box/area averaging (downscale) */
void fe_resize_area(const FeImage *src, int nw, int nh, FeImage *dst);
/* INTER_LINEAR: bilinear, half-pixel centres */
void fe_resize_linear(const FeImage *src, int nw, int nh, FeImage *dst);
/* INTER_CUBIC: Catmull-Rom (a = -0.5), half-pixel centres */
void fe_resize_cubic(const FeImage *src, int nw, int nh, FeImage *dst);
void fe_downscale_maxedge(FeImage *img, int max_edge);

/* ---- blurs ------------------------------------------------------------- */
/* Separable Gaussian, BORDER_REFLECT_101 like OpenCV. */
void fe_gaussian_blur(const FeImage *src, float sigma, FeImage *dst);
/* The Python app's cheap_blur(): shrink by sigma/6, blur, grow back. */
void fe_cheap_blur(const FeImage *src, float sigma, FeImage *dst);

/* ---- deterministic RNG (numpy-free, reproducible) ---------------------- */
typedef struct { unsigned long long s[4]; } FeRng;
void fe_rng_seed(FeRng *r, unsigned long long seed);
unsigned int fe_rng_u32(FeRng *r);
double fe_rng_uniform(FeRng *r);        /* [0,1) */
double fe_rng_normal(FeRng *r);         /* standard normal */

/* ---- PCHIP (scipy parity) --------------------------------------------- */
void fe_pchip_fit(const double *xs, const double *ys, int n, double *d);
void fe_pchip_eval(const double *xs, const double *ys, const double *d, int n,
                   const double *q, int nq, double *out);
void fe_tone_curve_lut(double contrast, double blacks, double whites, double fade,
                       float *lut, int n);

#endif
