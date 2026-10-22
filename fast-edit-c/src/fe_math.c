#include "fe_math.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Rec. 709 luma weights — the same ones the Python app uses. */
static const float LUMA_W[3] = {0.2126f, 0.7152f, 0.0722f};

/* ======================================================================
   COLOUR PRIMITIVES
   ====================================================================== */

void fe_luma(const FeImage *img, float *out_hw) {
    int n = img->w * img->h;
    const float *p = img->data;
    for (int i = 0; i < n; i++) {
        out_hw[i] = p[0] * LUMA_W[0] + p[1] * LUMA_W[1] + p[2] * LUMA_W[2];
        p += 3;
    }
}

void fe_srgb_to_linear(FeImage *img) {
    int n = img->w * img->h * 3;
    float *p = img->data;
    for (int i = 0; i < n; i++) {
        float x = p[i];
        if (x <= 0.04045f) p[i] = x / 12.92f;
        else p[i] = powf((x + 0.055f) / 1.055f, 2.4f);
    }
}

void fe_linear_to_srgb(FeImage *img) {
    int n = img->w * img->h * 3;
    float *p = img->data;
    for (int i = 0; i < n; i++) {
        float x = p[i] < 0.0f ? 0.0f : p[i];
        if (x <= 0.0031308f) p[i] = x * 12.92f;
        else p[i] = 1.055f * powf(x, 1.0f / 2.4f) - 0.055f;
    }
}

void fe_soft_clip(FeImage *img, float shoulder) {
    int n = img->w * img->h * 3;
    float *p = img->data;
    for (int i = 0; i < n; i++) {
        float x = p[i] < 0.0f ? 0.0f : p[i];
        if (x <= shoulder) { p[i] = x; continue; }
        float denom = 1.0f - shoulder;
        p[i] = shoulder + denom * tanhf((x - shoulder) / denom);
    }
}

void fe_rgb_to_hsv(const FeImage *img, float *h, float *s, float *v) {
    int n = img->w * img->h;
    const float *p = img->data;
    for (int i = 0; i < n; i++) {
        float r = p[0], g = p[1], b = p[2];
        float maxc = r > g ? r : g;
        if (b > maxc) maxc = b;
        float minc = r < g ? r : g;
        if (b < minc) minc = b;
        float chroma = maxc - minc;

        s[i] = (maxc > 1e-8f) ? chroma / fmaxf(maxc, 1e-8f) : 0.0f;
        v[i] = maxc;

        float hue = 0.0f;
        if (chroma > 1e-8f) {
            if (maxc == r) {
                hue = fmodf((g - b) / chroma, 6.0f);
            } else if (maxc == g) {
                hue = (b - r) / chroma + 2.0f;
            } else {
                hue = (r - g) / chroma + 4.0f;
            }
        }
        hue = fmodf(hue * 60.0f, 360.0f);
        if (hue < 0.0f) hue += 360.0f;
        h[i] = hue;
        p += 3;
    }
}

void fe_hsv_to_rgb_img(const float *hue, const float *s, const float *v, int w, int h,
                       FeImage *out) {
    int n = w * h;
    float *p = out->data;
    for (int i = 0; i < n; i++) {
        float H = fmodf(hue[i], 360.0f) / 60.0f;
        if (H < 0.0f) H += 6.0f;
        int sector = (int)floorf(H) % 6;
        float f = H - floorf(H);
        float V = v[i], S = s[i];
        float p_ = V * (1.0f - S);
        float q = V * (1.0f - S * f);
        float t = V * (1.0f - S * (1.0f - f));
        float r, g, b;
        switch (sector) {
            case 0: r = V;   g = t;   b = p_;  break;
            case 1: r = q;   g = V;   b = p_;  break;
            case 2: r = p_;  g = V;   b = t;   break;
            case 3: r = p_;  g = q;   b = V;   break;
            case 4: r = t;   g = p_;  b = V;   break;
            default:r = V;   g = p_;  b = q;   break;
        }
        p[0] = r; p[1] = g; p[2] = b;
        p += 3;
    }
}

float fe_smoothstep(float e0, float e1, float x) {
    float t = (x - e0) / (e1 - e0 + 1e-9f);
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;
    return t * t * (3.0f - 2.0f * t);
}

void fe_apply_lut(FeImage *img, const float *lut, int lutn) {
    int n = img->w * img->h * 3;
    float *p = img->data;
    for (int i = 0; i < n; i++) {
        float x = fe_clampf(p[i], 0.0f, 1.0f);
        float idxf = x * (float)(lutn - 1);
        int idx = (int)idxf;
        if (idx >= lutn - 1) { p[i] = lut[lutn - 1]; continue; }
        float t = idxf - (float)idx;
        p[i] = lut[idx] * (1.0f - t) + lut[idx + 1] * t;
    }
}

/* ======================================================================
   RESAMPLING
   OpenCV's conventions, so the C render matches the Python render:
     * half-pixel centres for INTER_LINEAR / INTER_CUBIC
     * exact area averaging for INTER_AREA
   ====================================================================== */

void fe_resize_area(const FeImage *src, int nw, int nh, FeImage *dst) {
    float sw = (float)src->w / (float)nw;
    float sh = (float)src->h / (float)nh;
    for (int y = 0; y < nh; y++) {
        float y0 = (float)y * sh, y1 = y0 + sh;
        int iy0 = (int)floorf(y0), iy1 = (int)ceilf(y1) - 1;
        if (iy1 < iy0) iy1 = iy0;
        if (iy0 < 0) iy0 = 0;
        if (iy1 > src->h - 1) iy1 = src->h - 1;
        for (int x = 0; x < nw; x++) {
            float x0 = (float)x * sw, x1 = x0 + sw;
            int ix0 = (int)floorf(x0), ix1 = (int)ceilf(x1) - 1;
            if (ix1 < ix0) ix1 = ix0;
            if (ix0 < 0) ix0 = 0;
            if (ix1 > src->w - 1) ix1 = src->w - 1;
            float acc[3] = {0.0f, 0.0f, 0.0f};
            float wsum = 0.0f;
            for (int yy = iy0; yy <= iy1; yy++) {
                float wy = fminf((float)yy + 1.0f, y1) - fmaxf((float)yy, y0);
                if (wy <= 0.0f) continue;
                for (int xx = ix0; xx <= ix1; xx++) {
                    float wx = fminf((float)xx + 1.0f, x1) - fmaxf((float)xx, x0);
                    if (wx <= 0.0f) continue;
                    float w = wx * wy;
                    const float *p = src->data + ((size_t)yy * src->w + xx) * 3;
                    acc[0] += p[0] * w; acc[1] += p[1] * w; acc[2] += p[2] * w;
                    wsum += w;
                }
            }
            float *o = dst->data + ((size_t)y * nw + x) * 3;
            if (wsum > 0.0f) {
                o[0] = acc[0] / wsum; o[1] = acc[1] / wsum; o[2] = acc[2] / wsum;
            } else {
                const float *p = src->data + ((size_t)(y < src->h ? y : src->h - 1) * src->w +
                                             (x < src->w ? x : src->w - 1)) * 3;
                o[0] = p[0]; o[1] = p[1]; o[2] = p[2];
            }
        }
    }
}

/* Reflect-101 index mapping (OpenCV BORDER_REFLECT_101). */
static int reflect101(int i, int n) {
    if (n == 1) return 0;
    int period = 2 * (n - 1);
    i %= period;
    if (i < 0) i += period;
    if (i >= n) i = period - i;
    return i;
}

void fe_resize_linear(const FeImage *src, int nw, int nh, FeImage *dst) {
    float sx = (float)src->w / (float)nw;
    float sy = (float)src->h / (float)nh;
    for (int y = 0; y < nh; y++) {
        float fy = ((float)y + 0.5f) * sy - 0.5f;
        int y0 = (int)floorf(fy);
        float wy = fy - (float)y0;
        int y1 = y0 + 1;
        /* cv2.resize uses BORDER_REPLICATE by default */
        int ry0 = y0 < 0 ? 0 : (y0 > src->h - 1 ? src->h - 1 : y0);
        int ry1 = y1 < 0 ? 0 : (y1 > src->h - 1 ? src->h - 1 : y1);
        for (int x = 0; x < nw; x++) {
            float fx = ((float)x + 0.5f) * sx - 0.5f;
            int x0 = (int)floorf(fx);
            float wx = fx - (float)x0;
            int x1 = x0 + 1;
            int rx0 = x0 < 0 ? 0 : (x0 > src->w - 1 ? src->w - 1 : x0);
            int rx1 = x1 < 0 ? 0 : (x1 > src->w - 1 ? src->w - 1 : x1);
            const float *p00 = src->data + ((size_t)ry0 * src->w + rx0) * 3;
            const float *p01 = src->data + ((size_t)ry0 * src->w + rx1) * 3;
            const float *p10 = src->data + ((size_t)ry1 * src->w + rx0) * 3;
            const float *p11 = src->data + ((size_t)ry1 * src->w + rx1) * 3;
            float *o = dst->data + ((size_t)y * nw + x) * 3;
            for (int c = 0; c < 3; c++) {
                float top = p00[c] * (1.0f - wx) + p01[c] * wx;
                float bot = p10[c] * (1.0f - wx) + p11[c] * wx;
                o[c] = top * (1.0f - wy) + bot * wy;
            }
        }
    }
}

/* Catmull-Rom cubic (a = -0.5), the kernel OpenCV's INTER_CUBIC uses. */
static float cubic_weight(float t) {
    const float a = -0.5f;
    t = fabsf(t);
    if (t < 1.0f) return ((a + 2.0f) * t - (a + 3.0f)) * t * t + 1.0f;
    if (t < 2.0f) return ((a * t - 5.0f * a) * t + 8.0f * a) * t - 4.0f * a;
    return 0.0f;
}

void fe_resize_cubic(const FeImage *src, int nw, int nh, FeImage *dst) {
    float sx = (float)src->w / (float)nw;
    float sy = (float)src->h / (float)nh;
    for (int y = 0; y < nh; y++) {
        float fy = ((float)y + 0.5f) * sy - 0.5f;
        int y0 = (int)floorf(fy);
        float wy = fy - (float)y0;
        for (int x = 0; x < nw; x++) {
            float fx = ((float)x + 0.5f) * sx - 0.5f;
            int x0 = (int)floorf(fx);
            float wx = fx - (float)x0;
            float acc[3] = {0.0f, 0.0f, 0.0f};
            for (int dy = -1; dy <= 2; dy++) {
                int sy_i = y0 + dy;
                if (sy_i < 0) sy_i = 0;
                if (sy_i > src->h - 1) sy_i = src->h - 1;
                float wyc = cubic_weight(wy - (float)dy);
                for (int dx = -1; dx <= 2; dx++) {
                    int sx_i = x0 + dx;
                    if (sx_i < 0) sx_i = 0;
                    if (sx_i > src->w - 1) sx_i = src->w - 1;
                    float wxc = cubic_weight(wx - (float)dx);
                    float w = wyc * wxc;
                    const float *p = src->data + ((size_t)sy_i * src->w + sx_i) * 3;
                    acc[0] += p[0] * w; acc[1] += p[1] * w; acc[2] += p[2] * w;
                }
            }
            /* OpenCV does NOT renormalise the cubic weights — the kernel
               already sums to one, and renormalising would change the
               border behaviour compared to cv2.resize. */
            float *o = dst->data + ((size_t)y * nw + x) * 3;
            o[0] = acc[0]; o[1] = acc[1]; o[2] = acc[2];
        }
    }
}

void fe_downscale_maxedge(FeImage *img, int max_edge) {
    int w = img->w, h = img->h;
    if (w <= 0 || h <= 0) return;
    int me = w > h ? w : h;
    if (me <= max_edge) return;
    float scale = (float)max_edge / (float)me;
    int nw = (int)((float)w * scale);
    int nh = (int)((float)h * scale);
    if (nw < 1) nw = 1;
    if (nh < 1) nh = 1;
    FeImage out = fe_img_new(nw, nh);
    fe_resize_area(img, nw, nh, &out);
    free(img->data);
    img->data = out.data;
    img->w = nw; img->h = nh;
}

/* ======================================================================
   BLURS
   ====================================================================== */

void fe_gaussian_blur(const FeImage *src, float sigma, FeImage *dst) {
    if (sigma <= 0.0f) { memcpy(dst->data, src->data, (size_t)src->w * src->h * 3 * sizeof(float)); return; }
    int radius = (int)ceilf(sigma * 3.0f);
    if (radius < 1) radius = 1;
    int ksize = radius * 2 + 1;
    float *kernel = (float *)malloc((size_t)ksize * sizeof(float));
    double sum = 0.0;
    for (int i = 0; i < ksize; i++) {
        double x = (double)(i - radius);
        double v = exp(-(x * x) / (2.0 * (double)sigma * (double)sigma));
        kernel[i] = (float)v;
        sum += v;
    }
    for (int i = 0; i < ksize; i++) kernel[i] = (float)(kernel[i] / sum);

    int w = src->w, h = src->h;
    FeImage tmp = fe_img_new(w, h);
    /* horizontal */
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float acc[3] = {0, 0, 0};
            for (int k = -radius; k <= radius; k++) {
                int xx = reflect101(x + k, w);
                const float *p = src->data + ((size_t)y * w + xx) * 3;
                float kw = kernel[k + radius];
                acc[0] += p[0] * kw; acc[1] += p[1] * kw; acc[2] += p[2] * kw;
            }
            float *o = tmp.data + ((size_t)y * w + x) * 3;
            o[0] = acc[0]; o[1] = acc[1]; o[2] = acc[2];
        }
    }
    /* vertical */
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float acc[3] = {0, 0, 0};
            for (int k = -radius; k <= radius; k++) {
                int yy = reflect101(y + k, h);
                const float *p = tmp.data + ((size_t)yy * w + x) * 3;
                float kw = kernel[k + radius];
                acc[0] += p[0] * kw; acc[1] += p[1] * kw; acc[2] += p[2] * kw;
            }
            float *o = dst->data + ((size_t)y * w + x) * 3;
            o[0] = acc[0]; o[1] = acc[1]; o[2] = acc[2];
        }
    }
    free(kernel);
    fe_img_free(&tmp);
}

void fe_cheap_blur(const FeImage *src, float sigma, FeImage *dst) {
    int w = src->w, h = src->h;
    float step = sigma / 6.0f;
    if (step < 1.0f) step = 1.0f;
    int sw = (int)((float)w / step);
    int sh = (int)((float)h / step);
    if (sw < 2) sw = 2;
    if (sh < 2) sh = 2;
    FeImage small = fe_img_new(sw, sh);
    fe_resize_area(src, sw, sh, &small);
    float sigma_small = sigma / step;
    if (sigma_small < 0.5f) sigma_small = 0.5f;
    FeImage blurred = fe_img_new(sw, sh);
    fe_gaussian_blur(&small, sigma_small, &blurred);
    fe_resize_linear(&blurred, w, h, dst);
    fe_img_free(&small);
    fe_img_free(&blurred);
}

/* ======================================================================
   RNG — xoshiro256** seeded through splitmix64.
   Reproducible per run, statistically identical to numpy's PCG64 for the
   purposes of grain (the *realisation* differs, the statistics do not).
   ====================================================================== */

static unsigned long long splitmix64(unsigned long long *state) {
    unsigned long long z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

void fe_rng_seed(FeRng *r, unsigned long long seed) {
    unsigned long long state = seed;
    for (int i = 0; i < 4; i++) r->s[i] = splitmix64(&state);
}

static inline unsigned long long rotl(const unsigned long long x, int k) {
    return (x << k) | (x >> (64 - k));
}

unsigned int fe_rng_u32(FeRng *r) {
    unsigned long long *s = r->s;
    unsigned long long result = rotl(s[1] * 5ULL, 7) * 9ULL;
    unsigned long long t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t;
    s[3] = rotl(s[3], 45);
    return (unsigned int)(result >> 32);
}

double fe_rng_uniform(FeRng *r) {
    return (double)(fe_rng_u32(r) >> 8) * (1.0 / 16777216.0); /* [0,1) with 24 bits */
}

double fe_rng_normal(FeRng *r) {
    /* Box-Muller: two uniforms in, one normal out (the second is discarded). */
    double u1 = fe_rng_uniform(r);
    double u2 = fe_rng_uniform(r);
    if (u1 < 1e-12) u1 = 1e-12;
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}

/* ======================================================================
   PCHIP — reproduces scipy.interpolate.PchipInterpolator exactly
   (verified against scipy to 9e-16 in the probe script).
   ====================================================================== */

static double pchip_edge_case(double h0, double h1, double m0, double m1) {
    double d = ((2.0 * h0 + h1) * m0 - h0 * m1) / (h0 + h1);
    if ((d > 0.0 && m0 < 0.0) || (d < 0.0 && m0 > 0.0)) {
        d = 0.0;
    } else if (((m0 > 0.0 && m1 < 0.0) || (m0 < 0.0 && m1 > 0.0)) && fabs(d) > fabs(3.0 * m0)) {
        d = 3.0 * m0;
    }
    return d;
}

void fe_pchip_fit(const double *xs, const double *ys, int n, double *d) {
    if (n <= 0) return;
    if (n == 1) { d[0] = 0.0; return; }
    if (n == 2) {
        if (ys[1] == ys[0]) { d[0] = d[1] = 0.0; return; }
        double s = (ys[1] - ys[0]) / (xs[1] - xs[0]);
        d[0] = d[1] = s;
        return;
    }
    double *h = (double *)malloc((size_t)(n - 1) * sizeof(double));
    double *s = (double *)malloc((size_t)(n - 1) * sizeof(double));
    for (int i = 0; i < n - 1; i++) {
        h[i] = xs[i + 1] - xs[i];
        s[i] = (ys[i + 1] - ys[i]) / h[i];
    }
    for (int i = 0; i < n; i++) d[i] = 0.0;
    for (int i = 1; i < n - 1; i++) {
        if (s[i - 1] * s[i] > 0.0) {
            double w1 = 2.0 * h[i] + h[i - 1];
            double w2 = h[i] + 2.0 * h[i - 1];
            d[i] = (w1 + w2) / (w1 / s[i - 1] + w2 / s[i]);
        } else {
            d[i] = 0.0;
        }
    }
    d[0] = pchip_edge_case(h[0], h[1], s[0], s[1]);
    /* the last point mirrors it: h0 = last interval, h1 = the one before,
       m0 = last secant, m1 = the one before that */
    d[n - 1] = pchip_edge_case(h[n - 2], h[n - 3], s[n - 2], s[n - 3]);
    free(h);
    free(s);
}

void fe_pchip_eval(const double *xs, const double *ys, const double *d, int n,
                   const double *q, int nq, double *out) {
    for (int qi = 0; qi < nq; qi++) {
        double qq = q[qi];
        if (qq <= xs[0]) { out[qi] = ys[0]; continue; }
        if (qq >= xs[n - 1]) { out[qi] = ys[n - 1]; continue; }
        int idx = 0;
        for (int i = 0; i < n - 1; i++) {
            if (qq >= xs[i] && qq <= xs[i + 1]) { idx = i; break; }
        }
        double h = xs[idx + 1] - xs[idx];
        double t = (qq - xs[idx]) / h;
        double t2 = t * t, t3 = t2 * t;
        out[qi] = (2.0 * t3 - 3.0 * t2 + 1.0) * ys[idx] +
                  (t3 - 2.0 * t2 + t) * h * d[idx] +
                  (-2.0 * t3 + 3.0 * t2) * ys[idx + 1] +
                  (t3 - t2) * h * d[idx + 1];
    }
}

void fe_tone_curve_lut(double contrast, double blacks, double whites, double fade,
                       float *lut, int n) {
    if (n <= 0) return;
    static const double XS[7] = {0.0, 0.08, 0.25, 0.5, 0.75, 0.92, 1.0};
    double ys[7];
    const double pivot = 0.45;
    for (int i = 0; i < 7; i++) {
        double y = pivot + (XS[i] - pivot) * (1.0 + contrast);
        y += blacks * pow(1.0 - XS[i], 1.6);
        y -= whites * pow(XS[i], 1.6);
        y += fade * pow(1.0 - XS[i], 3.0);
        if (y < 0.0) y = 0.0;
        if (y > 1.0) y = 1.0;
        ys[i] = y;
    }
    double d[7];
    fe_pchip_fit(XS, ys, 7, d);
    double *q = (double *)malloc((size_t)n * sizeof(double));
    double *vals = (double *)malloc((size_t)n * sizeof(double));
    for (int i = 0; i < n; i++) q[i] = (double)i / (double)(n - 1);
    fe_pchip_eval(XS, ys, d, 7, q, n, vals);
    for (int i = 0; i < n; i++) {
        double v = vals[i];
        if (v < 0.0) v = 0.0;
        if (v > 1.0) v = 1.0;
        lut[i] = (float)v;
    }
    free(q);
    free(vals);
}
