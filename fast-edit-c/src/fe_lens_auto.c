/* The Makefile already passes -D_GNU_SOURCE; only define it when building
   without it so the two never clash. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "fe_lens.h"
#include "fe_math.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

typedef struct {
    double k1;           /* estimated Brown-Conrady k1 (negative = barrel) */
    double tca_r;        /* radial scale for red */
    double tca_b;        /* radial scale for blue */
    double vignette;     /* correction amount */
    double confidence;
    int has_distortion;
    int has_tca;
    int has_vignette;
} FeLensAutoResult;

static inline float fe_luma_px(const float *p) {
    return 0.2126f * p[0] + 0.7152f * p[1] + 0.0722f * p[2];
}

/* Estimate vignetting from radial luminance profile */
static void estimate_vignetting(const FeImage *im, FeLensAutoResult *res) {
    int w = im->w, h = im->h;
    float cx = w * 0.5f, cy = h * 0.5f;
    float maxr = sqrtf(cx*cx + cy*cy);
    if (maxr < 1.0f) return;
    
    const int NB = 24;
    double sum_lum[24] = {0}, cnt[24] = {0};
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float dx = (x - cx) / maxr;
            float dy = (y - cy) / maxr;
            float r = sqrtf(dx*dx + dy*dy);
            if (r > 1.0f) continue;
            int b = (int)(r * NB);
            if (b < 0) b = 0;
            if (b >= NB) b = NB - 1;
            const float *p = im->data + ((size_t)y * w + x) * 3;
            float lum = fe_luma_px(p);
            /* skip very dark pixels that might be shadows */
            if (lum < 0.01f) continue;
            sum_lum[b] += lum;
            cnt[b]++;
        }
    }
    
    /* Get average per bin */
    double mean_lum[24];
    for (int i = 0; i < NB; i++) {
        if (cnt[i] > 0) mean_lum[i] = sum_lum[i] / cnt[i];
        else mean_lum[i] = mean_lum[i > 0 ? i-1 : 0];
    }
    
    /* Fit: log10(mean) vs r^2 for outer portion */
    double Sx = 0, Sy = 0, Sxx = 0, Sxy = 0, Sn = 0;
    for (int i = (int)(NB * 0.3); i < NB; i++) {
        double r = (i + 0.5) / (double)NB;
        double u = r * r;
        double yv = log10(mean_lum[i] > 1e-6 ? mean_lum[i] : 1e-6);
        Sx += u; Sy += yv; Sxx += u*u; Sxy += u*yv; Sn++;
    }
    if (Sn < 3) return;
    
    double denom = Sn * Sxx - Sx * Sx;
    if (fabs(denom) < 1e-18) return;
    double a1 = (Sn * Sxy - Sx * Sy) / denom; /* slope */
    
    /* amount such that corr = 1 + amount*(r^2 + 0.5*r^4) brightens edges if needed */
    /* We want to compensate for falloff: if a1 < 0, corners darker */
    double amount = -a1 * 3.5; /* scale factor - be more aggressive for dark corners */
    if (amount < -0.6) amount = -0.6;
    if (amount > 0.8) amount = 0.8;
    if (fabs(amount) > 0.02) {
        res->vignette = amount;
        res->has_vignette = 1;
    }
}

/* Estimate TCA by measuring channel shifts at edges */
static void estimate_tca(const FeImage *im, FeLensAutoResult *res) {
    int w = im->w, h = im->h;
    if (w < 64 || h < 64) return;
    float cx = w * 0.5f, cy = h * 0.5f;
    float maxr = sqrtf(cx*cx + cy*cy);
    if (maxr < 1.0f) return;
    
    /* Compute Sobel to find edges */
    float *mag = calloc((size_t)w * h, sizeof(float));
    for (int y = 1; y < h-1; y++) {
        for (int x = 1; x < w-1; x++) {
            const float *c00 = im->data + ((size_t)(y-1)*w + (x-1))*3;
            const float *c10 = im->data + ((size_t)(y-1)*w + x)*3;
            const float *c20 = im->data + ((size_t)(y-1)*w + (x+1))*3;
            const float *c01 = im->data + ((size_t)y*w + (x-1))*3;
            const float *c21 = im->data + ((size_t)y*w + (x+1))*3;
            const float *c02 = im->data + ((size_t)(y+1)*w + (x-1))*3;
            const float *c12 = im->data + ((size_t)(y+1)*w + x)*3;
            const float *c22 = im->data + ((size_t)(y+1)*w + (x+1))*3;
            float gx_r = -c00[0]-2*c10[0]-c20[0]+c02[0]+2*c12[0]+c22[0];
            float gy_r = -c00[0]-2*c01[0]-c02[0]+c20[0]+2*c21[0]+c22[0];
            float gx_b = -c00[2]-2*c10[2]-c20[2]+c02[2]+2*c12[2]+c22[2];
            float gy_b = -c00[2]-2*c01[2]-c02[2]+c20[2]+2*c21[2]+c22[2];
            mag[y*w + x] = sqrtf((gx_r-gx_b)*(gx_r-gx_b) + (gy_r-gy_b)*(gy_r-gy_b));
        }
    }
    
    /* Find threshold for strong edges */
    float *mcopy = malloc((size_t)w * h * sizeof(float));
    memcpy(mcopy, mag, (size_t)w * h * sizeof(float));
    /* simple nth element approximation */
    for (int i = 0; i < 1000; i++) {
        int a = rand() % (w*h);
        int b = rand() % (w*h);
        float t = mcopy[a]; mcopy[a] = mcopy[b]; mcopy[b] = t;
    }
    float thresh = mcopy[w*h/20] * 2.0f;
    free(mcopy);
    
    double shift_r_accum = 0, shift_b_accum = 0, wacc = 0;
    /* Sample at strong edges in outer regions */
    for (int y = 32; y < h-32; y += 4) {
        for (int x = 32; x < w-32; x += 4) {
            if (mag[y*w + x] < thresh) continue;
            float dx = (x - cx) / maxr;
            float dy = (y - cy) / maxr;
            float r = sqrtf(dx*dx + dy*dy);
            if (r < 0.4f) continue;
            /* sample along radial direction */
            float ux = dx / r, uy = dy / r;
            /* find best shift for red vs green */
            double best_r = 0, best_b = 0, best_corr_r = -1e9, best_corr_b = -1e9;
            for (double s = -1.5; s <= 1.5; s += 0.25) {
                /* correlation between shifted R and G */
                double corr_r = 0, corr_b = 0;
                int cnt = 0;
                for (int t = -3; t <= 3; t++) {
                    int sx = (int)(x + s*ux + t*uy);
                    int sy = (int)(y + s*uy - t*ux);
                    int sx0 = x + t*uy;
                    int sy0 = y - t*ux;
                    if (sx < 0 || sx >= w || sy < 0 || sy >= h) continue;
                    if (sx0 < 0 || sx0 >= w || sy0 < 0 || sy0 >= h) continue;
                    const float *pr = im->data + ((size_t)sy * w + sx) * 3;
                    const float *pb = im->data + ((size_t)sy * w + sx) * 3;
                    const float *pg0 = im->data + ((size_t)sy0 * w + sx0) * 3;
                    corr_r += pr[0] * pg0[1];
                    corr_b += pb[2] * pg0[1];
                    cnt++;
                }
                if (cnt > 0) { corr_r /= cnt; corr_b /= cnt; }
                if (corr_r > best_corr_r) { best_corr_r = corr_r; best_r = s; }
                if (corr_b > best_corr_b) { best_corr_b = corr_b; best_b = s; }
            }
            if (fabs(best_r) > 0.01 && fabs(best_r) < 0.8) {
                /* convert pixel shift to tca coefficient */
                double a_r = (best_r / r) / (r*r) * 0.15;
                shift_r_accum += a_r;
                wacc++;
            }
            if (fabs(best_b) > 0.01 && fabs(best_b) < 0.8) {
                double a_b = (best_b / r) / (r*r) * 0.15;
                shift_b_accum += a_b;
            }
        }
    }
    free(mag);
    if (wacc > 20) {
        double ar = shift_r_accum / wacc;
        double ab = shift_b_accum / wacc;
        if (fabs(ar) > 0.001 && fabs(ar) < 0.15) { res->tca_r = ar; res->has_tca = 1; }
        if (fabs(ab) > 0.001 && fabs(ab) < 0.15) { res->tca_b = ab; res->has_tca = 1; }
    }
}

/* Estimate distortion from curved edges */
static void estimate_distortion(const FeImage *im, FeLensAutoResult *res) {
    int w = im->w, h = im->h;
    if (w < 64 || h < 64) return;
    float cx = w * 0.5f, cy = h * 0.5f;
    float maxr = sqrtf(cx*cx + cy*cy);
    if (maxr < 1.0f) return;
    
    /* Sobel */
    float *gx = calloc((size_t)w * h, sizeof(float));
    float *gy = calloc((size_t)w * h, sizeof(float));
    float *mag = calloc((size_t)w * h, sizeof(float));
    for (int y = 1; y < h-1; y++) {
        for (int x = 1; x < w-1; x++) {
            const float *p[9];
            for (int dy = -1; dy <= 1; dy++)
                for (int dx = -1; dx <= 1; dx++)
                    p[(dy+1)*3+(dx+1)] = im->data + ((size_t)(y+dy)*w + (x+dx))*3;
            float l[9];
            for (int i = 0; i < 9; i++) l[i] = fe_luma_px(p[i]);
            gx[y*w+x] = -l[0]-2*l[1]-l[2]+l[6]+2*l[7]+l[8];
            gy[y*w+x] = -l[0]-2*l[3]-l[6]+l[2]+2*l[5]+l[8];
            mag[y*w+x] = sqrtf(gx[y*w+x]*gx[y*w+x] + gy[y*w+x]*gy[y*w+x]);
        }
    }
    
    /* Find strong edges */
    float *mcopy = malloc((size_t)w * h * sizeof(float));
    memcpy(mcopy, mag, (size_t)w * h * sizeof(float));
    for (int i = 0; i < 2000; i++) {
        int a = rand() % (w*h);
        int b = rand() % (w*h);
        float t = mcopy[a]; mcopy[a] = mcopy[b]; mcopy[b] = t;
    }
    float thresh = mcopy[w*h/10];
    if (thresh < 0.01f) thresh = 0.01f;
    free(mcopy);
    
    /* Look for curved vertical/horizontal edges - sample and fit */
    double k1_accum = 0, wacc = 0;
    /* Sample vertical edges (|gx| > |gy|) */
    for (int x = (int)(w*0.2); x < (int)(w*0.8); x += 8) {
        /* trace vertical-ish edges */
        for (int y = 64; y < h-64; y += 2) {
            if (mag[y*w+x] < thresh) continue;
            if (fabsf(gx[y*w+x]) < fabsf(gy[y*w+x])) continue;
            /* collect a segment */
            double pts_x[64], pts_y[64];
            int npts = 0;
            for (int yy = y; yy < y+64 && yy < h-1 && npts < 64; yy++) {
                if (mag[yy*w+x] > thresh * 0.5f) {
                    pts_x[npts] = x - cx;
                    pts_y[npts] = yy - cy;
                    npts++;
                } else if (npts > 5) break;
            }
            if (npts < 15) continue;
            /* fit x' = a + b*y' + c*y'^2 (relative coords) */
            double Sx=0,Sy=0,Sxx=0,Sxxx=0,Sxxxx=0,Sxy=0,Sxxy=0;
            for (int i = 0; i < npts; i++) {
                double xx = pts_y[i]; /* use y as variable */
                double yy = pts_x[i];
                double xx2 = xx*xx, xx3 = xx2*xx, xx4 = xx2*xx2;
                Sx+=xx; Sy+=yy; Sxx+=xx2; Sxxx+=xx3; Sxxxx+=xx4;
                Sxy+=xx*yy; Sxxy+=xx2*yy;
            }
            double N = npts;
            /* 3x3 normal equations */
            double M[3][3] = {
                {Sxxxx, Sxxx, Sxx},
                {Sxxx,  Sxx,  Sx},
                {Sxx,   Sx,   N}
            };
            double V[3] = {Sxxy, Sxy, Sy};
            /* invert */
            double det = M[0][0]*(M[1][1]*M[2][2]-M[1][2]*M[2][1]) -
                         M[0][1]*(M[1][0]*M[2][2]-M[1][2]*M[2][0]) +
                         M[0][2]*(M[1][0]*M[2][1]-M[1][1]*M[2][0]);
            if (fabs(det) < 1e-18) continue;
            /* simple solve */
            double inv[3][3];
            inv[0][0]=(M[1][1]*M[2][2]-M[1][2]*M[2][1])/det;
            inv[0][1]=(M[0][2]*M[2][1]-M[0][1]*M[2][2])/det;
            inv[0][2]=(M[0][1]*M[1][2]-M[0][2]*M[1][1])/det;
            inv[1][0]=(M[1][2]*M[2][0]-M[1][0]*M[2][2])/det;
            inv[1][1]=(M[0][0]*M[2][2]-M[0][2]*M[2][0])/det;
            inv[1][2]=(M[0][2]*M[1][0]-M[0][0]*M[1][2])/det;
            inv[2][0]=(M[1][0]*M[2][1]-M[1][1]*M[2][0])/det;
            inv[2][1]=(M[0][1]*M[2][0]-M[0][0]*M[2][1])/det;
            inv[2][2]=(M[0][0]*M[1][1]-M[0][1]*M[1][0])/det;
            double c = inv[0][0]*V[0]+inv[0][1]*V[1]+inv[0][2]*V[2];
            /* D is distance of line from center in x direction */
            double D = (x - cx) / maxr;
            if (fabs(D) > 0.25) {
                /* k1 ≈ c*maxr/D (in normalized units) */
                double k1_line = c * maxr / D;
                if (fabs(k1_line) < 0.3) {
                    k1_accum += k1_line;
                    wacc++;
                }
            }
            y += 30; /* skip ahead */
        }
    }
    
    free(gx); free(gy); free(mag);
    if (wacc > 15) {
        double k1_est = k1_accum / wacc;
        /* clamp to reasonable range */
        if (k1_est > 0.15) k1_est = 0.15;
        if (k1_est < -0.15) k1_est = -0.15;
        if (fabs(k1_est) > 0.01) {
            res->k1 = k1_est;
            res->has_distortion = 1;
        }
    }
}

void fe_lens_auto_analyze(const FeImage *linear, FeLensAutoResult *out) {
    memset(out, 0, sizeof(*out));
    if (!linear || linear->w < 32 || linear->h < 32) return;
    
    /* Work on a downscaled copy for speed */
    FeImage work = fe_img_copy(linear);
    fe_downscale_maxedge(&work, 1200);
    
    estimate_vignetting(&work, out);
    estimate_tca(&work, out);
    estimate_distortion(&work, out);
    
    /* Compute confidence */
    out->confidence = 0;
    if (out->has_distortion) out->confidence += 0.4;
    if (out->has_tca) out->confidence += 0.3;
    if (out->has_vignette) out->confidence += 0.3;
    
    fe_img_free(&work);
} 

/* Export wrapper with C linkage */
void fe_lens_auto_analyze_image(const FeImage *linear, double *k1_out, double *tca_r_out, double *tca_b_out, 
                                 double *vignette_out, int *has_any) {
    FeLensAutoResult res;
    fe_lens_auto_analyze(linear, &res);
    if (k1_out) *k1_out = res.k1;
    if (tca_r_out) *tca_r_out = res.tca_r;
    if (tca_b_out) *tca_b_out = res.tca_b;
    if (vignette_out) *vignette_out = res.vignette;
    if (has_any) *has_any = (res.has_distortion || res.has_tca || res.has_vignette) ? 1 : 0;
}
