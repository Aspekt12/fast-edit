#include "fe_pipeline.h"
#include "fe_math.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ======================================================================
   THE DEVELOP PIPELINE — a faithful port of fast-edit-v2.py's develop().
   Order matters and matches the Python app exactly:
     calibration -> white balance -> exposure + filmic shoulder -> sRGB
     -> shadows/highlights -> tone curve -> mixer -> saturation/vibrance
     -> colour grading -> presence -> bloom -> vignette -> film grain.
   ====================================================================== */

/* ---- 1. calibration ---------------------------------------------------- */
static void fe_calibration_matrix(double red_h, double green_h, double blue_h,
                                  double red_s, double green_s, double blue_s,
                                  double M[3][3]) {
    /* Build the three new primaries (pure hue, rotated by the hue slider,
       scaled by the saturation slider) and normalise each ROW so neutral
       grey stays neutral grey. */
    const double base_hue[3] = {0.0, 120.0, 240.0};
    const double shift[3] = {red_h, green_h, blue_h};
    const double sat[3] = {red_s, green_s, blue_s};
    double prim[3][3]; /* prim[primary][channel] */
    for (int p = 0; p < 3; p++) {
        double h = fmod(base_hue[p] + shift[p], 360.0) / 360.0;
        if (h < 0.0) h += 1.0;
        double c = 1.0, x = c * (1.0 - fabs(fmod(h * 6.0, 2.0) - 1.0)), m = 1.0 - c;
        double r, g, b;
        if (h < 1.0 / 6.0)      { r = c; g = x; b = 0; }
        else if (h < 2.0 / 6.0) { r = x; g = c; b = 0; }
        else if (h < 3.0 / 6.0) { r = 0; g = c; b = x; }
        else if (h < 4.0 / 6.0) { r = 0; g = x; b = c; }
        else if (h < 5.0 / 6.0) { r = x; g = 0; b = c; }
        else                   { r = c; g = 0; b = x; }
        r += m; g += m; b += m;
        prim[p][0] = r * sat[p];
        prim[p][1] = g * sat[p];
        prim[p][2] = b * sat[p];
    }
    for (int r = 0; r < 3; r++) {
        double sum = prim[0][r] + prim[1][r] + prim[2][r];
        if (fabs(sum) < 1e-18) sum = 1.0;
        for (int c = 0; c < 3; c++) M[r][c] = prim[c][r] / sum;
    }
}

static void fe_apply_calibration(FeImage *img, const double M[3][3]) {
    int n = img->w * img->h;
    const float *p = img->data;
    float *d = img->data;
    for (int i = 0; i < n; i++) {
        float r = p[0], g = p[1], b = p[2];
        /* Python does `img @ m.T`, which for a (H,W,3) image contracts the
           pixel vector with the ROWS of m: out = M · pixel.  Row r of M is
           output channel r, so the row index selects the output channel. */
        d[0] = (float)(r * M[0][0] + g * M[0][1] + b * M[0][2]);
        d[1] = (float)(r * M[1][0] + g * M[1][1] + b * M[1][2]);
        d[2] = (float)(r * M[2][0] + g * M[2][1] + b * M[2][2]);
        p += 3; d += 3;
    }
}

/* ---- 2. white balance -------------------------------------------------- */
static void fe_white_balance(FeImage *img, double temperature, double tint) {
    if (fabs(temperature) < 1e-6 && fabs(tint) < 1e-6) return;
    double g0 = 1.0 + 0.30 * temperature - 0.08 * tint;
    double g1 = 1.0 + 0.16 * tint;
    double g2 = 1.0 - 0.30 * temperature - 0.08 * tint;
    double norm = g0 * 0.2126 + g1 * 0.7152 + g2 * 0.0722;
    if (fabs(norm) < 1e-18) norm = 1.0;
    g0 /= norm; g1 /= norm; g2 /= norm;
    int n = img->w * img->h;
    float *p = img->data;
    for (int i = 0; i < n; i++) {
        p[0] = (float)(p[0] * g0);
        p[1] = (float)(p[1] * g1);
        p[2] = (float)(p[2] * g2);
        p += 3;            /* next pixel */
    }
}

/* ---- 3. shadows / highlights ------------------------------------------- */
static void fe_range_adjust(FeImage *img, double shadows, double highlights) {
    if (fabs(shadows) < 1e-6 && fabs(highlights) < 1e-6) return;
    int n = img->w * img->h;
    float *L = (float *)malloc((size_t)n * sizeof(float));
    fe_luma(img, L);
    float *p = img->data;
    for (int i = 0; i < n; i++) {
        float lv = L[i];
        /* near-black protection: shadows lift, true black stays black */
        float shadow_mask = (1.0f - fe_smoothstep(0.0f, 0.62f, lv)) * fe_smoothstep(0.0f, 0.08f, lv);
        float high_mask = fe_smoothstep(0.38f, 1.0f, lv);
        float delta = (float)shadows * shadow_mask + (float)highlights * high_mask;
        float factor = (lv + delta) / fmaxf(lv, 1e-4f);
        p[0] = fe_clampf(p[0] * factor, 0.0f, 1.0f);
        p[1] = fe_clampf(p[1] * factor, 0.0f, 1.0f);
        p[2] = fe_clampf(p[2] * factor, 0.0f, 1.0f);
        p += 3;            /* next pixel */
    }
    free(L);
}

/* ---- 4. colour mixer ---------------------------------------------------- */
static void fe_color_mixer(FeImage *img, const FeSettings *s) {
    int any = 0;
    for (int i = 0; i < MIXER_BAND_COUNT; i++) if (s->mixer_enabled[i]) any = 1;
    if (!any) return;

    int n = img->w * img->h;
    float *H = (float *)malloc((size_t)n * sizeof(float));
    float *S = (float *)malloc((size_t)n * sizeof(float));
    float *V = (float *)malloc((size_t)n * sizeof(float));
    fe_rgb_to_hsv(img, H, S, V);

    float *dH = (float *)calloc((size_t)n, sizeof(float));
    float *dS = (float *)calloc((size_t)n, sizeof(float));
    float *dV = (float *)calloc((size_t)n, sizeof(float));

    for (int b = 0; b < MIXER_BAND_COUNT; b++) {
        if (!s->mixer_enabled[b]) continue;
        double centre = MIXER_BAND_CENTRES[b];
        double sh = s->mixer_h[b], sd = s->mixer_s[b], vd = s->mixer_v[b];
        for (int i = 0; i < n; i++) {
            double dist = fabs(fmod(H[i] - centre + 180.0, 360.0) - 180.0);
            /* smoothstep(75, 35, dist): full weight inside ±35°, zero at ±75° */
            float t = (75.0f - (float)dist) / (75.0f - 35.0f + 1e-9f);
            if (t < 0.0f) t = 0.0f;
            if (t > 1.0f) t = 1.0f;
            t = t * t * (3.0f - 2.0f * t);
            if (t == 0.0f) continue;
            dH[i] += (float)(t * sh);
            dS[i] += (float)(t * sd);
            dV[i] += (float)(t * vd);
        }
    }

    FeImage out = fe_img_new(img->w, img->h);
    for (int i = 0; i < n; i++) {
        float nh = fmodf(H[i] + dH[i], 360.0f);
        if (nh < 0.0f) nh += 360.0f;
        H[i] = nh;
        S[i] = fe_clampf(S[i] * (1.0f + dS[i]), 0.0f, 1.0f);
        V[i] = fe_clampf(V[i] * (1.0f + dV[i]), 0.0f, 1.0f);
    }
    fe_hsv_to_rgb_img(H, S, V, img->w, img->h, &out);
    free(H); free(S); free(V); free(dH); free(dS); free(dV);
    memcpy(img->data, out.data, (size_t)n * 3 * sizeof(float));
    fe_img_free(&out);
}

/* ---- 5. colour grading (three-way wheels) ------------------------------ */
static void fe_wheel_colour(double hue, double strength, double rgb[3]) {
    double h = hue / 60.0;
    int sector = ((int)floor(h)) % 6;
    if (sector < 0) sector += 6;
    double f = h - floor(h);
    double v = 1.0, s = strength;
    double p = v * (1.0 - s);
    double q = v * (1.0 - s * f);
    double t = v * (1.0 - s * (1.0 - f));
    double r, g, b;
    switch (sector) {
        case 0: r = v; g = t; b = p; break;
        case 1: r = q; g = v; b = p; break;
        case 2: r = p; g = v; b = t; break;
        case 3: r = p; g = q; b = v; break;
        case 4: r = t; g = p; b = v; break;
        default:r = v; g = p; b = q; break;
    }
    /* strip the brightness so the grade adds COLOUR, not exposure */
    double luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    rgb[0] = r - luma;
    rgb[1] = g - luma;
    rgb[2] = b - luma;
}

static void fe_color_grading(FeImage *img, const FeSettings *s) {
    if (fabs(s->grade_shadow_s) < 1e-6 && fabs(s->grade_midtone_s) < 1e-6 &&
        fabs(s->grade_highlight_s) < 1e-6)
        return;

    int n = img->w * img->h;
    float *L = (float *)malloc((size_t)n * sizeof(float));
    fe_luma(img, L);

    double width = 0.5 + s->grade_blending * 0.5;
    double sh_col[3], mid_col[3], hi_col[3];
    fe_wheel_colour(s->grade_shadow_h, s->grade_shadow_s, sh_col);
    fe_wheel_colour(s->grade_midtone_h, s->grade_midtone_s, mid_col);
    fe_wheel_colour(s->grade_highlight_h, s->grade_highlight_s, hi_col);

    float *p = img->data;
    for (int i = 0; i < n; i++) {
        float lv = L[i];
        float sw = 1.0f - fe_smoothstep(0.0f, (float)width, lv);
        float hw = fe_smoothstep((float)(1.0 - width), 1.0f, lv);
        float mw = 1.0f - sw - hw;
        if (mw < 0.0f) mw = 0.0f;
        float tint[3] = {0.0f, 0.0f, 0.0f};
        if (fabs(s->grade_shadow_s) > 1e-6)
            for (int c = 0; c < 3; c++) tint[c] += (float)sh_col[c] * sw;
        if (fabs(s->grade_midtone_s) > 1e-6)
            for (int c = 0; c < 3; c++) tint[c] += (float)mid_col[c] * mw;
        if (fabs(s->grade_highlight_s) > 1e-6)
            for (int c = 0; c < 3; c++) tint[c] += (float)hi_col[c] * hw;
        p[0] = fe_clampf(p[0] + tint[0], 0.0f, 1.0f);
        p[1] = fe_clampf(p[1] + tint[1], 0.0f, 1.0f);
        p[2] = fe_clampf(p[2] + tint[2], 0.0f, 1.0f);
        p += 3;            /* next pixel */
    }
    free(L);
}

/* ---- 6. saturation + vibrance ------------------------------------------ */
static void fe_saturation_vibrance(FeImage *img, double saturation, double vibrance) {
    if (fabs(saturation) < 1e-6 && fabs(vibrance) < 1e-6) return;
    int n = img->w * img->h;
    float *H = (float *)malloc((size_t)n * sizeof(float));
    float *S = (float *)malloc((size_t)n * sizeof(float));
    float *V = (float *)malloc((size_t)n * sizeof(float));
    fe_rgb_to_hsv(img, H, S, V);
    for (int i = 0; i < n; i++) {
        float s = S[i] * (1.0f + (float)saturation);
        if (fabs(vibrance) > 1e-6) {
            float muted = (1.0f - S[i]) * (1.0f - S[i]);
            float skin = fe_smoothstep(5.0f, 12.0f, H[i]) * (1.0f - fe_smoothstep(45.0f, 55.0f, H[i]));
            muted *= 1.0f - 0.5f * skin;
            s = s * (1.0f + (float)vibrance * muted);
        }
        S[i] = fe_clampf(s, 0.0f, 1.0f);
    }
    FeImage out = fe_img_new(img->w, img->h);
    fe_hsv_to_rgb_img(H, S, V, img->w, img->h, &out);
    free(H); free(S); free(V);
    memcpy(img->data, out.data, (size_t)n * 3 * sizeof(float));
    fe_img_free(&out);
}

/* ---- 7. presence (texture / clarity / dehaze) --------------------------- */
static void fe_presence(FeImage *img, double texture, double clarity, double dehaze,
                        double pixel_scale) {
    int n = img->w * img->h;
    int maxdim = img->w > img->h ? img->w : img->h;

    if (fabs(texture) > 1e-3) {
        FeImage blur = fe_img_new(img->w, img->h);
        fe_gaussian_blur(img, 2.0f * (float)pixel_scale, &blur);
        float *p = img->data, *b = blur.data;
        float amt = (float)texture * 0.7f;
        for (int i = 0; i < n * 3; i++) p[i] = p[i] + amt * (p[i] - b[i]);
        fe_img_free(&blur);
    }
    if (fabs(clarity) > 1e-3) {
        FeImage blur = fe_img_new(img->w, img->h);
        fe_cheap_blur(img, (float)maxdim / 45.0f, &blur);
        float *p = img->data, *b = blur.data;
        float amt = (float)clarity * 0.6f;
        for (int i = 0; i < n * 3; i++) p[i] = p[i] + amt * (p[i] - b[i]);
        fe_img_free(&blur);
    }
    if (fabs(dehaze) > 1e-3) {
        FeImage glare = fe_img_new(img->w, img->h);
        fe_cheap_blur(img, (float)maxdim / 14.0f, &glare);
        float *p = img->data, *g = glare.data;
        float amt = (float)dehaze * 0.8f;
        for (int i = 0; i < n * 3; i++) p[i] = p[i] + amt * (p[i] - g[i]);
        fe_img_free(&glare);
    }
    float *p = img->data;
    for (int i = 0; i < n * 3; i++) p[i] = fe_clampf(p[i], 0.0f, 1.0f);
}

/* ---- 8. threshold bloom ------------------------------------------------- */
static void fe_threshold_bloom(FeImage *img, double amount, double radius, double threshold) {
    if (amount < 1e-4) return;
    int n = img->w * img->h;
    FeImage highs = fe_img_copy(img);
    float *p = highs.data;
    for (int i = 0; i < n * 3; i++) {
        float v = p[i] - (float)threshold;
        p[i] = v > 0.0f ? v : 0.0f;
    }
    FeImage glow = fe_img_new(img->w, img->h);
    fe_cheap_blur(&highs, (float)radius, &glow);
    float *ip = img->data, *gp = glow.data;
    for (int i = 0; i < n * 3; i++) ip[i] = fe_clampf(ip[i] + gp[i] * (float)amount, 0.0f, 1.0f);
    fe_img_free(&highs);
    fe_img_free(&glow);
}

/* ---- 9. vignette -------------------------------------------------------- */
static void fe_vignette(FeImage *img, double amount) {
    if (amount < 1e-4) return;
    int w = img->w, h = img->h;
    float *p = img->data;
    for (int y = 0; y < h; y++) {
        float yy = ((float)y - ((float)h - 1.0f) / 2.0f) / ((float)h / 2.0f);
        for (int x = 0; x < w; x++) {
            float xx = ((float)x - ((float)w - 1.0f) / 2.0f) / ((float)w / 2.0f);
            float dist = sqrtf(yy * yy + xx * xx) / sqrtf(2.0f);
            float mask = 1.0f - (float)amount * fe_smoothstep(0.35f, 1.0f, dist);
            int idx = (y * w + x) * 3;
            p[idx] = fe_clampf(p[idx] * mask, 0.0f, 1.0f);
            p[idx + 1] = fe_clampf(p[idx + 1] * mask, 0.0f, 1.0f);
            p[idx + 2] = fe_clampf(p[idx + 2] * mask, 0.0f, 1.0f);
        }
    }
}

/* ---- 10. film grain (darktable's paper-response model) ----------------- */
static const double GRAIN_FREQ[3] = {GRAIN_OCTAVE_FREQ0, GRAIN_OCTAVE_FREQ1, GRAIN_OCTAVE_FREQ2};
static const double GRAIN_AMP[3] = {GRAIN_OCTAVE_AMP0, GRAIN_OCTAVE_AMP1, GRAIN_OCTAVE_AMP2};

/* Multi-octave value noise: white noise on a coarse lattice, cubicly
   upsampled.  Band-limited noise with a sharp cutoff at the lattice
   frequency — the same role darktable's simplex lattice plays. */
static void fe_value_noise_octaves(int w, int h, double cell_px, FeRng *rng, float *total) {
    memset(total, 0, (size_t)w * h * sizeof(float));
    for (int o = 0; o < 3; o++) {
        double cell = cell_px / GRAIN_FREQ[o];
        if (cell < 0.75) cell = 0.75;
        int nh = (int)((double)h / cell) + 2;
        int nw = (int)((double)w / cell) + 2;
        if (nh < 2) nh = 2;
        if (nw < 2) nw = 2;
        /* lattice (stored in 3 channels so the shared cubic resizer can be used) */
        FeImage lattice = fe_img_new(nw, nh);
        for (int i = 0; i < nw * nh; i++) {
            float v = (float)fe_rng_normal(rng);
            lattice.data[i * 3 + 0] = v;
            lattice.data[i * 3 + 1] = v;
            lattice.data[i * 3 + 2] = v;
        }
        FeImage up = fe_img_new(w, h);
        fe_resize_cubic(&lattice, w, h, &up);
        for (int i = 0; i < w * h; i++)
            total[i] += up.data[i * 3] * (float)GRAIN_AMP[o];
        fe_img_free(&lattice);
        fe_img_free(&up);
    }
    for (int i = 0; i < w * h; i++) total[i] /= (float)GRAIN_OCTAVE_NORM;
}

static double fe_paper_curve(double density, double bias, double gamma) {
    double delta = 2.0 * exp((bias / 100.0) * log(1e-4));
    double z = exp(4.0 * gamma * (0.5 - density) / (1.0 + 2.0 * delta));
    if (z > exp(60.0)) z = exp(60.0);
    if (z < exp(-60.0)) z = exp(-60.0);
    return (1.0 + 2.0 * delta) / (1.0 + z) - delta;
}

static double fe_paper_curve_inverse(double density, double bias, double gamma) {
    double delta = 2.0 * exp((bias / 100.0) * log(1e-4));
    double d = density;
    if (d < 1e-4) d = 1e-4;
    if (d > 1.0 - 1e-4) d = 1.0 - 1e-4;
    double arg = (1.0 + 2.0 * delta) / (d + delta) - 1.0;
    if (arg < 1e-300) arg = 1e-300;
    return -log(arg) * (1.0 + 2.0 * delta) / (4.0 * gamma) + 0.5;
}

static void fe_film_grain(FeImage *img, double amount, double size, double color,
                           double bias, double pixel_scale) {
    if (amount < 1e-4) return;
    int w = img->w, h = img->h;
    int n = w * h;
    FeRng rng;
    fe_rng_seed(&rng, 20261021ULL);   /* seeded: renders are reproducible */

    double cell_px = size * pixel_scale * 2.0;
    if (cell_px < 1.0) cell_px = 1.0;

    float *noise = (float *)malloc((size_t)n * sizeof(float));
    fe_value_noise_octaves(w, h, cell_px, &rng, noise);

    float *L = (float *)malloc((size_t)n * sizeof(float));
    fe_luma(img, L);

    float *p = img->data;
    for (int i = 0; i < n; i++) {
        float exposure = (float)fe_paper_curve_inverse(L[i], bias, 1.0);
        float displaced = exposure + noise[i] * (float)(amount * GRAIN_STRENGTH_SCALE);
        float newL = (float)fe_paper_curve(displaced, bias, 1.0);
        float factor = newL / fmaxf(L[i], 1e-3f);
        if (factor < 0.2f) factor = 0.2f;
        if (factor > 5.0f) factor = 5.0f;
        p[i * 3 + 0] = fe_clampf(p[i * 3 + 0] * factor, 0.0f, 1.0f);
        p[i * 3 + 1] = fe_clampf(p[i * 3 + 1] * factor, 0.0f, 1.0f);
        p[i * 3 + 2] = fe_clampf(p[i * 3 + 2] * factor, 0.0f, 1.0f);
    }
    free(noise);
    free(L);

    /* dye clouds: a whisper of chroma, but only where colour already exists */
    if (color > 1e-4) {
        float *clouds = (float *)malloc((size_t)n * sizeof(float));
        fe_value_noise_octaves(w, h, cell_px * 1.6, &rng, clouds);
        for (int i = 0; i < n; i++) {
            float mx = p[i * 3], mn = p[i * 3];
            if (p[i * 3 + 1] > mx) mx = p[i * 3 + 1];
            if (p[i * 3 + 1] < mn) mn = p[i * 3 + 1];
            if (p[i * 3 + 2] > mx) mx = p[i * 3 + 2];
            if (p[i * 3 + 2] < mn) mn = p[i * 3 + 2];
            float chroma = mx - mn;
            float mask = fe_smoothstep(0.01f, 0.08f, chroma);
            float amt = clouds[i] * mask * (float)(color * amount * 0.8);
            p[i * 3 + 0] = fe_clampf(p[i * 3 + 0] + amt, 0.0f, 1.0f);
            p[i * 3 + 1] = fe_clampf(p[i * 3 + 1] + amt, 0.0f, 1.0f);
            p[i * 3 + 2] = fe_clampf(p[i * 3 + 2] + amt, 0.0f, 1.0f);
        }
        free(clouds);
    }
}

/* ======================================================================
   DEVELOP
   ====================================================================== */
void fe_develop(const FeImage *linear_rgb, const FeSettings *s, FeImage *out) {
    FeImage img = fe_img_copy(linear_rgb);

    /* 1 — calibration: redefine the primaries before anything else sees them */
    double M[3][3];
    fe_calibration_matrix(s->cal_red_hue, s->cal_green_hue, s->cal_blue_hue,
                          s->cal_red_sat, s->cal_green_sat, s->cal_blue_sat, M);
    fe_apply_calibration(&img, M);

    /* 2 — white balance, in linear light where it physically belongs */
    fe_white_balance(&img, s->temperature, s->tint);

    /* 3 — exposure + filmic shoulder, then back to display gamma */
    {
        int n = img.w * img.h * 3;
        float expf = powf(2.0f, (float)s->exposure);
        float *p = img.data;
        for (int i = 0; i < n; i++) p[i] *= expf;
    }
    fe_soft_clip(&img, 0.72f);
    fe_linear_to_srgb(&img);

    /* 4 — basic tonality */
    fe_range_adjust(&img, s->shadows, s->highlights);
    {
        float lut[1024];
        fe_tone_curve_lut(s->contrast, s->blacks, s->whites, s->fade, lut, 1024);
        fe_apply_lut(&img, lut, 1024);
    }

    /* 5 — colour work (mixer, then saturation, then grading — order matters) */
    fe_color_mixer(&img, s);
    fe_saturation_vibrance(&img, s->saturation, s->vibrance);
    fe_color_grading(&img, s);

    /* Pixelation for pixel art looks */
    if (s->pixel_size > 1.0) {
        int psize = (int)(s->pixel_size);
        if (psize < 2) psize = 2;
        if (psize > 32) psize = 32;
        int nw = img.w / psize;
        int nh = img.h / psize;
        if (nw < 1) nw = 1;
        if (nh < 1) nh = 1;
        FeImage small = fe_img_new(nw, nh);
        fe_resize_area(&img, nw, nh, &small);
        /* Quantize to limited palette */
        for (int i = 0; i < nw * nh * 3; i++) {
            small.data[i] = floorf(small.data[i] * 4.0f + 0.5f) / 4.0f;
        }
        FeImage out_px = fe_img_new(img.w, img.h);
        fe_resize_linear(&small, img.w, img.h, &out_px);
        fe_img_free(&img);
        img = out_px;
        fe_img_free(&small);
    }

    /* 6 — presence and effects, scaled to the output resolution */
    double px = (double)(img.w > img.h ? img.w : img.h) / (double)PREVIEW_LONG_EDGE;
    fe_presence(&img, s->texture, s->clarity, s->dehaze, px);
    /* Additional sharpening */
    if (s->sharpening_amount > 1e-4) {
        FeImage blur;
        memset(&blur, 0, sizeof(blur));
        float sr = s->sharpening_radius > 0 ? (float)s->sharpening_radius : 0.5f;
        fe_gaussian_blur(&img, sr, &blur);
        float amt = (float)s->sharpening_amount * 0.5f;
        int n2 = img.w * img.h;
        float *p = img.data;
        const float *b = blur.data;
        for (int i = 0; i < n2; i++) {
            for (int c = 0; c < 3; c++) {
                float detail = p[c] - b[c];
                p[c] = fe_clampf(p[c] + amt * detail, 0.0f, 1.0f);
            }
            p += 3; b += 3;
        }
        fe_img_free(&blur);
    }
    /* Lens correction */
    if (s->lens_enable || s->lens_distortion != 0 || s->lens_tca_r != 0 || s->lens_tca_b != 0 ||
        s->lens_vignette_corr != 0 || s->lens_keystone_h != 0 || s->lens_keystone_v != 0 ||
        s->lens_k1 != 0 || s->lens_k2 != 0 || s->lens_k3 != 0) {
        extern void fe_lens_auto_correct(FeImage *img, const FeSettings *s);
        fe_lens_auto_correct(&img, s);
    }
    fe_threshold_bloom(&img, s->bloom_amount, s->bloom_radius * px, s->bloom_threshold);
    fe_vignette(&img, s->vignette_amount);
    double grain_size = s->grain_size < 0.8 ? 0.8 : s->grain_size;
    fe_film_grain(&img, s->grain_amount, grain_size, s->grain_color, s->grain_bias, px);

    int n = img.w * img.h * 3;
    float *p = img.data;
    for (int i = 0; i < n; i++) p[i] = fe_clampf(p[i], 0.0f, 1.0f);

    /* hand the result over */
    free(out->data);
    out->w = img.w; out->h = img.h;
    out->data = img.data;
}

/* ======================================================================
   AUTO EXPOSURE
   Log-average key solve + highlight anchor (see the Python docstring for
   the full rationale).
   ====================================================================== */
static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

double fe_auto_exposure(const FeImage *linear_rgb) {
    int n = linear_rgb->w * linear_rgb->h;
    const float *p = linear_rgb->data;
    double sumlog = 0.0;
    int cnt = 0;
    double *L = (double *)malloc((size_t)n * sizeof(double));
    for (int i = 0; i < n; i++) {
        double l = p[i * 3] * 0.2126 + p[i * 3 + 1] * 0.7152 + p[i * 3 + 2] * 0.0722;
        if (l > 1e-4) { L[cnt++] = l; sumlog += log(l); }
    }
    if (cnt == 0) { free(L); return 0.0; }
    double log_avg = exp(sumlog / (double)cnt);
    if (log_avg <= 1e-9) { free(L); return 0.0; }
    double stops = log2(0.18 / log_avg);

    /* highlight anchor: the 99.9th percentile may reach at most 1.6x white */
    qsort(L, (size_t)cnt, sizeof(double), cmp_double);
    double pos = 0.999 * (double)(cnt - 1);
    int lo = (int)floor(pos);
    int hi = lo + 1 < cnt ? lo + 1 : cnt - 1;
    double frac = pos - (double)lo;
    double anchor = L[lo] * (1.0 - frac) + L[hi] * frac;
    if (anchor > 1e-6) {
        double capped = log2(1.6 / anchor);
        if (capped < stops) stops = capped;
    }
    free(L);
    return stops;
}

/* ======================================================================
   QUANTISE — float [0,1] to 8 bit, optional TPDF dither (export only).
   ====================================================================== */
void fe_quantize_u8(const FeImage *img, unsigned char *out_rgb, int dither) {
    int n = img->w * img->h;
    const float *p = img->data;
    FeRng rng;
    fe_rng_seed(&rng, 20261022ULL);
    for (int i = 0; i < n; i++) {
        /* The Python app dithers MONOCHROME: one noise value per pixel,
           shared by R, G and B (per-channel dither reads as chroma dirt). */
        float noise = 0.0f;
        if (dither) {
            /* difference of two uniforms = triangular PDF on (-1, 1) */
            noise = (float)(fe_rng_uniform(&rng) - fe_rng_uniform(&rng));
        }
        for (int c = 0; c < 3; c++) {
            float x = fe_clampf(p[i * 3 + c], 0.0f, 1.0f) * 255.0f + noise;
            if (dither) {
                /* round-half-to-even, matching numpy's np.rint */
                x = nearbyintf(x);
            } else {
                /* plain truncation — byte-identical to numpy's astype(uint8) */
                x = floorf(x);
            }
            if (x < 0.0f) x = 0.0f;
            if (x > 255.0f) x = 255.0f;
            out_rgb[i * 3 + c] = (unsigned char)x;
        }
    }
}
