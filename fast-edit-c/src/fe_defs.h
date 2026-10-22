#ifndef FE_DEFS_H
#define FE_DEFS_H

#include <stddef.h>
#include <stdint.h>

#define PREVIEW_LONG_EDGE 1800
#define DRAFT_LONG_EDGE   850

#define GRAIN_OCTAVE_FREQ0 0.4910
#define GRAIN_OCTAVE_FREQ1 0.9441
#define GRAIN_OCTAVE_FREQ2 1.7280
#define GRAIN_OCTAVE_AMP0  0.2340
#define GRAIN_OCTAVE_AMP1  0.7850
#define GRAIN_OCTAVE_AMP2  1.2150
#define GRAIN_OCTAVE_NORM  1.465
#define GRAIN_STRENGTH_SCALE 0.6

/* mixer's 8 bands */
#define MIXER_BAND_RED     0
#define MIXER_BAND_ORANGE  1
#define MIXER_BAND_YELLOW  2
#define MIXER_BAND_GREEN   3
#define MIXER_BAND_AQUA    4
#define MIXER_BAND_BLUE    5
#define MIXER_BAND_PURPLE  6
#define MIXER_BAND_MAGENTA 7
#define MIXER_BAND_COUNT   8

extern const double MIXER_BAND_CENTRES[8];
extern const char *MIXER_BAND_NAMES[8];

/* Image: row-major float RGB in [0,1] (linear or srgb depending on stage) */
typedef struct {
    int w, h;
    float *data; /* h*w*3 */
} FeImage;

typedef struct {
    double h, s, l; /* luma */
} FeGradeWheel;

typedef struct {
    /* white balance / calibration */
    double temperature, tint;
    double cal_red_hue, cal_green_hue, cal_blue_hue;
    double cal_red_sat, cal_green_sat, cal_blue_sat;
    /* basic tonality */
    double exposure, contrast, highlights, shadows, blacks, whites, fade;
    /* color mixer: per-band (hue_shift, sat_delta, lum_delta) */
    double mixer_h[8], mixer_s[8], mixer_v[8];
    int mixer_enabled[8];
    /* grading: (hue_deg, strength) */
    double grade_shadow_h, grade_shadow_s;
    double grade_midtone_h, grade_midtone_s;
    double grade_highlight_h, grade_highlight_s;
    double grade_blending;
    /* colour */
    double saturation, vibrance;
    /* presence */
    double texture, clarity, dehaze;
    /* effects */
    double grain_amount, grain_size, grain_color, grain_bias;
    double vignette_amount;
    double bloom_amount, bloom_radius, bloom_threshold;
    /* lens correction */
    double lens_distortion;
    double lens_tca_r, lens_tca_b;
    double lens_vignette_corr;
    double lens_k1, lens_k2, lens_k3;
    double lens_ca_r, lens_ca_b;
    double lens_keystone_h, lens_keystone_v;
    double lens_rotate;
    int lens_enable;
    /* sharpening & noise reduction */
    double sharpening_amount, sharpening_radius, sharpening_detail, sharpening_masking;
    double nr_luminance, nr_color;
    /* curves */
    double curve_highlights, curve_lights, curve_darks, curve_shadows;
    /* transform */
    double crop_x, crop_y, crop_w, crop_h;
    int has_crop;
    double straighten_angle;
    /* pixel art */
    double pixel_size;
} FeSettings;

typedef struct {
    char *key;
    char *category_dir;
    char *category_label;
    char *label;
    char *description;
    char *colour;
    FeSettings settings;
    /* effects */
    int has_ca, has_scanlines, has_vcr;
    double ca_shift, scanlines_intensity;
} FeFilter;

FeSettings fe_settings_default(void);
void fe_img_free(FeImage *img);
FeImage fe_img_copy(const FeImage *src);
FeImage fe_img_new(int w, int h);

#endif
