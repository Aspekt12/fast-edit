#include "fe_defs.h"
#include <stdlib.h>
#include <string.h>

const double MIXER_BAND_CENTRES[8] = {0.0, 30.0, 60.0, 120.0, 180.0, 240.0, 285.0, 330.0};
const char *MIXER_BAND_NAMES[8] = {"red","orange","yellow","green","aqua","blue","purple","magenta"};

FeSettings fe_settings_default(void) {
    FeSettings s = {0};
    s.cal_red_sat = s.cal_green_sat = s.cal_blue_sat = 1.0;
    s.grain_size = 1.2;
    s.grain_color = 0.02;
    s.grain_bias = 100.0;
    s.grade_blending = 0.45;
    s.bloom_radius = 25.0;
    s.bloom_threshold = 0.72;
    s.saturation = 1.0;
    s.vibrance = 1.0;
    return s;
}

FeImage fe_img_new(int w, int h) {
    FeImage img;
    img.w = w; img.h = h;
    img.data = (float *)calloc((size_t)w * h * 3, sizeof(float));
    return img;
}

FeImage fe_img_copy(const FeImage *src) {
    FeImage img = fe_img_new(src->w, src->h);
    memcpy(img.data, src->data, (size_t)src->w * src->h * 3 * sizeof(float));
    return img;
}

void fe_img_free(FeImage *img) {
    if (img && img->data) {
        free(img->data);
        img->data = NULL;
    }
}
