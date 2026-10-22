#ifndef FE_PIPELINE_H
#define FE_PIPELINE_H

#include "fe_defs.h"

void fe_develop(const FeImage *linear_rgb, const FeSettings *s, FeImage *out);
double fe_auto_exposure(const FeImage *linear_rgb);
void fe_apply_lut(FeImage *img, const float *lut, int lutn);
void fe_quantize_u8(const FeImage *img, unsigned char *out_rgb, int dither);

#endif
