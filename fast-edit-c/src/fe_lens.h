#ifndef FE_LENS_H
#define FE_LENS_H

#include "fe_defs.h"

/* Lens correction - quality preserving
   Implements Brown-Conrady distortion model, TCA correction,
   vignetting correction, and perspective correction.
   All operations use high-quality resampling to avoid quality loss. */

void fe_lens_distortion_correct(FeImage *img, double k1, double k2, double k3);
void fe_lens_tca_correct(FeImage *img, double r_shift, double b_shift);
void fe_lens_vignette_correct(FeImage *img, double amount);
void fe_lens_perspective_correct(FeImage *img, double keystone_h, double keystone_v);
void fe_lens_auto_correct(FeImage *img, const FeSettings *s);

#endif
