#ifndef FE_OVERLAY_H
#define FE_OVERLAY_H

#include "fe_defs.h"

/* Fake lateral CA: pull red and blue apart horizontally (red left, blue right). */
void fe_chromatic_aberration(FeImage *img, double shift);
/* Tape interlace lines: every third row slightly darkened. */
void fe_scanlines(FeImage *img, double intensity);
/* The vintage camcorder stamp: "PLAY >" plus a date, drawn in ember tones. */
void fe_vcr_overlay(unsigned char *rgb, int w, int h, const char *date_str);

#endif
