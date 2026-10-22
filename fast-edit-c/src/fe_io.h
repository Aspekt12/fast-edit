#ifndef FE_IO_H
#define FE_IO_H

#include "fe_defs.h"

#include <stdio.h>

/* Decode a RAW/ARW or an ordinary image into LINEAR float RGB.
   `max_edge` > 0 downscales the long edge (the preview path); 0 keeps the
   full resolution (the export path).  `date_out` receives the VCR-style
   timestamp, "" when unknown. */
int fe_load_image(const char *path, int max_edge, FeImage *out,
                  char *date_out, size_t date_len);

int fe_save_png(const char *path, const unsigned char *rgb, int w, int h);
int fe_save_jpg(const char *path, const unsigned char *rgb, int w, int h, int quality);

/* Decode a fast 8-bit RGB preview for the browser.  For a RAW file this
   pulls the camera's embedded JPEG thumbnail out of the file — a few
   milliseconds instead of a full demosaic pass, which is what makes a
   film-strip of previews possible.  Returns 1 on success; *rgb is a
   malloc'd buffer (free it with free()) and *w and *h are its size. */
int fe_load_thumb_rgb(const char *path, int max_edge,
                      unsigned char **rgb, int *w, int *h);

/* 16-bit PPM (netpbm P6, big endian) — for lossless comparisons. */
int fe_save_ppm16(const char *path, const unsigned short *rgb, int w, int h);

/* "OCT. 20 2026  PM 11:35" — the VCR stamp's date, from EXIF when present. */
void fe_read_exif_date(const char *path, char *out, size_t out_len);

/* Read a binary PPM (netpbm P6) from an open stream.  Exposed for the
   self-tests: the byte order of 16-bit samples is the sort of thing that
   silently ruins every RAW preview, so it needs a test of its own. */
int fe_read_ppm(FILE *f, FeImage *out, int linear_input);

#endif
