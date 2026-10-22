#ifndef FE_APP_H
#define FE_APP_H

#include "fe_defs.h"
#include "fe_filters.h"

#define FE_PATH_MAX 4096

typedef struct {
    FeFilterSet filters;
    char filter_root[FE_PATH_MAX];

    FeImage source_linear;   /* working-resolution linear RGB (preview) */
    FeImage full_linear;     /* full-resolution linear RGB (lazy)      */
    int full_loaded;
    int has_image;

    char file_path[FE_PATH_MAX];
    char camera_date[64];
} FeApp;

/* Locate the filters folder: ./filters, ../filters or $FE_FILTERS. */
void fe_app_find_filters_root(char *out, size_t out_len);
int  fe_app_init(FeApp *app);
int  fe_app_load(FeApp *app, const char *path);

/* Develop the current source with `settings` + the filter's effect chain
   and return an 8-bit RGB buffer (caller frees). */
unsigned char *fe_app_render(FeApp *app, const FeFilter *flt, const FeSettings *settings,
                              int *w_out, int *h_out, int dither);

/* Same, but from an arbitrary source image (the GUI uses it for the draft
   preview, which is the same photo downscaled). */
unsigned char *fe_app_render_from(const FeImage *src, const FeFilter *flt,
                                   const FeSettings *settings, int *w_out, int *h_out,
                                   int dither, const char *date);

/* Write the current look to disk (PNG or JPEG by extension). */
int fe_app_save(FeApp *app, const char *path, const FeFilter *flt,
                const FeSettings *settings, int fullres);

void fe_app_free(FeApp *app);

#endif
