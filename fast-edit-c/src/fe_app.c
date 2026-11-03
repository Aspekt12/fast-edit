#include "fe_app.h"
#include "fe_io.h"
#include "fe_math.h"
#include "fe_overlay.h"
#include "fe_pipeline.h"

#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <unistd.h>

/* Join "dir/name" into out, refusing (returning 0) when it would not fit.
   Done by hand rather than with snprintf so the compiler can prove the
   result never overflows and stays quiet about it. */
static int fe_path_join(char *out, size_t out_len, const char *dir, const char *name) {
    size_t dl = strlen(dir), nl = strlen(name);
    if (dl + 1 + nl + 1 > out_len) return 0;
    memcpy(out, dir, dl);
    out[dl] = '/';
    memcpy(out + dl + 1, name, nl + 1);
    return 1;
}

/* A folder qualifies as a filters root when it holds at least one
   subfolder (each subfolder is a category). */
static int fe_is_filters_root(const char *dir) {
    DIR *d = opendir(dir);
    if (!d) return 0;
    struct dirent *ent;
    int ok = 0;
    while ((ent = readdir(d)) != NULL) {
        if (ent->d_name[0] == '.') continue;
        char sub[FE_PATH_MAX];
        if (!fe_path_join(sub, sizeof(sub), dir, ent->d_name)) continue;
        struct stat st;
        if (stat(sub, &st) == 0 && S_ISDIR(st.st_mode)) { ok = 1; break; }
    }
    closedir(d);
    return ok;
}

/* Where does this binary live?  Used to find the filter folder no matter
   which directory the user launched the editor from. */
static void fe_exe_dir(char *out, size_t out_len) {
    out[0] = '\0';
    ssize_t n = readlink("/proc/self/exe", out, out_len - 1);
    if (n <= 0) { out[0] = '\0'; return; }
    out[n] = '\0';
    char *slash = strrchr(out, '/');
    if (slash) *slash = '\0';   /* keep the directory, drop the binary name */
    else out[0] = '\0';
}

/* Find the folder that holds the filter presets.

   The presets live in ONE place — <repo>/filters — and the Python editor
   (fast-edit-v2.py) reaches them through
   Path(__file__).resolve().parent / "filters".  The C editor lives one
   level down in fast-edit-c/, so it looks for the same folder relative to
   its own binary first (build/../../filters), and only then falls back to
   the working directory.  That way "make run", "./build/fast-edit" from the
   repository root and a double-clicked binary all find the SAME presets. */
void fe_app_find_filters_root(char *out, size_t out_len) {
    const char *env = getenv("FE_FILTERS");
    if (env && *env) {
        snprintf(out, out_len, "%s", env);
        return;
    }

    char candidates[8][FE_PATH_MAX];
    int n = 0;

    /* 1. relative to the running binary */
    char exedir[FE_PATH_MAX];
    fe_exe_dir(exedir, sizeof(exedir));
    if (exedir[0]) {
        snprintf(candidates[n++], FE_PATH_MAX, "%s/filters", exedir);
        snprintf(candidates[n++], FE_PATH_MAX, "%s/../../filters", exedir);
        snprintf(candidates[n++], FE_PATH_MAX, "%s/../filters", exedir);
    }

    /* 2. relative to the working directory */
    snprintf(candidates[n++], FE_PATH_MAX, "filters");
    snprintf(candidates[n++], FE_PATH_MAX, "../filters");

    for (int i = 0; i < n; i++) {
        if (fe_is_filters_root(candidates[i])) {
            snprintf(out, out_len, "%s", candidates[i]);
            return;
        }
    }
    snprintf(out, out_len, "filters");
}

int fe_app_init(FeApp *app) {
    memset(app, 0, sizeof(*app));
    fe_app_find_filters_root(app->filter_root, sizeof(app->filter_root));
    int n = fe_filters_load(app->filter_root, &app->filters);
    if (n == 0) {
        printf("no filters found in %s\n", app->filter_root);
        return 0;
    }
    return n;
}

int fe_app_load(FeApp *app, const char *path) {
    FeImage img;
    char date[64] = "";
    if (!fe_load_image(path, PREVIEW_LONG_EDGE, &img, date, sizeof(date))) return 0;
    if (app->has_image) {
        fe_img_free(&app->source_linear);
        fe_img_free(&app->full_linear);
    }
    app->source_linear = img;
    app->full_loaded = 0;
    app->has_image = 1;
    snprintf(app->file_path, sizeof(app->file_path), "%s", path);
    snprintf(app->camera_date, sizeof(app->camera_date), "%s", date);
    return 1;
}

unsigned char *fe_app_render(FeApp *app, const FeFilter *flt, const FeSettings *settings,
                              int *w_out, int *h_out, int dither) {
    if (!app->has_image) return NULL;
    return fe_app_render_from(&app->source_linear, flt, settings, w_out, h_out,
                              dither, app->camera_date);
}

/* The workhorse: develop `src` with `settings`, run the filter's analog-damage
   chain and quantise to 8 bit.  Kept separate from fe_app_render() so the GUI
   can render a DRAFT (downscaled) source through exactly the same code. */
unsigned char *fe_app_render_from(const FeImage *src, const FeFilter *flt,
                                   const FeSettings *settings, int *w_out, int *h_out,
                                   int dither, const char *date) {
    FeImage out;
    memset(&out, 0, sizeof(out));
    fe_develop(src, settings, &out);

    /* the analog-damage chain runs AFTER develop, in float */
    double px = (double)(out.w > out.h ? out.w : out.h) / (double)PREVIEW_LONG_EDGE;
    if (flt && flt->has_ca)
        fe_chromatic_aberration(&out, flt->ca_shift * px);
    if (flt && flt->has_scanlines)
        fe_scanlines(&out, flt->scanlines_intensity);

    size_t bytes = (size_t)out.w * out.h * 3;
    unsigned char *rgb = (unsigned char *)malloc(bytes);
    fe_quantize_u8(&out, rgb, dither);
    if (flt && flt->has_vcr)
        fe_vcr_overlay(rgb, out.w, out.h, date ? date : "");
    if (w_out) *w_out = out.w;
    if (h_out) *h_out = out.h;
    fe_img_free(&out);
    return rgb;
}

int fe_app_save(FeApp *app, const char *path, const FeFilter *flt,
                const FeSettings *settings, int fullres) {
    if (!app->has_image) return 0;

    /* Full-res export: decode the original again at full resolution. */
    if (fullres && !app->full_loaded) {
        FeImage full;
        char date[64] = "";
        if (fe_load_image(app->file_path, 0, &full, date, sizeof(date))) {
            if (app->full_loaded) fe_img_free(&app->full_linear);
            app->full_linear = full;
            app->full_loaded = 1;
        }
    }

    FeImage source = app->source_linear;
    if (fullres && app->full_loaded) source = app->full_linear;

    FeImage out;
    memset(&out, 0, sizeof(out));
    fe_develop(&source, settings, &out);

    double px = (double)(out.w > out.h ? out.w : out.h) / (double)PREVIEW_LONG_EDGE;
    if (flt && flt->has_ca) fe_chromatic_aberration(&out, flt->ca_shift * px);
    if (flt && flt->has_scanlines) fe_scanlines(&out, flt->scanlines_intensity);

    size_t bytes = (size_t)out.w * out.h * 3;
    unsigned char *rgb = (unsigned char *)malloc(bytes);
    /* dither on export only: a saved file is viewed at 100%, where bands show */
    fe_quantize_u8(&out, rgb, 1);
    if (flt && flt->has_vcr) fe_vcr_overlay(rgb, out.w, out.h, app->camera_date);

    int ok;
    if (strcasecmp(path + strlen(path) - 4, ".jpg") == 0 ||
        strcasecmp(path + strlen(path) - 5, ".jpeg") == 0)
        ok = fe_save_jpg(path, rgb, out.w, out.h, 98);
    else
        ok = fe_save_png(path, rgb, out.w, out.h);

    free(rgb);
    fe_img_free(&out);
    return ok;
}

void fe_app_free(FeApp *app) {
    if (app->has_image) {
        fe_img_free(&app->source_linear);
        if (app->full_loaded) fe_img_free(&app->full_linear);
    }
    fe_filters_free(&app->filters);
}
