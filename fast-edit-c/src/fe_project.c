#include "fe_project.h"
#include "fe_io.h"
#include "fe_math.h"
#include "fe_optimize.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>

#ifdef _OPENMP
#include <omp.h>
#endif

FeProject *fe_project_new(void) {
    FeProject *proj = (FeProject *)calloc(1, sizeof(FeProject));
    proj->capacity = 64;
    proj->items = (FeProjectItem *)calloc(proj->capacity, sizeof(FeProjectItem));
    proj->export_dir = strdup("./export");
    proj->export_prefix = strdup("");
    proj->export_format = strdup("png");
    proj->export_quality = 95;
    proj->use_dither = 0;
    proj->active_filter_key = strdup("00-basics/original");
    return proj;
}

void fe_project_free(FeProject *proj) {
    if (!proj) return;
    for (int i = 0; i < proj->count; i++) {
        free(proj->items[i].path);
        free(proj->items[i].thumbnail);
        free(proj->items[i].name);
        if (proj->items[i].thumb_img) { fe_img_free(proj->items[i].thumb_img); free(proj->items[i].thumb_img); }
        if (proj->items[i].has_source) fe_img_free(&proj->items[i].source_linear);
    }
    free(proj->items);
    free(proj->project_path);
    free(proj->export_dir);
    free(proj->export_prefix);
    free(proj->export_format);
    free(proj->active_filter_key);
    free(proj);
}

static const char *fe_basename(const char *path) {
    const char *p = strrchr(path, '/');
    return p ? p + 1 : path;
}

int fe_project_add(FeProject *proj, const char *path) {
    if (!proj || !path) return 0;
    /* Check if already added */
    for (int i = 0; i < proj->count; i++) {
        if (strcmp(proj->items[i].path, path) == 0) return 1;
    }
    if (proj->count >= proj->capacity) {
        proj->capacity *= 2;
        proj->items = (FeProjectItem *)realloc(proj->items, proj->capacity * sizeof(FeProjectItem));
    }
    FeProjectItem *it = &proj->items[proj->count++];
    memset(it, 0, sizeof(FeProjectItem));
    it->path = strdup(path);
    it->name = strdup(fe_basename(path));
    it->selected = 1;
    it->processed = 0;
    return 1;
}

int fe_project_add_many(FeProject *proj, const char **paths, int count) {
    int added = 0;
    for (int i = 0; i < count; i++) {
        if (fe_project_add(proj, paths[i])) added++;
    }
    return added;
}

void fe_project_select_all(FeProject *proj, int selected) {
    if (!proj) return;
    for (int i = 0; i < proj->count; i++) proj->items[i].selected = selected;
}

void fe_project_select_none(FeProject *proj) { fe_project_select_all(proj, 0); }
void fe_project_select_invert(FeProject *proj) {
    if (!proj) return;
    for (int i = 0; i < proj->count; i++) proj->items[i].selected = !proj->items[i].selected;
}

int fe_project_selected_count(FeProject *proj) {
    if (!proj) return 0;
    int n = 0;
    for (int i = 0; i < proj->count; i++) if (proj->items[i].selected) n++;
    return n;
}

void fe_project_remove_selected(FeProject *proj) {
    if (!proj) return;
    int w = 0;
    for (int i = 0; i < proj->count; i++) {
        if (proj->items[i].selected) {
            free(proj->items[i].path);
            free(proj->items[i].thumbnail);
            free(proj->items[i].name);
            if (proj->items[i].thumb_img) { fe_img_free(proj->items[i].thumb_img); free(proj->items[i].thumb_img); }
            if (proj->items[i].has_source) fe_img_free(&proj->items[i].source_linear);
            continue;
        }
        proj->items[w++] = proj->items[i];
    }
    proj->count = w;
}

int fe_project_generate_thumbnails(FeProject *proj, int thumb_size) {
    if (!proj || proj->count == 0) return 0;
    int nt = fe_thread_count();
    if (nt < 1) nt = 1;
    
    /* Multi-threaded thumbnail generation */
    #pragma omp parallel for num_threads(nt) schedule(dynamic)
    for (int i = 0; i < proj->count; i++) {
        FeProjectItem *it = &proj->items[i];
        if (it->thumb_img) { fe_img_free(it->thumb_img); free(it->thumb_img); it->thumb_img = NULL; }
        if (it->has_source) fe_img_free(&it->source_linear);
    }
    return proj->count;
}

/* Thread-safe export context */
typedef struct {
    FeProject *proj;
    FeApp *app;
    const FeFilter *flt;
    int start_idx;
    int end_idx;
    int use_dither;
    const char *export_dir;
    const char *prefix;
    const char *fmt;
    int quality;
    int *success_count;
    int *fail_count;
    FeProjectProgressFn progress;
    void *user;
    int *next_idx; /* atomic counter */
#ifdef _OPENMP
    omp_lock_t lock;
#endif
} ExportCtx;

static void export_worker(void *ctxv) {
    ExportCtx *ctx = (ExportCtx *)ctxv;
    for (int i = ctx->start_idx; i < ctx->end_idx; i++) {
        FeProjectItem *it = &ctx->proj->items[i];
        if (!it->selected) continue;
        
        /* Load image */
        FeApp tmp_app;
        memset(&tmp_app, 0, sizeof(tmp_app));
        /* Use a copy of filter list */
        tmp_app.filters = ctx->app->filters;
        tmp_app.camera_date[0] = '\0';
        
        if (!fe_app_load(&tmp_app, it->path)) {
#ifdef _OPENMP
            omp_set_lock(&ctx->lock);
#endif
            (*ctx->fail_count)++;
#ifdef _OPENMP
            omp_unset_lock(&ctx->lock);
#endif
            continue;
        }
        
        int w = 0, h = 0;
        unsigned char *rgb = fe_app_render(&tmp_app, ctx->flt, &ctx->flt->settings, &w, &h, ctx->use_dither);
        
        if (!rgb) {
            fe_app_free(&tmp_app);
#ifdef _OPENMP
            omp_set_lock(&ctx->lock);
#endif
            (*ctx->fail_count)++;
#ifdef _OPENMP
            omp_unset_lock(&ctx->lock);
#endif
            continue;
        }
        
        /* Build output filename */
        char outpath[1024];
        const char *base = fe_basename(it->path);
        char stem[512];
        strncpy(stem, base, 511);
        stem[511] = '\0';
        /* strip extension */
        char *dot = strrchr(stem, '.');
        if (dot) *dot = '\0';
        
        snprintf(outpath, sizeof(outpath), "%s/%s%s.%s", ctx->export_dir, ctx->prefix, stem, ctx->fmt);
        
        /* Ensure directory exists */
        mkdir(ctx->export_dir, 0755);
        
        int ok = 0;
        if (strcmp(ctx->fmt, "jpg") == 0 || strcmp(ctx->fmt, "jpeg") == 0) {
            ok = fe_save_jpg(outpath, rgb, w, h, ctx->quality);
        } else {
            ok = fe_save_png(outpath, rgb, w, h);
        }
        
        free(rgb);
        fe_app_free(&tmp_app);
        
#ifdef _OPENMP
        omp_set_lock(&ctx->lock);
#endif
        if (ok) (*ctx->success_count)++; else (*ctx->fail_count)++;
        int done = (*ctx->success_count) + (*ctx->fail_count);
        int total = ctx->proj->count;
        if (ctx->progress) ctx->progress(done, total, outpath, ctx->user);
#ifdef _OPENMP
        omp_unset_lock(&ctx->lock);
#endif
    }
}

int fe_project_export(FeProject *proj, FeApp *app, const char *filter_key,
                      FeProjectProgressFn progress, void *user) {
    if (!proj || !app || proj->count == 0) return 0;
    
    const FeFilter *flt = fe_filters_find(&app->filters, filter_key);
    if (!flt) flt = fe_filters_find(&app->filters, proj->active_filter_key);
    if (!flt) return 0;
    
    mkdir(proj->export_dir, 0755);
    
    int success = 0, fail = 0;
    int nt = fe_thread_count();
    if (nt < 1) nt = 1;
    /* Be a bit conservative with memory for large batches */
    if (nt > 4) nt = 4;
    
    ExportCtx ctx = {
        .proj = proj, .app = app, .flt = flt,
        .use_dither = proj->use_dither,
        .export_dir = proj->export_dir,
        .prefix = proj->export_prefix,
        .fmt = proj->export_format,
        .quality = proj->export_quality,
        .success_count = &success,
        .fail_count = &fail,
        .progress = progress,
        .user = user
    };
    
#ifdef _OPENMP
    omp_init_lock(&ctx.lock);
#endif
    
    int per_thread = proj->count / nt;
    if (per_thread < 1) per_thread = 1;
    
#pragma omp parallel for num_threads(nt) schedule(dynamic)
    for (int t = 0; t < nt; t++) {
        ExportCtx local = ctx;
        local.start_idx = t * per_thread;
        local.end_idx = local.start_idx + per_thread;
        if (t == nt - 1) local.end_idx = proj->count;
        if (local.start_idx < proj->count) {
            export_worker(&local);
        }
    }
    
#ifdef _OPENMP
    omp_destroy_lock(&ctx.lock);
#endif
    
    printf("Mass export complete: %d success, %d failed\n", success, fail);
    return success;
}

int fe_project_save(FeProject *proj, const char *path) {
    /* Simple JSON-like format for now - can use cJSON */
    FILE *f = fopen(path, "w");
    if (!f) return 0;
    fprintf(f, "{\"version\":1,\"items\":[");
    for (int i = 0; i < proj->count; i++) {
        if (i > 0) fprintf(f, ",");
        fprintf(f, "{\"path\":\"%s\",\"selected\":%d}", proj->items[i].path, proj->items[i].selected);
    }
    fprintf(f, "],\"export_dir\":\"%s\",\"filter\":\"%s\",\"format\":\"%s\"}\n",
            proj->export_dir, proj->active_filter_key, proj->export_format);
    fclose(f);
    return 1;
}

int fe_project_load(FeProject *proj, const char *path) {
    /* Placeholder - would use cJSON */
    (void)proj; (void)path;
    return 0;
}
