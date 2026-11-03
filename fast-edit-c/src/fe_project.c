#include "fe_fields.h"
#include "fe_filters.h"
#include "fe_io.h"
#include "fe_optimize.h"
#include "fe_overlay.h"
#include "fe_pipeline.h"
#include "fe_project.h"
#include "fe_state.h"

#include "../vendor/cJSON.h"

#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <unistd.h>

#ifdef _OPENMP
#include <omp.h>
#endif

int fe_is_raw_image_file(const char *name) {
    if (!name) return 0;
    const char *ext = strrchr(name, '.');
    if (!ext) return 0;
    return strcasecmp(ext, ".arw") == 0 || strcasecmp(ext, ".raw") == 0 ||
           strcasecmp(ext, ".cr2") == 0 || strcasecmp(ext, ".nef") == 0 ||
           strcasecmp(ext, ".dng") == 0 || strcasecmp(ext, ".raf") == 0 ||
           strcasecmp(ext, ".orf") == 0 || strcasecmp(ext, ".rw2") == 0;
}

int fe_is_supported_image_file(const char *name) {
    if (!name) return 0;
    if (fe_is_raw_image_file(name)) return 1;
    const char *ext = strrchr(name, '.');
    if (!ext) return 0;
    return strcasecmp(ext, ".jpg") == 0 || strcasecmp(ext, ".jpeg") == 0 ||
           strcasecmp(ext, ".png") == 0 || strcasecmp(ext, ".tif") == 0 ||
           strcasecmp(ext, ".tiff") == 0;
}

FeProject *fe_project_new(void) {
    FeProject *proj = (FeProject *)calloc(1, sizeof(FeProject));
    proj->capacity = 64;
    proj->items = (FeProjectItem *)calloc(proj->capacity, sizeof(FeProjectItem));
    proj->export_dir = strdup("./export");
    proj->export_prefix = strdup("");
    proj->export_format = strdup("png");
    proj->export_quality = 95;
    proj->use_dither = 1;
    proj->fullres = 1;   /* quality first: batch exports re-decode at full size */
    proj->active_filter_key = strdup("00-basics/original");
    return proj;
}

static void fe_project_item_free(FeProjectItem *it) {
    if (!it) return;
    free(it->path);
    free(it->thumbnail);
    free(it->name);
    if (it->thumb_img) { fe_img_free(it->thumb_img); free(it->thumb_img); }
    if (it->has_source) fe_img_free(&it->source_linear);
    memset(it, 0, sizeof(*it));
}

void fe_project_clear(FeProject *proj) {
    if (!proj) return;
    for (int i = 0; i < proj->count; i++) {
        fe_project_item_free(&proj->items[i]);
    }
    proj->count = 0;
}

void fe_project_free(FeProject *proj) {
    if (!proj) return;
    fe_project_clear(proj);
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

int fe_project_find(const FeProject *proj, const char *path) {
    if (!proj || !path) return -1;
    for (int i = 0; i < proj->count; i++) {
        if (strcmp(proj->items[i].path, path) == 0) return i;
    }
    return -1;
}

int fe_project_add(FeProject *proj, const char *path) {
    if (!proj || !path || !path[0]) return 0;
    /* Check if already added */
    if (fe_project_find(proj, path) >= 0) return 0;
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

static int cmp_str_case(const void *a, const void *b) {
    const char *sa = *(const char *const *)a;
    const char *sb = *(const char *const *)b;
    int c = strcasecmp(sa, sb);
    return c != 0 ? c : strcmp(sa, sb);
}

int fe_project_add_folder(FeProject *proj, const char *folder_path) {
    if (!proj || !folder_path || !folder_path[0]) return 0;
    DIR *d = opendir(folder_path);
    if (!d) return 0;

    char **names = NULL;
    int n = 0, cap = 0;
    int has_raw = 0;
    struct dirent *ent;
    while ((ent = readdir(d)) != NULL) {
        if (ent->d_name[0] == '.') continue;
        if (!fe_is_supported_image_file(ent->d_name)) continue;
        char full[FE_PATH_MAX];
        snprintf(full, sizeof(full), "%s/%s", folder_path, ent->d_name);
        struct stat st;
        if (stat(full, &st) != 0 || !S_ISREG(st.st_mode)) continue;
        if (fe_is_raw_image_file(ent->d_name)) has_raw = 1;
        if (n == cap) {
            cap = cap ? cap * 2 : 32;
            names = (char **)realloc(names, (size_t)cap * sizeof(char *));
        }
        names[n++] = strdup(ent->d_name);
    }
    closedir(d);
    if (n == 0) return 0;

    qsort(names, (size_t)n, sizeof(char *), cmp_str_case);

    int added = 0;
    for (int i = 0; i < n; i++) {
        /* When a camera folder contains RAWs alongside in-camera JPEGs,
           prefer the RAW files so every shot appears once. */
        if (!has_raw || fe_is_raw_image_file(names[i])) {
            char full[FE_PATH_MAX];
            snprintf(full, sizeof(full), "%s/%s", folder_path, names[i]);
            if (fe_project_add(proj, full)) added++;
        }
        free(names[i]);
    }
    free(names);
    return added;
}

void fe_project_select_all(FeProject *proj, int selected) {
    if (!proj) return;
    for (int i = 0; i < proj->count; i++) proj->items[i].selected = selected ? 1 : 0;
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
            fe_project_item_free(&proj->items[i]);
            continue;
        }
        proj->items[w++] = proj->items[i];
    }
    proj->count = w;
}

int fe_project_generate_thumbnails(FeProject *proj, int thumb_size) {
    (void)thumb_size;
    if (!proj || proj->count == 0) return 0;
    int nt = fe_thread_count();
    if (nt < 1) nt = 1;

    #pragma omp parallel for num_threads(nt) schedule(dynamic)
    for (int i = 0; i < proj->count; i++) {
        FeProjectItem *it = &proj->items[i];
        if (it->thumb_img) { fe_img_free(it->thumb_img); free(it->thumb_img); it->thumb_img = NULL; }
        if (it->has_source) { fe_img_free(&it->source_linear); it->has_source = 0; }
    }
    return proj->count;
}

int fe_project_export(FeProject *proj, FeApp *app, const char *filter_key,
                      FeProjectProgressFn progress, void *user) {
    if (!proj || !app || proj->count == 0) return 0;
    int total_selected = fe_project_selected_count(proj);
    if (total_selected == 0) return 0;

    const FeFilter *default_flt = fe_filters_find(&app->filters, filter_key ? filter_key : "");
    if (!default_flt) default_flt = fe_filters_find(&app->filters, proj->active_filter_key);
    if (!default_flt) default_flt = fe_filters_find(&app->filters, "00-basics/original");
    if (!default_flt) return 0;

    mkdir(proj->export_dir, 0755);

    int success = 0, fail = 0;
    int nt = fe_thread_count();
    if (nt < 1) nt = 1;
    /* A full-resolution decode of a 24-60 MP frame takes significant RAM,
       so keep concurrency modest when decoding at full resolution. */
    if (nt > (proj->fullres ? 2 : 4)) nt = proj->fullres ? 2 : 4;

#ifdef _OPENMP
    omp_lock_t lock;
    omp_init_lock(&lock);
#endif

    #pragma omp parallel for num_threads(nt) schedule(dynamic)
    for (int i = 0; i < proj->count; i++) {
        FeProjectItem *it = &proj->items[i];
        if (!it->selected) continue;

        FeImage src;
        memset(&src, 0, sizeof(src));
        char date[64] = "";
        if (!fe_load_image(it->path, proj->fullres ? 0 : PREVIEW_LONG_EDGE,
                           &src, date, sizeof(date))) {
#ifdef _OPENMP
            omp_set_lock(&lock);
#endif
            fail++;
#ifdef _OPENMP
            omp_unset_lock(&lock);
#endif
            continue;
        }

        /* Every photo uses its own saved preset and slider offsets from its
           sidecar (<photo>.fastedit.json).  If the photo has never been edited,
           fall back to the batch default filter. */
        const FeFilter *use = default_flt;
        FeSettings settings = use->settings;
        FeEditState st;
        memset(&st, 0, sizeof(st));
        if (fe_state_load(it->path, &st)) {
            if (st.filter_key[0]) {
                const FeFilter *own = fe_filters_find(&app->filters, st.filter_key);
                if (own) use = own;
            }
            settings = use->settings;
            for (int k = 0; k < st.nfields; k++) {
                fe_settings_apply_offset(&settings, st.fields[k].name, st.fields[k].value);
            }
        }

        FeImage out;
        memset(&out, 0, sizeof(out));
        fe_develop(&src, &settings, &out);

        double px = (double)(out.w > out.h ? out.w : out.h) / (double)PREVIEW_LONG_EDGE;
        if (use->has_ca) fe_chromatic_aberration(&out, use->ca_shift * px);
        if (use->has_scanlines) fe_scanlines(&out, use->scanlines_intensity);

        unsigned char *rgb = (unsigned char *)malloc((size_t)out.w * out.h * 3);
        fe_quantize_u8(&out, rgb, proj->use_dither);
        if (use->has_vcr) fe_vcr_overlay(rgb, out.w, out.h, date);

        char outpath[1024];
        char stem[512];
        const char *base = fe_basename(it->path);
        snprintf(stem, sizeof(stem), "%s", base);
        char *dot = strrchr(stem, '.');
        if (dot) *dot = '\0';
        snprintf(outpath, sizeof(outpath), "%s/%s%s.%s", proj->export_dir,
                 proj->export_prefix ? proj->export_prefix : "", stem, proj->export_format);

        int ok = (strcasecmp(proj->export_format, "jpg") == 0 ||
                  strcasecmp(proj->export_format, "jpeg") == 0)
                     ? fe_save_jpg(outpath, rgb, out.w, out.h, proj->export_quality)
                     : fe_save_png(outpath, rgb, out.w, out.h);

        free(rgb);
        fe_img_free(&out);
        fe_img_free(&src);

#ifdef _OPENMP
        omp_set_lock(&lock);
#endif
        if (ok) {
            success++;
            it->processed = 1;
        } else {
            fail++;
        }
        int done = success + fail;
        if (progress) progress(done, total_selected, outpath, user);
#ifdef _OPENMP
        omp_unset_lock(&lock);
#endif
    }

#ifdef _OPENMP
    omp_destroy_lock(&lock);
#endif

    printf("Mass export complete: %d success, %d failed\n", success, fail);
    return success;
}

/* Give every SELECTED photo the chosen filter while preserving any per-photo
   slider adjustments (e.g. exposure or white balance tweaks) that photo
   already had in its sidecar. */
int fe_project_apply_filter(FeProject *proj, FeApp *app, const char *filter_key) {
    if (!proj || !app || !filter_key) return 0;
    const FeFilter *flt = fe_filters_find(&app->filters, filter_key);
    if (!flt) return 0;

    int done = 0;
    for (int i = 0; i < proj->count; i++) {
        FeProjectItem *it = &proj->items[i];
        if (!it->selected) continue;
        FeEditState st;
        memset(&st, 0, sizeof(st));
        /* Load existing sidecar if present so per-photo slider tweaks survive */
        fe_state_load(it->path, &st);
        snprintf(st.filter_key, sizeof(st.filter_key), "%s", flt->key);
        if (fe_state_save(it->path, &st)) done++;
    }

    free(proj->active_filter_key);
    proj->active_filter_key = strdup(flt->key);
    return done;
}

/* Copy a complete edit state (filter + all slider offsets) to all selected
   photos in the project. */
int fe_project_sync_state(FeProject *proj, FeApp *app, const FeEditState *src_state) {
    if (!proj || !app || !src_state) return 0;
    int done = 0;
    for (int i = 0; i < proj->count; i++) {
        FeProjectItem *it = &proj->items[i];
        if (!it->selected) continue;
        if (fe_state_save(it->path, src_state)) done++;
    }
    if (src_state->filter_key[0]) {
        free(proj->active_filter_key);
        proj->active_filter_key = strdup(src_state->filter_key);
    }
    return done;
}

/* Reset all selected photos in the project back to unedited (removes sidecars). */
int fe_project_reset_selected(FeProject *proj) {
    if (!proj) return 0;
    int done = 0;
    for (int i = 0; i < proj->count; i++) {
        FeProjectItem *it = &proj->items[i];
        if (!it->selected) continue;
        if (fe_state_remove(it->path)) done++;
    }
    return done;
}

int fe_project_save(FeProject *proj, const char *path) {
    if (!proj || !path || !path[0]) return 0;

    cJSON *root = cJSON_CreateObject();
    if (!root) return 0;

    cJSON_AddNumberToObject(root, "version", 1);
    cJSON_AddStringToObject(root, "export_dir", proj->export_dir ? proj->export_dir : "./export");
    cJSON_AddStringToObject(root, "export_prefix", proj->export_prefix ? proj->export_prefix : "");
    cJSON_AddStringToObject(root, "format", proj->export_format ? proj->export_format : "png");
    cJSON_AddNumberToObject(root, "quality", proj->export_quality);
    cJSON_AddNumberToObject(root, "dither", proj->use_dither);
    cJSON_AddNumberToObject(root, "fullres", proj->fullres);
    cJSON_AddStringToObject(root, "filter",
                            proj->active_filter_key ? proj->active_filter_key : "00-basics/original");

    cJSON *items = cJSON_CreateArray();
    for (int i = 0; i < proj->count; i++) {
        cJSON *it = cJSON_CreateObject();
        cJSON_AddStringToObject(it, "path", proj->items[i].path);
        cJSON_AddNumberToObject(it, "selected", proj->items[i].selected);

        /* Embed the photo's current sidecar state inside the project file too,
           so saving a project captures every photo's filter and slider tweaks. */
        FeEditState st;
        memset(&st, 0, sizeof(st));
        if (fe_state_load(proj->items[i].path, &st)) {
            cJSON *st_obj = cJSON_CreateObject();
            cJSON_AddStringToObject(st_obj, "filter", st.filter_key);
            cJSON *offs = cJSON_CreateObject();
            for (int k = 0; k < st.nfields; k++) {
                if (st.fields[k].value == 0.0) continue;
                cJSON_AddNumberToObject(offs, st.fields[k].name, st.fields[k].value);
            }
            cJSON_AddItemToObject(st_obj, "offsets", offs);
            cJSON_AddItemToObject(it, "state", st_obj);
        }
        cJSON_AddItemToArray(items, it);
    }
    cJSON_AddItemToObject(root, "items", items);

    char *text = cJSON_Print(root);
    cJSON_Delete(root);
    if (!text) return 0;

    char tmp[4200];
    snprintf(tmp, sizeof(tmp), "%s.tmp", path);
    FILE *f = fopen(tmp, "w");
    if (!f) {
        free(text);
        return 0;
    }
    fputs(text, f);
    fclose(f);
    free(text);

    if (rename(tmp, path) != 0) {
        remove(tmp);
        return 0;
    }
    free(proj->project_path);
    proj->project_path = strdup(path);
    return 1;
}

int fe_project_load(FeProject *proj, const char *path) {
    if (!proj || !path || !path[0]) return 0;

    FILE *f = fopen(path, "r");
    if (!f) return 0;
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (len <= 0) { fclose(f); return 0; }

    char *buf = (char *)malloc((size_t)len + 1);
    if (!buf) { fclose(f); return 0; }
    size_t got = fread(buf, 1, (size_t)len, f);
    fclose(f);
    buf[got] = '\0';

    cJSON *root = cJSON_Parse(buf);
    free(buf);
    if (!root) return 0;

    fe_project_clear(proj);

    cJSON *edir = cJSON_GetObjectItem(root, "export_dir");
    if (cJSON_IsString(edir) && edir->valuestring) {
        free(proj->export_dir);
        proj->export_dir = strdup(edir->valuestring);
    }
    cJSON *eprefix = cJSON_GetObjectItem(root, "export_prefix");
    if (cJSON_IsString(eprefix) && eprefix->valuestring) {
        free(proj->export_prefix);
        proj->export_prefix = strdup(eprefix->valuestring);
    }
    cJSON *fmt = cJSON_GetObjectItem(root, "format");
    if (cJSON_IsString(fmt) && fmt->valuestring) {
        free(proj->export_format);
        proj->export_format = strdup(fmt->valuestring);
    }
    cJSON *qual = cJSON_GetObjectItem(root, "quality");
    if (cJSON_IsNumber(qual)) proj->export_quality = (int)qual->valuedouble;
    cJSON *dith = cJSON_GetObjectItem(root, "dither");
    if (cJSON_IsNumber(dith)) proj->use_dither = (int)dith->valuedouble;
    cJSON *fres = cJSON_GetObjectItem(root, "fullres");
    if (cJSON_IsNumber(fres)) proj->fullres = (int)fres->valuedouble;
    cJSON *flt = cJSON_GetObjectItem(root, "filter");
    if (cJSON_IsString(flt) && flt->valuestring) {
        free(proj->active_filter_key);
        proj->active_filter_key = strdup(flt->valuestring);
    }

    cJSON *items = cJSON_GetObjectItem(root, "items");
    if (cJSON_IsArray(items)) {
        int n = cJSON_GetArraySize(items);
        for (int i = 0; i < n; i++) {
            cJSON *it = cJSON_GetArrayItem(items, i);
            if (!cJSON_IsObject(it)) continue;
            cJSON *p = cJSON_GetObjectItem(it, "path");
            if (!cJSON_IsString(p) || !p->valuestring) continue;
            fe_project_add(proj, p->valuestring);
            int idx = fe_project_find(proj, p->valuestring);
            if (idx >= 0) {
                cJSON *sel = cJSON_GetObjectItem(it, "selected");
                if (cJSON_IsNumber(sel)) proj->items[idx].selected = (int)sel->valuedouble ? 1 : 0;
            }
            /* If the project file embeds per-photo state and no sidecar exists
               on disk (or to restore the saved project snapshot), sync it. */
            cJSON *st_obj = cJSON_GetObjectItem(it, "state");
            if (cJSON_IsObject(st_obj)) {
                FeEditState existing;
                if (!fe_state_load(p->valuestring, &existing)) {
                    FeEditState st;
                    memset(&st, 0, sizeof(st));
                    cJSON *sflt = cJSON_GetObjectItem(st_obj, "filter");
                    if (cJSON_IsString(sflt) && sflt->valuestring)
                        snprintf(st.filter_key, sizeof(st.filter_key), "%s", sflt->valuestring);
                    cJSON *offs = cJSON_GetObjectItem(st_obj, "offsets");
                    if (cJSON_IsObject(offs)) {
                        int noffs = cJSON_GetArraySize(offs);
                        for (int k = 0; k < noffs; k++) {
                            cJSON *o = cJSON_GetArrayItem(offs, k);
                            if (o && o->string && cJSON_IsNumber(o))
                                fe_state_set(&st, o->string, o->valuedouble);
                        }
                    }
                    fe_state_save(p->valuestring, &st);
                }
            }
        }
    }

    cJSON_Delete(root);
    free(proj->project_path);
    proj->project_path = strdup(path);
    return 1;
}
