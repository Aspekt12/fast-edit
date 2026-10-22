#include "fe_filters.h"

#include <ctype.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "../vendor/cJSON.h"

/* ------------------------------------------------------------------ */
/* small helpers                                                        */
/* ------------------------------------------------------------------ */

static char *fe_strdup(const char *s) {
    size_t n = strlen(s) + 1;
    char *d = (char *)malloc(n);
    memcpy(d, s, n);
    return d;
}

/* Python's str.title(): first letter of every word upper, rest lower. */
static char *fe_title_case(const char *s) {
    size_t n = strlen(s);
    char *out = (char *)malloc(n + 1);
    int start_of_word = 1;
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        if (isalpha(c)) {
            out[i] = (char)(start_of_word ? toupper(c) : tolower(c));
            start_of_word = 0;
        } else {
            out[i] = (char)c;
            start_of_word = 1; /* digits do NOT break a word in Python's title() */
            if (isdigit(c)) start_of_word = 0;
        }
    }
    out[n] = '\0';
    return out;
}

/* "01-sage-muted" -> "Sage Muted" (strip the ordering prefix, title-case) */
static char *fe_category_default_label(const char *dir) {
    const char *name = dir;
    if (isdigit((unsigned char)name[0]) && isdigit((unsigned char)name[1]) && name[2] == '-')
        name += 3;
    size_t n = strlen(name);
    char *tmp = (char *)malloc(n + 1);
    for (size_t i = 0; i < n; i++) tmp[i] = name[i] == '-' ? ' ' : name[i];
    tmp[n] = '\0';
    char *out = fe_title_case(tmp);
    free(tmp);
    return out;
}

/* ------------------------------------------------------------------ */
/* settings parsing                                                     */
/* ------------------------------------------------------------------ */

/* kind 0 = plain double, 1 = (hue, strength) pair, 2 = mixer object */
typedef struct {
    const char *name;
    size_t offset;
    int kind;
} FeSettingRef;

#define FIELD(name) offsetof(FeSettings, name)

static const FeSettingRef SETTING_TABLE[] = {
    {"temperature", FIELD(temperature), 0},
    {"tint", FIELD(tint), 0},
    {"cal_red_hue", FIELD(cal_red_hue), 0},
    {"cal_green_hue", FIELD(cal_green_hue), 0},
    {"cal_blue_hue", FIELD(cal_blue_hue), 0},
    {"cal_red_sat", FIELD(cal_red_sat), 0},
    {"cal_green_sat", FIELD(cal_green_sat), 0},
    {"cal_blue_sat", FIELD(cal_blue_sat), 0},
    {"exposure", FIELD(exposure), 0},
    {"contrast", FIELD(contrast), 0},
    {"highlights", FIELD(highlights), 0},
    {"shadows", FIELD(shadows), 0},
    {"blacks", FIELD(blacks), 0},
    {"whites", FIELD(whites), 0},
    {"fade", FIELD(fade), 0},
    {"mixer", 0, 2},
    {"grade_shadow", FIELD(grade_shadow_h), 1},
    {"grade_midtone", FIELD(grade_midtone_h), 1},
    {"grade_highlight", FIELD(grade_highlight_h), 1},
    {"grade_blending", FIELD(grade_blending), 0},
    {"saturation", FIELD(saturation), 0},
    {"vibrance", FIELD(vibrance), 0},
    {"texture", FIELD(texture), 0},
    {"clarity", FIELD(clarity), 0},
    {"dehaze", FIELD(dehaze), 0},
    {"grain_amount", FIELD(grain_amount), 0},
    {"grain_size", FIELD(grain_size), 0},
    {"grain_color", FIELD(grain_color), 0},
    {"grain_bias", FIELD(grain_bias), 0},
    {"vignette_amount", FIELD(vignette_amount), 0},
    {"bloom_amount", FIELD(bloom_amount), 0},
    {"bloom_radius", FIELD(bloom_radius), 0},
    {"bloom_threshold", FIELD(bloom_threshold), 0},
    {"pixel_size", FIELD(pixel_size), 0},
};
static const int SETTING_COUNT = (int)(sizeof(SETTING_TABLE) / sizeof(SETTING_TABLE[0]));

static int fe_mixer_band_index(const char *name) {
    for (int i = 0; i < MIXER_BAND_COUNT; i++)
        if (strcmp(MIXER_BAND_NAMES[i], name) == 0) return i;
    return -1;
}

/* Returns 0 on success, or a message (caller frees) on failure. */
static char *fe_parse_settings(const cJSON *obj, FeSettings *out) {
    *out = fe_settings_default();
    if (!obj) return NULL;

    const cJSON *item = NULL;
    cJSON_ArrayForEach(item, obj) {
        const char *key = item->string;
        if (!key) continue;
        int found = -1;
        for (int i = 0; i < SETTING_COUNT; i++) {
            if (strcmp(SETTING_TABLE[i].name, key) == 0) { found = i; break; }
        }
        if (found < 0) {
            /* Unknown keys are a hard error: silently ignoring a typo like
               "exposre" would produce a look nobody can reproduce. */
            char *msg = (char *)malloc(256);
            snprintf(msg, 256, "unknown setting: %s", key);
            return msg;
        }
        const FeSettingRef *ref = &SETTING_TABLE[found];
        if (ref->kind == 0) {
            if (!cJSON_IsNumber(item)) {
                char *msg = (char *)malloc(256);
                snprintf(msg, 256, "%s must be a number", key);
                return msg;
            }
            double v = cJSON_GetNumberValue(item);
            *(double *)((char *)out + ref->offset) = v;
        } else if (ref->kind == 1) {
            if (!cJSON_IsArray(item) || cJSON_GetArraySize(item) < 2) {
                char *msg = (char *)malloc(256);
                snprintf(msg, 256, "%s must be [hue, strength]", key);
                return msg;
            }
            const cJSON *a = cJSON_GetArrayItem(item, 0);
            const cJSON *b = cJSON_GetArrayItem(item, 1);
            double *base = (double *)((char *)out + ref->offset);
            base[0] = cJSON_GetNumberValue(a);
            base[1] = cJSON_GetNumberValue(b);
        } else { /* mixer */
            if (!cJSON_IsObject(item)) {
                char *msg = (char *)malloc(256);
                snprintf(msg, 256, "mixer must be an object");
                return msg;
            }
            const cJSON *band = NULL;
            cJSON_ArrayForEach(band, item) {
                int bi = fe_mixer_band_index(band->string ? band->string : "");
                if (bi < 0) {
                    char *msg = (char *)malloc(256);
                    snprintf(msg, 256, "unknown mixer band: %s", band->string ? band->string : "?");
                    return msg;
                }
                if (!cJSON_IsArray(band) || cJSON_GetArraySize(band) < 3) {
                    char *msg = (char *)malloc(256);
                    snprintf(msg, 256, "mixer band %s must be [hue, sat, lum]", band->string);
                    return msg;
                }
                out->mixer_h[bi] = cJSON_GetNumberValue(cJSON_GetArrayItem(band, 0));
                out->mixer_s[bi] = cJSON_GetNumberValue(cJSON_GetArrayItem(band, 1));
                out->mixer_v[bi] = cJSON_GetNumberValue(cJSON_GetArrayItem(band, 2));
                out->mixer_enabled[bi] = 1;
            }
        }
    }
    return NULL;
}

/* ------------------------------------------------------------------ */
/* directory scanning                                                   */
/* ------------------------------------------------------------------ */

static int cmp_str(const void *a, const void *b) {
    return strcmp(*(const char *const *)a, *(const char *const *)b);
}

static char **fe_list_dir(const char *path, int *count_out, int want_dirs) {
    DIR *d = opendir(path);
    if (!d) { *count_out = 0; return NULL; }
    char **names = NULL;
    int count = 0, cap = 0;
    struct dirent *ent;
    while ((ent = readdir(d)) != NULL) {
        if (ent->d_name[0] == '.') continue;
        char full[4096];
        snprintf(full, sizeof(full), "%s/%s", path, ent->d_name);
        struct stat st;
        if (stat(full, &st) != 0) continue;
        int is_dir = S_ISDIR(st.st_mode);
        if (want_dirs && !is_dir) continue;
        if (!want_dirs) {
            if (is_dir) continue;
            size_t len = strlen(ent->d_name);
            if (len < 6 || strcmp(ent->d_name + len - 5, ".json") != 0) continue;
        }
        if (count == cap) {
            cap = cap ? cap * 2 : 16;
            names = (char **)realloc(names, (size_t)cap * sizeof(char *));
        }
        names[count++] = fe_strdup(ent->d_name);
    }
    closedir(d);
    if (count > 1) qsort(names, (size_t)count, sizeof(char *), cmp_str);
    *count_out = count;
    return names;
}

static char *fe_read_file(const char *path, size_t *len_out) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (len < 0) { fclose(f); return NULL; }
    char *buf = (char *)malloc((size_t)len + 1);
    size_t got = fread(buf, 1, (size_t)len, f);
    fclose(f);
    buf[got] = '\0';
    if (len_out) *len_out = got;
    return buf;
}

static void fe_parse_effects(const cJSON *effects, FeFilter *f) {
    if (!effects || !cJSON_IsObject(effects)) return;
    const cJSON *ca = cJSON_GetObjectItem(effects, "chromatic_aberration");
    if (cJSON_IsNumber(ca)) { f->has_ca = 1; f->ca_shift = cJSON_GetNumberValue(ca); }
    const cJSON *sl = cJSON_GetObjectItem(effects, "scanlines");
    if (cJSON_IsNumber(sl)) { f->has_scanlines = 1; f->scanlines_intensity = cJSON_GetNumberValue(sl); }
    const cJSON *vcr = cJSON_GetObjectItem(effects, "vcr_overlay");
    if (cJSON_IsTrue(vcr)) f->has_vcr = 1;
}

int fe_filters_load(const char *root, FeFilterSet *out) {
    memset(out, 0, sizeof(*out));
    struct stat st;
    if (stat(root, &st) != 0 || !S_ISDIR(st.st_mode)) {
        printf("filter folder not found: %s\n", root);
        return 0;
    }
    int ndirs = 0;
    char **dirs = fe_list_dir(root, &ndirs, 1);
    if (!dirs) return 0;

    out->cats = (FeCategory *)calloc((size_t)ndirs, sizeof(FeCategory));
    for (int i = 0; i < ndirs; i++) {
        char catpath[4096];
        snprintf(catpath, sizeof(catpath), "%s/%s", root, dirs[i]);

        FeCategory *cat = &out->cats[out->count];
        cat->dir = fe_strdup(dirs[i]);
        cat->label = fe_category_default_label(dirs[i]);
        cat->colour = fe_strdup("#4A6FA5");

        /* category.json — label + colour */
        char metapath[FE_PATH_MAX * 2];
        snprintf(metapath, sizeof(metapath), "%s/category.json", catpath);
        char *metatext = fe_read_file(metapath, NULL);
        if (metatext) {
            cJSON *meta = cJSON_Parse(metatext);
            if (meta) {
                const cJSON *lbl = cJSON_GetObjectItem(meta, "label");
                const cJSON *col = cJSON_GetObjectItem(meta, "color");
                if (cJSON_IsString(lbl)) { free(cat->label); cat->label = fe_strdup(lbl->valuestring); }
                if (cJSON_IsString(col)) { free(cat->colour); cat->colour = fe_strdup(col->valuestring); }
                cJSON_Delete(meta);
            } else {
                printf("warning: %s: invalid JSON\n", metapath);
            }
            free(metatext);
        }

        /* the filters themselves */
        int nfiles = 0;
        char **files = fe_list_dir(catpath, &nfiles, 0);
        if (files) {
            cat->filters = (FeFilter *)calloc((size_t)(nfiles ? nfiles : 1), sizeof(FeFilter));
            for (int j = 0; j < nfiles; j++) {
                if (strcmp(files[j], "category.json") == 0) { free(files[j]); continue; }
                char fpath[FE_PATH_MAX * 2];
                snprintf(fpath, sizeof(fpath), "%s/%s", catpath, files[j]);
                char *text = fe_read_file(fpath, NULL);
                if (!text) { free(files[j]); continue; }
                cJSON *doc = cJSON_Parse(text);
                free(text);
                if (!doc) {
                    printf("warning: skipping filter %s: invalid JSON\n", fpath);
                    free(files[j]);
                    continue;
                }
                FeSettings settings;
                char *err = fe_parse_settings(cJSON_GetObjectItem(doc, "settings"), &settings);
                if (err) {
                    printf("warning: skipping filter %s: %s\n", fpath, err);
                    free(err);
                    cJSON_Delete(doc);
                    free(files[j]);
                    continue;
                }
                FeFilter *f = &cat->filters[cat->count];
                memset(f, 0, sizeof(*f));
                size_t klen = strlen(dirs[i]) + strlen(files[j]) + 2;
                f->key = (char *)malloc(klen);
                snprintf(f->key, klen, "%s/%s", dirs[i], files[j]);
                /* strip the .json suffix for the key */
                char *dot = strrchr(f->key, '.');
                if (dot) *dot = '\0';
                f->category_dir = fe_strdup(dirs[i]);
                f->category_label = fe_strdup(cat->label);
                f->colour = fe_strdup(cat->colour);

                const cJSON *lbl = cJSON_GetObjectItem(doc, "label");
                if (cJSON_IsString(lbl)) {
                    f->label = fe_strdup(lbl->valuestring);
                } else {
                    /* default: file name, title-cased */
                    char stem[512];
                    snprintf(stem, sizeof(stem), "%s", files[j]);
                    char *dot2 = strrchr(stem, '.');
                    if (dot2) *dot2 = '\0';
                    for (char *ch = stem; *ch; ch++) if (*ch == '-') *ch = ' ';
                    f->label = fe_title_case(stem);
                }
                const cJSON *desc = cJSON_GetObjectItem(doc, "description");
                f->description = cJSON_IsString(desc) ? fe_strdup(desc->valuestring) : fe_strdup("");
                f->settings = settings;
                fe_parse_effects(cJSON_GetObjectItem(doc, "effects"), f);
                cat->count++;
                cJSON_Delete(doc);
                free(files[j]);
            }
            free(files);
        }

        if (cat->count > 0) out->count++;
        else { /* empty category: keep the slot but do not count it */
            free(cat->dir); free(cat->label); free(cat->colour); free(cat->filters);
        }
        free(dirs[i]);
    }
    free(dirs);
    /* compact the list (drop empty categories) */
    if (out->count < ndirs) {
        int w = 0;
        for (int i = 0; i < ndirs; i++) {
            if (out->cats[i].filters && out->cats[i].count > 0) {
                if (w != i) out->cats[w] = out->cats[i];
                w++;
            }
        }
        out->count = w;
    }
    return out->count;
}

void fe_filters_free(FeFilterSet *set) {
    for (int i = 0; i < set->count; i++) {
        FeCategory *cat = &set->cats[i];
        for (int j = 0; j < cat->count; j++) {
            FeFilter *f = &cat->filters[j];
            free(f->key); free(f->category_dir); free(f->category_label);
            free(f->label); free(f->description); free(f->colour);
        }
        free(cat->filters);
        free(cat->dir); free(cat->label); free(cat->colour);
    }
    free(set->cats);
    memset(set, 0, sizeof(*set));
}

const FeFilter *fe_filters_find(const FeFilterSet *set, const char *key) {
    for (int i = 0; i < set->count; i++) {
        for (int j = 0; j < set->cats[i].count; j++) {
            if (strcmp(set->cats[i].filters[j].key, key) == 0)
                return &set->cats[i].filters[j];
        }
    }
    return NULL;
}
