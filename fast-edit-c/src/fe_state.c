#include "fe_state.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "../vendor/cJSON.h"

/* ------------------------------------------------------------------ */
/* path helpers                                                        */
/* ------------------------------------------------------------------ */

void fe_state_path(const char *image_path, char *out, size_t out_len) {
    if (!image_path || !image_path[0]) {
        if (out && out_len) out[0] = '\0';
        return;
    }
    snprintf(out, out_len, "%s.fastedit.json", image_path);
}

/* ------------------------------------------------------------------ */
/* the field list                                                      */
/* ------------------------------------------------------------------ */

int fe_state_get(const FeEditState *st, const char *name, double *out) {
    for (int i = 0; i < st->nfields; i++) {
        if (strcmp(st->fields[i].name, name) == 0) {
            if (out) *out = st->fields[i].value;
            return 1;
        }
    }
    return 0;
}

void fe_state_set(FeEditState *st, const char *name, double value) {
    for (int i = 0; i < st->nfields; i++) {
        if (strcmp(st->fields[i].name, name) == 0) {
            st->fields[i].value = value;
            return;
        }
    }
    if (st->nfields >= FE_STATE_MAX_FIELDS) return;
    snprintf(st->fields[st->nfields].name, FE_STATE_NAME_MAX, "%s", name);
    st->fields[st->nfields].value = value;
    st->nfields++;
}

/* ------------------------------------------------------------------ */
/* writing                                                             */
/* ------------------------------------------------------------------ */

int fe_state_save(const char *image_path, const FeEditState *st) {
    if (!image_path || !image_path[0]) return 0;

    char path[4096];
    fe_state_path(image_path, path, sizeof(path));

    cJSON *root = cJSON_CreateObject();
    if (!root) return 0;
    cJSON_AddNumberToObject(root, "version", 1);
    cJSON_AddStringToObject(root, "filter", st->filter_key[0] ? st->filter_key : "");

    cJSON *offsets = cJSON_CreateObject();
    if (!offsets) { cJSON_Delete(root); return 0; }
    for (int i = 0; i < st->nfields; i++) {
        /* An offset of exactly 0 means "the slider sits on the preset",
           which is indistinguishable from "never touched".  Leaving it out
           keeps the file down to what the user actually changed, and means
           a slider nudged back to rest disappears from the sidecar. */
        if (st->fields[i].value == 0.0) continue;
        cJSON_AddNumberToObject(offsets, st->fields[i].name, st->fields[i].value);
    }
    cJSON_AddItemToObject(root, "offsets", offsets);

    char *text = cJSON_Print(root);
    cJSON_Delete(root);
    if (!text) return 0;

    /* Write to a temporary file and rename it into place: a half-written
       sidecar would lose the photo's edits, a rename cannot. */
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
    return 1;
}

/* ------------------------------------------------------------------ */
/* reading                                                             */
/* ------------------------------------------------------------------ */

int fe_state_load(const char *image_path, FeEditState *st) {
    if (!image_path || !image_path[0] || !st) return 0;

    char path[4096];
    fe_state_path(image_path, path, sizeof(path));

    FILE *f = fopen(path, "r");
    if (!f) return 0;

    /* slurp the file (they are tiny) */
    size_t cap = 8192, len = 0;
    char *buf = (char *)malloc(cap);
    if (!buf) { fclose(f); return 0; }
    size_t got;
    while ((got = fread(buf + len, 1, cap - len - 1, f)) > 0) {
        len += got;
        if (len + 1 >= cap) {
            char *bigger = (char *)realloc(buf, cap * 2);
            if (!bigger) { free(buf); fclose(f); return 0; }
            buf = bigger;
            cap *= 2;
        }
    }
    fclose(f);
    buf[len] = '\0';

    cJSON *root = cJSON_Parse(buf);
    free(buf);
    if (!root) return 0;

    int ok = 0;
    FeEditState tmp;
    memset(&tmp, 0, sizeof(tmp));

    cJSON *flt = cJSON_GetObjectItem(root, "filter");
    if (flt && cJSON_IsString(flt) && flt->valuestring)
        snprintf(tmp.filter_key, sizeof(tmp.filter_key), "%s", flt->valuestring);

    cJSON *offsets = cJSON_GetObjectItem(root, "offsets");
    if (offsets && cJSON_IsObject(offsets)) {
        int n = cJSON_GetArraySize(offsets);
        for (int i = 0; i < n && tmp.nfields < FE_STATE_MAX_FIELDS; i++) {
            cJSON *item = cJSON_GetArrayItem(offsets, i);
            if (!item) continue;
            /* cJSON keeps the member names as children of the object */
            if (!item->string || !cJSON_IsNumber(item)) continue;
            fe_state_set(&tmp, item->string, item->valuedouble);
        }
        ok = 1;
    } else {
        /* Older sidecars (and hand-written ones) may store the values flat
           at the top level.  Accept those too so nothing is lost. */
        int n = cJSON_GetArraySize(root);
        for (int i = 0; i < n && tmp.nfields < FE_STATE_MAX_FIELDS; i++) {
            cJSON *item = cJSON_GetArrayItem(root, i);
            if (!item || !item->string || !cJSON_IsNumber(item)) continue;
            if (strcmp(item->string, "version") == 0) continue;
            fe_state_set(&tmp, item->string, item->valuedouble);
        }
        ok = (tmp.nfields > 0);
    }

    cJSON_Delete(root);
    if (!ok) return 0;
    *st = tmp;
    return 1;
}
