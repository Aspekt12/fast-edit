#ifndef FE_PROJECT_H
#define FE_PROJECT_H

#include "fe_defs.h"
#include "fe_app.h"

#include <stddef.h>

/* Project / Batch Manager
   Add many photos, select them, mass-apply filters, mass-export.
   Designed for speed with multi-threading and quality preservation. */

typedef struct {
    char *path;          /* full path to source image */
    char *thumbnail;     /* path to cached thumbnail (optional) */
    char *name;          /* display name */
    int selected;
    int processed;
    /* Cached data */
    FeImage *thumb_img;  /* small thumbnail for UI */
    FeImage source_linear; /* decoded source (cached on demand) */
    int has_source;
} FeProjectItem;

typedef struct {
    FeProjectItem *items;
    int count;
    int capacity;
    char *project_path;  /* .feproj file location */
    char *export_dir;
    char *export_prefix;
    char *export_format; /* "png", "jpg" */
    int export_quality;  /* for jpg */
    int use_dither;
    int fullres;         /* 1 = re-decode each source at full resolution */
    char *active_filter_key;
} FeProject;

FeProject *fe_project_new(void);
void fe_project_free(FeProject *proj);

/* Add images to project */
int fe_project_add(FeProject *proj, const char *path);
int fe_project_add_many(FeProject *proj, const char **paths, int count);

/* Selection */
void fe_project_select_all(FeProject *proj, int selected);
void fe_project_select_none(FeProject *proj);
void fe_project_select_invert(FeProject *proj);
int  fe_project_selected_count(FeProject *proj);

/* Remove items */
void fe_project_remove_selected(FeProject *proj);
void fe_project_clear(FeProject *proj);

/* Generate thumbnails (multi-threaded) */
int fe_project_generate_thumbnails(FeProject *proj, int thumb_size);

/* Mass apply: apply filter to all selected items */
int fe_project_apply_filter(FeProject *proj, FeApp *app, const char *filter_key);

/* Mass export with multi-threading, no quality loss */
typedef void (*FeProjectProgressFn)(int done, int total, const char *current_file, void *user);
int fe_project_export(FeProject *proj, FeApp *app, const char *filter_key,
                      FeProjectProgressFn progress, void *user);

/* Save/load project */
int fe_project_save(FeProject *proj, const char *path);
int fe_project_load(FeProject *proj, const char *path);

#endif
