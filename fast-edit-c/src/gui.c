/* =============================================================================
   THE GTK3 EDITOR — a native window around the C develop pipeline.

   Layout (three columns, like a real Lightroom):

     ┌──────────────────┬───────────────────────────┬───────────────────┐
     │ fine-tune tabs   │        preview            │  filter browser   │
     │ (sliders)        │        (drawing area)     │  (categories)     │
     ├──────────────────┴───────────────────────────┴───────────────────┤
     │ folder / project film-strip                                       │
     ├───────────────────────────────────────────────────────────────────┤
     │ status line                                                       │
     └───────────────────────────────────────────────────────────────────┘

   THE SLIDER MODEL
   ----------------
   Every slider is an OFFSET on the active filter's value, so 0 always means
   "the filter exactly as authored".  Selecting a different filter re-bases
   the sliders; the filter's own numbers are the baseline and the user's
   drag is added on top.  A handful of values may never go below a floor
   (a negative grain amount would be nonsense, and a grain size below one
   pixel would break the noise lattice) — those are clamped in
   gui_current_settings().

   EVERY PHOTO KEEPS ITS OWN EDIT
   ------------------------------
   The preset a photo was developed with plus how far each slider was moved
   away from it are written to a sidecar JSON next to the photo:

       <photo>.ARW.fastedit.json        see fe_state.c

   That is what makes the edits travel WITH the picture: open the photo
   again (or export a batch) and it comes back exactly as it was left.
   Two rules keep it honest:

     * nothing carries over between photos — a photo that has never been
       edited starts from the preset with every slider at rest;
     * autosave is DEBOUNCED (a drag fires value-changed sixty times a
       second) and SUPPRESSED while the program itself moves sliders, so
       loading a photo can never overwrite it with a half-applied state.

   WHILE DRAGGING
   --------------
   Dragging a slider asks for a DRAFT render (the photo downscaled to
   DRAFT_LONG_EDGE) so the preview keeps up with the mouse; when the drag
   settles (no movement for a moment) one full-quality render is paid for.
   Renders are debounced with a timeout, exactly like the Python app.
   ============================================================================= */

#include "fe_app.h"
#include "fe_defs.h"
#include "fe_fields.h"
#include "fe_filters.h"
#include "fe_io.h"
#include "fe_math.h"
#include "fe_overlay.h"
#include "fe_pipeline.h"
#include "fe_project.h"
#include "fe_state.h"

#include <gdk-pixbuf/gdk-pixbuf.h>
#include <gtk/gtk.h>

#include <math.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <strings.h>
#include <sys/stat.h>

/* declared in fe_lens_auto.c */
void fe_lens_auto_analyze_image(const FeImage *linear, double *k1_out, double *tca_r_out,
                                double *tca_b_out, double *vignette_out, int *has_any);

/* the long edge of a film-strip preview (the camera's embedded JPEG) */
#define THUMB_LONG_EDGE 320

/* how many milliseconds of thumbnail decoding one idle slice may take */
#define THUMB_SLICE_MS 12

/* ------------------------------------------------------------------ */
/* the slider table                                                     */
/* ------------------------------------------------------------------ */

enum { TAB_FINE = 0, TAB_GRAIN = 1, TAB_EFFECTS = 2, TAB_LENS = 3 };

typedef struct {
    const char *label;      /* what the user sees */
    const char *key;        /* the field's name in JSON (see fe_fields.c) */
    size_t offset;          /* the field's offset inside FeSettings */
    double lo, hi;          /* the offset's range */
    double step;
    double floor_value;     /* the value may never drop below this (-1e9: no floor) */
    int tab;
} FeSlider;

#define S(field) offsetof(FeSettings, field)

/* The ranges and floors mirror the Python app exactly, so the two editors
   behave identically.  `key` is what the per-image sidecar is written
   with, so the file survives any reordering of this table. */
static const FeSlider SLIDERS[] = {
    /* fine tune */
    {"Exposure",     "exposure",         S(exposure),         -2.0,  2.0,  0.010, -1e9, TAB_FINE},
    {"Contrast",     "contrast",         S(contrast),         -0.5,  0.5,  0.005, -1e9, TAB_FINE},
    {"Temp",         "temperature",      S(temperature),      -0.4,  0.4,  0.005, -1e9, TAB_FINE},
    {"Tint",         "tint",             S(tint),             -0.4,  0.4,  0.005, -1e9, TAB_FINE},
    {"Highlights",   "highlights",       S(highlights),       -0.5,  0.5,  0.005, -1e9, TAB_FINE},
    {"Shadows",      "shadows",          S(shadows),          -0.5,  0.5,  0.005, -1e9, TAB_FINE},
    {"Fade (matte)", "fade",             S(fade),             -0.1,  0.2,  0.005,  0.0, TAB_FINE},
    {"Vibrance",     "vibrance",         S(vibrance),         -0.5,  0.5,  0.005, -1e9, TAB_FINE},
    /* grain */
    {"Amount",       "grain_amount",     S(grain_amount),     -0.05, 0.15, 0.002,  0.0, TAB_GRAIN},
    {"Coarseness",   "grain_size",       S(grain_size),       -0.5,  1.5,  0.010,  0.6, TAB_GRAIN},
    {"Dye colour",   "grain_color",      S(grain_color),      -0.02, 0.10, 0.002,  0.0, TAB_GRAIN},
    {"Mid bias",     "grain_bias",       S(grain_bias),       -60.0, 80.0, 0.500, 20.0, TAB_GRAIN},
    /* effects */
    {"Bloom",        "bloom_amount",     S(bloom_amount),     -0.2,  0.4,  0.005,  0.0, TAB_EFFECTS},
    {"Vignette",     "vignette_amount",  S(vignette_amount),  -0.2,  0.3,  0.005,  0.0, TAB_EFFECTS},
    {"Texture",      "texture",          S(texture),          -0.3,  0.3,  0.005, -1e9, TAB_EFFECTS},
    {"Clarity",      "clarity",          S(clarity),          -0.5,  0.5,  0.005, -1e9, TAB_EFFECTS},
    /* lens correction */
    {"Distortion",   "lens_distortion",  S(lens_distortion),  -1.0,  1.0,  0.005, -1e9, TAB_LENS},
    {"TCA Red",      "lens_tca_r",       S(lens_tca_r),       -0.3,  0.3,  0.005, -1e9, TAB_LENS},
    {"TCA Blue",     "lens_tca_b",       S(lens_tca_b),       -0.3,  0.3,  0.005, -1e9, TAB_LENS},
    {"Vignette Corr","lens_vignette_corr", S(lens_vignette_corr), -0.5,  0.5,  0.005, -1e9, TAB_LENS},
    {"Keystone H",   "lens_keystone_h",  S(lens_keystone_h),  -0.3,  0.3,  0.005, -1e9, TAB_LENS},
    {"Keystone V",   "lens_keystone_v",  S(lens_keystone_v),  -0.3,  0.3,  0.005, -1e9, TAB_LENS},
    {"Sharpening",   "sharpening_amount", S(sharpening_amount), -0.0,  1.0,  0.005, -1e9, TAB_LENS},
    {"Pixel Size",   "pixel_size",       S(pixel_size),       -0.0,  12.0, 0.5,    0.0, TAB_LENS},
};
#define NSLIDERS ((int)(sizeof(SLIDERS) / sizeof(SLIDERS[0])))

/* which slider holds the exposure (auto-exposure writes into it) */
#define SLIDER_EXPOSURE 0

/* The slider that owns a JSON field name, or -1.  Everything that used to
   be a magic index (the sidecar loader, the auto-lens button) goes through
   this, so reordering SLIDERS can no longer silently write the wrong
   number into the wrong setting. */
static int slider_by_key(const char *key) {
    for (int i = 0; i < NSLIDERS; i++)
        if (strcmp(SLIDERS[i].key, key) == 0) return i;
    return -1;
}

/* ------------------------------------------------------------------ */
/* GUI state                                                            */
/* ------------------------------------------------------------------ */

typedef struct {
    FeApp *app;

    GtkWidget *window;
    GtkWidget *notebook;
    GtkWidget *preview;      /* drawing area */
    GtkWidget *status;       /* status line */
    GtkWidget *fullres;      /* "full-res on export" check box */

    /* the active filter and the baseline its sliders offset from */
    const FeFilter *current;
    FeSettings base;

    /* slider widgets + their resting values */
    GtkWidget *scale[NSLIDERS];
    double rest[NSLIDERS];

    /* every filter button, in a flat array */
    GtkWidget **buttons;
    int nbuttons;
    GtkWidget **expanders;   /* one per category, for search + reveal */
    int nexpanders;
    int active_index;

    /* the rendered preview */
    unsigned char *rgb;
    int rgb_w, rgb_h;
    cairo_surface_t *surface;

    /* render scheduling */
    guint timeout_id;        /* the debounced render */
    guint settle_id;         /* "the drag stopped" timer */
    int draft;               /* 1 = render the cheap draft */

    /* per-image edit state */
    FeEditState state;       /* the photo currently open */
    guint save_id;           /* the debounced autosave */
    int suppress;            /* >0: the PROGRAM moved a slider, save nothing */

    /* the film-strip: previews waiting to be decoded one slice at a time */
    GQueue *thumb_pending;   /* char * paths, owned by the queue */
    guint thumb_idle;
    int alive;               /* 0 once the widgets are gone */

    /* the bottom panel */
    GtkWidget *bottom_tabs;
    GtkWidget *folder_box;
    GtkWidget *project_box;
    GtkWidget *project_page;        /* the notebook page wrapping project_box */
    GtkWidget *project_placeholder; /* the "no photos yet" hint */
    FeProject *project;
} FeGui;


/* forward declarations */
static void gui_render(FeGui *g);
static void gui_schedule_render(FeGui *g);
static void gui_select_filter(FeGui *g, int index);
static void gui_status(FeGui *g, const char *fmt, ...);
static void gui_load_image(FeGui *g, const char *path);
static void gui_save_state_now(FeGui *g);
static void gui_update_project_label(FeGui *g);

/* ------------------------------------------------------------------ */
/* small helpers                                                        */
/* ------------------------------------------------------------------ */

static void gui_status(FeGui *g, const char *fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    gtk_label_set_text(GTK_LABEL(g->status), buf);
}

/* Read a double straight out of the settings struct (by field offset). */
static double settings_get(const FeSettings *s, size_t offset) {
    return *(const double *)((const char *)s + offset);
}

static void settings_set(FeSettings *s, size_t offset, double v) {
    *(double *)((char *)s + offset) = v;
}

/* The filter's settings plus every slider offset, with the floors applied. */
static void gui_current_settings(FeGui *g, FeSettings *out) {
    *out = g->base;
    for (int i = 0; i < NSLIDERS; i++) {
        double offset = gtk_range_get_value(GTK_RANGE(g->scale[i])) - g->rest[i];
        double value = settings_get(&g->base, SLIDERS[i].offset) + offset;
        if (value < SLIDERS[i].floor_value) value = SLIDERS[i].floor_value;
        settings_set(out, SLIDERS[i].offset, value);
    }
}

/* ------------------------------------------------------------------ */
/* the preview                                                          */
/* ------------------------------------------------------------------ */

/* Copy an 8-bit RGB buffer into a cairo image surface (RGB24 wants 4 bytes
   per pixel, our buffer has 3 — so the padding byte is filled in here). */
static cairo_surface_t *surface_from_rgb(const unsigned char *rgb, int w, int h) {
    cairo_surface_t *s = cairo_image_surface_create(CAIRO_FORMAT_RGB24, w, h);
    if (!s) return NULL;
    unsigned char *dst = cairo_image_surface_get_data(s);
    int stride = cairo_image_surface_get_stride(s);
    for (int y = 0; y < h; y++) {
        const unsigned char *src = rgb + (size_t)y * w * 3;
        unsigned int *row = (unsigned int *)(dst + (size_t)y * stride);
        for (int x = 0; x < w; x++) {
            row[x] = ((unsigned int)src[x * 3 + 0] << 16) |
                     ((unsigned int)src[x * 3 + 1] << 8) |
                     ((unsigned int)src[x * 3 + 2]);
        }
    }
    cairo_surface_mark_dirty(s);
    return s;
}

static gboolean on_draw_preview(GtkWidget *da, cairo_t *cr, gpointer data) {
    FeGui *g = (FeGui *)data;
    int aw = gtk_widget_get_allocated_width(da);
    int ah = gtk_widget_get_allocated_height(da);

    if (getenv("FE_DEBUG_GUI"))
        fprintf(stderr, "[gui] draw area %dx%d surface=%p rgb=%dx%d\n", aw, ah,
                (void *)g->surface, g->rgb_w, g->rgb_h);

    /* the dark studio backdrop */
    cairo_set_source_rgb(cr, 0.08, 0.08, 0.09);
    cairo_paint(cr);

    if (!g->surface) {
        cairo_set_source_rgb(cr, 0.55, 0.55, 0.58);
        cairo_set_font_size(cr, 15.0);
        cairo_move_to(cr, 24, 34);
        cairo_show_text(cr, g->app->has_image ? "Developing…" : "Open an image to start");
        return TRUE;
    }

    /* fit the frame into the panel without distorting it (never upscale) */
    double iw = g->rgb_w, ih = g->rgb_h;
    double s = MIN((aw - 24.0) / iw, (ah - 24.0) / ih);
    if (s > 1.0) s = 1.0;
    if (s <= 0.0) s = 1.0;
    double w = iw * s, h = ih * s;
    double x = (aw - w) * 0.5, y = (ah - h) * 0.5;

    /* a soft drop shadow so the photo sits on the backdrop */
    cairo_set_source_rgba(cr, 0, 0, 0, 0.45);
    cairo_rectangle(cr, x + 5, y + 6, w, h);
    cairo_fill(cr);

    cairo_save(cr);
    cairo_translate(cr, x, y);
    cairo_scale(cr, s, s);
    cairo_set_source_surface(cr, g->surface, 0, 0);
    cairo_pattern_set_filter(cairo_get_source(cr), CAIRO_FILTER_BILINEAR);
    cairo_paint(cr);
    cairo_restore(cr);
    return TRUE;
}

/* ------------------------------------------------------------------ */
/* rendering                                                            */
/* ------------------------------------------------------------------ */

static gboolean on_render_timeout(gpointer data) {
    FeGui *g = (FeGui *)data;
    g->timeout_id = 0;
    gui_render(g);
    return G_SOURCE_REMOVE;
}

/* The drag stopped: no new slider movement for a moment — now pay for one
   full-quality render. */
static gboolean on_settle_timeout(gpointer data) {
    FeGui *g = (FeGui *)data;
    g->settle_id = 0;
    if (!g->draft) return G_SOURCE_REMOVE;
    g->draft = 0;
    gui_schedule_render(g);
    return G_SOURCE_REMOVE;
}

static void gui_schedule_render(FeGui *g) {
    if (g->timeout_id) g_source_remove(g->timeout_id);
    /* dragging gets a snappier delay than a final render */
    g->timeout_id = g_timeout_add(g->draft ? 50 : 90, on_render_timeout, g);

    /* (re)arm the "the drag stopped" timer while a drag is running */
    if (g->settle_id) g_source_remove(g->settle_id);
    g->settle_id = 0;
    if (g->draft) g->settle_id = g_timeout_add(260, on_settle_timeout, g);
}

static void gui_render(FeGui *g) {
    if (!g->app->has_image) return;

    FeSettings settings;
    gui_current_settings(g, &settings);

    /* while a slider is being dragged, develop a downscaled copy so the
       preview keeps up with the mouse */
    FeImage draft;
    const FeImage *src = &g->app->source_linear;
    if (g->draft) {
        draft = fe_img_copy(src);
        fe_downscale_maxedge(&draft, DRAFT_LONG_EDGE);
        src = &draft;
    }

    int w = 0, h = 0;
    unsigned char *rgb = fe_app_render_from(src, g->current, &settings, &w, &h, 0,
                                            g->app->camera_date);
    if (g->draft) fe_img_free(&draft);
    if (!rgb) return;

    if (g->surface) cairo_surface_destroy(g->surface);
    g->surface = surface_from_rgb(rgb, w, h);
    free(g->rgb);
    g->rgb = rgb;
    g->rgb_w = w;
    g->rgb_h = h;

    gtk_widget_queue_draw(g->preview);

    if (getenv("FE_DEBUG_GUI"))
        fprintf(stderr, "[gui] render %dx%d surface=%p draft=%d\n", w, h,
                (void *)g->surface, g->draft);
}

/* ------------------------------------------------------------------ */
/* per-image edit state                                                 */
/* ------------------------------------------------------------------ */

/* The flat browser index of the preset called `key` (-1 when unknown). */
static int gui_filter_index_by_key(FeGui *g, const char *key) {
    if (!key || !key[0]) return -1;
    int k = 0;
    for (int c = 0; c < g->app->filters.count; c++) {
        FeCategory *cat = &g->app->filters.cats[c];
        for (int j = 0; j < cat->count; j++, k++) {
            if (strcmp(cat->filters[j].key, key) == 0) return k;
        }
    }
    return -1;
}

/* Snapshot the current look: which preset is active and how far each
   slider was moved away from it.  Sliders sitting at rest are left out —
   the file stays small and "the preset exactly as authored" is implied. */
static void gui_collect_state(FeGui *g) {
    snprintf(g->state.filter_key, sizeof(g->state.filter_key), "%s",
             g->current ? g->current->key : "");
    g->state.nfields = 0;
    for (int i = 0; i < NSLIDERS; i++) {
        double offset = gtk_range_get_value(GTK_RANGE(g->scale[i])) - g->rest[i];
        if (offset == 0.0) continue;
        fe_state_set(&g->state, SLIDERS[i].key, offset);
    }
}

/* Write the sidecar for the photo that is open right now. */
static void gui_save_state_now(FeGui *g) {
    if (g->save_id) {
        g_source_remove(g->save_id);
        g->save_id = 0;
    }
    if (!g->app->has_image || !g->app->file_path[0]) return;

    gui_collect_state(g);
    if (fe_state_save(g->app->file_path, &g->state)) {
        if (getenv("FE_DEBUG_GUI"))
            fprintf(stderr, "[gui] autosaved %s (filter %s, %d adjustments)\n",
                    g->app->file_path, g->state.filter_key, g->state.nfields);
    } else {
        gui_status(g, "Could not write %s.fastedit.json", g->app->file_path);
    }
}

static gboolean on_save_timeout(gpointer data) {
    FeGui *g = (FeGui *)data;
    g->save_id = 0;
    gui_save_state_now(g);
    return G_SOURCE_REMOVE;
}

/* Debounced: dragging a slider fires value-changed sixty times a second,
   and rewriting a file on every tick would only make the editor stutter. */
static void gui_schedule_save(FeGui *g) {
    if (g->suppress) return;                 /* the program moved it, not the user */
    if (!g->app->has_image || !g->app->file_path[0]) return;
    if (g->save_id) g_source_remove(g->save_id);
    g->save_id = g_timeout_add(400, on_save_timeout, g);
}

/* Put the sliders where `st` says they were, on top of the right preset. */
static void gui_apply_state(FeGui *g, const FeEditState *st) {
    /* 1. the preset becomes the baseline again */
    int idx = gui_filter_index_by_key(g, st->filter_key);
    if (idx >= 0 && idx != g->active_index) gui_select_filter(g, idx);

    /* 2. every saved delta goes back on top of it */
    for (int i = 0; i < NSLIDERS; i++) {
        double offset = 0.0;
        if (!fe_state_get(st, SLIDERS[i].key, &offset)) offset = 0.0;
        if (offset < SLIDERS[i].lo) offset = SLIDERS[i].lo;
        if (offset > SLIDERS[i].hi) offset = SLIDERS[i].hi;
        gtk_range_set_value(GTK_RANGE(g->scale[i]), g->rest[i] + offset);
    }
}

/* Open `path` and give the photo back ITS OWN edit.

   This is the function the whole "changes stay with the picture" promise
   rests on, so the order matters:

     1. flush the previous photo's pending autosave FIRST — the sidecar
        path and the sliders still describe the photo being left;
     2. decode;
     3. suppress autosave while the sliders are repositioned, otherwise
        the reset would overwrite the very file we are about to read;
     4. restore the preset and the deltas, or start clean when the photo
        has never been edited (nothing carries over). */
static void gui_load_image(FeGui *g, const char *path) {
    gui_save_state_now(g);                 /* step 1 */

    if (!fe_app_load(g->app, path)) {
        gui_status(g, "Could not decode %s", path);
        return;
    }

    FeEditState st;
    memset(&st, 0, sizeof(st));
    int has = fe_state_load(path, &st);    /* step 2 */

    g->suppress++;                         /* step 3 */
    if (has) {
        gui_apply_state(g, &st);
        g->state = st;
    } else {
        for (int i = 0; i < NSLIDERS; i++)
            gtk_range_set_value(GTK_RANGE(g->scale[i]), g->rest[i]);
        g->state.filter_key[0] = '\0';
        g->state.nfields = 0;
    }
    g->suppress--;

    g->draft = 0;
    gui_schedule_render(g);
    gui_status(g, "%s%s", path, has ? "   (edit restored)" : "   (new photo — sliders at the preset)");
}

/* ------------------------------------------------------------------ */
/* the sliders                                                          */
/* ------------------------------------------------------------------ */

static void on_slider_changed(GtkRange *range, gpointer data) {
    (void)range;
    FeGui *g = (FeGui *)data;
    g->draft = 1;                     /* a draft render while dragging */
    gui_schedule_save(g);
    gui_schedule_render(g);
}

static void gui_reset_sliders(FeGui *g) {
    g->suppress++;                    /* a reset is not an edit of its own */
    for (int i = 0; i < NSLIDERS; i++)
        gtk_range_set_value(GTK_RANGE(g->scale[i]), g->rest[i]);
    g->suppress--;
    g->draft = 0;
    gui_schedule_save(g);
    gui_status(g, "Sliders reset to the preset");
    gui_schedule_render(g);
}

static void gui_set_slider_abs(FeGui *g, const char *key, double target) {
    int i = slider_by_key(key);
    if (i < 0 || !g->scale[i]) return;
    double base = settings_get(&g->base, SLIDERS[i].offset);
    double offset = target - base;
    if (offset < SLIDERS[i].lo) offset = SLIDERS[i].lo;
    if (offset > SLIDERS[i].hi) offset = SLIDERS[i].hi;
    gtk_range_set_value(GTK_RANGE(g->scale[i]), g->rest[i] + offset);
}

static void on_auto_lens(GtkButton *btn, gpointer data) {
    (void)btn;
    FeGui *g = (FeGui *)data;
    if (!g->app->has_image) {
        gui_status(g, "Open an image first");
        return;
    }
    gui_status(g, "Analyzing lens corrections…");
    while (gtk_events_pending()) gtk_main_iteration();

    double k1 = 0, tca_r = 0, tca_b = 0, vig = 0;
    int has_any = 0;
    fe_lens_auto_analyze_image(&g->app->source_linear, &k1, &tca_r, &tca_b, &vig, &has_any);

    /* Each result is written into the slider that owns its field — the
       slider is found by name, never by position in the table.  A positive
       slider corrects barrel distortion, and barrel means k1 < 0. */
    if (k1 != 0) {
        double v = -k1 / 0.2;
        gui_set_slider_abs(g, "lens_distortion", v);
    }
    if (tca_r != 0) gui_set_slider_abs(g, "lens_tca_r", tca_r);
    if (tca_b != 0) gui_set_slider_abs(g, "lens_tca_b", tca_b);
    if (vig != 0)   gui_set_slider_abs(g, "lens_vignette_corr", vig);

    g->draft = 0;
    gui_schedule_save(g);
    gui_status(g, has_any ? "Auto lens correction applied" : "No strong lens issues detected");
    gui_schedule_render(g);
}

/* Auto exposure: solve the stops this photo needs and write the answer into
   the exposure slider (as an offset from the filter's own exposure). */
static void on_auto_exposure(GtkButton *btn, gpointer data) {
    FeGui *g = (FeGui *)data;
    if (!g->app->has_image) {
        gui_status(g, "Open an image first");
        return;
    }
    double target = fe_auto_exposure(&g->app->source_linear);
    double base_exposure = settings_get(&g->base, SLIDERS[SLIDER_EXPOSURE].offset);
    double offset = target - base_exposure;
    if (offset < -2.0) offset = -2.0;
    if (offset > 2.0) offset = 2.0;
    gtk_range_set_value(GTK_RANGE(g->scale[SLIDER_EXPOSURE]),
                        g->rest[SLIDER_EXPOSURE] + offset);
    g->draft = 0;
    gui_schedule_save(g);
    gui_status(g, "Auto exposure: %+.2f stops", base_exposure + offset);
    gui_schedule_render(g);
}

/* ------------------------------------------------------------------ */
/* the filter browser                                                   */
/* ------------------------------------------------------------------ */

static void on_filter_clicked(GtkButton *btn, gpointer data) {
    FeGui *g = (FeGui *)data;
    gpointer idx = g_object_get_data(G_OBJECT(btn), "fe-index");
    if (idx) gui_select_filter(g, GPOINTER_TO_INT(idx));
}

/* Scroll the browser so the active preset is on screen — with a dozen
   categories it is easy to lose track of which button is lit. */
static void gui_browser_reveal(FeGui *g, int index) {
    if (index < 0 || index >= g->nbuttons || !g->buttons[index]) return;
    GtkWidget *btn = g->buttons[index];
    GtkWidget *scroller = gtk_widget_get_ancestor(btn, GTK_TYPE_SCROLLED_WINDOW);
    if (!scroller) return;
    GtkAdjustment *adj = gtk_scrolled_window_get_vadjustment(GTK_SCROLLED_WINDOW(scroller));
    if (!adj) return;

    GtkAllocation a;
    gtk_widget_get_allocation(btn, &a);
    gdouble value = gtk_adjustment_get_value(adj);
    gdouble page = gtk_adjustment_get_page_size(adj);
    if (a.y < value) value = a.y;
    else if (a.y + a.height > value + page) value = a.y + a.height - page;

    gdouble lower = gtk_adjustment_get_lower(adj);
    gdouble upper = gtk_adjustment_get_upper(adj) - page;
    gtk_adjustment_set_value(adj, CLAMP(value, lower, MAX(lower, upper)));
}

/* The search box above the list: hide every preset whose name does not
   contain the text, and hide the categories left with nothing in them.
   Both the friendly label and the preset's key are matched, so "obra",
   "portra" and "90-pixelart" all find what they are looking for. */
static void on_search_changed(GtkEditable *entry, gpointer data) {
    FeGui *g = (FeGui *)data;
    const char *text = gtk_entry_get_text(GTK_ENTRY(entry));
    int filtering = (text && text[0]);

    for (int i = 0; i < g->nbuttons; i++) {
        if (!g->buttons[i]) continue;
        int show = 1;
        if (filtering) {
            const char *label = gtk_button_get_label(GTK_BUTTON(g->buttons[i]));
            const char *key = (const char *)g_object_get_data(G_OBJECT(g->buttons[i]),
                                                              "fe-key");
            show = (label && strcasestr(label, text) != NULL) ||
                   (key && strcasestr(key, text) != NULL);
        }
        gtk_widget_set_visible(g->buttons[i], show);
    }

    for (int c = 0; c < g->nexpanders; c++) {
        if (!g->expanders[c]) continue;
        int show = !filtering;   /* while searching, keep a category only if a match survived */
        if (filtering) {
            GList *kids = gtk_container_get_children(
                GTK_CONTAINER(gtk_bin_get_child(GTK_BIN(g->expanders[c]))));
            for (GList *l = kids; l; l = l->next) {
                if (gtk_widget_get_visible(GTK_WIDGET(l->data))) { show = 1; break; }
            }
            g_list_free(kids);
        }
        gtk_widget_set_visible(g->expanders[c], show);
        if (filtering && show) gtk_expander_set_expanded(GTK_EXPANDER(g->expanders[c]), TRUE);
    }
}

/* Build the right-hand browser: one expander per category, each holding a
   button per filter.  The category colour comes from category.json. */
static void gui_build_browser(FeGui *g, GtkWidget *parent) {
    int total = 0;
    for (int i = 0; i < g->app->filters.count; i++)
        total += g->app->filters.cats[i].count;
    g->buttons = (GtkWidget **)calloc((size_t)(total > 0 ? total : 1), sizeof(GtkWidget *));
    g->nbuttons = total;
    g->expanders = (GtkWidget **)calloc((size_t)(g->app->filters.count > 0
                                                     ? g->app->filters.count : 1),
                                        sizeof(GtkWidget *));
    g->nexpanders = g->app->filters.count;

    int index = 0;
    for (int c = 0; c < g->app->filters.count; c++) {
        FeCategory *cat = &g->app->filters.cats[c];

        /* one CSS class per category so the buttons wear the category colour */
        char *css = g_strdup_printf(
            ".fecat%d { background-image: none; background-color: %s; color: #ffffff; }\n"
            ".fecat%d:hover { background-color: shade(%s, 1.18); }\n",
            c, cat->colour, c, cat->colour);
        GtkCssProvider *prov = gtk_css_provider_new();
        gtk_css_provider_load_from_data(prov, css, -1, NULL);
        gtk_style_context_add_provider_for_screen(gdk_screen_get_default(),
                                                  GTK_STYLE_PROVIDER(prov), 600 + c);
        g_object_unref(prov);
        g_free(css);

        char cls[32];
        snprintf(cls, sizeof(cls), "fecat%d", c);

        char *title = g_strdup_printf("%s  (%d)", cat->label, cat->count);
        GtkWidget *exp = gtk_expander_new(title);
        g_free(title);
        gtk_expander_set_expanded(GTK_EXPANDER(exp), TRUE);
        g->expanders[c] = exp;

        GtkWidget *box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 4);
        gtk_container_set_border_width(GTK_CONTAINER(box), 8);

        for (int j = 0; j < cat->count; j++) {
            FeFilter *f = &cat->filters[j];
            GtkWidget *btn = gtk_button_new_with_label(f->label);
            gtk_button_set_relief(GTK_BUTTON(btn), GTK_RELIEF_NONE);
            gtk_style_context_add_class(gtk_widget_get_style_context(btn), cls);
            g_object_set_data(G_OBJECT(btn), "fe-index", GINT_TO_POINTER(index));
            g_object_set_data_full(G_OBJECT(btn), "fe-key", g_strdup(f->key), g_free);
            g_signal_connect(btn, "clicked", G_CALLBACK(on_filter_clicked), g);
            gtk_box_pack_start(GTK_BOX(box), btn, FALSE, FALSE, 0);
            g->buttons[index] = btn;
            index++;
        }
        gtk_container_add(GTK_CONTAINER(exp), box);
        gtk_box_pack_start(GTK_BOX(parent), exp, FALSE, FALSE, 0);
    }
}

/* Make filter `index` the active one: re-base the sliders, highlight the
   button and re-render.  The preset is part of the photo's edit, so it is
   remembered in the sidecar too. */
static void gui_select_filter(FeGui *g, int index) {
    /* find the filter behind the flat index */
    const FeFilter *flt = NULL;
    int k = 0;
    for (int c = 0; c < g->app->filters.count && !flt; c++) {
        FeCategory *cat = &g->app->filters.cats[c];
        for (int j = 0; j < cat->count; j++) {
            if (k == index) { flt = &cat->filters[j]; break; }
            k++;
        }
    }
    if (!flt) return;

    g->current = flt;
    g->base = flt->settings;
    g->active_index = index;

    /* the sliders rest at 0 — the filter exactly as authored.  The reset
       raises value-changed, which would normally queue an autosave; during
       a photo load the caller has raised `suppress`, and everywhere else a
       preset change genuinely IS something worth saving. */
    for (int i = 0; i < NSLIDERS; i++)
        gtk_range_set_value(GTK_RANGE(g->scale[i]), g->rest[i]);

    /* …and it is not a drag: every reset raised value-changed, which flags
       the next render as a cheap draft.  Choosing a preset is a deliberate
       click, so ask for the real thing right away instead of waiting for a
       settle timer that is about to expire. */
    g->draft = 0;

    /* highlight the active button */
    for (int i = 0; i < g->nbuttons; i++) {
        if (!g->buttons[i]) continue;
        GtkStyleContext *ctx = gtk_widget_get_style_context(g->buttons[i]);
        if (i == index) gtk_style_context_add_class(ctx, "feactive");
        else gtk_style_context_remove_class(ctx, "feactive");
    }

    gui_browser_reveal(g, index);
    gui_status(g, "%s — %s", flt->label,
               flt->description && flt->description[0] ? flt->description : flt->key);
    gui_schedule_save(g);
    gui_schedule_render(g);
}

/* ------------------------------------------------------------------ */
/* film-strip previews                                                  */
/* ------------------------------------------------------------------ */

/* GdkPixbuf calls this when the last picture showing the pixels is gone.
   Its signature is (guchar *pixels, gpointer data) — writing the callback
   out (instead of casting `free` to fit) is what keeps
   -Wincompatible-pointer-types quiet. */
static void thumb_pixels_free(guchar *pixels, gpointer data) {
    (void)data;
    free(pixels);
}

/* Put the picture into every thumbnail strip entry that shows `path`.
   The folder strip and the project strip may both hold the same photo. */
static void gui_thumb_attach(FeGui *g, const char *path, GdkPixbuf *pb) {
    GtkWidget *boxes[2] = {g->folder_box, g->project_box};
    for (int b = 0; b < 2; b++) {
        if (!boxes[b]) continue;
        GList *children = gtk_container_get_children(GTK_CONTAINER(boxes[b]));
        for (GList *l = children; l; l = l->next) {
            GtkWidget *owner = GTK_WIDGET(l->data);
            const char *p = (const char *)g_object_get_data(G_OBJECT(owner), "fe-path");
            if (!p || strcmp(p, path) != 0) continue;
            GtkWidget *img = GTK_WIDGET(g_object_get_data(G_OBJECT(owner), "fe-thumb"));
            if (img && GTK_IS_IMAGE(img)) gtk_image_set_from_pixbuf(GTK_IMAGE(img), pb);
        }
        g_list_free(children);
    }
}

static void gui_thumbs_drain(FeGui *g) {
    if (!g->thumb_pending) return;
    char *p;
    while ((p = (char *)g_queue_pop_head(g->thumb_pending)) != NULL) g_free(p);
}

static void gui_thumbs_kick(FeGui *g);

/* One slice of background work: decode previews until the time budget is
   spent, then hand the main loop back.  Working in slices keeps the editor
   responsive — the film strip fills in while the user keeps clicking. */
static gboolean on_thumb_slice(gpointer data) {
    FeGui *g = (FeGui *)data;
    g->thumb_idle = 0;

    gint64 deadline = g_get_monotonic_time() +
                      THUMB_SLICE_MS * 1000;   /* 1 ms == 1000 µs */
    while (g_get_monotonic_time() < deadline) {
        char *path = (char *)g_queue_pop_head(g->thumb_pending);
        if (!path) break;

        unsigned char *pixels = NULL;
        int w = 0, h = 0;
        if (fe_load_thumb_rgb(path, THUMB_LONG_EDGE, &pixels, &w, &h)) {
            GdkPixbuf *pb = gdk_pixbuf_new_from_data(pixels, GDK_COLORSPACE_RGB, FALSE,
                                                     8, w, h, w * 3, thumb_pixels_free, NULL);
            if (pb) {
                /* from here on the pixbuf owns `pixels` and frees them
                   once the last thumbnail showing them is gone */
                gui_thumb_attach(g, path, pb);
                g_object_unref(pb);
            } else {
                free(pixels);
            }
        }
        g_free(path);
    }

    gui_thumbs_kick(g);
    return G_SOURCE_REMOVE;
}

static void gui_thumbs_kick(FeGui *g) {
    if (g->thumb_idle || !g->thumb_pending) return;
    if (g_queue_is_empty(g->thumb_pending)) return;
    if (!g->alive) return;
    g->thumb_idle = g_timeout_add(5, on_thumb_slice, g);
}

/* ------------------------------------------------------------------ */
/* the bottom panel: folder + project                                   */
/* ------------------------------------------------------------------ */

static void gui_load_folder(FeGui *g, const char *folder);

static void on_folder_open(GtkButton *btn, gpointer data) {
    FeGui *g = (FeGui *)data;
    const char *path = (const char *)g_object_get_data(G_OBJECT(btn), "fe-path");
    if (!path) return;
    gui_status(g, "Decoding…");
    while (gtk_events_pending()) gtk_main_iteration();
    gui_load_image(g, path);
}

static void on_project_open(GtkButton *btn, gpointer data) {
    FeGui *g = (FeGui *)data;
    const char *path = (const char *)g_object_get_data(G_OBJECT(btn), "fe-path");
    if (!path) return;
    gui_status(g, "Decoding…");
    while (gtk_events_pending()) gtk_main_iteration();
    gui_load_image(g, path);
    gtk_notebook_set_current_page(GTK_NOTEBOOK(g->bottom_tabs), 0);
}

/* Selection lives in the project itself; the check box only mirrors it. */
static void on_project_toggled(GtkToggleButton *tog, gpointer data) {
    FeGui *g = (FeGui *)data;
    int index = GPOINTER_TO_INT(g_object_get_data(G_OBJECT(tog), "fe-index")) - 1;
    if (!g->project || index < 0 || index >= g->project->count) return;
    g->project->items[index].selected = gtk_toggle_button_get_active(tog) ? 1 : 0;
    gui_update_project_label(g);
}

/* Only RAW families show up in the strip — a folder full of JPEGs next to
   every RAW would make the browser useless for culling. */
static int is_raw_name(const char *name) {
    const char *ext = strrchr(name, '.');
    if (!ext) return 0;
    return strcasecmp(ext, ".arw") == 0 || strcasecmp(ext, ".raw") == 0 ||
           strcasecmp(ext, ".cr2") == 0 || strcasecmp(ext, ".nef") == 0 ||
           strcasecmp(ext, ".dng") == 0 || strcasecmp(ext, ".raf") == 0 ||
           strcasecmp(ext, ".orf") == 0 || strcasecmp(ext, ".rw2") == 0;
}

static void gui_clear_box(GtkWidget *box) {
    GList *children = gtk_container_get_children(GTK_CONTAINER(box));
    for (GList *l = children; l; l = l->next) gtk_widget_destroy(GTK_WIDGET(l->data));
    g_list_free(children);
}

/* Keep the "Project" tab title and the empty-project hint in step with the
   project's contents.  The tab label names the PAGE (the wrapping box), not
   the film-strip inside it — passing the inner box would relabel the wrong
   notebook page. */
static void gui_update_project_label(FeGui *g) {
    if (!g->bottom_tabs || !g->project_page) return;
    char buf[64];
    snprintf(buf, sizeof(buf), "Project  (%d/%d selected)",
             fe_project_selected_count(g->project), g->project->count);
    gtk_notebook_set_tab_label_text(GTK_NOTEBOOK(g->bottom_tabs), g->project_page, buf);
    if (g->project_placeholder)
        gtk_widget_set_visible(g->project_placeholder, g->project->count == 0);
}

static void gui_load_folder(FeGui *g, const char *folder) {
    if (!g->folder_box) return;
    gui_save_state_now(g);            /* the photo we are leaving keeps its edits */
    gui_clear_box(g->folder_box);
    gui_thumbs_drain(g);              /* previews for the old listing are stale */

    DIR *d = opendir(folder);
    if (!d) {
        gui_status(g, "Cannot open %s", folder);
        return;
    }
    struct dirent *ent;
    int count = 0;
    while ((ent = readdir(d))) {
        if (ent->d_name[0] == '.') continue;
        if (!is_raw_name(ent->d_name)) continue;

        char path[FE_PATH_MAX];
        snprintf(path, sizeof(path), "%s/%s", folder, ent->d_name);

        GtkWidget *box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 6);
        gtk_container_set_border_width(GTK_CONTAINER(box), 4);

        GtkWidget *thumb = gtk_image_new_from_icon_name("image-x-generic",
                                                        GTK_ICON_SIZE_DND);
        gtk_box_pack_start(GTK_BOX(box), thumb, FALSE, FALSE, 0);

        GtkWidget *lbl = gtk_label_new(ent->d_name);
        gtk_label_set_xalign(GTK_LABEL(lbl), 0.0f);
        gtk_label_set_ellipsize(GTK_LABEL(lbl), PANGO_ELLIPSIZE_MIDDLE);
        gtk_label_set_max_width_chars(GTK_LABEL(lbl), 16);
        gtk_box_pack_start(GTK_BOX(box), lbl, TRUE, TRUE, 0);

        GtkWidget *btn = gtk_button_new();
        gtk_container_add(GTK_CONTAINER(btn), box);
        gtk_button_set_relief(GTK_BUTTON(btn), GTK_RELIEF_NONE);
        g_object_set_data_full(G_OBJECT(btn), "fe-path", g_strdup(path), g_free);
        g_object_set_data(G_OBJECT(btn), "fe-thumb", thumb);
        g_signal_connect(btn, "clicked", G_CALLBACK(on_folder_open), g);
        gtk_box_pack_start(GTK_BOX(g->folder_box), btn, FALSE, FALSE, 2);

        g_queue_push_tail(g->thumb_pending, g_strdup(path));
        count++;
    }
    closedir(d);

    gtk_widget_show_all(g->folder_box);
    gui_status(g, "Folder: %s (%d RAW/ARW files)", folder, count);
    gui_thumbs_kick(g);
}

static void on_browse_folder(GtkButton *btn, gpointer data) {
    FeGui *g = (FeGui *)data;
    GtkWidget *dlg = gtk_file_chooser_dialog_new(
        "Browse Folder", GTK_WINDOW(g->window),
        GTK_FILE_CHOOSER_ACTION_SELECT_FOLDER,
        "_Cancel", GTK_RESPONSE_CANCEL, "_Select", GTK_RESPONSE_ACCEPT, NULL);
    if (gtk_dialog_run(GTK_DIALOG(dlg)) == GTK_RESPONSE_ACCEPT) {
        char *path = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(dlg));
        if (path) {
            gui_load_folder(g, path);
            gtk_notebook_set_current_page(GTK_NOTEBOOK(g->bottom_tabs), 0);
            g_free(path);
        }
    }
    gtk_widget_destroy(dlg);
}

/* ---- the project -------------------------------------------------- */

static void gui_build_project(FeGui *g);

static void on_add_project(GtkButton *btn, gpointer data) {
    FeGui *g = (FeGui *)data;
    GtkWidget *dlg = gtk_file_chooser_dialog_new(
        "Add to Project", GTK_WINDOW(g->window),
        GTK_FILE_CHOOSER_ACTION_OPEN,
        "_Cancel", GTK_RESPONSE_CANCEL, "_Add", GTK_RESPONSE_ACCEPT, NULL);
    gtk_file_chooser_set_select_multiple(GTK_FILE_CHOOSER(dlg), TRUE);
    GtkFileFilter *ff = gtk_file_filter_new();
    gtk_file_filter_set_name(ff, "Images");
    gtk_file_filter_add_pattern(ff, "*.ARW"); gtk_file_filter_add_pattern(ff, "*.arw");
    gtk_file_filter_add_pattern(ff, "*.RAW"); gtk_file_filter_add_pattern(ff, "*.raw");
    gtk_file_filter_add_pattern(ff, "*.CR2"); gtk_file_filter_add_pattern(ff, "*.cr2");
    gtk_file_filter_add_pattern(ff, "*.NEF"); gtk_file_filter_add_pattern(ff, "*.nef");
    gtk_file_filter_add_pattern(ff, "*.DNG"); gtk_file_filter_add_pattern(ff, "*.dng");
    gtk_file_filter_add_pattern(ff, "*.RAF"); gtk_file_filter_add_pattern(ff, "*.raf");
    gtk_file_filter_add_pattern(ff, "*.jpg"); gtk_file_filter_add_pattern(ff, "*.jpeg");
    gtk_file_filter_add_pattern(ff, "*.png");
    gtk_file_chooser_add_filter(GTK_FILE_CHOOSER(dlg), ff);

    if (gtk_dialog_run(GTK_DIALOG(dlg)) == GTK_RESPONSE_ACCEPT) {
        GSList *files = gtk_file_chooser_get_filenames(GTK_FILE_CHOOSER(dlg));
        int added = 0;
        for (GSList *l = files; l; l = l->next) {
            if (fe_project_add(g->project, (char *)l->data)) added++;
            g_free(l->data);
        }
        g_slist_free(files);
        gui_build_project(g);
        gtk_notebook_set_current_page(GTK_NOTEBOOK(g->bottom_tabs), 1);
        gui_status(g, "Added %d photo(s) — project now %d, %d selected", added,
                   g->project->count, fe_project_selected_count(g->project));
    }
    gtk_widget_destroy(dlg);
}

static void on_project_select(GtkButton *btn, gpointer data) {
    FeGui *g = (FeGui *)data;
    const char *mode = (const char *)g_object_get_data(G_OBJECT(btn), "fe-mode");
    if (!mode || !g->project) return;
    if (strcmp(mode, "all") == 0) fe_project_select_all(g->project, 1);
    else if (strcmp(mode, "none") == 0) fe_project_select_none(g->project);
    else if (strcmp(mode, "invert") == 0) fe_project_select_invert(g->project);
    else if (strcmp(mode, "remove") == 0) fe_project_remove_selected(g->project);
    gui_build_project(g);
    gui_status(g, "%d/%d selected", fe_project_selected_count(g->project),
               g->project->count);
}

/* Give every selected photo the current look — one click for a whole
   batch.  The preset lands in each photo's sidecar, so it sticks. */
static void on_apply_filter_selected(GtkButton *btn, gpointer data) {
    FeGui *g = (FeGui *)data;
    if (!g->project || g->project->count == 0) { gui_status(g, "Project empty"); return; }
    if (!g->current) { gui_status(g, "Pick a filter first"); return; }
    int n = fe_project_selected_count(g->project);
    if (n == 0) { gui_status(g, "Nothing selected"); return; }

    gui_status(g, "Applying “%s” to %d photo(s)…", g->current->label, n);
    while (gtk_events_pending()) gtk_main_iteration();

    int done = fe_project_apply_filter(g->project, g->app, g->current->key);
    gui_status(g, "Applied “%s” to %d photo(s)", g->current->label, done);
}

/* Export the selected photos.  Each one is rendered with ITS OWN saved
   edit, so the batch comes out looking like the editor. */
static void on_export_selected(GtkButton *btn, gpointer data) {
    FeGui *g = (FeGui *)data;
    if (!g->project || g->project->count == 0) { gui_status(g, "Project empty"); return; }
    int n = fe_project_selected_count(g->project);
    if (n == 0) { gui_status(g, "Nothing selected"); return; }

    GtkWidget *dlg = gtk_file_chooser_dialog_new(
        "Export selected photos", GTK_WINDOW(g->window),
        GTK_FILE_CHOOSER_ACTION_SELECT_FOLDER,
        "_Cancel", GTK_RESPONSE_CANCEL, "_Export", GTK_RESPONSE_ACCEPT, NULL);
    if (gtk_dialog_run(GTK_DIALOG(dlg)) == GTK_RESPONSE_ACCEPT) {
        char *dir = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(dlg));
        if (dir) {
            free(g->project->export_dir);
            g->project->export_dir = strdup(dir);
            /* honour the same quality switch a single export uses */
            g->project->fullres = gtk_toggle_button_get_active(
                GTK_TOGGLE_BUTTON(g->fullres));

            gui_save_state_now(g);     /* make sure what we show is on disk */
            gui_status(g, "Exporting %d photo(s)…", n);
            while (gtk_events_pending()) gtk_main_iteration();

            int ok = fe_project_export(g->project, g->app,
                                       g->current ? g->current->key : NULL, NULL, NULL);
            gui_status(g, "Exported %d/%d photo(s) to %s", ok, n, dir);
            g_free(dir);
        }
    }
    gtk_widget_destroy(dlg);
}

/* Rebuild the project film-strip.  Selection is a check box over each
   thumbnail; clicking the thumbnail opens that photo with its own edit. */
static void gui_build_project(FeGui *g) {
    if (!g->project_box) return;
    gui_clear_box(g->project_box);

    for (int i = 0; i < g->project->count; i++) {
        FeProjectItem *it = &g->project->items[i];

        GtkWidget *col = gtk_box_new(GTK_ORIENTATION_VERTICAL, 2);
        gtk_container_set_border_width(GTK_CONTAINER(col), 4);

        GtkWidget *chk = gtk_check_button_new_with_label(it->name);
        gtk_label_set_ellipsize(GTK_LABEL(gtk_bin_get_child(GTK_BIN(chk))),
                                PANGO_ELLIPSIZE_MIDDLE);
        gtk_label_set_max_width_chars(GTK_LABEL(gtk_bin_get_child(GTK_BIN(chk))), 16);
        gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(chk), it->selected);
        g_object_set_data(G_OBJECT(chk), "fe-index", GINT_TO_POINTER(i + 1));
        g_signal_connect(chk, "toggled", G_CALLBACK(on_project_toggled), g);
        gtk_box_pack_start(GTK_BOX(col), chk, FALSE, FALSE, 0);

        GtkWidget *btn = gtk_button_new();
        gtk_button_set_relief(GTK_BUTTON(btn), GTK_RELIEF_NONE);
        GtkWidget *thumb = gtk_image_new_from_icon_name("image-x-generic",
                                                        GTK_ICON_SIZE_DND);
        gtk_container_add(GTK_CONTAINER(btn), thumb);
        g_object_set_data_full(G_OBJECT(btn), "fe-path", g_strdup(it->path), g_free);
        g_object_set_data(G_OBJECT(btn), "fe-thumb", thumb);
        g_signal_connect(btn, "clicked", G_CALLBACK(on_project_open), g);
        gtk_box_pack_start(GTK_BOX(col), btn, FALSE, FALSE, 0);

        gtk_box_pack_start(GTK_BOX(g->project_box), col, FALSE, FALSE, 2);
        g_queue_push_tail(g->thumb_pending, g_strdup(it->path));
    }

    gtk_widget_show_all(g->project_box);
    gui_update_project_label(g);
    gui_thumbs_kick(g);
}

/* The window is going away.  This is the LAST moment at which the sliders
   can still be read: the default handler for ::destroy tears the children
   down afterwards, and user-connected handlers run first.  Flushing the
   autosave here rather than after gtk_main() returns is what stops a save
   on exit from reading freed widgets. */
static void on_window_destroy(GtkWidget *w, gpointer data) {
    (void)w;
    FeGui *g = (FeGui *)data;
    gui_save_state_now(g);
    g->alive = 0;
    gtk_main_quit();
}

/* ------------------------------------------------------------------ */
/* open / export                                                        */
/* ------------------------------------------------------------------ */

static void on_open(GtkButton *btn, gpointer data) {
    FeGui *g = (FeGui *)data;
    GtkWidget *dlg = gtk_file_chooser_dialog_new(
        "Open Image", GTK_WINDOW(g->window),
        GTK_FILE_CHOOSER_ACTION_OPEN,
        "_Cancel", GTK_RESPONSE_CANCEL,
        "_Open", GTK_RESPONSE_ACCEPT, NULL);
    GtkFileFilter *ff = gtk_file_filter_new();
    gtk_file_filter_set_name(ff, "Images");
    gtk_file_filter_add_pattern(ff, "*.ARW");
    gtk_file_filter_add_pattern(ff, "*.arw");
    gtk_file_filter_add_pattern(ff, "*.RAW");
    gtk_file_filter_add_pattern(ff, "*.raw");
    gtk_file_filter_add_pattern(ff, "*.CR2");
    gtk_file_filter_add_pattern(ff, "*.cr2");
    gtk_file_filter_add_pattern(ff, "*.NEF");
    gtk_file_filter_add_pattern(ff, "*.nef");
    gtk_file_filter_add_pattern(ff, "*.DNG");
    gtk_file_filter_add_pattern(ff, "*.dng");
    gtk_file_filter_add_pattern(ff, "*.RAF");
    gtk_file_filter_add_pattern(ff, "*.raf");
    gtk_file_filter_add_pattern(ff, "*.jpg");
    gtk_file_filter_add_pattern(ff, "*.jpeg");
    gtk_file_filter_add_pattern(ff, "*.png");
    gtk_file_filter_add_pattern(ff, "*.tif");
    gtk_file_filter_add_pattern(ff, "*.tiff");
    gtk_file_chooser_add_filter(GTK_FILE_CHOOSER(dlg), ff);

    if (gtk_dialog_run(GTK_DIALOG(dlg)) == GTK_RESPONSE_ACCEPT) {
        char *path = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(dlg));
        if (path) {
            gui_status(g, "Decoding…");
            while (gtk_events_pending()) gtk_main_iteration();
            gui_load_image(g, path);        /* restores this photo's own edit */
            g_free(path);
        }
    }
    gtk_widget_destroy(dlg);
}

static void on_export(GtkButton *btn, gpointer data) {
    FeGui *g = (FeGui *)data;
    if (!g->app->has_image) {
        gui_status(g, "Nothing to export");
        return;
    }
    GtkWidget *dlg = gtk_file_chooser_dialog_new(
        "Export Image", GTK_WINDOW(g->window),
        GTK_FILE_CHOOSER_ACTION_SAVE,
        "_Cancel", GTK_RESPONSE_CANCEL,
        "_Export", GTK_RESPONSE_ACCEPT, NULL);
    gtk_file_chooser_set_do_overwrite_confirmation(GTK_FILE_CHOOSER(dlg), TRUE);
    gtk_file_chooser_set_current_name(GTK_FILE_CHOOSER(dlg), "edit.png");
    GtkFileFilter *ff = gtk_file_filter_new();
    gtk_file_filter_set_name(ff, "PNG (lossless)");
    gtk_file_filter_add_pattern(ff, "*.png");
    gtk_file_chooser_add_filter(GTK_FILE_CHOOSER(dlg), ff);
    GtkFileFilter *ff2 = gtk_file_filter_new();
    gtk_file_filter_set_name(ff2, "JPEG");
    gtk_file_filter_add_pattern(ff2, "*.jpg");
    gtk_file_filter_add_pattern(ff2, "*.jpeg");
    gtk_file_chooser_add_filter(GTK_FILE_CHOOSER(dlg), ff2);

    if (gtk_dialog_run(GTK_DIALOG(dlg)) == GTK_RESPONSE_ACCEPT) {
        char *path = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(dlg));
        if (path) {
            FeSettings settings;
            gui_current_settings(g, &settings);
            int fullres = gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(g->fullres));
            gui_status(g, fullres ? "Exporting full resolution…" : "Exporting…");
            while (gtk_events_pending()) gtk_main_iteration();
            if (fe_app_save(g->app, path, g->current, &settings, fullres))
                gui_status(g, "Saved %s", path);
            else
                gui_status(g, "Could not write %s", path);
            g_free(path);
        }
    }
    gtk_widget_destroy(dlg);
}

/* ------------------------------------------------------------------ */
/* keyboard shortcuts                                                   */
/* ------------------------------------------------------------------ */

static void on_next_filter(GtkWidget *w, gpointer data) {
    FeGui *g = (FeGui *)data;
    if (g->nbuttons <= 0) return;
    gui_select_filter(g, (g->active_index + 1) % g->nbuttons);
}

static void on_prev_filter(GtkWidget *w, gpointer data) {
    FeGui *g = (FeGui *)data;
    if (g->nbuttons <= 0) return;
    gui_select_filter(g, (g->active_index - 1 + g->nbuttons) % g->nbuttons);
}

static void on_reset(GtkWidget *w, gpointer data) {
    (void)w;
    gui_reset_sliders((FeGui *)data);
}

/* Left/right step through the filters while the preview has the focus. */
static gboolean on_preview_key(GtkWidget *w, GdkEventKey *ev, gpointer data) {
    (void)w;
    FeGui *g = (FeGui *)data;
    if (ev->keyval == GDK_KEY_Right) { on_next_filter(w, g); return TRUE; }
    if (ev->keyval == GDK_KEY_Left)  { on_prev_filter(w, g); return TRUE; }
    return FALSE;
}

/* ------------------------------------------------------------------ */
/* building the window                                                  */
/* ------------------------------------------------------------------ */

static GtkWidget *build_slider_page(FeGui *g, int tab) {
    GtkWidget *page = gtk_box_new(GTK_ORIENTATION_VERTICAL, 4);
    gtk_container_set_border_width(GTK_CONTAINER(page), 8);

    for (int i = 0; i < NSLIDERS; i++) {
        if (SLIDERS[i].tab != tab) continue;

        GtkWidget *label = gtk_label_new(SLIDERS[i].label);
        gtk_label_set_xalign(GTK_LABEL(label), 0.0f);
        gtk_box_pack_start(GTK_BOX(page), label, FALSE, FALSE, 0);

        GtkWidget *scale = gtk_scale_new_with_range(GTK_ORIENTATION_HORIZONTAL,
                                                    SLIDERS[i].lo, SLIDERS[i].hi,
                                                    SLIDERS[i].step);
        gtk_scale_set_draw_value(GTK_SCALE(scale), FALSE);
        gtk_widget_set_hexpand(scale, TRUE);
        gtk_range_set_value(GTK_RANGE(scale), 0.0);
        g->scale[i] = scale;
        /* The resting value cancels any widget-level snapping, exactly like
           the Python app's _slider_rest: "no user input" must mean the
           filter exactly as authored. */
        g->rest[i] = gtk_range_get_value(GTK_RANGE(scale));

        g_signal_connect(scale, "value-changed", G_CALLBACK(on_slider_changed), g);
        gtk_box_pack_start(GTK_BOX(page), scale, FALSE, FALSE, 0);
    }
    if (tab == TAB_LENS) {
        GtkWidget *auto_lens_btn = gtk_button_new_with_label("Auto Lens Correction");
        g_signal_connect(auto_lens_btn, "clicked", G_CALLBACK(on_auto_lens), g);
        gtk_box_pack_start(GTK_BOX(page), auto_lens_btn, FALSE, FALSE, 6);
    }
    return page;
}

/* The two JSON formats must agree about what a field is called, or the
   sidecar would quietly write a number into the wrong setting. */
static void gui_check_slider_keys(void) {
    for (int i = 0; i < NSLIDERS; i++) {
        ptrdiff_t off = fe_settings_field_offset(SLIDERS[i].key);
        if (off < 0 || (size_t)off != SLIDERS[i].offset) {
            fprintf(stderr,
                    "fast-edit: slider \"%s\" uses key \"%s\", which does not name "
                    "the same field in FeSettings\n",
                    SLIDERS[i].label, SLIDERS[i].key);
        }
    }
}

int fe_gui_run(FeApp *app) {
    FeGui g;
    memset(&g, 0, sizeof(g));
    g.app = app;
    g.alive = 1;
    g.active_index = -1;
    g.thumb_pending = g_queue_new();
    g.project = fe_project_new();

    gui_check_slider_keys();

    gtk_init(NULL, NULL);

    g.window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    gtk_window_set_title(GTK_WINDOW(g.window), "SAGE RAW Studio — C edition");
    gtk_window_set_default_size(GTK_WINDOW(g.window), 1400, 880);
    gtk_window_set_position(GTK_WINDOW(g.window), GTK_WIN_POS_CENTER);
    g_signal_connect(g.window, "destroy", G_CALLBACK(on_window_destroy), &g);

    /* the header bar carries the two file commands */
    GtkWidget *bar = gtk_header_bar_new();
    gtk_header_bar_set_title(GTK_HEADER_BAR(bar), "SAGE RAW STUDIO");
    gtk_header_bar_set_subtitle(GTK_HEADER_BAR(bar), "C edition — Lightroom-style pipeline");
    gtk_header_bar_set_show_close_button(GTK_HEADER_BAR(bar), TRUE);
    GtkWidget *open_btn = gtk_button_new_with_label("Open");
    g_signal_connect(open_btn, "clicked", G_CALLBACK(on_open), &g);
    gtk_header_bar_pack_start(GTK_HEADER_BAR(bar), open_btn);
    GtkWidget *export_btn = gtk_button_new_with_label("Export");
    g_signal_connect(export_btn, "clicked", G_CALLBACK(on_export), &g);
    gtk_header_bar_pack_end(GTK_HEADER_BAR(bar), export_btn);
    gtk_window_set_titlebar(GTK_WINDOW(g.window), bar);

    /* --- left column: the fine-tune tabs -------------------------------- */
    GtkWidget *left = gtk_box_new(GTK_ORIENTATION_VERTICAL, 6);
    gtk_container_set_border_width(GTK_CONTAINER(left), 8);

    GtkWidget *fine_page = build_slider_page(&g, TAB_FINE);
    GtkWidget *auto_btn = gtk_button_new_with_label("Auto Exposure");
    g_signal_connect(auto_btn, "clicked", G_CALLBACK(on_auto_exposure), &g);
    gtk_box_pack_start(GTK_BOX(fine_page), auto_btn, FALSE, FALSE, 6);

    g.notebook = gtk_notebook_new();
    gtk_notebook_append_page(GTK_NOTEBOOK(g.notebook), fine_page, gtk_label_new("Fine-tune"));
    gtk_notebook_append_page(GTK_NOTEBOOK(g.notebook), build_slider_page(&g, TAB_GRAIN),
                             gtk_label_new("Grain"));
    gtk_notebook_append_page(GTK_NOTEBOOK(g.notebook), build_slider_page(&g, TAB_EFFECTS),
                             gtk_label_new("Effects"));
    gtk_notebook_append_page(GTK_NOTEBOOK(g.notebook), build_slider_page(&g, TAB_LENS),
                             gtk_label_new("Lens"));
    GtkWidget *tabs_scroll = gtk_scrolled_window_new(NULL, NULL);
    gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(tabs_scroll),
                                   GTK_POLICY_NEVER, GTK_POLICY_AUTOMATIC);
    gtk_container_add(GTK_CONTAINER(tabs_scroll), g.notebook);
    gtk_box_pack_start(GTK_BOX(left), tabs_scroll, TRUE, TRUE, 0);

    GtkWidget *reset_btn = gtk_button_new_with_label("Reset Sliders");
    g_signal_connect(reset_btn, "clicked", G_CALLBACK(on_reset), &g);
    gtk_box_pack_start(GTK_BOX(left), reset_btn, FALSE, FALSE, 0);

    g.fullres = gtk_check_button_new_with_label("Full-res on export (slow)");
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g.fullres), TRUE);
    gtk_box_pack_start(GTK_BOX(left), g.fullres, FALSE, FALSE, 0);

    /* --- centre column: the preview ------------------------------------- */
    g.preview = gtk_drawing_area_new();
    gtk_widget_set_size_request(g.preview, 480, 400);
    gtk_widget_set_hexpand(g.preview, TRUE);
    gtk_widget_set_vexpand(g.preview, TRUE);
    gtk_widget_set_can_focus(g.preview, TRUE);
    g_signal_connect(g.preview, "draw", G_CALLBACK(on_draw_preview), &g);
    g_signal_connect(g.preview, "key-press-event", G_CALLBACK(on_preview_key), &g);

    /* --- right column: the filter browser ------------------------------- */
    GtkWidget *browser_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 4);

    GtkWidget *browser_title = gtk_label_new("FILTERS");
    gtk_label_set_xalign(GTK_LABEL(browser_title), 0.0f);
    gtk_box_pack_start(GTK_BOX(browser_box), browser_title, FALSE, FALSE, 4);

    /* the search box: with 60+ presets a category that starts collapsed is
       a preset nobody will ever find */
    GtkWidget *search = gtk_search_entry_new();
    gtk_entry_set_placeholder_text(GTK_ENTRY(search), "Search filters…");
    gtk_widget_set_hexpand(search, TRUE);
    g_signal_connect(search, "changed", G_CALLBACK(on_search_changed), &g);
    gtk_box_pack_start(GTK_BOX(browser_box), search, FALSE, FALSE, 0);

    GtkWidget *browser_scroll = gtk_scrolled_window_new(NULL, NULL);
    gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(browser_scroll),
                                   GTK_POLICY_NEVER, GTK_POLICY_AUTOMATIC);
    gtk_box_pack_start(GTK_BOX(browser_box), browser_scroll, TRUE, TRUE, 0);

    GtkWidget *browser_list = gtk_box_new(GTK_ORIENTATION_VERTICAL, 4);
    gtk_container_set_border_width(GTK_CONTAINER(browser_list), 4);
    gtk_container_add(GTK_CONTAINER(browser_scroll), browser_list);
    gui_build_browser(&g, browser_list);
    /* the width request belongs on the COLUMN the paned is given, not on the
       scrolled window inside it — a widget may only ever have one parent */
    gtk_widget_set_size_request(browser_box, 250, -1);

    /* the preview + sidebar share one paned widget, the browser another */
    GtkWidget *paned = gtk_paned_new(GTK_ORIENTATION_HORIZONTAL);
    gtk_paned_set_position(GTK_PANED(paned), 320);
    gtk_paned_pack1(GTK_PANED(paned), left, FALSE, TRUE);
    gtk_paned_pack2(GTK_PANED(paned), g.preview, TRUE, TRUE);

    GtkWidget *outer = gtk_paned_new(GTK_ORIENTATION_HORIZONTAL);
    gtk_paned_set_position(GTK_PANED(outer), 1080);
    gtk_paned_pack1(GTK_PANED(outer), paned, TRUE, TRUE);
    gtk_paned_pack2(GTK_PANED(outer), browser_box, FALSE, TRUE);

    /* --- the bottom panel: folder strip + project strip ------------------ */
    GtkWidget *bottom = gtk_box_new(GTK_ORIENTATION_VERTICAL, 4);
    gtk_container_set_border_width(GTK_CONTAINER(bottom), 6);

    GtkWidget *top_row = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 6);
    GtkWidget *ftitle = gtk_label_new("FOLDER / PROJECT");
    gtk_label_set_xalign(GTK_LABEL(ftitle), 0.0f);
    gtk_box_pack_start(GTK_BOX(top_row), ftitle, FALSE, FALSE, 0);

    GtkWidget *bf = gtk_button_new_with_label("Browse Folder");
    g_signal_connect(bf, "clicked", G_CALLBACK(on_browse_folder), &g);
    gtk_box_pack_start(GTK_BOX(top_row), bf, FALSE, FALSE, 0);

    GtkWidget *apb = gtk_button_new_with_label("Add to Project");
    g_signal_connect(apb, "clicked", G_CALLBACK(on_add_project), &g);
    gtk_box_pack_start(GTK_BOX(top_row), apb, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(bottom), top_row, FALSE, FALSE, 0);

    g.bottom_tabs = gtk_notebook_new();

    /* tab 0 — the folder we are working in */
    GtkWidget *folder_scroll = gtk_scrolled_window_new(NULL, NULL);
    gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(folder_scroll),
                                   GTK_POLICY_AUTOMATIC, GTK_POLICY_NEVER);
    gtk_widget_set_size_request(folder_scroll, -1, 130);
    g.folder_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 4);
    gtk_container_add(GTK_CONTAINER(folder_scroll), g.folder_box);
    gtk_notebook_append_page(GTK_NOTEBOOK(g.bottom_tabs), folder_scroll,
                             gtk_label_new("Folder"));

    /* tab 1 — the batch: select, mass-apply, mass-export */
    GtkWidget *project_page = gtk_box_new(GTK_ORIENTATION_VERTICAL, 4);
    gtk_container_set_border_width(GTK_CONTAINER(project_page), 4);

    GtkWidget *pt = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 4);

    GtkWidget *applyb = gtk_button_new_with_label("Apply Filter to Selected");
    g_signal_connect(applyb, "clicked", G_CALLBACK(on_apply_filter_selected), &g);
    gtk_box_pack_start(GTK_BOX(pt), applyb, FALSE, FALSE, 0);

    const char *modes[4][2] = {{"Select All", "all"}, {"None", "none"},
                               {"Invert", "invert"}, {"Remove Selected", "remove"}};
    for (int i = 0; i < 4; i++) {
        GtkWidget *b = gtk_button_new_with_label(modes[i][0]);
        g_object_set_data(G_OBJECT(b), "fe-mode", (gpointer)modes[i][1]);
        g_signal_connect(b, "clicked", G_CALLBACK(on_project_select), &g);
        gtk_box_pack_start(GTK_BOX(pt), b, FALSE, FALSE, 0);
    }

    GtkWidget *exportb = gtk_button_new_with_label("Export Selected");
    g_signal_connect(exportb, "clicked", G_CALLBACK(on_export_selected), &g);
    gtk_box_pack_start(GTK_BOX(pt), exportb, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(project_page), pt, FALSE, FALSE, 0);

    GtkWidget *project_scroll = gtk_scrolled_window_new(NULL, NULL);
    gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(project_scroll),
                                   GTK_POLICY_AUTOMATIC, GTK_POLICY_NEVER);
    gtk_widget_set_size_request(project_scroll, -1, 110);
    g.project_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 4);
    gtk_container_add(GTK_CONTAINER(project_scroll), g.project_box);
    gtk_box_pack_start(GTK_BOX(project_page), project_scroll, TRUE, TRUE, 0);

    GtkWidget *project_placeholder = gtk_label_new(
        "No photos yet — press “Add to Project”, or use “Browse Folder”.");
    gtk_label_set_xalign(GTK_LABEL(project_placeholder), 0.0f);
    gtk_box_pack_start(GTK_BOX(project_page), project_placeholder, FALSE, FALSE, 4);

    g.project_page = project_page;
    g.project_placeholder = project_placeholder;

    gtk_notebook_append_page(GTK_NOTEBOOK(g.bottom_tabs), project_page,
                             gtk_label_new("Project (0)"));

    gtk_box_pack_start(GTK_BOX(bottom), g.bottom_tabs, FALSE, FALSE, 0);
    GtkWidget *ff = gtk_frame_new(NULL);
    gtk_container_add(GTK_CONTAINER(ff), bottom);

    /* --- the status line ------------------------------------------------ */
    g.status = gtk_label_new("Open an image to start");
    gtk_label_set_xalign(GTK_LABEL(g.status), 0.0f);
    gtk_label_set_ellipsize(GTK_LABEL(g.status), PANGO_ELLIPSIZE_END);
    GtkWidget *status_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);
    gtk_container_set_border_width(GTK_CONTAINER(status_box), 6);
    gtk_box_pack_start(GTK_BOX(status_box), g.status, TRUE, TRUE, 0);

    GtkWidget *root = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
    gtk_box_pack_start(GTK_BOX(root), outer, TRUE, TRUE, 0);
    gtk_box_pack_start(GTK_BOX(root), ff, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(root), status_box, FALSE, FALSE, 0);
    gtk_container_add(GTK_CONTAINER(g.window), root);

    /* --- keyboard shortcuts --------------------------------------------- */
    GtkAccelGroup *accel = gtk_accel_group_new();
    gtk_window_add_accel_group(GTK_WINDOW(g.window), accel);
    gtk_widget_add_accelerator(open_btn, "clicked", accel, GDK_KEY_o, GDK_CONTROL_MASK,
                               GTK_ACCEL_VISIBLE);
    gtk_widget_add_accelerator(export_btn, "clicked", accel, GDK_KEY_s, GDK_CONTROL_MASK,
                               GTK_ACCEL_VISIBLE);
    gtk_widget_add_accelerator(reset_btn, "clicked", accel, GDK_KEY_r, GDK_CONTROL_MASK,
                               GTK_ACCEL_VISIBLE);

    /* the "active filter" outline */
    GtkCssProvider *active_css = gtk_css_provider_new();
    gtk_css_provider_load_from_data(
        active_css,
        ".feactive { border: 2px solid #FFD479; border-radius: 4px; }\n", -1, NULL);
    gtk_style_context_add_provider_for_screen(gdk_screen_get_default(),
                                              GTK_STYLE_PROVIDER(active_css), 900);
    g_object_unref(active_css);

    gtk_widget_show_all(g.window);
    /* after show_all, so hiding the "no photos yet" hint actually sticks */
    gui_update_project_label(&g);
    gtk_widget_grab_focus(g.preview);

    /* start on the first filter, like the Python app */
    if (g.nbuttons > 0) gui_select_filter(&g, 0);

    gtk_main();

    /* on_window_destroy() has already flushed the edit and marked the GUI
       dead; all that is left is to retire the timers we own and free the
       plain C memory — no widget is touched from here on. */
    if (g.timeout_id) g_source_remove(g.timeout_id);
    if (g.settle_id) g_source_remove(g.settle_id);
    if (g.save_id) g_source_remove(g.save_id);
    if (g.thumb_idle) g_source_remove(g.thumb_idle);
    gui_thumbs_drain(&g);
    g_queue_free(g.thumb_pending);

    if (g.surface) cairo_surface_destroy(g.surface);
    free(g.rgb);
    free(g.buttons);
    free(g.expanders);
    fe_project_free(g.project);
    return 0;
}
