/* =============================================================================
   THE GTK3 EDITOR — a native window around the C develop pipeline.

   Layout (three columns, like a real Lightroom):

     ┌──────────────────┬───────────────────────────┬───────────────────┐
     │ fine-tune tabs   │        preview            │  filter browser   │
     │ (sliders)        │        (drawing area)     │  (categories)     │
     ├──────────────────┴───────────────────────────┴───────────────────┤
     │ status line                                                         │
     └────────────────────────────────────────────────────────────────────┘

   THE SLIDER MODEL
   ----------------
   Every slider is an OFFSET on the active filter's value, so 0 always means
   "the filter exactly as authored".  Selecting a different filter re-bases
   the sliders; the filter's own numbers are the baseline and the user's
   drag is added on top.  A handful of values may never go below a floor
   (a negative grain amount would be nonsense, and a grain size below one
   pixel would break the noise lattice) — those are clamped in
   gui_current_settings().

   WHILE DRAGGING
   --------------
   Dragging a slider asks for a DRAFT render (the photo downscaled to
   DRAFT_LONG_EDGE) so the preview keeps up with the mouse; when the drag
   settles (no movement for a moment) one full-quality render is paid for.
   Renders are debounced with a timeout, exactly like the Python app.
   ============================================================================= */

#include "fe_app.h"
#include "fe_defs.h"
#include "fe_io.h"
#include "fe_math.h"
#include "fe_pipeline.h"
#include "fe_project.h"

#include <gtk/gtk.h>
#include <gdk-pixbuf/gdk-pixbuf.h>

#include <math.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>
#include <strings.h>

void fe_lens_auto_analyze_image(const FeImage *linear, double *k1_out, double *tca_r_out, double *tca_b_out, 
                                 double *vignette_out, int *has_any);

/* ------------------------------------------------------------------ */
/* the slider table                                                     */
/* ------------------------------------------------------------------ */

enum { TAB_FINE = 0, TAB_GRAIN = 1, TAB_EFFECTS = 2, TAB_LENS = 3 };

typedef struct {
    const char *label;
    size_t offset;       /* the field's offset inside FeSettings */
    double lo, hi;       /* the offset's range */
    double step;
    double floor_value;  /* the value may never drop below this (-1e9: no floor) */
    int tab;
} FeSlider;

#define S(field) offsetof(FeSettings, field)

/* The ranges and floors mirror the Python app exactly, so the two editors
   behave identically. */
static const FeSlider SLIDERS[] = {
    /* fine tune */
    {"Exposure",     S(exposure),         -2.0,  2.0,  0.010, -1e9,   TAB_FINE},
    {"Contrast",     S(contrast),         -0.5,  0.5,  0.005, -1e9,   TAB_FINE},
    {"Temp",         S(temperature),      -0.4,  0.4,  0.005, -1e9,   TAB_FINE},
    {"Tint",         S(tint),             -0.4,  0.4,  0.005, -1e9,   TAB_FINE},
    {"Highlights",   S(highlights),       -0.5,  0.5,  0.005, -1e9,   TAB_FINE},
    {"Shadows",      S(shadows),          -0.5,  0.5,  0.005, -1e9,   TAB_FINE},
    {"Fade (matte)", S(fade),             -0.1,  0.2,  0.005,  0.0,   TAB_FINE},
    {"Vibrance",     S(vibrance),         -0.5,  0.5,  0.005, -1e9,   TAB_FINE},
    /* grain */
    {"Amount",       S(grain_amount),     -0.05, 0.15, 0.002,  0.0,   TAB_GRAIN},
    {"Coarseness",   S(grain_size),       -0.5,  1.5,  0.010,  0.6,   TAB_GRAIN},
    {"Dye colour",   S(grain_color),      -0.02, 0.10, 0.002,  0.0,   TAB_GRAIN},
    {"Mid bias",     S(grain_bias),       -60.0, 80.0, 0.500, 20.0,   TAB_GRAIN},
    /* effects */
    {"Bloom",        S(bloom_amount),     -0.2,  0.4,  0.005,  0.0,   TAB_EFFECTS},
    {"Vignette",     S(vignette_amount),  -0.2,  0.3,  0.005,  0.0,   TAB_EFFECTS},
    {"Texture",      S(texture),          -0.3,  0.3,  0.005, -1e9,   TAB_EFFECTS},
    {"Clarity",      S(clarity),          -0.5,  0.5,  0.005, -1e9,   TAB_EFFECTS},
    /* lens correction */
    {"Distortion",   S(lens_distortion),  -1.0,  1.0,  0.005, -1e9,   TAB_LENS},
    {"TCA Red",      S(lens_tca_r),       -0.3,  0.3,  0.005, -1e9,   TAB_LENS},
    {"TCA Blue",     S(lens_tca_b),       -0.3,  0.3,  0.005, -1e9,   TAB_LENS},
    {"Vignette Corr",S(lens_vignette_corr),-0.5, 0.5,  0.005, -1e9,   TAB_LENS},
    {"Keystone H",   S(lens_keystone_h),  -0.3,  0.3,  0.005, -1e9,   TAB_LENS},
    {"Keystone V",   S(lens_keystone_v),  -0.3,  0.3,  0.005, -1e9,   TAB_LENS},
    {"Sharpening",   S(sharpening_amount),-0.0,  1.0,  0.005, -1e9,   TAB_LENS},
    {"Pixel Size",   S(pixel_size),       -0.0,  12.0, 0.5,   0.0,   TAB_LENS},
};
#define NSLIDERS ((int)(sizeof(SLIDERS) / sizeof(SLIDERS[0])))

/* which slider holds the exposure (auto-exposure writes into it) */
#define SLIDER_EXPOSURE 0

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
    int active_index;

    /* the rendered preview */
    unsigned char *rgb;
    int rgb_w, rgb_h;
    cairo_surface_t *surface;

    /* render scheduling */
    guint timeout_id;        /* the debounced render */
    guint settle_id;         /* "the drag stopped" timer */
    int draft;               /* 1 = render the cheap draft */
    /* folder/browser/project */
    GtkWidget *folder_box;
    GtkWidget *folder_scroll;
    FeProject *project;
} FeGui;


/* forward declarations */
static void gui_render(FeGui *g);
static void gui_schedule_render(FeGui *g);
static void gui_select_filter(FeGui *g, int index);
static void gui_status(FeGui *g, const char *fmt, ...);

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
/* the sliders                                                          */
/* ------------------------------------------------------------------ */

static void on_slider_changed(GtkRange *range, gpointer data) {
    FeGui *g = (FeGui *)data;
    g->draft = 1;                     /* a draft render while dragging */
    gui_schedule_render(g);
}

static void gui_reset_sliders(FeGui *g) {
    for (int i = 0; i < NSLIDERS; i++)
        gtk_range_set_value(GTK_RANGE(g->scale[i]), g->rest[i]);
    g->draft = 0;
    gui_status(g, "Sliders reset");
    gui_schedule_render(g);
}

static void gui_set_slider_abs(FeGui *g, int slider_idx, double target) {
    if (slider_idx < 0 || slider_idx >= NSLIDERS) return;
    if (!g->scale[slider_idx]) return;
    double base = settings_get(&g->base, SLIDERS[slider_idx].offset);
    double offset = target - base;
    if (offset < SLIDERS[slider_idx].lo) offset = SLIDERS[slider_idx].lo;
    if (offset > SLIDERS[slider_idx].hi) offset = SLIDERS[slider_idx].hi;
    gtk_range_set_value(GTK_RANGE(g->scale[slider_idx]), g->rest[slider_idx] + offset);
}

static void on_auto_lens(GtkButton *btn, gpointer data) {
    (void)btn;
    FeGui *g = (FeGui *)data;
    if (!g->app->has_image) {
        gui_status(g, "Open an image first");
        return;
    }
    gui_status(g, "Analyzing lens corrections...");
    while (gtk_events_pending()) gtk_main_iteration();
    
    double k1 = 0, tca_r = 0, tca_b = 0, vig = 0;
    int has_any = 0;
    fe_lens_auto_analyze_image(&g->app->source_linear, &k1, &tca_r, &tca_b, &vig, &has_any);
    
    /* Find slider indices */
    /* Distortion is at index 15 in our list? Let's look: 
     * 0:Exposure,1:Contrast,2:Temp,3:Tint,4:Highlights,5:Shadows,6:Fade,7:Vibrance,
     * 8:Amount(grain),9:Coarseness,10:Dye,11:Mid bias,
     * 12:Bloom,13:Vignette,14:Texture,15:Clarity,
     * 16:Distortion,17:TCA Red,18:TCA Blue,19:Vignette Corr,20:Keystone H,21:Keystone V,22:Sharpening
     */
    /* Set values: positive slider corrects barrel -> k1 < 0 for barrel means slider = -k1/0.2 */
    if (k1 != 0) {
        double v = -k1 / 0.2;
        if (v < -1.0) v = -1.0;
        if (v > 1.0) v = 1.0;
        gui_set_slider_abs(g, 16, v);
    }
    if (tca_r != 0) {
        double v = tca_r;
        if (v < -0.3) v = -0.3;
        if (v > 0.3) v = 0.3;
        gui_set_slider_abs(g, 17, v);
    }
    if (tca_b != 0) {
        double v = tca_b;
        if (v < -0.3) v = -0.3;
        if (v > 0.3) v = 0.3;
        gui_set_slider_abs(g, 18, v);
    }
    if (vig != 0) {
        double v = vig;
        if (v < -0.5) v = -0.5;
        if (v > 0.5) v = 0.5;
        gui_set_slider_abs(g, 19, v);
    }
    
    g->draft = 0;
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

/* Build the right-hand browser: one expander per category, each holding a
   button per filter.  The category colour comes from category.json. */

/* folder browser */
static void on_folder_open(GtkButton *btn, gpointer data) {
    FeGui *g = (FeGui *)data;
    const char *path = (const char *)g_object_get_data(G_OBJECT(btn), "fe-path");
    if (path) {
        gui_status(g, "Decoding…");
        while (gtk_events_pending()) gtk_main_iteration();
        if (fe_app_load(g->app, path)) {
            gui_status(g, "%s", path);
            g->draft = 0;
            gui_schedule_render(g);
        } else {
            gui_status(g, "Could not decode %s", path);
        }
    }
}

static GdkPixbuf *generate_tiny_thumb(const char *path) {
    /* Load a small preview for the thumbnail */
    FeImage img;
    memset(&img, 0, sizeof(img));
    char date[64] = "";
    if (!fe_load_image(path, 120, &img, date, sizeof(date))) return NULL;
    /* Apply quick display-like tonemap */
    {
        int n = img.w * img.h * 3;
        float *p = img.data;
        for (int i = 0; i < n; i++) {
            float v = p[i];
            v = v * 1.15f;
            if (v <= 0.0031308f) p[i] = v * 12.92f;
            else p[i] = 1.055f * powf(v, 1.0f / 2.4f) - 0.055f;
            if (p[i] < 0.0f) p[i] = 0.0f;
            if (p[i] > 1.0f) p[i] = 1.0f;
        }
    }
    /* Convert to 8-bit RGB */
    int w = img.w, h = img.h;
    unsigned char *rgb = (unsigned char *)malloc((size_t)w * h * 3);
    for (int i = 0; i < w * h * 3; i++) {
        rgb[i] = (unsigned char)(img.data[i] * 255.0f);
    }
    fe_img_free(&img);
    GdkPixbuf *pixbuf = gdk_pixbuf_new_from_data(rgb, GDK_COLORSPACE_RGB, FALSE, 8, w, h, w * 3,
                                                   (GdkPixbufDestroyNotify)free, NULL);
    return pixbuf;
}

static void gui_load_folder(FeGui *g, const char *folder) {
    if (!g->folder_box) return;
    GList *children = gtk_container_get_children(GTK_CONTAINER(g->folder_box));
    for (GList *l = children; l; l = l->next) gtk_widget_destroy(GTK_WIDGET(l->data));
    g_list_free(children);
    DIR *d = opendir(folder);
    if (!d) return;
    struct dirent *ent;
    while ((ent = readdir(d))) {
        if (ent->d_name[0] == '.') continue;
        const char *ext = strrchr(ent->d_name, '.');
        if (!ext) continue;
        /* Only show ARW files as requested */
        if (strcasecmp(ext, ".arw") != 0 && strcasecmp(ext, ".raw") != 0 && 
            strcasecmp(ext, ".cr2") != 0 && strcasecmp(ext, ".nef") != 0 &&
            strcasecmp(ext, ".dng") != 0 && strcasecmp(ext, ".raf") != 0) continue;
        char path[2048];
        snprintf(path, sizeof(path), "%s/%s", folder, ent->d_name);
        GtkWidget *box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 4);
        GtkWidget *img = gtk_image_new();
        /* Generate thumbnail */
        GdkPixbuf *thumb = generate_tiny_thumb(path);
        if (thumb) {
            /* Scale to small size */
            GdkPixbuf *scaled = gdk_pixbuf_scale_simple(thumb, 48, 32, GDK_INTERP_BILINEAR);
            gtk_image_set_from_pixbuf(GTK_IMAGE(img), scaled);
            g_object_unref(scaled);
            g_object_unref(thumb);
        } else {
            gtk_image_set_from_icon_name(GTK_IMAGE(img), "image-x-generic", GTK_ICON_SIZE_BUTTON);
        }
        gtk_box_pack_start(GTK_BOX(box), img, FALSE, FALSE, 0);
        GtkWidget *lbl = gtk_label_new(ent->d_name);
        gtk_label_set_xalign(GTK_LABEL(lbl), 0.0f);
        gtk_label_set_ellipsize(GTK_LABEL(lbl), PANGO_ELLIPSIZE_END);
        gtk_box_pack_start(GTK_BOX(box), lbl, TRUE, TRUE, 0);
        GtkWidget *btn = gtk_button_new();
        gtk_container_add(GTK_CONTAINER(btn), box);
        gtk_button_set_relief(GTK_BUTTON(btn), GTK_RELIEF_NONE);
        g_object_set_data_full(G_OBJECT(btn), "fe-path", g_strdup(path), g_free);
        g_signal_connect(btn, "clicked", G_CALLBACK(on_folder_open), g);
        gtk_box_pack_start(GTK_BOX(g->folder_box), btn, FALSE, FALSE, 2);
    }
    closedir(d);
    gtk_widget_show_all(g->folder_scroll);
    gui_status(g, "Folder: %s (ARW/RAW only)", folder);
}

static void on_browse_folder(GtkButton *btn, gpointer data) {
    FeGui *g = (FeGui *)data;
    GtkWidget *dlg = gtk_file_chooser_dialog_new("Browse Folder", GTK_WINDOW(g->window),
        GTK_FILE_CHOOSER_ACTION_SELECT_FOLDER, "_Cancel", GTK_RESPONSE_CANCEL, "_Select", GTK_RESPONSE_ACCEPT, NULL);
    if (gtk_dialog_run(GTK_DIALOG(dlg)) == GTK_RESPONSE_ACCEPT) {
        char *path = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(dlg));
        if (path) { gui_load_folder(g, path); g_free(path); }
    }
    gtk_widget_destroy(dlg);
}

static void on_add_project(GtkButton *btn, gpointer data) {
    FeGui *g = (FeGui *)data;
    if (!g->project) g->project = fe_project_new();
    GtkWidget *dlg = gtk_file_chooser_dialog_new("Add to Project", GTK_WINDOW(g->window),
        GTK_FILE_CHOOSER_ACTION_OPEN, "_Cancel", GTK_RESPONSE_CANCEL, "_Add", GTK_RESPONSE_ACCEPT, NULL);
    gtk_file_chooser_set_select_multiple(GTK_FILE_CHOOSER(dlg), TRUE);
    GtkFileFilter *ff = gtk_file_filter_new();
    gtk_file_filter_set_name(ff, "Images");
    gtk_file_filter_add_pattern(ff, "*.ARW"); gtk_file_filter_add_pattern(ff, "*.arw");
    gtk_file_filter_add_pattern(ff, "*.jpg"); gtk_file_filter_add_pattern(ff, "*.jpeg");
    gtk_file_filter_add_pattern(ff, "*.png"); gtk_file_filter_add_pattern(ff, "*.raw");
    gtk_file_chooser_add_filter(GTK_FILE_CHOOSER(dlg), ff);
    if (gtk_dialog_run(GTK_DIALOG(dlg)) == GTK_RESPONSE_ACCEPT) {
        GSList *files = gtk_file_chooser_get_filenames(GTK_FILE_CHOOSER(dlg));
        int added = 0;
        for (GSList *l = files; l; l = l->next) { if (fe_project_add(g->project, (char *)l->data)) added++; g_free(l->data); }
        g_slist_free(files);
        gui_status(g, "Project: %d images added", added);
    }
    gtk_widget_destroy(dlg);
}

static void on_mass_export_btn(GtkButton *btn, gpointer data) {
    FeGui *g = (FeGui *)data;
    if (!g->project || g->project->count == 0) { gui_status(g, "Project empty"); return; }
    GtkWidget *dlg = gtk_file_chooser_dialog_new("Mass Export", GTK_WINDOW(g->window),
        GTK_FILE_CHOOSER_ACTION_SELECT_FOLDER, "_Cancel", GTK_RESPONSE_CANCEL, "_Export", GTK_RESPONSE_ACCEPT, NULL);
    if (gtk_dialog_run(GTK_DIALOG(dlg)) == GTK_RESPONSE_ACCEPT) {
        char *dir = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(dlg));
        if (dir) {
            free(g->project->export_dir);
            g->project->export_dir = strdup(dir);
            gui_status(g, "Exporting...");
            while (gtk_events_pending()) gtk_main_iteration();
            int ok = fe_project_export(g->project, g->app, g->current ? g->current->key : "00-basics/original", NULL, NULL);
            gui_status(g, "Exported %d", ok);
            g_free(dir);
        }
    }
    gtk_widget_destroy(dlg);
}

static void gui_build_browser(FeGui *g, GtkWidget *parent) {
    int total = 0;
    for (int i = 0; i < g->app->filters.count; i++)
        total += g->app->filters.cats[i].count;
    g->buttons = (GtkWidget **)calloc((size_t)(total > 0 ? total : 1), sizeof(GtkWidget *));
    g->nbuttons = total;

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
        gtk_expander_set_expanded(GTK_EXPANDER(exp), c == 0);

        GtkWidget *box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 4);
        gtk_container_set_border_width(GTK_CONTAINER(box), 8);

        for (int j = 0; j < cat->count; j++) {
            FeFilter *f = &cat->filters[j];
            GtkWidget *btn = gtk_button_new_with_label(f->label);
            gtk_button_set_relief(GTK_BUTTON(btn), GTK_RELIEF_NONE);
            gtk_style_context_add_class(gtk_widget_get_style_context(btn), cls);
            g_object_set_data(G_OBJECT(btn), "fe-index", GINT_TO_POINTER(index));
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
   button and re-render. */
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
    g->draft = 0;

    /* the sliders rest at 0 — the filter exactly as authored */
    for (int i = 0; i < NSLIDERS; i++)
        gtk_range_set_value(GTK_RANGE(g->scale[i]), g->rest[i]);

    /* highlight the active button */
    for (int i = 0; i < g->nbuttons; i++) {
        if (!g->buttons[i]) continue;
        GtkStyleContext *ctx = gtk_widget_get_style_context(g->buttons[i]);
        if (i == index) gtk_style_context_add_class(ctx, "feactive");
        else gtk_style_context_remove_class(ctx, "feactive");
    }

    gui_status(g, "%s — %s", flt->label,
               flt->description && flt->description[0] ? flt->description : flt->key);
    gui_schedule_render(g);
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
    gtk_file_filter_add_pattern(ff, "*.jpg");
    gtk_file_filter_add_pattern(ff, "*.jpeg");
    gtk_file_filter_add_pattern(ff, "*.png");
    gtk_file_filter_add_pattern(ff, "*.tif");
    gtk_file_filter_add_pattern(ff, "*.tiff");
    gtk_file_chooser_add_filter(GTK_FILE_CHOOSER(dlg), ff);
    if (g->app->filter_root[0])
        gtk_file_chooser_set_current_folder(GTK_FILE_CHOOSER(dlg), g->app->filter_root);

    if (gtk_dialog_run(GTK_DIALOG(dlg)) == GTK_RESPONSE_ACCEPT) {
        char *path = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(dlg));
        if (path) {
            gui_status(g, "Decoding…");
            while (gtk_events_pending()) gtk_main_iteration();
            if (fe_app_load(g->app, path)) {
                gui_status(g, "%s", path);
                g->draft = 0;
                gui_schedule_render(g);
            } else {
                gui_status(g, "Could not decode %s", path);
            }
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
    gui_reset_sliders((FeGui *)data);
}

/* Left/right step through the filters while the preview has the focus. */
static gboolean on_preview_key(GtkWidget *w, GdkEventKey *ev, gpointer data) {
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

int fe_gui_run(FeApp *app) {
    FeGui g;
    memset(&g, 0, sizeof(g));
    g.app = app;

    gtk_init(NULL, NULL);

    g.window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    gtk_window_set_title(GTK_WINDOW(g.window), "SAGE RAW Studio — C edition");
    gtk_window_set_default_size(GTK_WINDOW(g.window), 1400, 880);
    gtk_window_set_position(GTK_WINDOW(g.window), GTK_WIN_POS_CENTER);
    g_signal_connect(g.window, "destroy", G_CALLBACK(gtk_main_quit), NULL);

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
    GtkWidget *browser_scroll = gtk_scrolled_window_new(NULL, NULL);
    gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(browser_scroll),
                                   GTK_POLICY_NEVER, GTK_POLICY_AUTOMATIC);
    gtk_widget_set_size_request(browser_scroll, 250, -1);
    gtk_container_add(GTK_CONTAINER(browser_scroll), browser_box);
    gui_build_browser(&g, browser_box);

    /* the preview + sidebar share one paned widget, the browser another */
    GtkWidget *paned = gtk_paned_new(GTK_ORIENTATION_HORIZONTAL);
    gtk_paned_set_position(GTK_PANED(paned), 320);
    gtk_paned_pack1(GTK_PANED(paned), left, FALSE, TRUE);
    gtk_paned_pack2(GTK_PANED(paned), g.preview, TRUE, TRUE);

    GtkWidget *outer = gtk_paned_new(GTK_ORIENTATION_HORIZONTAL);
    gtk_paned_set_position(GTK_PANED(outer), 1080);
    gtk_paned_pack1(GTK_PANED(outer), paned, TRUE, TRUE);
    gtk_paned_pack2(GTK_PANED(outer), browser_scroll, FALSE, TRUE);

    /* --- the status line ------------------------------------------------ */
    g.status = gtk_label_new("Open an image to start");
    gtk_label_set_xalign(GTK_LABEL(g.status), 0.0f);
    gtk_label_set_ellipsize(GTK_LABEL(g.status), PANGO_ELLIPSIZE_END);
    GtkWidget *status_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);
    gtk_container_set_border_width(GTK_CONTAINER(status_box), 6);
    gtk_box_pack_start(GTK_BOX(status_box), g.status, TRUE, TRUE, 0);

    /* bottom slider/browser */
    g.folder_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 4);
    GtkWidget *fb = gtk_box_new(GTK_ORIENTATION_VERTICAL, 2);
    GtkWidget *ft = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 4);
    GtkWidget *bf = gtk_button_new_with_label("Browse Folder");
    g_signal_connect(bf, "clicked", G_CALLBACK(on_browse_folder), &g);
    GtkWidget *apb = gtk_button_new_with_label("Add to Project");
    g_signal_connect(apb, "clicked", G_CALLBACK(on_add_project), &g);
    GtkWidget *meb = gtk_button_new_with_label("Mass Export");
    g_signal_connect(meb, "clicked", G_CALLBACK(on_mass_export_btn), &g);
    gtk_box_pack_start(GTK_BOX(ft), bf, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(ft), apb, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(ft), meb, FALSE, FALSE, 0);
    GtkWidget *ftitle = gtk_label_new("FOLDER / PROJECT");
    gtk_label_set_xalign(GTK_LABEL(ftitle), 0.0f);
    gtk_box_pack_start(GTK_BOX(fb), ftitle, FALSE, FALSE, 2);
    gtk_box_pack_start(GTK_BOX(fb), ft, FALSE, FALSE, 2);
    g.folder_scroll = gtk_scrolled_window_new(NULL, NULL);
    gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(g.folder_scroll), GTK_POLICY_AUTOMATIC, GTK_POLICY_NEVER);
    gtk_widget_set_size_request(g.folder_scroll, -1, 95);
    gtk_container_add(GTK_CONTAINER(g.folder_scroll), g.folder_box);
    gtk_box_pack_start(GTK_BOX(fb), g.folder_scroll, TRUE, TRUE, 0);
    GtkWidget *ff = gtk_frame_new(NULL);
    gtk_container_add(GTK_CONTAINER(ff), fb);

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
    gtk_widget_grab_focus(g.preview);

    /* start on the first filter, like the Python app */
    if (g.nbuttons > 0) gui_select_filter(&g, 0);

    gtk_main();

    if (g.timeout_id) g_source_remove(g.timeout_id);
    if (g.settle_id) g_source_remove(g.settle_id);
    if (g.surface) cairo_surface_destroy(g.surface);
    free(g.rgb);
    free(g.buttons);
    return 0;
}
