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
   Three rules keep it honest:

     * nothing carries over between photos — a photo that has never been
       edited starts from Original (00-basics/original) with every slider
       at rest;
     * merely viewing a photo never writes a sidecar — `dirty` is raised
       only when the user actually changes a filter or slider;
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
#include "../vendor/cJSON.h"

#include <gdk-pixbuf/gdk-pixbuf.h>
#include <gtk/gtk.h>

#include <dirent.h>
#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>

/* declared in fe_lens_auto.c */
void fe_lens_auto_analyze_image(const FeImage *linear, double *k1_out, double *tca_r_out,
                                double *tca_b_out, double *vignette_out, int *has_any);

/* the long edge of a film-strip preview (the camera's embedded JPEG) */
#define THUMB_LONG_EDGE 140

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
    {"Sharpening",   "sharpening_amount", S(sharpening_amount), -0.0,  1.0,  0.005,  0.0, TAB_LENS},
    {"Pixel Size",   "pixel_size",       S(pixel_size),       -0.0,  12.0, 0.5,    0.0, TAB_LENS},
};
#define NSLIDERS ((int)(sizeof(SLIDERS) / sizeof(SLIDERS[0])))

/* which slider holds the exposure (auto-exposure writes into it) */
#define SLIDER_EXPOSURE 0

static int slider_by_key(const char *key) {
    for (int i = 0; i < NSLIDERS; i++)
        if (strcmp(SLIDERS[i].key, key) == 0) return i;
    return -1;
}

/* ------------------------------------------------------------------ */
/* UI themes & customization                                            */
/* ------------------------------------------------------------------ */

typedef struct {
    const char *key;
    const char *label;
    int is_dark;
    const char *bg_main;       /* window / workspace background */
    const char *bg_sidebar;    /* sidebars & elevated panels */
    const char *bg_surface;    /* cards, buttons, slider troughs */
    const char *bg_hover;      /* hover surface */
    const char *canvas_bg;     /* default preview viewport backdrop */
    const char *fg_primary;    /* primary text */
    const char *fg_muted;      /* secondary / muted text */
    const char *accent;        /* primary accent (highlights, badges, sliders) */
    const char *accent_fg;     /* text on accent background */
    const char *border;        /* subtle panel/card border */
    const char *danger;        /* destructive button color */
    const char *success;       /* export / primary CTA color */
} FeTheme;

static const FeTheme FE_THEMES[] = {
    {
        "gruvbox-dark", "Gruvbox Dark", 1,
        "#1d2021", "#282828", "#3c3836", "#504945", "#141617",
        "#ebdbb2", "#a89984", "#fabd2f", "#1d2021", "#504945",
        "#cc241d", "#689d6a"
    },
    {
        "gruvbox-light", "Gruvbox Light", 0,
        "#f9f5d7", "#fbf1c7", "#ebdbb2", "#d5c4a1", "#e0d3af",
        "#3c3836", "#7c6f64", "#d65d0e", "#fbf1c7", "#d5c4a1",
        "#9d0006", "#427b58"
    },
    {
        "studio-dark", "Studio Dark (Lightroom)", 1,
        "#121316", "#1a1b1f", "#26282e", "#32353d", "#0b0c0e",
        "#eceef2", "#8e939e", "#ffd479", "#121316", "#2f323a",
        "#9e3b3b", "#237a4b"
    },
    {
        "sage-atelier", "Sage Atelier", 1,
        "#131815", "#1b221e", "#26312b", "#324038", "#0d110f",
        "#e2ebe5", "#8fa396", "#7fa98e", "#111714", "#2e3b33",
        "#9e4444", "#3d855c"
    },
    {
        "catppuccin-mocha", "Catppuccin Mocha", 1,
        "#181825", "#1e1e2e", "#313244", "#45475a", "#11111b",
        "#cdd6f4", "#a6adc8", "#cba6f7", "#11111b", "#45475a",
        "#f38ba8", "#a6e3a1"
    },
    {
        "nord", "Nord Polar", 1,
        "#242933", "#2e3440", "#3b4252", "#434c5e", "#1d212a",
        "#eceff4", "#9aa5b8", "#88c0d0", "#242933", "#4c566a",
        "#bf616a", "#a3be8c"
    },
    {
        "tokyo-night", "Tokyo Night", 1,
        "#16161e", "#1a1b26", "#24283b", "#2f3549", "#101016",
        "#c0caf5", "#7982a9", "#7aa2f7", "#16161e", "#3b4261",
        "#f7768e", "#73daca"
    },
    {
        "dracula", "Dracula", 1,
        "#1e1f29", "#282a36", "#383a4c", "#44475a", "#15161d",
        "#f8f8f2", "#9499b3", "#bd93f9", "#1e1f29", "#44475a",
        "#ff5555", "#50fa7b"
    },
    {
        "rose-pine", "Rosé Pine", 1,
        "#191724", "#1f1d2e", "#26233a", "#393552", "#12101a",
        "#e0def4", "#908caa", "#f6c177", "#191724", "#403d52",
        "#eb6f92", "#31748f"
    },
    {
        "solarized-dark", "Solarized Dark", 1,
        "#00212b", "#002b36", "#073642", "#0f4654", "#001920",
        "#eee8d5", "#839496", "#b58900", "#00212b", "#164e5c",
        "#dc322f", "#2aa198"
    }
};
#define NTHEMES ((int)(sizeof(FE_THEMES) / sizeof(FE_THEMES[0])))

typedef struct {
    const char *key;
    const char *label;
    const char *hex;   /* NULL = follow active theme */
} FeColorOption;

static const FeColorOption FE_ACCENTS[] = {
    {"theme",   "Theme Default",   NULL},
    {"yellow",  "Gruvbox Yellow",  "#fabd2f"},
    {"orange",  "Gruvbox Orange",  "#fe8019"},
    {"aqua",    "Gruvbox Aqua",    "#8ec07c"},
    {"gold",    "Warm Studio Gold","#ffd479"},
    {"sage",    "Botanical Sage",  "#7fa98e"},
    {"frost",   "Nord Frost Cyan", "#88c0d0"},
    {"mauve",   "Catppuccin Mauve","#cba6f7"},
    {"blue",    "Tokyo Neon Blue", "#7aa2f7"},
    {"rose",    "Rosé Gold",       "#ebbcba"},
    {"crimson", "Warm Coral",      "#fb4934"}
};
#define NACCENTS ((int)(sizeof(FE_ACCENTS) / sizeof(FE_ACCENTS[0])))

static const FeColorOption FE_BACKDROPS[] = {
    {"theme",     "Theme Default",       NULL},
    {"black",     "Pitch Black",         "#050505"},
    {"charcoal",  "Darkroom Charcoal",   "#101113"},
    {"gruvbox",   "Gruvbox Hard Dark",   "#1d2021"},
    {"midgrey",   "18% Neutral Grey",    "#4a4a4a"},
    {"warmpaper", "Warm Gallery Paper",  "#ebdbb2"},
    {"white",     "Pure Gallery White",  "#f7f7f7"}
};
#define NBACKDROPS ((int)(sizeof(FE_BACKDROPS) / sizeof(FE_BACKDROPS[0])))

enum { FRAME_SHADOW = 0, FRAME_MAT = 1, FRAME_BORDER = 2, FRAME_NONE = 3 };
enum { DENSITY_COMPACT = 0, DENSITY_COMFORTABLE = 1, DENSITY_SPACIOUS = 2 };
enum { FILTER_STYLE_MODERN = 0, FILTER_STYLE_CLASSIC = 1 };
enum { GRID_NONE = 0, GRID_THIRDS = 1, GRID_GOLDEN = 2, GRID_CROSSHAIR = 3 };
enum { COMPARE_OFF = 0, COMPARE_SPLIT = 1, COMPARE_BEFORE = 2 };

typedef struct {
    int theme_idx;
    int accent_idx;
    int backdrop_idx;
    int frame_style;      /* FRAME_* */
    int density;          /* DENSITY_* */
    int filter_style;     /* FILTER_STYLE_* */
    int show_histogram;   /* 1 = draw live RGB histogram on preview */
    int grid_mode;        /* GRID_* */
    int show_hud;         /* 1 = draw translucent photo info pill on preview */
} FeUiPrefs;

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

    /* headerbar quick widgets */
    GtkWidget *theme_combo;
    GtkWidget *compare_btn;
    GtkWidget *hist_btn;
    GtkWidget *grid_btn;
    GtkCssProvider *theme_css;

    /* the active filter and the baseline its sliders offset from */
    const FeFilter *current;
    FeSettings base;

    /* slider widgets + their resting values + live numeric value labels */
    GtkWidget *scale[NSLIDERS];
    GtkWidget *val_label[NSLIDERS];
    double rest[NSLIDERS];

    /* every filter button, in a flat array */
    GtkWidget **buttons;
    int nbuttons;
    GtkWidget **expanders;   /* one per category, for search + reveal */
    int nexpanders;
    int active_index;

    /* the rendered preview + original (before) surface for split comparison */
    unsigned char *rgb;
    int rgb_w, rgb_h;
    cairo_surface_t *surface;
    unsigned char *orig_rgb;
    int orig_w, orig_h;
    cairo_surface_t *orig_surface;

    /* before/after comparison state */
    int compare_mode;        /* COMPARE_OFF, COMPARE_SPLIT, COMPARE_BEFORE */
    double split_ratio;      /* 0.05 .. 0.95 for split wipe */
    int dragging_split;

    /* UI customization preferences */
    FeUiPrefs prefs;

    /* render scheduling */
    guint timeout_id;        /* the debounced render */
    guint settle_id;         /* "the drag stopped" timer */
    int draft;               /* 1 = render the cheap draft */

    /* per-image edit state */
    FeEditState state;       /* the photo currently open */
    FeEditState clipboard;   /* copied edits for Copy/Paste Edits */
    int has_clipboard;
    guint save_id;           /* the debounced autosave */
    int dirty;               /* 1 = user changed filter or sliders on this photo */
    int suppress;            /* >0: the PROGRAM moved a slider, save nothing */

    /* the film-strip: previews waiting to be decoded one slice at a time */
    GQueue *thumb_pending;   /* char * paths, owned by the queue */
    guint thumb_idle;
    int alive;               /* 0 once the widgets are gone */

    /* the bottom panel */
    GtkWidget *bottom_tabs;
    GtkWidget *folder_box;
    GtkWidget *folder_page;
    GtkWidget *project_box;
    GtkWidget *project_page;        /* the notebook page wrapping project_box */
    GtkWidget *project_placeholder; /* the "no photos yet" hint */
    FeProject *project;

    /* current folder listing (sorted) */
    char current_folder[FE_PATH_MAX];
    char **folder_paths;
    int folder_count;
} FeGui;


/* forward declarations */
static void gui_render(FeGui *g);
static void gui_schedule_render(FeGui *g);
static void gui_select_filter(FeGui *g, int index);
static void gui_status(FeGui *g, const char *fmt, ...);
static void gui_load_image(FeGui *g, const char *path);
static void gui_save_state_now(FeGui *g);
static void gui_update_project_label(FeGui *g);
static void gui_update_folder_label(FeGui *g);
static void gui_refresh_strip_badges(FeGui *g);
static void gui_highlight_active_photo(FeGui *g);
static void gui_load_folder(FeGui *g, const char *folder, int auto_select_first);
static void gui_build_project(FeGui *g);
static void gui_apply_theme(FeGui *g);
static void gui_save_ui_prefs(const FeGui *g);
static void gui_update_slider_labels(FeGui *g);

/* ------------------------------------------------------------------ */
/* UI preferences persistence & CSS theme engine                        */
/* ------------------------------------------------------------------ */

static void gui_prefs_default(FeUiPrefs *p) {
    memset(p, 0, sizeof(*p));
    p->theme_idx = 0;                     /* Gruvbox Dark by default */
    p->accent_idx = 0;                    /* Theme Default */
    p->backdrop_idx = 0;                  /* Theme Default */
    p->frame_style = FRAME_SHADOW;
    p->density = DENSITY_COMFORTABLE;
    p->filter_style = FILTER_STYLE_MODERN;
    p->show_histogram = 1;
    p->grid_mode = GRID_NONE;
    p->show_hud = 1;
}

static void gui_prefs_path(char *out, size_t out_len) {
    const char *xdg = getenv("XDG_CONFIG_HOME");
    const char *home = getenv("HOME");
    if (xdg && xdg[0]) {
        char dir[FE_PATH_MAX - 32];
        snprintf(dir, sizeof(dir), "%s/fast-edit", xdg);
        mkdir(dir, 0755);
        snprintf(out, out_len, "%s/ui.json", dir);
    } else if (home && home[0]) {
        char cfg[FE_PATH_MAX - 64], dir[FE_PATH_MAX - 32];
        snprintf(cfg, sizeof(cfg), "%s/.config", home);
        mkdir(cfg, 0755);
        snprintf(dir, sizeof(dir), "%s/.config/fast-edit", home);
        mkdir(dir, 0755);
        snprintf(out, out_len, "%s/ui.json", dir);
    } else {
        snprintf(out, out_len, ".fastedit-ui.json");
    }
}

static int theme_index_by_key(const char *key) {
    if (!key) return 0;
    for (int i = 0; i < NTHEMES; i++)
        if (strcmp(FE_THEMES[i].key, key) == 0) return i;
    return 0;
}

static int accent_index_by_key(const char *key) {
    if (!key) return 0;
    for (int i = 0; i < NACCENTS; i++)
        if (strcmp(FE_ACCENTS[i].key, key) == 0) return i;
    return 0;
}

static int backdrop_index_by_key(const char *key) {
    if (!key) return 0;
    for (int i = 0; i < NBACKDROPS; i++)
        if (strcmp(FE_BACKDROPS[i].key, key) == 0) return i;
    return 0;
}

static void gui_load_ui_prefs(FeGui *g) {
    gui_prefs_default(&g->prefs);
    char path[FE_PATH_MAX];
    gui_prefs_path(path, sizeof(path));
    FILE *fp = fopen(path, "rb");
    if (!fp) return;
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (sz <= 0 || sz > 65536) { fclose(fp); return; }
    char *buf = (char *)malloc((size_t)sz + 1);
    if (!buf) { fclose(fp); return; }
    size_t rd = fread(buf, 1, (size_t)sz, fp);
    buf[rd] = '\0';
    fclose(fp);

    cJSON *root = cJSON_Parse(buf);
    free(buf);
    if (!root) return;

    cJSON *it;
    if ((it = cJSON_GetObjectItemCaseSensitive(root, "theme")) && cJSON_IsString(it))
        g->prefs.theme_idx = theme_index_by_key(it->valuestring);
    if ((it = cJSON_GetObjectItemCaseSensitive(root, "accent")) && cJSON_IsString(it))
        g->prefs.accent_idx = accent_index_by_key(it->valuestring);
    if ((it = cJSON_GetObjectItemCaseSensitive(root, "backdrop")) && cJSON_IsString(it))
        g->prefs.backdrop_idx = backdrop_index_by_key(it->valuestring);
    if ((it = cJSON_GetObjectItemCaseSensitive(root, "frame_style")) && cJSON_IsNumber(it))
        g->prefs.frame_style = CLAMP(it->valueint, 0, 3);
    if ((it = cJSON_GetObjectItemCaseSensitive(root, "density")) && cJSON_IsNumber(it))
        g->prefs.density = CLAMP(it->valueint, 0, 2);
    if ((it = cJSON_GetObjectItemCaseSensitive(root, "filter_style")) && cJSON_IsNumber(it))
        g->prefs.filter_style = CLAMP(it->valueint, 0, 1);
    if ((it = cJSON_GetObjectItemCaseSensitive(root, "show_histogram")) && cJSON_IsBool(it))
        g->prefs.show_histogram = cJSON_IsTrue(it) ? 1 : 0;
    else if ((it = cJSON_GetObjectItemCaseSensitive(root, "show_histogram")) && cJSON_IsNumber(it))
        g->prefs.show_histogram = it->valueint ? 1 : 0;
    if ((it = cJSON_GetObjectItemCaseSensitive(root, "grid_mode")) && cJSON_IsNumber(it))
        g->prefs.grid_mode = CLAMP(it->valueint, 0, 3);
    if ((it = cJSON_GetObjectItemCaseSensitive(root, "show_hud")) && cJSON_IsBool(it))
        g->prefs.show_hud = cJSON_IsTrue(it) ? 1 : 0;
    else if ((it = cJSON_GetObjectItemCaseSensitive(root, "show_hud")) && cJSON_IsNumber(it))
        g->prefs.show_hud = it->valueint ? 1 : 0;

    cJSON_Delete(root);
}

static void gui_save_ui_prefs(const FeGui *g) {
    char path[FE_PATH_MAX];
    gui_prefs_path(path, sizeof(path));
    cJSON *root = cJSON_CreateObject();
    if (!root) return;
    cJSON_AddStringToObject(root, "theme", FE_THEMES[g->prefs.theme_idx].key);
    cJSON_AddStringToObject(root, "accent", FE_ACCENTS[g->prefs.accent_idx].key);
    cJSON_AddStringToObject(root, "backdrop", FE_BACKDROPS[g->prefs.backdrop_idx].key);
    cJSON_AddNumberToObject(root, "frame_style", g->prefs.frame_style);
    cJSON_AddNumberToObject(root, "density", g->prefs.density);
    cJSON_AddNumberToObject(root, "filter_style", g->prefs.filter_style);
    cJSON_AddBoolToObject(root, "show_histogram", g->prefs.show_histogram);
    cJSON_AddNumberToObject(root, "grid_mode", g->prefs.grid_mode);
    cJSON_AddBoolToObject(root, "show_hud", g->prefs.show_hud);

    char *txt = cJSON_Print(root);
    cJSON_Delete(root);
    if (!txt) return;
    FILE *fp = fopen(path, "wb");
    if (fp) {
        fwrite(txt, 1, strlen(txt), fp);
        fputc('\n', fp);
        fclose(fp);
    }
    free(txt);
}

static void parse_hex_rgb(const char *hex, double *r, double *gr, double *b) {
    *r = 0.08; *gr = 0.08; *b = 0.09;
    if (!hex || hex[0] != '#' || strlen(hex) < 7) return;
    unsigned int rv = 0, gv = 0, bv = 0;
    if (sscanf(hex + 1, "%02x%02x%02x", &rv, &gv, &bv) == 3) {
        *r = rv / 255.0;
        *gr = gv / 255.0;
        *b = bv / 255.0;
    }
}

static const FeTheme *gui_active_theme(const FeGui *g) {
    int idx = CLAMP(g->prefs.theme_idx, 0, NTHEMES - 1);
    return &FE_THEMES[idx];
}

static const char *gui_active_accent(const FeGui *g) {
    int idx = CLAMP(g->prefs.accent_idx, 0, NACCENTS - 1);
    if (FE_ACCENTS[idx].hex) return FE_ACCENTS[idx].hex;
    return gui_active_theme(g)->accent;
}

static const char *gui_active_backdrop(const FeGui *g) {
    int idx = CLAMP(g->prefs.backdrop_idx, 0, NBACKDROPS - 1);
    if (FE_BACKDROPS[idx].hex) return FE_BACKDROPS[idx].hex;
    return gui_active_theme(g)->canvas_bg;
}

static void gui_apply_theme(FeGui *g) {
    const FeTheme *t = gui_active_theme(g);
    const char *acc = gui_active_accent(g);

    GtkSettings *settings = gtk_settings_get_default();
    if (settings) {
        g_object_set(settings, "gtk-application-prefer-dark-theme",
                     t->is_dark ? TRUE : FALSE, NULL);
    }

    int pad_v = (g->prefs.density == DENSITY_COMPACT) ? 3
              : (g->prefs.density == DENSITY_SPACIOUS) ? 7 : 5;
    int font_px = (g->prefs.density == DENSITY_COMPACT) ? 12
                : (g->prefs.density == DENSITY_SPACIOUS) ? 14 : 13;

    GString *css = g_string_new(NULL);
    g_string_append_printf(
        css,
        "window, .background {\n"
        "  background-color: %s;\n"
        "  color: %s;\n"
        "  font-size: %dpx;\n"
        "}\n"
        "headerbar {\n"
        "  background-image: none;\n"
        "  background-color: %s;\n"
        "  color: %s;\n"
        "  border-bottom: 1px solid %s;\n"
        "  padding: 4px 8px;\n"
        "}\n"
        "headerbar .title {\n"
        "  color: %s;\n"
        "  font-weight: 800;\n"
        "  letter-spacing: 0.6px;\n"
        "}\n"
        "headerbar .subtitle {\n"
        "  color: %s;\n"
        "  font-size: 11px;\n"
        "}\n"
        ".fe-sidebar, .fe-bottom-panel {\n"
        "  background-color: %s;\n"
        "  border-color: %s;\n"
        "}\n"
        ".fe-section-title {\n"
        "  color: %s;\n"
        "  font-weight: bold;\n"
        "  font-size: 11px;\n"
        "  letter-spacing: 1.1px;\n"
        "}\n"
        ".fe-slider-name {\n"
        "  color: %s;\n"
        "  font-weight: 500;\n"
        "}\n"
        ".fe-slider-val {\n"
        "  color: %s;\n"
        "  font-family: monospace;\n"
        "  font-size: 11px;\n"
        "  background-color: %s;\n"
        "  border-radius: 4px;\n"
        "  padding: 1px 6px;\n"
        "}\n"
        ".fe-slider-val-active {\n"
        "  color: %s;\n"
        "  font-family: monospace;\n"
        "  font-size: 11px;\n"
        "  font-weight: bold;\n"
        "  background-color: %s;\n"
        "  border: 1px solid %s;\n"
        "  border-radius: 4px;\n"
        "  padding: 1px 6px;\n"
        "}\n"
        "notebook, notebook > stack {\n"
        "  background-color: %s;\n"
        "  border: none;\n"
        "}\n"
        "notebook > header {\n"
        "  background-color: %s;\n"
        "  border-bottom: 1px solid %s;\n"
        "}\n"
        "notebook > header > tabs > tab {\n"
        "  background-color: transparent;\n"
        "  color: %s;\n"
        "  padding: %dpx 10px;\n"
        "  border: none;\n"
        "  border-bottom: 2px solid transparent;\n"
        "  font-weight: 600;\n"
        "}\n"
        "notebook > header > tabs > tab:checked {\n"
        "  color: %s;\n"
        "  border-bottom: 2px solid %s;\n"
        "  background-color: %s;\n"
        "}\n"
        "button {\n"
        "  background-image: none;\n"
        "  background-color: %s;\n"
        "  color: %s;\n"
        "  border: 1px solid %s;\n"
        "  border-radius: 6px;\n"
        "  padding: %dpx 10px;\n"
        "  box-shadow: none;\n"
        "}\n"
        "button:hover {\n"
        "  background-color: %s;\n"
        "  border-color: %s;\n"
        "}\n"
        "button.fe-btn-accent {\n"
        "  background-color: %s;\n"
        "  color: %s;\n"
        "  border-color: %s;\n"
        "  font-weight: bold;\n"
        "}\n"
        "button.fe-btn-accent:hover {\n"
        "  background-color: shade(%s, 1.1);\n"
        "}\n"
        "button.fe-btn-success {\n"
        "  background-color: %s;\n"
        "  color: #ffffff;\n"
        "  border-color: shade(%s, 1.15);\n"
        "  font-weight: bold;\n"
        "}\n"
        "button.fe-btn-success:hover {\n"
        "  background-color: shade(%s, 1.12);\n"
        "}\n"
        "button.fe-btn-danger {\n"
        "  background-color: %s;\n"
        "  color: #ffffff;\n"
        "  border-color: shade(%s, 1.15);\n"
        "}\n"
        "button.fe-btn-danger:hover {\n"
        "  background-color: shade(%s, 1.12);\n"
        "}\n"
        "button.fe-toggle-on {\n"
        "  background-color: %s;\n"
        "  color: %s;\n"
        "  border-color: %s;\n"
        "  font-weight: bold;\n"
        "}\n"
        "entry, searchentry {\n"
        "  background-image: none;\n"
        "  background-color: %s;\n"
        "  color: %s;\n"
        "  border: 1px solid %s;\n"
        "  border-radius: 6px;\n"
        "  padding: 5px 8px;\n"
        "}\n"
        "entry:focus, searchentry:focus {\n"
        "  border-color: %s;\n"
        "}\n"
        "scale {\n"
        "  padding: 4px 0px;\n"
        "}\n"
        "scale contents trough {\n"
        "  background-color: %s;\n"
        "  border-radius: 4px;\n"
        "  min-height: 6px;\n"
        "  border: 1px solid %s;\n"
        "}\n"
        "scale contents trough highlight {\n"
        "  background-color: %s;\n"
        "  border-radius: 4px;\n"
        "  border-color: %s;\n"
        "}\n"
        "scale contents trough slider {\n"
        "  background-image: none;\n"
        "  background-color: %s;\n"
        "  border: 2px solid %s;\n"
        "  border-radius: 50%%;\n"
        "  min-width: 14px;\n"
        "  min-height: 14px;\n"
        "  margin: -5px 0px;\n"
        "  box-shadow: none;\n"
        "}\n"
        "scale contents trough slider:hover {\n"
        "  background-color: %s;\n"
        "}\n"
        "expander > title {\n"
        "  background-color: %s;\n"
        "  color: %s;\n"
        "  border-radius: 6px;\n"
        "  padding: 5px 8px;\n"
        "  font-weight: bold;\n"
        "}\n"
        "expander > title:hover {\n"
        "  background-color: %s;\n"
        "}\n"
        ".feactive {\n"
        "  border: 2px solid %s;\n"
        "  border-radius: 6px;\n"
        "  font-weight: bold;\n"
        "}\n"
        ".fethumb-card {\n"
        "  background-color: %s;\n"
        "  border: 1px solid %s;\n"
        "  border-radius: 8px;\n"
        "  padding: 4px;\n"
        "}\n"
        ".fethumb-active {\n"
        "  border: 2px solid %s;\n"
        "  background-color: %s;\n"
        "  border-radius: 8px;\n"
        "}\n"
        ".febadge-edited {\n"
        "  color: %s;\n"
        "  font-size: 11px;\n"
        "  font-weight: bold;\n"
        "}\n"
        ".febadge-clean {\n"
        "  color: %s;\n"
        "  font-size: 11px;\n"
        "}\n"
        ".fe-status-bar {\n"
        "  background-color: %s;\n"
        "  border-top: 1px solid %s;\n"
        "  color: %s;\n"
        "  font-size: 12px;\n"
        "}\n",
        t->bg_main, t->fg_primary, font_px,
        t->bg_sidebar, t->fg_primary, t->border,
        acc, t->fg_muted,
        t->bg_sidebar, t->border,
        acc,
        t->fg_primary,
        t->fg_muted, t->bg_surface,
        acc, t->bg_surface, acc,
        t->bg_sidebar,
        t->bg_main, t->border,
        t->fg_muted, pad_v,
        acc, acc, t->bg_sidebar,
        t->bg_surface, t->fg_primary, t->border, pad_v,
        t->bg_hover, acc,
        acc, t->accent_fg, acc, acc,
        t->success, t->success, t->success,
        t->danger, t->danger, t->danger,
        acc, t->accent_fg, acc,
        t->bg_surface, t->fg_primary, t->border,
        acc,
        t->bg_surface, t->border,
        acc, acc,
        t->fg_primary, acc, acc,
        t->bg_surface, t->fg_primary, t->bg_hover,
        acc,
        t->bg_surface, t->border,
        acc, t->bg_hover,
        acc, t->fg_muted,
        t->bg_main, t->border, t->fg_muted
    );

    /* Per-category filter button styles: either Modern (sleek surface card with
       colored left border indicator) or Classic (full category color fill). */
    for (int c = 0; c < g->app->filters.count; c++) {
        FeCategory *cat = &g->app->filters.cats[c];
        if (g->prefs.filter_style == FILTER_STYLE_MODERN) {
            g_string_append_printf(
                css,
                "button.fecat%d {\n"
                "  background-image: none;\n"
                "  background-color: %s;\n"
                "  color: %s;\n"
                "  border: 1px solid %s;\n"
                "  border-left: 5px solid %s;\n"
                "  border-radius: 6px;\n"
                "  padding: %dpx 8px;\n"
                "}\n"
                "button.fecat%d:hover {\n"
                "  background-color: %s;\n"
                "  border-color: %s;\n"
                "  border-left: 5px solid %s;\n"
                "}\n"
                "button.fecat%d.feactive {\n"
                "  background-color: %s;\n"
                "  border: 2px solid %s;\n"
                "  border-left: 6px solid %s;\n"
                "  color: %s;\n"
                "}\n",
                c, t->bg_surface, t->fg_primary, t->border, cat->colour, pad_v,
                c, t->bg_hover, cat->colour, cat->colour,
                c, t->bg_hover, acc, cat->colour, acc
            );
        } else {
            g_string_append_printf(
                css,
                "button.fecat%d {\n"
                "  background-image: none;\n"
                "  background-color: %s;\n"
                "  color: #ffffff;\n"
                "  border-radius: 6px;\n"
                "  padding: %dpx 8px;\n"
                "}\n"
                "button.fecat%d:hover {\n"
                "  background-color: shade(%s, 1.15);\n"
                "}\n"
                "button.fecat%d.feactive {\n"
                "  border: 2px solid %s;\n"
                "}\n",
                c, cat->colour, pad_v,
                c, cat->colour,
                c, acc
            );
        }
    }

    if (!g->theme_css) {
        g->theme_css = gtk_css_provider_new();
        gtk_style_context_add_provider_for_screen(
            gdk_screen_get_default(), GTK_STYLE_PROVIDER(g->theme_css), 950);
    }
    gtk_css_provider_load_from_data(g->theme_css, css->str, -1, NULL);
    g_string_free(css, TRUE);

    if (g->preview) gtk_widget_queue_draw(g->preview);
}

/* ------------------------------------------------------------------ */
/* small helpers                                                        */
/* ------------------------------------------------------------------ */

static const char *gui_basename(const char *path) {
    if (!path) return "";
    const char *p = strrchr(path, '/');
    return p ? p + 1 : path;
}

static void gui_status(FeGui *g, const char *fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    if (g->status) gtk_label_set_text(GTK_LABEL(g->status), buf);
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

static void gui_update_slider_labels(FeGui *g) {
    for (int i = 0; i < NSLIDERS; i++) {
        if (!g->scale[i] || !g->val_label[i]) continue;
        double offset = gtk_range_get_value(GTK_RANGE(g->scale[i])) - g->rest[i];
        char buf[32];
        if (SLIDERS[i].hi - SLIDERS[i].lo >= 20.0)
            snprintf(buf, sizeof(buf), "%+.1f", offset);
        else if (SLIDERS[i].step < 0.005)
            snprintf(buf, sizeof(buf), "%+.3f", offset);
        else
            snprintf(buf, sizeof(buf), "%+.2f", offset);
        gtk_label_set_text(GTK_LABEL(g->val_label[i]), buf);
        GtkStyleContext *ctx = gtk_widget_get_style_context(g->val_label[i]);
        if (fabs(offset) >= 1e-4) {
            gtk_style_context_add_class(ctx, "fe-slider-val-active");
            gtk_style_context_remove_class(ctx, "fe-slider-val");
        } else {
            gtk_style_context_add_class(ctx, "fe-slider-val");
            gtk_style_context_remove_class(ctx, "fe-slider-val-active");
        }
    }
}

/* Format a short summary of a photo's saved sidecar for its film-strip badge. */
static void gui_format_badge(FeGui *g, const char *image_path, char *out, size_t out_len) {
    FeEditState st;
    memset(&st, 0, sizeof(st));
    if (!fe_state_load(image_path, &st)) {
        snprintf(out, out_len, "Unedited");
        return;
    }
    const char *flt_label = "Original";
    if (st.filter_key[0]) {
        const FeFilter *flt = fe_filters_find(&g->app->filters, st.filter_key);
        if (flt && flt->label) flt_label = flt->label;
        else flt_label = st.filter_key;
    }
    int is_orig = (!st.filter_key[0] || strcmp(st.filter_key, "00-basics/original") == 0);
    if (is_orig && st.nfields == 0) {
        snprintf(out, out_len, "Original");
    } else if (st.nfields > 0) {
        snprintf(out, out_len, "● %s (+%d)", flt_label, st.nfields);
    } else {
        snprintf(out, out_len, "● %s", flt_label);
    }
}

static int gui_photo_has_meaningful_edit(const char *image_path) {
    FeEditState st;
    memset(&st, 0, sizeof(st));
    if (!fe_state_load(image_path, &st)) return 0;
    int is_orig = (!st.filter_key[0] || strcmp(st.filter_key, "00-basics/original") == 0);
    return (!is_orig || st.nfields > 0);
}

/* ------------------------------------------------------------------ */
/* the preview & Cairo overlays                                         */
/* ------------------------------------------------------------------ */

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

static void cairo_rounded_rect(cairo_t *cr, double x, double y, double w, double h, double r) {
    cairo_new_sub_path(cr);
    cairo_arc(cr, x + w - r, y + r, r, -M_PI / 2.0, 0.0);
    cairo_arc(cr, x + w - r, y + h - r, r, 0.0, M_PI / 2.0);
    cairo_arc(cr, x + r, y + h - r, r, M_PI / 2.0, M_PI);
    cairo_arc(cr, x + r, y + r, r, M_PI, 3.0 * M_PI / 2.0);
    cairo_close_path(cr);
}

/* Draw a live 64-bin RGB + Luma histogram card in the top-right corner. */
static void draw_histogram_overlay(cairo_t *cr, const FeGui *g, int aw) {
    if (!g->rgb || g->rgb_w <= 0 || g->rgb_h <= 0) return;

    const int BINS = 64;
    int hr[64] = {0}, hg[64] = {0}, hb[64] = {0}, hl[64] = {0};
    int total = g->rgb_w * g->rgb_h;
    int step = MAX(1, total / 16000);
    for (int i = 0; i < total; i += step) {
        unsigned int r = g->rgb[i * 3 + 0];
        unsigned int gr = g->rgb[i * 3 + 1];
        unsigned int b = g->rgb[i * 3 + 2];
        unsigned int lum = (r * 54 + gr * 183 + b * 19) >> 8;
        hr[(r * BINS) >> 8]++;
        hg[(gr * BINS) >> 8]++;
        hb[(b * BINS) >> 8]++;
        hl[(lum * BINS) >> 8]++;
    }
    int peak = 1;
    for (int b = 1; b < BINS - 1; b++) {
        if (hr[b] > peak) peak = hr[b];
        if (hg[b] > peak) peak = hg[b];
        if (hb[b] > peak) peak = hb[b];
    }

    double cw = 196.0, ch = 88.0;
    double cx = aw - cw - 16.0, cy = 16.0;
    if (cx < 12.0) return;

    cairo_save(cr);
    cairo_rounded_rect(cr, cx, cy, cw, ch, 8.0);
    cairo_set_source_rgba(cr, 0.07, 0.08, 0.09, 0.80);
    cairo_fill_preserve(cr);
    cairo_set_source_rgba(cr, 1.0, 1.0, 1.0, 0.14);
    cairo_set_line_width(cr, 1.0);
    cairo_stroke(cr);

    double px = cx + 10.0, py = cy + 20.0;
    double pw = cw - 20.0, ph = ch - 28.0;

    /* channel curves: Red, Green, Blue, Luma */
    const int *chans[4] = {hr, hg, hb, hl};
    const double cols[4][4] = {
        {0.98, 0.32, 0.32, 0.38},
        {0.35, 0.88, 0.48, 0.38},
        {0.38, 0.62, 0.98, 0.38},
        {0.92, 0.90, 0.84, 0.25}
    };
    for (int c = 0; c < 4; c++) {
        cairo_new_path(cr);
        cairo_move_to(cr, px, py + ph);
        for (int b = 0; b < BINS; b++) {
            double bx = px + (double)b / (BINS - 1) * pw;
            double v = MIN(1.0, (double)chans[c][b] / (double)peak);
            double by = py + ph - v * ph;
            cairo_line_to(cr, bx, by);
        }
        cairo_line_to(cr, px + pw, py + ph);
        cairo_close_path(cr);
        cairo_set_source_rgba(cr, cols[c][0], cols[c][1], cols[c][2], cols[c][3]);
        cairo_fill_preserve(cr);
        cairo_set_source_rgba(cr, cols[c][0], cols[c][1], cols[c][2], 0.85);
        cairo_set_line_width(cr, 1.1);
        cairo_stroke(cr);
    }

    cairo_set_source_rgba(cr, 0.85, 0.85, 0.82, 0.80);
    cairo_set_font_size(cr, 9.5);
    cairo_move_to(cr, cx + 10.0, cy + 13.0);
    cairo_show_text(cr, "HISTOGRAM  RGB / LUMA");
    cairo_restore(cr);
}

static void draw_grid_overlay(cairo_t *cr, int mode, double x, double y, double w, double h) {
    if (mode == GRID_NONE) return;
    cairo_save(cr);
    cairo_set_source_rgba(cr, 1.0, 1.0, 1.0, 0.38);
    cairo_set_line_width(cr, 1.0);

    if (mode == GRID_THIRDS) {
        for (int i = 1; i <= 2; i++) {
            double gx = x + w * (i / 3.0);
            double gy = y + h * (i / 3.0);
            cairo_move_to(cr, gx, y); cairo_line_to(cr, gx, y + h);
            cairo_move_to(cr, x, gy); cairo_line_to(cr, x + w, gy);
        }
    } else if (mode == GRID_GOLDEN) {
        const double r1 = 0.381966, r2 = 0.618034;
        double xs[2] = {x + w * r1, x + w * r2};
        double ys[2] = {y + h * r1, y + h * r2};
        for (int i = 0; i < 2; i++) {
            cairo_move_to(cr, xs[i], y); cairo_line_to(cr, xs[i], y + h);
            cairo_move_to(cr, x, ys[i]); cairo_line_to(cr, x + w, ys[i]);
        }
    } else if (mode == GRID_CROSSHAIR) {
        cairo_move_to(cr, x + w * 0.5, y); cairo_line_to(cr, x + w * 0.5, y + h);
        cairo_move_to(cr, x, y + h * 0.5); cairo_line_to(cr, x + w, y + h * 0.5);
        cairo_move_to(cr, x, y); cairo_line_to(cr, x + w, y + h);
        cairo_move_to(cr, x + w, y); cairo_line_to(cr, x, y + h);
    }
    cairo_stroke(cr);
    cairo_restore(cr);
}

static gboolean on_draw_preview(GtkWidget *da, cairo_t *cr, gpointer data) {
    FeGui *g = (FeGui *)data;
    int aw = gtk_widget_get_allocated_width(da);
    int ah = gtk_widget_get_allocated_height(da);

    if (getenv("FE_DEBUG_GUI"))
        fprintf(stderr, "[gui] draw area %dx%d surface=%p rgb=%dx%d\n", aw, ah,
                (void *)g->surface, g->rgb_w, g->rgb_h);

    double br = 0.08, bg = 0.08, bb = 0.09;
    parse_hex_rgb(gui_active_backdrop(g), &br, &bg, &bb);
    cairo_set_source_rgb(cr, br, bg, bb);
    cairo_paint(cr);

    double ar = 0.98, ag = 0.74, ab = 0.18;
    parse_hex_rgb(gui_active_accent(g), &ar, &ag, &ab);

    if (!g->surface) {
        double cw = MIN(420.0, aw - 40.0), ch = 150.0;
        double cx = (aw - cw) * 0.5, cy = (ah - ch) * 0.5;
        cairo_rounded_rect(cr, cx, cy, cw, ch, 12.0);
        cairo_set_source_rgba(cr, 0.08, 0.09, 0.10, 0.72);
        cairo_fill_preserve(cr);
        cairo_set_source_rgba(cr, ar, ag, ab, 0.35);
        cairo_set_line_width(cr, 1.5);
        cairo_stroke(cr);

        cairo_set_source_rgba(cr, ar, ag, ab, 0.95);
        cairo_set_font_size(cr, 16.0);
        cairo_move_to(cr, cx + 28.0, cy + 44.0);
        cairo_show_text(cr, g->app->has_image ? "Developing…" : "SAGE RAW STUDIO");

        cairo_set_source_rgba(cr, 0.85, 0.85, 0.82, 0.85);
        cairo_set_font_size(cr, 13.0);
        cairo_move_to(cr, cx + 28.0, cy + 74.0);
        cairo_show_text(cr, "Open a RAW/image file (Ctrl+O) or browse a folder to begin.");

        cairo_set_source_rgba(cr, 0.65, 0.65, 0.62, 0.80);
        cairo_set_font_size(cr, 11.5);
        cairo_move_to(cr, cx + 28.0, cy + 104.0);
        cairo_show_text(cr, "Shortcuts:  [ / ] Step Photos   •   ← / → Step Filters   •   \\ Compare");
        cairo_move_to(cr, cx + 28.0, cy + 124.0);
        char tbuf[128];
        snprintf(tbuf, sizeof(tbuf), "Theme: %s  (click Customize in top bar to change)",
                 gui_active_theme(g)->label);
        cairo_show_text(cr, tbuf);
        return TRUE;
    }

    /* fit the frame into the panel without distorting it (never upscale) */
    double pad = (g->prefs.frame_style == FRAME_MAT) ? 56.0 : 32.0;
    double iw = g->rgb_w, ih = g->rgb_h;
    double s = MIN((aw - pad) / iw, (ah - pad) / ih);
    if (s > 1.0) s = 1.0;
    if (s <= 0.0) s = 1.0;
    double w = iw * s, h = ih * s;
    double x = (aw - w) * 0.5, y = (ah - h) * 0.5;

    /* frame / matboard styles */
    if (g->prefs.frame_style == FRAME_SHADOW) {
        cairo_set_source_rgba(cr, 0, 0, 0, 0.50);
        cairo_rectangle(cr, x + 6, y + 8, w, h);
        cairo_fill(cr);
    } else if (g->prefs.frame_style == FRAME_MAT) {
        double m = 18.0;
        cairo_set_source_rgba(cr, 0, 0, 0, 0.48);
        cairo_rectangle(cr, x - m + 6, y - m + 8, w + m * 2, h + m * 2);
        cairo_fill(cr);
        cairo_set_source_rgb(cr, 0.96, 0.95, 0.92);
        cairo_rectangle(cr, x - m, y - m, w + m * 2, h + m * 2);
        cairo_fill(cr);
        cairo_set_source_rgba(cr, 0.2, 0.2, 0.2, 0.35);
        cairo_set_line_width(cr, 1.0);
        cairo_rectangle(cr, x - 1.0, y - 1.0, w + 2.0, h + 2.0);
        cairo_stroke(cr);
    }

    /* paint the image (or before/after split comparison) */
    if (g->compare_mode == COMPARE_BEFORE && g->orig_surface) {
        cairo_save(cr);
        cairo_translate(cr, x, y);
        cairo_scale(cr, (double)w / g->orig_w, (double)h / g->orig_h);
        cairo_set_source_surface(cr, g->orig_surface, 0, 0);
        cairo_pattern_set_filter(cairo_get_source(cr), CAIRO_FILTER_BILINEAR);
        cairo_paint(cr);
        cairo_restore(cr);
    } else if (g->compare_mode == COMPARE_SPLIT && g->orig_surface) {
        double split_x = x + w * CLAMP(g->split_ratio, 0.05, 0.95);
        /* left side: original */
        cairo_save(cr);
        cairo_rectangle(cr, x, y, split_x - x, h);
        cairo_clip(cr);
        cairo_translate(cr, x, y);
        cairo_scale(cr, (double)w / g->orig_w, (double)h / g->orig_h);
        cairo_set_source_surface(cr, g->orig_surface, 0, 0);
        cairo_pattern_set_filter(cairo_get_source(cr), CAIRO_FILTER_BILINEAR);
        cairo_paint(cr);
        cairo_restore(cr);

        /* right side: edited */
        cairo_save(cr);
        cairo_rectangle(cr, split_x, y, (x + w) - split_x, h);
        cairo_clip(cr);
        cairo_translate(cr, x, y);
        cairo_scale(cr, s, s);
        cairo_set_source_surface(cr, g->surface, 0, 0);
        cairo_pattern_set_filter(cairo_get_source(cr), CAIRO_FILTER_BILINEAR);
        cairo_paint(cr);
        cairo_restore(cr);

        /* split divider line + handle */
        cairo_save(cr);
        cairo_set_source_rgba(cr, ar, ag, ab, 0.95);
        cairo_set_line_width(cr, 2.0);
        cairo_move_to(cr, split_x, y);
        cairo_line_to(cr, split_x, y + h);
        cairo_stroke(cr);

        cairo_arc(cr, split_x, y + h * 0.5, 11.0, 0, 2.0 * M_PI);
        cairo_set_source_rgba(cr, 0.1, 0.1, 0.12, 0.85);
        cairo_fill_preserve(cr);
        cairo_set_source_rgba(cr, ar, ag, ab, 1.0);
        cairo_stroke(cr);

        /* BEFORE / AFTER badges */
        cairo_rounded_rect(cr, x + 10, y + 10, 62, 22, 5.0);
        cairo_set_source_rgba(cr, 0.06, 0.06, 0.08, 0.75);
        cairo_fill(cr);
        cairo_set_source_rgb(cr, 0.9, 0.9, 0.9);
        cairo_set_font_size(cr, 10.5);
        cairo_move_to(cr, x + 19, y + 25);
        cairo_show_text(cr, "BEFORE");

        cairo_rounded_rect(cr, x + w - 66, y + 10, 56, 22, 5.0);
        cairo_set_source_rgba(cr, 0.06, 0.06, 0.08, 0.75);
        cairo_fill(cr);
        cairo_set_source_rgb(cr, ar, ag, ab);
        cairo_move_to(cr, x + w - 56, y + 25);
        cairo_show_text(cr, "AFTER");
        cairo_restore(cr);
    } else {
        cairo_save(cr);
        cairo_translate(cr, x, y);
        cairo_scale(cr, s, s);
        cairo_set_source_surface(cr, g->surface, 0, 0);
        cairo_pattern_set_filter(cairo_get_source(cr), CAIRO_FILTER_BILINEAR);
        cairo_paint(cr);
        cairo_restore(cr);
    }

    if (g->prefs.frame_style == FRAME_BORDER) {
        cairo_set_source_rgba(cr, ar, ag, ab, 0.85);
        cairo_set_line_width(cr, 1.5);
        cairo_rectangle(cr, x - 0.5, y - 0.5, w + 1.0, h + 1.0);
        cairo_stroke(cr);
    }

    /* composition grid */
    draw_grid_overlay(cr, g->prefs.grid_mode, x, y, w, h);

    /* live RGB histogram */
    if (g->prefs.show_histogram)
        draw_histogram_overlay(cr, g, aw);

    /* bottom-left HUD info pill */
    if (g->prefs.show_hud && g->app->has_image) {
        char hud[512];
        const char *fn = gui_basename(g->app->file_path);
        const char *fl = (g->current && g->current->label) ? g->current->label : "Original";
        const char *cmp = (g->compare_mode == COMPARE_BEFORE) ? "  •  [BEFORE]"
                        : (g->compare_mode == COMPARE_SPLIT)  ? "  •  [SPLIT WIPE]" : "";
        snprintf(hud, sizeof(hud), "%.64s   •   %dx%d   •   %.64s%s",
                 fn[0] ? fn : "Image", g->app->source_linear.w, g->app->source_linear.h, fl, cmp);
        cairo_save(cr);
        cairo_set_font_size(cr, 11.5);
        cairo_text_extents_t ext;
        cairo_text_extents(cr, hud, &ext);
        double pw = ext.width + 24.0, ph = 26.0;
        double px = 14.0, py = ah - ph - 14.0;
        cairo_rounded_rect(cr, px, py, pw, ph, 6.0);
        cairo_set_source_rgba(cr, 0.07, 0.08, 0.09, 0.78);
        cairo_fill_preserve(cr);
        cairo_set_source_rgba(cr, 1.0, 1.0, 1.0, 0.14);
        cairo_set_line_width(cr, 1.0);
        cairo_stroke(cr);
        cairo_set_source_rgba(cr, 0.92, 0.90, 0.85, 0.95);
        cairo_move_to(cr, px + 12.0, py + 17.0);
        cairo_show_text(cr, hud);
        cairo_restore(cr);
    }

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
    g->timeout_id = g_timeout_add(g->draft ? 50 : 90, on_render_timeout, g);

    if (g->settle_id) g_source_remove(g->settle_id);
    g->settle_id = 0;
    if (g->draft) g->settle_id = g_timeout_add(260, on_settle_timeout, g);
}

static void gui_ensure_orig_surface(FeGui *g) {
    if (!g->app->has_image) return;
    const FeImage *src = &g->app->source_linear;
    if (g->orig_surface && g->orig_w == src->w && g->orig_h == src->h) return;

    FeSettings def = fe_settings_default();
    int w = 0, h = 0;
    unsigned char *rgb = fe_app_render_from(src, NULL, &def, &w, &h, 0, NULL);
    if (!rgb) return;
    if (g->orig_surface) cairo_surface_destroy(g->orig_surface);
    g->orig_surface = surface_from_rgb(rgb, w, h);
    free(g->orig_rgb);
    g->orig_rgb = rgb;
    g->orig_w = w;
    g->orig_h = h;
}

static void gui_render(FeGui *g) {
    if (!g->app->has_image) return;

    FeSettings settings;
    gui_current_settings(g, &settings);

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

    if (g->compare_mode != COMPARE_OFF)
        gui_ensure_orig_surface(g);

    gtk_widget_queue_draw(g->preview);

    if (getenv("FE_DEBUG_GUI"))
        fprintf(stderr, "[gui] render %dx%d surface=%p draft=%d\n", w, h,
                (void *)g->surface, g->draft);
}

/* ------------------------------------------------------------------ */
/* per-image edit state                                                 */
/* ------------------------------------------------------------------ */

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

static void gui_collect_state(FeGui *g, FeEditState *out) {
    memset(out, 0, sizeof(*out));
    snprintf(out->filter_key, sizeof(out->filter_key), "%s",
             g->current ? g->current->key : "00-basics/original");
    for (int i = 0; i < NSLIDERS; i++) {
        double offset = gtk_range_get_value(GTK_RANGE(g->scale[i])) - g->rest[i];
        if (fabs(offset) < 1e-9) continue;
        fe_state_set(out, SLIDERS[i].key, offset);
    }
}

/* Write the sidecar for the photo that is open right now — ONLY if the user
   actually modified it (`g->dirty`).  Merely clicking a photo to view it
   never creates a sidecar. */
static void gui_save_state_now(FeGui *g) {
    if (g->save_id) {
        g_source_remove(g->save_id);
        g->save_id = 0;
    }
    if (!g->dirty) return;
    if (!g->app->has_image || !g->app->file_path[0]) {
        g->dirty = 0;
        return;
    }

    gui_collect_state(g, &g->state);
    if (fe_state_save(g->app->file_path, &g->state)) {
        g->dirty = 0;
        gui_refresh_strip_badges(g);
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

static void gui_schedule_save(FeGui *g) {
    if (g->suppress) return;                 /* the program moved it, not the user */
    if (!g->app->has_image || !g->app->file_path[0]) return;
    g->dirty = 1;
    if (g->save_id) g_source_remove(g->save_id);
    g->save_id = g_timeout_add(350, on_save_timeout, g);
}

/* Put the sliders and filter preset where `st` says they were.
   Caller must wrap in g->suppress++ / g->suppress-- when restoring on load. */
static void gui_apply_state(FeGui *g, const FeEditState *st) {
    int idx = gui_filter_index_by_key(g, st->filter_key);
    if (idx < 0) idx = 0;
    if (g->nbuttons > 0) gui_select_filter(g, idx);

    for (int i = 0; i < NSLIDERS; i++) {
        double offset = 0.0;
        if (!fe_state_get(st, SLIDERS[i].key, &offset)) offset = 0.0;
        if (offset < SLIDERS[i].lo) offset = SLIDERS[i].lo;
        if (offset > SLIDERS[i].hi) offset = SLIDERS[i].hi;
        gtk_range_set_value(GTK_RANGE(g->scale[i]), g->rest[i] + offset);
    }
    gui_update_slider_labels(g);
}

/* Ensure the folder strip shows the directory of `image_path` when no folder
   has been opened yet, so opening a single file still populates its sibling
   photos in the Folder strip. */
static void gui_ensure_parent_folder_loaded(FeGui *g, const char *image_path) {
    if (!image_path || !image_path[0] || g->folder_count > 0) return;
    char dir[FE_PATH_MAX];
    snprintf(dir, sizeof(dir), "%s", image_path);
    char *slash = strrchr(dir, '/');
    if (!slash) return;
    if (slash == dir) slash[1] = '\0';
    else *slash = '\0';
    gui_load_folder(g, dir, 0);
}

/* Open `path` and give the photo back ITS OWN edit.
     1. flush the previous photo's pending autosave FIRST (if dirty);
     2. decode the new photo;
     3. suppress autosave while the filter and sliders are repositioned;
     4. restore the saved preset + slider deltas from <path>.fastedit.json,
        or reset to Original (index 0) + all sliders at 0 when the photo
        has never been edited. */
static void gui_load_image(FeGui *g, const char *path) {
    if (!path || !path[0]) return;
    gui_save_state_now(g);                 /* step 1: save previous photo if modified */

    if (!fe_app_load(g->app, path)) {
        gui_status(g, "Could not decode %s", path);
        return;
    }

    if (g->orig_surface) {
        cairo_surface_destroy(g->orig_surface);
        g->orig_surface = NULL;
    }
    free(g->orig_rgb);
    g->orig_rgb = NULL;
    g->orig_w = 0;
    g->orig_h = 0;

    FeEditState st;
    memset(&st, 0, sizeof(st));
    int has = fe_state_load(path, &st);    /* step 2: read this photo's sidecar */

    g->suppress++;                         /* step 3: program-driven state restore */
    if (has) {
        gui_apply_state(g, &st);
        g->state = st;
    } else {
        if (g->nbuttons > 0) gui_select_filter(g, 0);
        for (int i = 0; i < NSLIDERS; i++)
            gtk_range_set_value(GTK_RANGE(g->scale[i]), g->rest[i]);
        memset(&g->state, 0, sizeof(g->state));
        if (g->current)
            snprintf(g->state.filter_key, sizeof(g->state.filter_key), "%s", g->current->key);
    }
    g->suppress--;
    gui_update_slider_labels(g);
    g->dirty = 0;
    g->draft = 0;

    gui_ensure_parent_folder_loaded(g, path);
    gui_highlight_active_photo(g);
    gui_schedule_render(g);

    const char *base = gui_basename(path);
    const char *flt_name = (g->current && g->current->label) ? g->current->label : "Original";
    if (has) {
        gui_status(g, "%s — %s (%d slider adjustment%s restored)",
                   base, flt_name, st.nfields, st.nfields == 1 ? "" : "s");
    } else {
        gui_status(g, "%s — Original (unedited photo)", base);
    }
}

/* ------------------------------------------------------------------ */
/* the sliders & per-photo actions                                      */
/* ------------------------------------------------------------------ */

static void on_slider_changed(GtkRange *range, gpointer data) {
    (void)range;
    FeGui *g = (FeGui *)data;
    gui_update_slider_labels(g);
    if (g->suppress) return;
    g->draft = 1;                     /* a draft render while dragging */
    gui_schedule_save(g);
    gui_schedule_render(g);
}

static void gui_reset_sliders(FeGui *g) {
    g->suppress++;
    for (int i = 0; i < NSLIDERS; i++)
        gtk_range_set_value(GTK_RANGE(g->scale[i]), g->rest[i]);
    g->suppress--;
    gui_update_slider_labels(g);
    g->draft = 0;
    gui_schedule_save(g);
    gui_save_state_now(g);
    gui_status(g, "Sliders reset to %s", g->current ? g->current->label : "preset");
    gui_schedule_render(g);
}

/* Reset the current photo completely: back to Original + 0 offsets, and
   delete its .fastedit.json sidecar. */
static void on_reset_photo(GtkButton *btn, gpointer data) {
    (void)btn;
    FeGui *g = (FeGui *)data;
    if (!g->app->has_image) {
        gui_status(g, "Open an image first");
        return;
    }
    if (g->save_id) {
        g_source_remove(g->save_id);
        g->save_id = 0;
    }
    g->dirty = 0;
    fe_state_remove(g->app->file_path);

    g->suppress++;
    if (g->nbuttons > 0) gui_select_filter(g, 0);
    for (int i = 0; i < NSLIDERS; i++)
        gtk_range_set_value(GTK_RANGE(g->scale[i]), g->rest[i]);
    memset(&g->state, 0, sizeof(g->state));
    g->suppress--;
    gui_update_slider_labels(g);

    g->draft = 0;
    gui_refresh_strip_badges(g);
    gui_schedule_render(g);
    gui_status(g, "Reset %s to Original (sidecar removed)", gui_basename(g->app->file_path));
}

static void on_copy_edits(GtkButton *btn, gpointer data) {
    (void)btn;
    FeGui *g = (FeGui *)data;
    if (!g->app->has_image) {
        gui_status(g, "Open an image first");
        return;
    }
    gui_collect_state(g, &g->clipboard);
    g->has_clipboard = 1;
    gui_status(g, "Copied edits (%s + %d adjustment%s)",
               g->current ? g->current->label : "Original",
               g->clipboard.nfields, g->clipboard.nfields == 1 ? "" : "s");
}

static void on_paste_edits(GtkButton *btn, gpointer data) {
    (void)btn;
    FeGui *g = (FeGui *)data;
    if (!g->app->has_image) {
        gui_status(g, "Open an image first");
        return;
    }
    if (!g->has_clipboard) {
        gui_status(g, "No edits copied yet — click Copy Edits first");
        return;
    }
    g->suppress++;
    gui_apply_state(g, &g->clipboard);
    g->suppress--;
    gui_update_slider_labels(g);
    g->draft = 0;
    gui_schedule_save(g);
    gui_save_state_now(g);
    gui_schedule_render(g);
    gui_status(g, "Pasted edits onto %s", gui_basename(g->app->file_path));
}

static void gui_set_slider_abs(FeGui *g, const char *key, double target) {
    int i = slider_by_key(key);
    if (i < 0 || !g->scale[i]) return;
    double base = settings_get(&g->base, SLIDERS[i].offset);
    double offset = target - base;
    if (offset < SLIDERS[i].lo) offset = SLIDERS[i].lo;
    if (offset > SLIDERS[i].hi) offset = SLIDERS[i].hi;
    gtk_range_set_value(GTK_RANGE(g->scale[i]), g->rest[i] + offset);
    gui_update_slider_labels(g);
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

    g->suppress++;
    if (k1 != 0) {
        double v = -k1 / 0.2;
        gui_set_slider_abs(g, "lens_distortion", v);
    }
    if (tca_r != 0) gui_set_slider_abs(g, "lens_tca_r", tca_r);
    if (tca_b != 0) gui_set_slider_abs(g, "lens_tca_b", tca_b);
    if (vig != 0)   gui_set_slider_abs(g, "lens_vignette_corr", vig);
    g->suppress--;
    gui_update_slider_labels(g);

    g->draft = 0;
    gui_schedule_save(g);
    gui_status(g, has_any ? "Auto lens correction applied" : "No strong lens issues detected");
    gui_schedule_render(g);
}

static void on_auto_exposure(GtkButton *btn, gpointer data) {
    (void)btn;
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
    g->suppress++;
    gtk_range_set_value(GTK_RANGE(g->scale[SLIDER_EXPOSURE]),
                        g->rest[SLIDER_EXPOSURE] + offset);
    g->suppress--;
    gui_update_slider_labels(g);
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
        int show = !filtering;
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

        char cls[32];
        snprintf(cls, sizeof(cls), "fecat%d", c);

        char *title = g_strdup_printf("%s  (%d)", cat->label, cat->count);
        GtkWidget *exp = gtk_expander_new(title);
        g_free(title);
        gtk_expander_set_expanded(GTK_EXPANDER(exp), c == 0);
        g->expanders[c] = exp;

        GtkWidget *box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 4);
        gtk_container_set_border_width(GTK_CONTAINER(box), 6);

        for (int j = 0; j < cat->count; j++) {
            FeFilter *f = &cat->filters[j];
            GtkWidget *btn = gtk_button_new_with_label(f->label);
            gtk_button_set_relief(GTK_BUTTON(btn), GTK_RELIEF_NONE);
            GtkWidget *child = gtk_bin_get_child(GTK_BIN(btn));
            if (child && GTK_IS_LABEL(child))
                gtk_label_set_xalign(GTK_LABEL(child), 0.0f);
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

static void gui_select_filter(FeGui *g, int index) {
    const FeFilter *flt = NULL;
    int k = 0, target_cat = -1;
    for (int c = 0; c < g->app->filters.count && !flt; c++) {
        FeCategory *cat = &g->app->filters.cats[c];
        for (int j = 0; j < cat->count; j++) {
            if (k == index) { flt = &cat->filters[j]; target_cat = c; break; }
            k++;
        }
    }
    if (!flt) return;

    if (target_cat >= 0 && target_cat < g->nexpanders && g->expanders[target_cat])
        gtk_expander_set_expanded(GTK_EXPANDER(g->expanders[target_cat]), TRUE);

    g->current = flt;
    g->base = flt->settings;
    g->active_index = index;

    g->suppress++;
    for (int i = 0; i < NSLIDERS; i++)
        gtk_range_set_value(GTK_RANGE(g->scale[i]), g->rest[i]);
    g->suppress--;
    gui_update_slider_labels(g);

    g->draft = 0;

    for (int i = 0; i < g->nbuttons; i++) {
        if (!g->buttons[i]) continue;
        GtkStyleContext *ctx = gtk_widget_get_style_context(g->buttons[i]);
        if (i == index) gtk_style_context_add_class(ctx, "feactive");
        else gtk_style_context_remove_class(ctx, "feactive");
    }

    gui_browser_reveal(g, index);
    if (!g->suppress) {
        gui_status(g, "%s — %s", flt->label,
                   flt->description && flt->description[0] ? flt->description : flt->key);
        gui_schedule_save(g);
        gui_schedule_render(g);
    }
}

/* ------------------------------------------------------------------ */
/* film-strip previews, active highlight & edit badges                  */
/* ------------------------------------------------------------------ */

static void thumb_pixels_free(guchar *pixels, gpointer data) {
    (void)data;
    free(pixels);
}

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

/* Update the edit badge text and style on every card in Folder and Project. */
static void gui_refresh_strip_badges(FeGui *g) {
    GtkWidget *boxes[2] = {g->folder_box, g->project_box};
    for (int b = 0; b < 2; b++) {
        if (!boxes[b]) continue;
        GList *children = gtk_container_get_children(GTK_CONTAINER(boxes[b]));
        for (GList *l = children; l; l = l->next) {
            GtkWidget *owner = GTK_WIDGET(l->data);
            const char *p = (const char *)g_object_get_data(G_OBJECT(owner), "fe-path");
            GtkWidget *badge = GTK_WIDGET(g_object_get_data(G_OBJECT(owner), "fe-badge"));
            if (!p || !badge || !GTK_IS_LABEL(badge)) continue;
            char text[128];
            gui_format_badge(g, p, text, sizeof(text));
            gtk_label_set_text(GTK_LABEL(badge), text);
            GtkStyleContext *ctx = gtk_widget_get_style_context(badge);
            if (gui_photo_has_meaningful_edit(p)) {
                gtk_style_context_add_class(ctx, "febadge-edited");
                gtk_style_context_remove_class(ctx, "febadge-clean");
            } else {
                gtk_style_context_add_class(ctx, "febadge-clean");
                gtk_style_context_remove_class(ctx, "febadge-edited");
            }
        }
        g_list_free(children);
    }
}

/* Highlight the card corresponding to `g->app->file_path` in both the Folder
   strip and the Project strip, and scroll it into view. */
static void gui_highlight_active_photo(FeGui *g) {
    const char *active = (g->app && g->app->has_image) ? g->app->file_path : "";
    GtkWidget *boxes[2] = {g->folder_box, g->project_box};
    for (int b = 0; b < 2; b++) {
        if (!boxes[b]) continue;
        GList *children = gtk_container_get_children(GTK_CONTAINER(boxes[b]));
        for (GList *l = children; l; l = l->next) {
            GtkWidget *owner = GTK_WIDGET(l->data);
            const char *p = (const char *)g_object_get_data(G_OBJECT(owner), "fe-path");
            GtkStyleContext *ctx = gtk_widget_get_style_context(owner);
            int is_curr = (p && active[0] && strcmp(p, active) == 0);
            if (is_curr) {
                gtk_style_context_add_class(ctx, "fethumb-active");
                GtkWidget *scroller = gtk_widget_get_ancestor(owner, GTK_TYPE_SCROLLED_WINDOW);
                if (scroller) {
                    GtkAdjustment *adj = gtk_scrolled_window_get_hadjustment(
                        GTK_SCROLLED_WINDOW(scroller));
                    if (adj) {
                        GtkAllocation a;
                        gtk_widget_get_allocation(owner, &a);
                        gdouble val = gtk_adjustment_get_value(adj);
                        gdouble page = gtk_adjustment_get_page_size(adj);
                        if (a.x < val) val = a.x;
                        else if (a.x + a.width > val + page) val = a.x + a.width - page;
                        gdouble lo = gtk_adjustment_get_lower(adj);
                        gdouble hi = gtk_adjustment_get_upper(adj) - page;
                        gtk_adjustment_set_value(adj, CLAMP(val, lo, MAX(lo, hi)));
                    }
                }
            } else {
                gtk_style_context_remove_class(ctx, "fethumb-active");
            }
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

static gboolean on_thumb_slice(gpointer data) {
    FeGui *g = (FeGui *)data;
    g->thumb_idle = 0;

    gint64 deadline = g_get_monotonic_time() + THUMB_SLICE_MS * 1000;
    while (g_get_monotonic_time() < deadline) {
        char *path = (char *)g_queue_pop_head(g->thumb_pending);
        if (!path) break;

        unsigned char *pixels = NULL;
        int w = 0, h = 0;
        if (fe_load_thumb_rgb(path, THUMB_LONG_EDGE, &pixels, &w, &h)) {
            GdkPixbuf *pb = gdk_pixbuf_new_from_data(pixels, GDK_COLORSPACE_RGB, FALSE,
                                                     8, w, h, w * 3, thumb_pixels_free, NULL);
            if (pb) {
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

static void on_folder_open(GtkButton *btn, gpointer data) {
    FeGui *g = (FeGui *)data;
    const char *path = (const char *)g_object_get_data(G_OBJECT(btn), "fe-path");
    if (!path) return;
    gui_status(g, "Decoding %s…", gui_basename(path));
    while (gtk_events_pending()) gtk_main_iteration();
    gui_load_image(g, path);
}

static void on_project_open(GtkButton *btn, gpointer data) {
    FeGui *g = (FeGui *)data;
    const char *path = (const char *)g_object_get_data(G_OBJECT(btn), "fe-path");
    if (!path) return;
    gui_status(g, "Decoding %s…", gui_basename(path));
    while (gtk_events_pending()) gtk_main_iteration();
    /* Keep the user on the Project tab while editing project photos! */
    gui_load_image(g, path);
}

static void on_project_toggled(GtkToggleButton *tog, gpointer data) {
    FeGui *g = (FeGui *)data;
    int index = GPOINTER_TO_INT(g_object_get_data(G_OBJECT(tog), "fe-index")) - 1;
    if (!g->project || index < 0 || index >= g->project->count) return;
    g->project->items[index].selected = gtk_toggle_button_get_active(tog) ? 1 : 0;
    gui_update_project_label(g);
}

static void gui_clear_box(GtkWidget *box) {
    GList *children = gtk_container_get_children(GTK_CONTAINER(box));
    for (GList *l = children; l; l = l->next) gtk_widget_destroy(GTK_WIDGET(l->data));
    g_list_free(children);
}

static void gui_free_folder_paths(FeGui *g) {
    if (!g->folder_paths) return;
    for (int i = 0; i < g->folder_count; i++) free(g->folder_paths[i]);
    free(g->folder_paths);
    g->folder_paths = NULL;
    g->folder_count = 0;
}

static void gui_update_folder_label(FeGui *g) {
    if (!g->bottom_tabs || !g->folder_page) return;
    char buf[96];
    if (g->folder_count > 0)
        snprintf(buf, sizeof(buf), "Folder  (%d photos)", g->folder_count);
    else
        snprintf(buf, sizeof(buf), "Folder");
    gtk_notebook_set_tab_label_text(GTK_NOTEBOOK(g->bottom_tabs), g->folder_page, buf);
}

static void gui_update_project_label(FeGui *g) {
    if (!g->bottom_tabs || !g->project_page) return;
    char buf[96];
    snprintf(buf, sizeof(buf), "Project  (%d/%d selected)",
             fe_project_selected_count(g->project), g->project->count);
    gtk_notebook_set_tab_label_text(GTK_NOTEBOOK(g->bottom_tabs), g->project_page, buf);
    if (g->project_placeholder)
        gtk_widget_set_visible(g->project_placeholder, g->project->count == 0);
}

static int cmp_filename(const void *a, const void *b) {
    const char *sa = *(const char *const *)a;
    const char *sb = *(const char *const *)b;
    int c = strcasecmp(sa, sb);
    return c != 0 ? c : strcmp(sa, sb);
}

/* Build a card for the Folder film-strip: thumbnail + filename + edit badge. */
static GtkWidget *gui_make_folder_card(FeGui *g, const char *path, const char *name) {
    GtkWidget *col = gtk_box_new(GTK_ORIENTATION_VERTICAL, 2);
    gtk_container_set_border_width(GTK_CONTAINER(col), 4);
    gtk_style_context_add_class(gtk_widget_get_style_context(col), "fethumb-card");

    GtkWidget *btn = gtk_button_new();
    gtk_button_set_relief(GTK_BUTTON(btn), GTK_RELIEF_NONE);

    GtkWidget *inner = gtk_box_new(GTK_ORIENTATION_VERTICAL, 2);
    GtkWidget *thumb = gtk_image_new_from_icon_name("image-x-generic", GTK_ICON_SIZE_DIALOG);
    gtk_widget_set_size_request(thumb, 120, 78);
    gtk_box_pack_start(GTK_BOX(inner), thumb, FALSE, FALSE, 0);

    GtkWidget *lbl = gtk_label_new(name);
    gtk_label_set_ellipsize(GTK_LABEL(lbl), PANGO_ELLIPSIZE_MIDDLE);
    gtk_label_set_max_width_chars(GTK_LABEL(lbl), 16);
    gtk_box_pack_start(GTK_BOX(inner), lbl, FALSE, FALSE, 0);

    char badge_text[128];
    gui_format_badge(g, path, badge_text, sizeof(badge_text));
    GtkWidget *badge = gtk_label_new(badge_text);
    gtk_label_set_ellipsize(GTK_LABEL(badge), PANGO_ELLIPSIZE_END);
    gtk_label_set_max_width_chars(GTK_LABEL(badge), 18);
    GtkStyleContext *bctx = gtk_widget_get_style_context(badge);
    gtk_style_context_add_class(
        bctx, gui_photo_has_meaningful_edit(path) ? "febadge-edited" : "febadge-clean");
    gtk_box_pack_start(GTK_BOX(inner), badge, FALSE, FALSE, 0);

    gtk_container_add(GTK_CONTAINER(btn), inner);
    g_object_set_data_full(G_OBJECT(btn), "fe-path", g_strdup(path), g_free);
    g_signal_connect(btn, "clicked", G_CALLBACK(on_folder_open), g);
    gtk_box_pack_start(GTK_BOX(col), btn, FALSE, FALSE, 0);

    g_object_set_data_full(G_OBJECT(col), "fe-path", g_strdup(path), g_free);
    g_object_set_data(G_OBJECT(col), "fe-thumb", thumb);
    g_object_set_data(G_OBJECT(col), "fe-badge", badge);
    return col;
}

static void gui_load_folder(FeGui *g, const char *folder, int auto_select_first) {
    if (!g->folder_box || !folder || !folder[0]) return;
    gui_save_state_now(g);
    gui_clear_box(g->folder_box);
    gui_free_folder_paths(g);

    DIR *d = opendir(folder);
    if (!d) {
        gui_status(g, "Cannot open %s", folder);
        return;
    }

    char **names = NULL;
    int n = 0, cap = 0;
    int has_raw = 0;
    struct dirent *ent;
    while ((ent = readdir(d)) != NULL) {
        if (ent->d_name[0] == '.') continue;
        if (!fe_is_supported_image_file(ent->d_name)) continue;
        char full[FE_PATH_MAX];
        snprintf(full, sizeof(full), "%s/%s", folder, ent->d_name);
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

    if (n > 1) qsort(names, (size_t)n, sizeof(char *), cmp_filename);

    snprintf(g->current_folder, sizeof(g->current_folder), "%s", folder);
    g->folder_paths = (char **)calloc((size_t)(n > 0 ? n : 1), sizeof(char *));
    g->folder_count = 0;

    for (int i = 0; i < n; i++) {
        /* If the folder has RAW files, show RAWs so companion JPEGs don't
           duplicate every shot; otherwise show all supported images. */
        if (!has_raw || fe_is_raw_image_file(names[i])) {
            char path[FE_PATH_MAX];
            snprintf(path, sizeof(path), "%s/%s", folder, names[i]);
            g->folder_paths[g->folder_count++] = strdup(path);

            GtkWidget *card = gui_make_folder_card(g, path, names[i]);
            gtk_box_pack_start(GTK_BOX(g->folder_box), card, FALSE, FALSE, 2);
            g_queue_push_tail(g->thumb_pending, g_strdup(path));
        }
        free(names[i]);
    }
    free(names);

    gtk_widget_show_all(g->folder_box);
    gui_update_folder_label(g);
    gui_highlight_active_photo(g);
    gui_thumbs_kick(g);

    if (auto_select_first && g->folder_count > 0) {
        gui_load_image(g, g->folder_paths[0]);
    } else {
        gui_status(g, "Folder: %s (%d photo%s)", folder, g->folder_count,
                   g->folder_count == 1 ? "" : "s");
    }
}

static void on_browse_folder(GtkButton *btn, gpointer data) {
    (void)btn;
    FeGui *g = (FeGui *)data;
    GtkWidget *dlg = gtk_file_chooser_dialog_new(
        "Browse Folder", GTK_WINDOW(g->window),
        GTK_FILE_CHOOSER_ACTION_SELECT_FOLDER,
        "_Cancel", GTK_RESPONSE_CANCEL, "_Select", GTK_RESPONSE_ACCEPT, NULL);
    if (gtk_dialog_run(GTK_DIALOG(dlg)) == GTK_RESPONSE_ACCEPT) {
        char *path = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(dlg));
        if (path) {
            gtk_notebook_set_current_page(GTK_NOTEBOOK(g->bottom_tabs), 0);
            gui_load_folder(g, path, 1);
            g_free(path);
        }
    }
    gtk_widget_destroy(dlg);
}

/* Step backward (dir = -1) or forward (dir = +1) in whichever strip (Folder or
   Project) is currently active in the bottom notebook. */
static void gui_step_photo(FeGui *g, int dir) {
    int page = gtk_notebook_get_current_page(GTK_NOTEBOOK(g->bottom_tabs));
    const char *curr = (g->app && g->app->has_image) ? g->app->file_path : "";

    if (page == 1 && g->project && g->project->count > 0) {
        int idx = fe_project_find(g->project, curr);
        int next = (idx < 0) ? 0 : (idx + dir + g->project->count) % g->project->count;
        gui_load_image(g, g->project->items[next].path);
        return;
    }
    if (g->folder_count > 0) {
        int idx = -1;
        for (int i = 0; i < g->folder_count; i++) {
            if (strcmp(g->folder_paths[i], curr) == 0) { idx = i; break; }
        }
        int next = (idx < 0) ? 0 : (idx + dir + g->folder_count) % g->folder_count;
        gui_load_image(g, g->folder_paths[next]);
        return;
    }
    if (g->project && g->project->count > 0) {
        int idx = fe_project_find(g->project, curr);
        int next = (idx < 0) ? 0 : (idx + dir + g->project->count) % g->project->count;
        gui_load_image(g, g->project->items[next].path);
    }
}

static void on_prev_photo(GtkButton *btn, gpointer data) {
    (void)btn;
    gui_step_photo((FeGui *)data, -1);
}

static void on_next_photo(GtkButton *btn, gpointer data) {
    (void)btn;
    gui_step_photo((FeGui *)data, +1);
}

/* ---- the project -------------------------------------------------- */

static void on_add_project(GtkButton *btn, gpointer data) {
    (void)btn;
    FeGui *g = (FeGui *)data;
    GtkWidget *dlg = gtk_file_chooser_dialog_new(
        "Add Photos to Project", GTK_WINDOW(g->window),
        GTK_FILE_CHOOSER_ACTION_OPEN,
        "_Cancel", GTK_RESPONSE_CANCEL, "_Add", GTK_RESPONSE_ACCEPT, NULL);
    gtk_file_chooser_set_select_multiple(GTK_FILE_CHOOSER(dlg), TRUE);
    if (g->current_folder[0])
        gtk_file_chooser_set_current_folder(GTK_FILE_CHOOSER(dlg), g->current_folder);

    GtkFileFilter *ff = gtk_file_filter_new();
    gtk_file_filter_set_name(ff, "Images (ARW / RAW / JPEG / PNG)");
    gtk_file_filter_add_pattern(ff, "*.ARW"); gtk_file_filter_add_pattern(ff, "*.arw");
    gtk_file_filter_add_pattern(ff, "*.RAW"); gtk_file_filter_add_pattern(ff, "*.raw");
    gtk_file_filter_add_pattern(ff, "*.CR2"); gtk_file_filter_add_pattern(ff, "*.cr2");
    gtk_file_filter_add_pattern(ff, "*.NEF"); gtk_file_filter_add_pattern(ff, "*.nef");
    gtk_file_filter_add_pattern(ff, "*.DNG"); gtk_file_filter_add_pattern(ff, "*.dng");
    gtk_file_filter_add_pattern(ff, "*.RAF"); gtk_file_filter_add_pattern(ff, "*.raf");
    gtk_file_filter_add_pattern(ff, "*.jpg"); gtk_file_filter_add_pattern(ff, "*.jpeg");
    gtk_file_filter_add_pattern(ff, "*.png"); gtk_file_filter_add_pattern(ff, "*.tif");
    gtk_file_chooser_add_filter(GTK_FILE_CHOOSER(dlg), ff);

    if (gtk_dialog_run(GTK_DIALOG(dlg)) == GTK_RESPONSE_ACCEPT) {
        GSList *files = gtk_file_chooser_get_filenames(GTK_FILE_CHOOSER(dlg));
        int added = 0;
        char *first_added = NULL;
        for (GSList *l = files; l; l = l->next) {
            if (fe_project_add(g->project, (char *)l->data)) {
                if (!first_added) first_added = g_strdup((char *)l->data);
                added++;
            }
            g_free(l->data);
        }
        g_slist_free(files);
        gui_build_project(g);
        gtk_notebook_set_current_page(GTK_NOTEBOOK(g->bottom_tabs), 1);
        if (!g->app->has_image && first_added) {
            gui_load_image(g, first_added);
        } else {
            gui_status(g, "Added %d photo(s) — project has %d (%d selected)", added,
                       g->project->count, fe_project_selected_count(g->project));
        }
        g_free(first_added);
    }
    gtk_widget_destroy(dlg);
}

/* Add the currently browsed folder (or ask for one) straight into the Project. */
static void on_add_folder_to_project(GtkButton *btn, gpointer data) {
    (void)btn;
    FeGui *g = (FeGui *)data;
    if (g->current_folder[0] && g->folder_count > 0) {
        int added = 0;
        for (int i = 0; i < g->folder_count; i++) {
            if (fe_project_add(g->project, g->folder_paths[i])) added++;
        }
        gui_build_project(g);
        gtk_notebook_set_current_page(GTK_NOTEBOOK(g->bottom_tabs), 1);
        if (!g->app->has_image && g->project->count > 0)
            gui_load_image(g, g->project->items[0].path);
        gui_status(g, "Added %d photo(s) from folder to Project (%d total)",
                   added, g->project->count);
        return;
    }

    GtkWidget *dlg = gtk_file_chooser_dialog_new(
        "Add Folder to Project", GTK_WINDOW(g->window),
        GTK_FILE_CHOOSER_ACTION_SELECT_FOLDER,
        "_Cancel", GTK_RESPONSE_CANCEL, "_Add Folder", GTK_RESPONSE_ACCEPT, NULL);
    if (gtk_dialog_run(GTK_DIALOG(dlg)) == GTK_RESPONSE_ACCEPT) {
        char *folder = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(dlg));
        if (folder) {
            int added = fe_project_add_folder(g->project, folder);
            gui_build_project(g);
            gtk_notebook_set_current_page(GTK_NOTEBOOK(g->bottom_tabs), 1);
            if (!g->app->has_image && g->project->count > 0)
                gui_load_image(g, g->project->items[0].path);
            gui_status(g, "Added %d photo(s) from %s to Project", added, folder);
            g_free(folder);
        }
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
    gui_status(g, "%d/%d selected in project", fe_project_selected_count(g->project),
               g->project->count);
}

static void on_apply_filter_selected(GtkButton *btn, gpointer data) {
    (void)btn;
    FeGui *g = (FeGui *)data;
    if (!g->project || g->project->count == 0) { gui_status(g, "Project empty"); return; }
    if (!g->current) { gui_status(g, "Pick a filter first"); return; }
    int n = fe_project_selected_count(g->project);
    if (n == 0) { gui_status(g, "Nothing selected"); return; }

    gui_save_state_now(g);
    gui_status(g, "Applying “%s” to %d photo(s)…", g->current->label, n);
    while (gtk_events_pending()) gtk_main_iteration();

    int done = fe_project_apply_filter(g->project, g->app, g->current->key);
    gui_refresh_strip_badges(g);
    gui_status(g, "Applied filter “%s” to %d photo(s) (individual slider tweaks preserved)",
               g->current->label, done);
}

/* Sync the currently open photo's ENTIRE edit (filter + all slider offsets)
   to all selected photos in the project. */
static void on_sync_edits_selected(GtkButton *btn, gpointer data) {
    (void)btn;
    FeGui *g = (FeGui *)data;
    if (!g->project || g->project->count == 0) { gui_status(g, "Project empty"); return; }
    int n = fe_project_selected_count(g->project);
    if (n == 0) { gui_status(g, "Nothing selected"); return; }

    FeEditState src_st;
    gui_collect_state(g, &src_st);
    int done = fe_project_sync_state(g->project, g->app, &src_st);
    g->dirty = 0;
    gui_refresh_strip_badges(g);
    gui_status(g, "Synced “%s” + %d slider adjustment(s) to %d selected photo(s)",
               g->current ? g->current->label : "Original", src_st.nfields, done);
}

/* Reset edits on all selected project photos. */
static void on_reset_selected_edits(GtkButton *btn, gpointer data) {
    (void)btn;
    FeGui *g = (FeGui *)data;
    if (!g->project || g->project->count == 0) { gui_status(g, "Project empty"); return; }
    int n = fe_project_selected_count(g->project);
    if (n == 0) { gui_status(g, "Nothing selected"); return; }

    int done = fe_project_reset_selected(g->project);
    /* If the currently open photo was among the selected items, reset the live
       view too so it reflects the cleared state. */
    if (g->app->has_image) {
        int idx = fe_project_find(g->project, g->app->file_path);
        if (idx >= 0 && g->project->items[idx].selected) {
            g->dirty = 0;
            g->suppress++;
            if (g->nbuttons > 0) gui_select_filter(g, 0);
            for (int i = 0; i < NSLIDERS; i++)
                gtk_range_set_value(GTK_RANGE(g->scale[i]), g->rest[i]);
            g->suppress--;
            gui_update_slider_labels(g);
            gui_schedule_render(g);
        }
    }
    gui_refresh_strip_badges(g);
    gui_status(g, "Reset edits on %d selected photo(s)", done);
}

static void on_save_project(GtkButton *btn, gpointer data) {
    (void)btn;
    FeGui *g = (FeGui *)data;
    if (!g->project || g->project->count == 0) {
        gui_status(g, "Project is empty — add photos first");
        return;
    }
    gui_save_state_now(g);

    GtkWidget *dlg = gtk_file_chooser_dialog_new(
        "Save Project (.feproj)", GTK_WINDOW(g->window),
        GTK_FILE_CHOOSER_ACTION_SAVE,
        "_Cancel", GTK_RESPONSE_CANCEL, "_Save", GTK_RESPONSE_ACCEPT, NULL);
    gtk_file_chooser_set_do_overwrite_confirmation(GTK_FILE_CHOOSER(dlg), TRUE);
    gtk_file_chooser_set_current_name(
        GTK_FILE_CHOOSER(dlg),
        g->project->project_path ? gui_basename(g->project->project_path) : "shoot.feproj");

    GtkFileFilter *ff = gtk_file_filter_new();
    gtk_file_filter_set_name(ff, "Fast-Edit Project (*.feproj)");
    gtk_file_filter_add_pattern(ff, "*.feproj");
    gtk_file_chooser_add_filter(GTK_FILE_CHOOSER(dlg), ff);

    if (gtk_dialog_run(GTK_DIALOG(dlg)) == GTK_RESPONSE_ACCEPT) {
        char *path = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(dlg));
        if (path) {
            if (fe_project_save(g->project, path))
                gui_status(g, "Saved project %s (%d photos)", path, g->project->count);
            else
                gui_status(g, "Could not save project %s", path);
            g_free(path);
        }
    }
    gtk_widget_destroy(dlg);
}

static void on_load_project(GtkButton *btn, gpointer data) {
    (void)btn;
    FeGui *g = (FeGui *)data;
    gui_save_state_now(g);

    GtkWidget *dlg = gtk_file_chooser_dialog_new(
        "Load Project (.feproj)", GTK_WINDOW(g->window),
        GTK_FILE_CHOOSER_ACTION_OPEN,
        "_Cancel", GTK_RESPONSE_CANCEL, "_Open", GTK_RESPONSE_ACCEPT, NULL);
    GtkFileFilter *ff = gtk_file_filter_new();
    gtk_file_filter_set_name(ff, "Fast-Edit Project (*.feproj)");
    gtk_file_filter_add_pattern(ff, "*.feproj");
    gtk_file_chooser_add_filter(GTK_FILE_CHOOSER(dlg), ff);

    if (gtk_dialog_run(GTK_DIALOG(dlg)) == GTK_RESPONSE_ACCEPT) {
        char *path = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(dlg));
        if (path) {
            if (fe_project_load(g->project, path)) {
                gui_build_project(g);
                gtk_notebook_set_current_page(GTK_NOTEBOOK(g->bottom_tabs), 1);
                if (g->project->count > 0)
                    gui_load_image(g, g->project->items[0].path);
                gui_status(g, "Loaded project %s (%d photos)",
                           gui_basename(path), g->project->count);
            } else {
                gui_status(g, "Could not load project %s", path);
            }
            g_free(path);
        }
    }
    gtk_widget_destroy(dlg);
}

static void on_export_selected(GtkButton *btn, gpointer data) {
    (void)btn;
    FeGui *g = (FeGui *)data;
    if (!g->project || g->project->count == 0) { gui_status(g, "Project empty"); return; }
    int n = fe_project_selected_count(g->project);
    if (n == 0) { gui_status(g, "Nothing selected"); return; }

    GtkWidget *dlg = gtk_file_chooser_dialog_new(
        "Export Selected Photos", GTK_WINDOW(g->window),
        GTK_FILE_CHOOSER_ACTION_SELECT_FOLDER,
        "_Cancel", GTK_RESPONSE_CANCEL, "_Export", GTK_RESPONSE_ACCEPT, NULL);
    if (gtk_dialog_run(GTK_DIALOG(dlg)) == GTK_RESPONSE_ACCEPT) {
        char *dir = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(dlg));
        if (dir) {
            free(g->project->export_dir);
            g->project->export_dir = strdup(dir);
            g->project->fullres = gtk_toggle_button_get_active(
                GTK_TOGGLE_BUTTON(g->fullres));

            gui_save_state_now(g);
            gui_status(g, "Exporting %d photo(s) with their individual edits…", n);
            while (gtk_events_pending()) gtk_main_iteration();

            int ok = fe_project_export(g->project, g->app,
                                       g->current ? g->current->key : NULL, NULL, NULL);
            gui_status(g, "Exported %d/%d photo(s) to %s", ok, n, dir);
            g_free(dir);
        }
    }
    gtk_widget_destroy(dlg);
}

static void gui_build_project(FeGui *g) {
    if (!g->project_box) return;
    gui_clear_box(g->project_box);

    for (int i = 0; i < g->project->count; i++) {
        FeProjectItem *it = &g->project->items[i];

        GtkWidget *col = gtk_box_new(GTK_ORIENTATION_VERTICAL, 2);
        gtk_container_set_border_width(GTK_CONTAINER(col), 4);
        gtk_style_context_add_class(gtk_widget_get_style_context(col), "fethumb-card");

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
                                                        GTK_ICON_SIZE_DIALOG);
        gtk_widget_set_size_request(thumb, 120, 72);
        gtk_container_add(GTK_CONTAINER(btn), thumb);
        g_object_set_data_full(G_OBJECT(btn), "fe-path", g_strdup(it->path), g_free);
        g_signal_connect(btn, "clicked", G_CALLBACK(on_project_open), g);
        gtk_box_pack_start(GTK_BOX(col), btn, FALSE, FALSE, 0);

        char badge_text[128];
        gui_format_badge(g, it->path, badge_text, sizeof(badge_text));
        GtkWidget *badge = gtk_label_new(badge_text);
        gtk_label_set_ellipsize(GTK_LABEL(badge), PANGO_ELLIPSIZE_END);
        gtk_label_set_max_width_chars(GTK_LABEL(badge), 18);
        GtkStyleContext *bctx = gtk_widget_get_style_context(badge);
        gtk_style_context_add_class(
            bctx, gui_photo_has_meaningful_edit(it->path) ? "febadge-edited" : "febadge-clean");
        gtk_box_pack_start(GTK_BOX(col), badge, FALSE, FALSE, 0);

        g_object_set_data_full(G_OBJECT(col), "fe-path", g_strdup(it->path), g_free);
        g_object_set_data(G_OBJECT(col), "fe-thumb", thumb);
        g_object_set_data(G_OBJECT(col), "fe-badge", badge);

        gtk_box_pack_start(GTK_BOX(g->project_box), col, FALSE, FALSE, 2);
        g_queue_push_tail(g->thumb_pending, g_strdup(it->path));
    }

    gtk_widget_show_all(g->project_box);
    gui_update_project_label(g);
    gui_highlight_active_photo(g);
    gui_thumbs_kick(g);
}

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
    (void)btn;
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
            gui_load_image(g, path);
            g_free(path);
        }
    }
    gtk_widget_destroy(dlg);
}

static void on_export(GtkButton *btn, gpointer data) {
    (void)btn;
    FeGui *g = (FeGui *)data;
    if (!g->app->has_image) {
        gui_status(g, "Nothing to export");
        return;
    }
    gui_save_state_now(g);

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
    (void)w;
    FeGui *g = (FeGui *)data;
    if (g->nbuttons <= 0) return;
    gui_select_filter(g, (g->active_index + 1) % g->nbuttons);
}

static void on_prev_filter(GtkWidget *w, gpointer data) {
    (void)w;
    FeGui *g = (FeGui *)data;
    if (g->nbuttons <= 0) return;
    gui_select_filter(g, (g->active_index - 1 + g->nbuttons) % g->nbuttons);
}

static void on_reset(GtkWidget *w, gpointer data) {
    (void)w;
    gui_reset_sliders((FeGui *)data);
}

static void gui_update_header_toggles(FeGui *g) {
    if (g->compare_btn) {
        const char *lbl = (g->compare_mode == COMPARE_SPLIT)  ? "Compare: Split"
                        : (g->compare_mode == COMPARE_BEFORE) ? "Compare: Before"
                                                              : "Compare";
        gtk_button_set_label(GTK_BUTTON(g->compare_btn), lbl);
        GtkStyleContext *ctx = gtk_widget_get_style_context(g->compare_btn);
        if (g->compare_mode != COMPARE_OFF) gtk_style_context_add_class(ctx, "fe-toggle-on");
        else gtk_style_context_remove_class(ctx, "fe-toggle-on");
    }
    if (g->hist_btn) {
        GtkStyleContext *ctx = gtk_widget_get_style_context(g->hist_btn);
        if (g->prefs.show_histogram) gtk_style_context_add_class(ctx, "fe-toggle-on");
        else gtk_style_context_remove_class(ctx, "fe-toggle-on");
    }
    if (g->grid_btn) {
        const char *glbl = (g->prefs.grid_mode == GRID_THIRDS)    ? "Grid: Thirds"
                         : (g->prefs.grid_mode == GRID_GOLDEN)    ? "Grid: Golden"
                         : (g->prefs.grid_mode == GRID_CROSSHAIR) ? "Grid: Cross"
                                                                  : "Grid";
        gtk_button_set_label(GTK_BUTTON(g->grid_btn), glbl);
        GtkStyleContext *ctx = gtk_widget_get_style_context(g->grid_btn);
        if (g->prefs.grid_mode != GRID_NONE) gtk_style_context_add_class(ctx, "fe-toggle-on");
        else gtk_style_context_remove_class(ctx, "fe-toggle-on");
    }
}

static void on_cycle_compare(GtkButton *btn, gpointer data) {
    (void)btn;
    FeGui *g = (FeGui *)data;
    g->compare_mode = (g->compare_mode + 1) % 3;
    if (g->compare_mode != COMPARE_OFF)
        gui_ensure_orig_surface(g);
    gui_update_header_toggles(g);
    if (g->preview) gtk_widget_queue_draw(g->preview);
    gui_status(g, g->compare_mode == COMPARE_SPLIT  ? "Comparison: Before / After Split Wipe (drag line on preview)"
                : g->compare_mode == COMPARE_BEFORE ? "Comparison: Viewing Original (Before)"
                                                    : "Comparison: Off (Viewing Edited)");
}

static void on_toggle_histogram(GtkButton *btn, gpointer data) {
    (void)btn;
    FeGui *g = (FeGui *)data;
    g->prefs.show_histogram = !g->prefs.show_histogram;
    gui_save_ui_prefs(g);
    gui_update_header_toggles(g);
    if (g->preview) gtk_widget_queue_draw(g->preview);
}

static void on_cycle_grid(GtkButton *btn, gpointer data) {
    (void)btn;
    FeGui *g = (FeGui *)data;
    g->prefs.grid_mode = (g->prefs.grid_mode + 1) % 4;
    gui_save_ui_prefs(g);
    gui_update_header_toggles(g);
    if (g->preview) gtk_widget_queue_draw(g->preview);
}

static void on_theme_combo_changed(GtkComboBox *combo, gpointer data) {
    FeGui *g = (FeGui *)data;
    if (g->suppress) return;
    int idx = gtk_combo_box_get_active(combo);
    if (idx >= 0 && idx < NTHEMES) {
        g->prefs.theme_idx = idx;
        gui_apply_theme(g);
        gui_save_ui_prefs(g);
        gui_status(g, "Theme switched to %s", FE_THEMES[idx].label);
    }
}

/* Mouse interaction on preview for dragging the Before/After Split Wipe divider */
static gboolean on_preview_press(GtkWidget *da, GdkEventButton *ev, gpointer data) {
    FeGui *g = (FeGui *)data;
    gtk_widget_grab_focus(da);
    if (g->compare_mode == COMPARE_SPLIT && ev->button == 1 && g->rgb_w > 0 && g->rgb_h > 0) {
        int aw = gtk_widget_get_allocated_width(da);
        int ah = gtk_widget_get_allocated_height(da);
        double pad = (g->prefs.frame_style == FRAME_MAT) ? 56.0 : 32.0;
        double s = MIN((aw - pad) / g->rgb_w, (ah - pad) / g->rgb_h);
        if (s > 1.0) s = 1.0;
        if (s <= 0.0) s = 1.0;
        double w = g->rgb_w * s;
        double x = (aw - w) * 0.5;
        if (w > 10.0) {
            g->split_ratio = CLAMP((ev->x - x) / w, 0.05, 0.95);
            g->dragging_split = 1;
            gtk_widget_queue_draw(da);
            return TRUE;
        }
    }
    return FALSE;
}

static gboolean on_preview_release(GtkWidget *da, GdkEventButton *ev, gpointer data) {
    (void)da; (void)ev;
    FeGui *g = (FeGui *)data;
    g->dragging_split = 0;
    return FALSE;
}

static gboolean on_preview_motion(GtkWidget *da, GdkEventMotion *ev, gpointer data) {
    FeGui *g = (FeGui *)data;
    if (g->dragging_split && g->compare_mode == COMPARE_SPLIT && g->rgb_w > 0 && g->rgb_h > 0) {
        int aw = gtk_widget_get_allocated_width(da);
        int ah = gtk_widget_get_allocated_height(da);
        double pad = (g->prefs.frame_style == FRAME_MAT) ? 56.0 : 32.0;
        double s = MIN((aw - pad) / g->rgb_w, (ah - pad) / g->rgb_h);
        if (s > 1.0) s = 1.0;
        if (s <= 0.0) s = 1.0;
        double w = g->rgb_w * s;
        double x = (aw - w) * 0.5;
        if (w > 10.0) {
            g->split_ratio = CLAMP((ev->x - x) / w, 0.05, 0.95);
            gtk_widget_queue_draw(da);
            return TRUE;
        }
    }
    return FALSE;
}

/* Left/right step through filters; [ and ] (or Up/Down) step through photos
   in the active Folder or Project strip; \ cycles Before/After compare. */
static gboolean on_preview_key(GtkWidget *w, GdkEventKey *ev, gpointer data) {
    FeGui *g = (FeGui *)data;
    if (ev->keyval == GDK_KEY_Right) { on_next_filter(w, g); return TRUE; }
    if (ev->keyval == GDK_KEY_Left)  { on_prev_filter(w, g); return TRUE; }
    if (ev->keyval == GDK_KEY_bracketright || ev->keyval == GDK_KEY_Page_Down) {
        gui_step_photo(g, +1);
        return TRUE;
    }
    if (ev->keyval == GDK_KEY_bracketleft || ev->keyval == GDK_KEY_Page_Up) {
        gui_step_photo(g, -1);
        return TRUE;
    }
    if (ev->keyval == GDK_KEY_backslash) {
        on_cycle_compare(NULL, g);
        return TRUE;
    }
    if (ev->keyval == GDK_KEY_h || ev->keyval == GDK_KEY_H) {
        on_toggle_histogram(NULL, g);
        return TRUE;
    }
    if (ev->keyval == GDK_KEY_g || ev->keyval == GDK_KEY_G) {
        on_cycle_grid(NULL, g);
        return TRUE;
    }
    return FALSE;
}

/* ------------------------------------------------------------------ */
/* Appearance & Workspace Customization Dialog                          */
/* ------------------------------------------------------------------ */

typedef struct {
    FeGui *g;
    GtkWidget *theme_cb;
    GtkWidget *accent_cb;
    GtkWidget *backdrop_cb;
    GtkWidget *frame_cb;
    GtkWidget *density_cb;
    GtkWidget *fstyle_cb;
    GtkWidget *hist_chk;
    GtkWidget *hud_chk;
    GtkWidget *grid_cb;
} FeCustDlg;

static void on_cust_changed(GtkWidget *w, gpointer data) {
    (void)w;
    FeCustDlg *cd = (FeCustDlg *)data;
    FeGui *g = cd->g;
    if (g->suppress) return;

    g->prefs.theme_idx     = CLAMP(gtk_combo_box_get_active(GTK_COMBO_BOX(cd->theme_cb)), 0, NTHEMES - 1);
    g->prefs.accent_idx    = CLAMP(gtk_combo_box_get_active(GTK_COMBO_BOX(cd->accent_cb)), 0, NACCENTS - 1);
    g->prefs.backdrop_idx  = CLAMP(gtk_combo_box_get_active(GTK_COMBO_BOX(cd->backdrop_cb)), 0, NBACKDROPS - 1);
    g->prefs.frame_style   = CLAMP(gtk_combo_box_get_active(GTK_COMBO_BOX(cd->frame_cb)), 0, 3);
    g->prefs.density       = CLAMP(gtk_combo_box_get_active(GTK_COMBO_BOX(cd->density_cb)), 0, 2);
    g->prefs.filter_style  = CLAMP(gtk_combo_box_get_active(GTK_COMBO_BOX(cd->fstyle_cb)), 0, 1);
    g->prefs.show_histogram = gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(cd->hist_chk)) ? 1 : 0;
    g->prefs.show_hud      = gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(cd->hud_chk)) ? 1 : 0;
    g->prefs.grid_mode     = CLAMP(gtk_combo_box_get_active(GTK_COMBO_BOX(cd->grid_cb)), 0, 3);

    if (g->theme_combo) {
        g->suppress++;
        gtk_combo_box_set_active(GTK_COMBO_BOX(g->theme_combo), g->prefs.theme_idx);
        g->suppress--;
    }
    gui_apply_theme(g);
    gui_update_header_toggles(g);
    gui_save_ui_prefs(g);
}

static void on_cust_reset_defaults(GtkButton *btn, gpointer data) {
    (void)btn;
    FeCustDlg *cd = (FeCustDlg *)data;
    FeGui *g = cd->g;
    gui_prefs_default(&g->prefs);
    g->suppress++;
    gtk_combo_box_set_active(GTK_COMBO_BOX(cd->theme_cb), g->prefs.theme_idx);
    gtk_combo_box_set_active(GTK_COMBO_BOX(cd->accent_cb), g->prefs.accent_idx);
    gtk_combo_box_set_active(GTK_COMBO_BOX(cd->backdrop_cb), g->prefs.backdrop_idx);
    gtk_combo_box_set_active(GTK_COMBO_BOX(cd->frame_cb), g->prefs.frame_style);
    gtk_combo_box_set_active(GTK_COMBO_BOX(cd->density_cb), g->prefs.density);
    gtk_combo_box_set_active(GTK_COMBO_BOX(cd->fstyle_cb), g->prefs.filter_style);
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(cd->hist_chk), g->prefs.show_histogram);
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(cd->hud_chk), g->prefs.show_hud);
    gtk_combo_box_set_active(GTK_COMBO_BOX(cd->grid_cb), g->prefs.grid_mode);
    if (g->theme_combo)
        gtk_combo_box_set_active(GTK_COMBO_BOX(g->theme_combo), g->prefs.theme_idx);
    g->suppress--;

    gui_apply_theme(g);
    gui_update_header_toggles(g);
    gui_save_ui_prefs(g);
}

static void on_open_customize(GtkButton *btn, gpointer data) {
    (void)btn;
    FeGui *g = (FeGui *)data;

    GtkWidget *dlg = gtk_dialog_new_with_buttons(
        "Customize Studio Appearance & Themes",
        GTK_WINDOW(g->window),
        GTK_DIALOG_MODAL | GTK_DIALOG_DESTROY_WITH_PARENT,
        "_Close", GTK_RESPONSE_CLOSE,
        NULL);
    gtk_window_set_default_size(GTK_WINDOW(dlg), 480, -1);

    GtkWidget *content = gtk_dialog_get_content_area(GTK_DIALOG(dlg));
    GtkWidget *grid = gtk_grid_new();
    gtk_grid_set_row_spacing(GTK_GRID(grid), 10);
    gtk_grid_set_column_spacing(GTK_GRID(grid), 14);
    gtk_container_set_border_width(GTK_CONTAINER(grid), 16);
    gtk_box_pack_start(GTK_BOX(content), grid, TRUE, TRUE, 0);

    FeCustDlg cd;
    memset(&cd, 0, sizeof(cd));
    cd.g = g;

    int row = 0;
    GtkWidget *hdr = gtk_label_new("COLOR THEME & PALETTE");
    gtk_label_set_xalign(GTK_LABEL(hdr), 0.0f);
    gtk_style_context_add_class(gtk_widget_get_style_context(hdr), "fe-section-title");
    gtk_grid_attach(GTK_GRID(grid), hdr, 0, row++, 2, 1);

    /* Theme selector */
    GtkWidget *l_theme = gtk_label_new("Color Theme");
    gtk_label_set_xalign(GTK_LABEL(l_theme), 0.0f);
    cd.theme_cb = gtk_combo_box_text_new();
    for (int i = 0; i < NTHEMES; i++)
        gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(cd.theme_cb), FE_THEMES[i].label);
    gtk_combo_box_set_active(GTK_COMBO_BOX(cd.theme_cb), g->prefs.theme_idx);
    gtk_widget_set_hexpand(cd.theme_cb, TRUE);
    gtk_grid_attach(GTK_GRID(grid), l_theme, 0, row, 1, 1);
    gtk_grid_attach(GTK_GRID(grid), cd.theme_cb, 1, row++, 1, 1);

    /* Accent color override */
    GtkWidget *l_acc = gtk_label_new("Accent Color");
    gtk_label_set_xalign(GTK_LABEL(l_acc), 0.0f);
    cd.accent_cb = gtk_combo_box_text_new();
    for (int i = 0; i < NACCENTS; i++)
        gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(cd.accent_cb), FE_ACCENTS[i].label);
    gtk_combo_box_set_active(GTK_COMBO_BOX(cd.accent_cb), g->prefs.accent_idx);
    gtk_grid_attach(GTK_GRID(grid), l_acc, 0, row, 1, 1);
    gtk_grid_attach(GTK_GRID(grid), cd.accent_cb, 1, row++, 1, 1);

    /* Canvas backdrop override */
    GtkWidget *l_bg = gtk_label_new("Canvas Backdrop");
    gtk_label_set_xalign(GTK_LABEL(l_bg), 0.0f);
    cd.backdrop_cb = gtk_combo_box_text_new();
    for (int i = 0; i < NBACKDROPS; i++)
        gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(cd.backdrop_cb), FE_BACKDROPS[i].label);
    gtk_combo_box_set_active(GTK_COMBO_BOX(cd.backdrop_cb), g->prefs.backdrop_idx);
    gtk_grid_attach(GTK_GRID(grid), l_bg, 0, row, 1, 1);
    gtk_grid_attach(GTK_GRID(grid), cd.backdrop_cb, 1, row++, 1, 1);

    GtkWidget *hdr2 = gtk_label_new("VIEWPORT & WORKSPACE LAYOUT");
    gtk_label_set_xalign(GTK_LABEL(hdr2), 0.0f);
    gtk_style_context_add_class(gtk_widget_get_style_context(hdr2), "fe-section-title");
    gtk_grid_attach(GTK_GRID(grid), hdr2, 0, row++, 2, 1);

    /* Photo Frame style */
    GtkWidget *l_frame = gtk_label_new("Photo Frame Style");
    gtk_label_set_xalign(GTK_LABEL(l_frame), 0.0f);
    cd.frame_cb = gtk_combo_box_text_new();
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(cd.frame_cb), "Soft Studio Drop Shadow");
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(cd.frame_cb), "Gallery White Matboard");
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(cd.frame_cb), "Thin Accent Hairline");
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(cd.frame_cb), "Flat (No Border)");
    gtk_combo_box_set_active(GTK_COMBO_BOX(cd.frame_cb), g->prefs.frame_style);
    gtk_grid_attach(GTK_GRID(grid), l_frame, 0, row, 1, 1);
    gtk_grid_attach(GTK_GRID(grid), cd.frame_cb, 1, row++, 1, 1);

    /* UI Density */
    GtkWidget *l_dens = gtk_label_new("UI Density & Scale");
    gtk_label_set_xalign(GTK_LABEL(l_dens), 0.0f);
    cd.density_cb = gtk_combo_box_text_new();
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(cd.density_cb), "Compact");
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(cd.density_cb), "Comfortable (Default)");
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(cd.density_cb), "Spacious");
    gtk_combo_box_set_active(GTK_COMBO_BOX(cd.density_cb), g->prefs.density);
    gtk_grid_attach(GTK_GRID(grid), l_dens, 0, row, 1, 1);
    gtk_grid_attach(GTK_GRID(grid), cd.density_cb, 1, row++, 1, 1);

    /* Filter Browser Button Style */
    GtkWidget *l_fstyle = gtk_label_new("Filter Browser Style");
    gtk_label_set_xalign(GTK_LABEL(l_fstyle), 0.0f);
    cd.fstyle_cb = gtk_combo_box_text_new();
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(cd.fstyle_cb), "Modern Cards (Category Color Bar)");
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(cd.fstyle_cb), "Classic Full-Color Buttons");
    gtk_combo_box_set_active(GTK_COMBO_BOX(cd.fstyle_cb), g->prefs.filter_style);
    gtk_grid_attach(GTK_GRID(grid), l_fstyle, 0, row, 1, 1);
    gtk_grid_attach(GTK_GRID(grid), cd.fstyle_cb, 1, row++, 1, 1);

    /* Composition Grid */
    GtkWidget *l_grid = gtk_label_new("Composition Grid");
    gtk_label_set_xalign(GTK_LABEL(l_grid), 0.0f);
    cd.grid_cb = gtk_combo_box_text_new();
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(cd.grid_cb), "None");
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(cd.grid_cb), "Rule of Thirds (3×3)");
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(cd.grid_cb), "Phi Golden Ratio");
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(cd.grid_cb), "Center Crosshair & Diagonals");
    gtk_combo_box_set_active(GTK_COMBO_BOX(cd.grid_cb), g->prefs.grid_mode);
    gtk_grid_attach(GTK_GRID(grid), l_grid, 0, row, 1, 1);
    gtk_grid_attach(GTK_GRID(grid), cd.grid_cb, 1, row++, 1, 1);

    /* Overlays checkboxes */
    cd.hist_chk = gtk_check_button_new_with_label("Show Live RGB / Luminance Histogram on Preview (H)");
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(cd.hist_chk), g->prefs.show_histogram);
    gtk_grid_attach(GTK_GRID(grid), cd.hist_chk, 0, row++, 2, 1);

    cd.hud_chk = gtk_check_button_new_with_label("Show Translucent Photo Info Pill on Preview");
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(cd.hud_chk), g->prefs.show_hud);
    gtk_grid_attach(GTK_GRID(grid), cd.hud_chk, 0, row++, 2, 1);

    GtkWidget *reset_ui_btn = gtk_button_new_with_label("Reset Appearance to Defaults (Gruvbox Dark)");
    gtk_grid_attach(GTK_GRID(grid), reset_ui_btn, 0, row++, 2, 1);

    g_signal_connect(cd.theme_cb, "changed", G_CALLBACK(on_cust_changed), &cd);
    g_signal_connect(cd.accent_cb, "changed", G_CALLBACK(on_cust_changed), &cd);
    g_signal_connect(cd.backdrop_cb, "changed", G_CALLBACK(on_cust_changed), &cd);
    g_signal_connect(cd.frame_cb, "changed", G_CALLBACK(on_cust_changed), &cd);
    g_signal_connect(cd.density_cb, "changed", G_CALLBACK(on_cust_changed), &cd);
    g_signal_connect(cd.fstyle_cb, "changed", G_CALLBACK(on_cust_changed), &cd);
    g_signal_connect(cd.grid_cb, "changed", G_CALLBACK(on_cust_changed), &cd);
    g_signal_connect(cd.hist_chk, "toggled", G_CALLBACK(on_cust_changed), &cd);
    g_signal_connect(cd.hud_chk, "toggled", G_CALLBACK(on_cust_changed), &cd);
    g_signal_connect(reset_ui_btn, "clicked", G_CALLBACK(on_cust_reset_defaults), &cd);

    gtk_widget_show_all(dlg);
    gtk_dialog_run(GTK_DIALOG(dlg));
    gtk_widget_destroy(dlg);
}

/* ------------------------------------------------------------------ */
/* building the window                                                  */
/* ------------------------------------------------------------------ */

static gboolean on_slider_button_press(GtkWidget *scale, GdkEventButton *ev, gpointer data) {
    FeGui *g = (FeGui *)data;
    /* Right-click or double-click on a slider resets that single slider to rest (0.0) */
    if (ev->button == 3 || (ev->button == 1 && ev->type == GDK_2BUTTON_PRESS)) {
        int idx = GPOINTER_TO_INT(g_object_get_data(G_OBJECT(scale), "fe-slider-idx")) - 1;
        if (idx >= 0 && idx < NSLIDERS) {
            g->suppress++;
            gtk_range_set_value(GTK_RANGE(g->scale[idx]), g->rest[idx]);
            g->suppress--;
            gui_update_slider_labels(g);
            g->draft = 0;
            gui_schedule_save(g);
            gui_schedule_render(g);
            gui_status(g, "Reset %s to preset default", SLIDERS[idx].label);
            return TRUE;
        }
    }
    return FALSE;
}

static GtkWidget *build_slider_page(FeGui *g, int tab) {
    GtkWidget *page = gtk_box_new(GTK_ORIENTATION_VERTICAL, 3);
    gtk_container_set_border_width(GTK_CONTAINER(page), 10);

    for (int i = 0; i < NSLIDERS; i++) {
        if (SLIDERS[i].tab != tab) continue;

        GtkWidget *hdr = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 4);
        GtkWidget *label = gtk_label_new(SLIDERS[i].label);
        gtk_label_set_xalign(GTK_LABEL(label), 0.0f);
        gtk_style_context_add_class(gtk_widget_get_style_context(label), "fe-slider-name");
        gtk_box_pack_start(GTK_BOX(hdr), label, TRUE, TRUE, 0);

        GtkWidget *vlabel = gtk_label_new("+0.00");
        gtk_style_context_add_class(gtk_widget_get_style_context(vlabel), "fe-slider-val");
        gtk_box_pack_end(GTK_BOX(hdr), vlabel, FALSE, FALSE, 0);
        g->val_label[i] = vlabel;
        gtk_box_pack_start(GTK_BOX(page), hdr, FALSE, FALSE, 2);

        GtkWidget *scale = gtk_scale_new_with_range(GTK_ORIENTATION_HORIZONTAL,
                                                    SLIDERS[i].lo, SLIDERS[i].hi,
                                                    SLIDERS[i].step);
        gtk_scale_set_draw_value(GTK_SCALE(scale), FALSE);
        gtk_widget_set_hexpand(scale, TRUE);
        gtk_widget_set_tooltip_text(scale, "Drag to adjust • Double-click or right-click to reset");
        gtk_range_set_value(GTK_RANGE(scale), 0.0);
        g_object_set_data(G_OBJECT(scale), "fe-slider-idx", GINT_TO_POINTER(i + 1));
        g->scale[i] = scale;
        g->rest[i] = gtk_range_get_value(GTK_RANGE(scale));

        g_signal_connect(scale, "value-changed", G_CALLBACK(on_slider_changed), g);
        g_signal_connect(scale, "button-press-event", G_CALLBACK(on_slider_button_press), g);
        gtk_box_pack_start(GTK_BOX(page), scale, FALSE, FALSE, 0);
    }
    if (tab == TAB_LENS) {
        GtkWidget *auto_lens_btn = gtk_button_new_with_label("Auto Lens Correction");
        gtk_style_context_add_class(gtk_widget_get_style_context(auto_lens_btn), "fe-btn-accent");
        g_signal_connect(auto_lens_btn, "clicked", G_CALLBACK(on_auto_lens), g);
        gtk_box_pack_start(GTK_BOX(page), auto_lens_btn, FALSE, FALSE, 8);
    }
    return page;
}

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
    g.split_ratio = 0.5;
    g.thumb_pending = g_queue_new();
    g.project = fe_project_new();

    gui_load_ui_prefs(&g);
    gui_check_slider_keys();

    gtk_init(NULL, NULL);

    g.window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    gtk_window_set_title(GTK_WINDOW(g.window), "SAGE RAW Studio — C edition");
    gtk_window_set_default_size(GTK_WINDOW(g.window), 1480, 920);
    gtk_window_set_position(GTK_WINDOW(g.window), GTK_WIN_POS_CENTER);
    g_signal_connect(g.window, "destroy", G_CALLBACK(on_window_destroy), &g);

    /* the header bar carries file, folder, comparison, theme & customize commands */
    GtkWidget *bar = gtk_header_bar_new();
    gtk_header_bar_set_title(GTK_HEADER_BAR(bar), "SAGE RAW STUDIO");
    gtk_header_bar_set_subtitle(GTK_HEADER_BAR(bar), "RAW Color & Film Pipeline");
    gtk_header_bar_set_show_close_button(GTK_HEADER_BAR(bar), TRUE);

    GtkWidget *open_btn = gtk_button_new_with_label("Open Photo");
    g_signal_connect(open_btn, "clicked", G_CALLBACK(on_open), &g);
    gtk_header_bar_pack_start(GTK_HEADER_BAR(bar), open_btn);

    GtkWidget *open_folder_btn = gtk_button_new_with_label("Open Folder");
    g_signal_connect(open_folder_btn, "clicked", G_CALLBACK(on_browse_folder), &g);
    gtk_header_bar_pack_start(GTK_HEADER_BAR(bar), open_folder_btn);

    g.compare_btn = gtk_button_new_with_label("Compare");
    gtk_widget_set_tooltip_text(g.compare_btn, "Cycle Before / After Split Comparison (\\)");
    g_signal_connect(g.compare_btn, "clicked", G_CALLBACK(on_cycle_compare), &g);
    gtk_header_bar_pack_start(GTK_HEADER_BAR(bar), g.compare_btn);

    g.hist_btn = gtk_button_new_with_label("Histogram");
    gtk_widget_set_tooltip_text(g.hist_btn, "Toggle Live RGB / Luminance Histogram (H)");
    g_signal_connect(g.hist_btn, "clicked", G_CALLBACK(on_toggle_histogram), &g);
    gtk_header_bar_pack_start(GTK_HEADER_BAR(bar), g.hist_btn);

    g.grid_btn = gtk_button_new_with_label("Grid");
    gtk_widget_set_tooltip_text(g.grid_btn, "Cycle Composition Grid Overlay (G)");
    g_signal_connect(g.grid_btn, "clicked", G_CALLBACK(on_cycle_grid), &g);
    gtk_header_bar_pack_start(GTK_HEADER_BAR(bar), g.grid_btn);

    GtkWidget *export_btn = gtk_button_new_with_label("Export Image");
    gtk_style_context_add_class(gtk_widget_get_style_context(export_btn), "fe-btn-success");
    g_signal_connect(export_btn, "clicked", G_CALLBACK(on_export), &g);
    gtk_header_bar_pack_end(GTK_HEADER_BAR(bar), export_btn);

    GtkWidget *cust_btn = gtk_button_new_with_label("Customize");
    gtk_widget_set_tooltip_text(cust_btn, "Customize Themes, Accents, Canvas Backdrop & Layout");
    g_signal_connect(cust_btn, "clicked", G_CALLBACK(on_open_customize), &g);
    gtk_header_bar_pack_end(GTK_HEADER_BAR(bar), cust_btn);

    g.theme_combo = gtk_combo_box_text_new();
    gtk_widget_set_tooltip_text(g.theme_combo, "Switch Color Theme");
    for (int i = 0; i < NTHEMES; i++)
        gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(g.theme_combo), FE_THEMES[i].label);
    gtk_combo_box_set_active(GTK_COMBO_BOX(g.theme_combo), g.prefs.theme_idx);
    g_signal_connect(g.theme_combo, "changed", G_CALLBACK(on_theme_combo_changed), &g);
    gtk_header_bar_pack_end(GTK_HEADER_BAR(bar), g.theme_combo);

    gtk_window_set_titlebar(GTK_WINDOW(g.window), bar);

    /* --- left column: the fine-tune tabs -------------------------------- */
    GtkWidget *left = gtk_box_new(GTK_ORIENTATION_VERTICAL, 6);
    gtk_style_context_add_class(gtk_widget_get_style_context(left), "fe-sidebar");
    gtk_container_set_border_width(GTK_CONTAINER(left), 10);

    GtkWidget *dev_title = gtk_label_new("DEVELOP CONTROLS");
    gtk_label_set_xalign(GTK_LABEL(dev_title), 0.0f);
    gtk_style_context_add_class(gtk_widget_get_style_context(dev_title), "fe-section-title");
    gtk_box_pack_start(GTK_BOX(left), dev_title, FALSE, FALSE, 2);

    GtkWidget *fine_page = build_slider_page(&g, TAB_FINE);
    GtkWidget *auto_btn = gtk_button_new_with_label("Auto Exposure");
    gtk_style_context_add_class(gtk_widget_get_style_context(auto_btn), "fe-btn-accent");
    g_signal_connect(auto_btn, "clicked", G_CALLBACK(on_auto_exposure), &g);
    gtk_box_pack_start(GTK_BOX(fine_page), auto_btn, FALSE, FALSE, 8);

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

    /* Per-photo edit tools: Reset Sliders, Reset Photo, Copy/Paste Edits */
    GtkWidget *reset_row = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 6);
    GtkWidget *reset_btn = gtk_button_new_with_label("Reset Sliders");
    g_signal_connect(reset_btn, "clicked", G_CALLBACK(on_reset), &g);
    gtk_box_pack_start(GTK_BOX(reset_row), reset_btn, TRUE, TRUE, 0);
    GtkWidget *reset_photo_btn = gtk_button_new_with_label("Reset Photo");
    gtk_style_context_add_class(gtk_widget_get_style_context(reset_photo_btn), "fe-btn-danger");
    g_signal_connect(reset_photo_btn, "clicked", G_CALLBACK(on_reset_photo), &g);
    gtk_box_pack_start(GTK_BOX(reset_row), reset_photo_btn, TRUE, TRUE, 0);
    gtk_box_pack_start(GTK_BOX(left), reset_row, FALSE, FALSE, 0);

    GtkWidget *clip_row = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 6);
    GtkWidget *copy_btn = gtk_button_new_with_label("Copy Edits");
    g_signal_connect(copy_btn, "clicked", G_CALLBACK(on_copy_edits), &g);
    gtk_box_pack_start(GTK_BOX(clip_row), copy_btn, TRUE, TRUE, 0);
    GtkWidget *paste_btn = gtk_button_new_with_label("Paste Edits");
    g_signal_connect(paste_btn, "clicked", G_CALLBACK(on_paste_edits), &g);
    gtk_box_pack_start(GTK_BOX(clip_row), paste_btn, TRUE, TRUE, 0);
    gtk_box_pack_start(GTK_BOX(left), clip_row, FALSE, FALSE, 0);

    g.fullres = gtk_check_button_new_with_label("Full-res on export (slow)");
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g.fullres), TRUE);
    gtk_box_pack_start(GTK_BOX(left), g.fullres, FALSE, FALSE, 2);

    /* --- centre column: the preview ------------------------------------- */
    g.preview = gtk_drawing_area_new();
    gtk_widget_set_size_request(g.preview, 480, 400);
    gtk_widget_set_hexpand(g.preview, TRUE);
    gtk_widget_set_vexpand(g.preview, TRUE);
    gtk_widget_set_can_focus(g.preview, TRUE);
    gtk_widget_add_events(g.preview, GDK_BUTTON_PRESS_MASK | GDK_BUTTON_RELEASE_MASK |
                                     GDK_POINTER_MOTION_MASK);
    g_signal_connect(g.preview, "draw", G_CALLBACK(on_draw_preview), &g);
    g_signal_connect(g.preview, "key-press-event", G_CALLBACK(on_preview_key), &g);
    g_signal_connect(g.preview, "button-press-event", G_CALLBACK(on_preview_press), &g);
    g_signal_connect(g.preview, "button-release-event", G_CALLBACK(on_preview_release), &g);
    g_signal_connect(g.preview, "motion-notify-event", G_CALLBACK(on_preview_motion), &g);

    /* --- right column: the filter browser ------------------------------- */
    GtkWidget *browser_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 6);
    gtk_style_context_add_class(gtk_widget_get_style_context(browser_box), "fe-sidebar");
    gtk_container_set_border_width(GTK_CONTAINER(browser_box), 10);

    GtkWidget *browser_title = gtk_label_new("FILTER PRESETS");
    gtk_label_set_xalign(GTK_LABEL(browser_title), 0.0f);
    gtk_style_context_add_class(gtk_widget_get_style_context(browser_title), "fe-section-title");
    gtk_box_pack_start(GTK_BOX(browser_box), browser_title, FALSE, FALSE, 2);

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
    gtk_container_set_border_width(GTK_CONTAINER(browser_list), 2);
    gtk_container_add(GTK_CONTAINER(browser_scroll), browser_list);
    gui_build_browser(&g, browser_list);
    gtk_widget_set_size_request(browser_box, 265, -1);

    GtkWidget *paned = gtk_paned_new(GTK_ORIENTATION_HORIZONTAL);
    gtk_paned_set_position(GTK_PANED(paned), 330);
    gtk_paned_pack1(GTK_PANED(paned), left, FALSE, TRUE);
    gtk_paned_pack2(GTK_PANED(paned), g.preview, TRUE, TRUE);

    GtkWidget *outer = gtk_paned_new(GTK_ORIENTATION_HORIZONTAL);
    gtk_paned_set_position(GTK_PANED(outer), 1150);
    gtk_paned_pack1(GTK_PANED(outer), paned, TRUE, TRUE);
    gtk_paned_pack2(GTK_PANED(outer), browser_box, FALSE, TRUE);

    /* --- the bottom panel: folder strip + project strip ------------------ */
    GtkWidget *bottom = gtk_box_new(GTK_ORIENTATION_VERTICAL, 4);
    gtk_style_context_add_class(gtk_widget_get_style_context(bottom), "fe-bottom-panel");
    gtk_container_set_border_width(GTK_CONTAINER(bottom), 8);

    GtkWidget *top_row = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 6);
    GtkWidget *ftitle = gtk_label_new("FILMSTRIP  •  FOLDER & PROJECT");
    gtk_label_set_xalign(GTK_LABEL(ftitle), 0.0f);
    gtk_style_context_add_class(gtk_widget_get_style_context(ftitle), "fe-section-title");
    gtk_box_pack_start(GTK_BOX(top_row), ftitle, FALSE, FALSE, 4);

    GtkWidget *bf = gtk_button_new_with_label("Browse Folder");
    g_signal_connect(bf, "clicked", G_CALLBACK(on_browse_folder), &g);
    gtk_box_pack_start(GTK_BOX(top_row), bf, FALSE, FALSE, 0);

    GtkWidget *afp = gtk_button_new_with_label("Add Folder to Project");
    g_signal_connect(afp, "clicked", G_CALLBACK(on_add_folder_to_project), &g);
    gtk_box_pack_start(GTK_BOX(top_row), afp, FALSE, FALSE, 0);

    GtkWidget *apb = gtk_button_new_with_label("Add Photos to Project");
    g_signal_connect(apb, "clicked", G_CALLBACK(on_add_project), &g);
    gtk_box_pack_start(GTK_BOX(top_row), apb, FALSE, FALSE, 0);

    GtkWidget *prev_p = gtk_button_new_with_label("◀ Prev Photo");
    g_signal_connect(prev_p, "clicked", G_CALLBACK(on_prev_photo), &g);
    gtk_box_pack_end(GTK_BOX(top_row), gtk_label_new(""), FALSE, FALSE, 0);
    GtkWidget *next_p = gtk_button_new_with_label("Next Photo ▶");
    g_signal_connect(next_p, "clicked", G_CALLBACK(on_next_photo), &g);
    gtk_box_pack_end(GTK_BOX(top_row), next_p, FALSE, FALSE, 0);
    gtk_box_pack_end(GTK_BOX(top_row), prev_p, FALSE, FALSE, 0);

    gtk_box_pack_start(GTK_BOX(bottom), top_row, FALSE, FALSE, 0);

    g.bottom_tabs = gtk_notebook_new();

    /* tab 0 — the folder we are working in */
    GtkWidget *folder_scroll = gtk_scrolled_window_new(NULL, NULL);
    gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(folder_scroll),
                                   GTK_POLICY_AUTOMATIC, GTK_POLICY_NEVER);
    gtk_widget_set_size_request(folder_scroll, -1, 145);
    g.folder_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 6);
    gtk_container_add(GTK_CONTAINER(folder_scroll), g.folder_box);
    g.folder_page = folder_scroll;
    gtk_notebook_append_page(GTK_NOTEBOOK(g.bottom_tabs), folder_scroll,
                             gtk_label_new("Folder"));

    /* tab 1 — the batch: select, mass-apply, sync edits, save/load, mass-export */
    GtkWidget *project_page = gtk_box_new(GTK_ORIENTATION_VERTICAL, 4);
    gtk_container_set_border_width(GTK_CONTAINER(project_page), 4);

    GtkWidget *pt = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 4);

    GtkWidget *applyb = gtk_button_new_with_label("Apply Filter to Selected");
    gtk_style_context_add_class(gtk_widget_get_style_context(applyb), "fe-btn-accent");
    g_signal_connect(applyb, "clicked", G_CALLBACK(on_apply_filter_selected), &g);
    gtk_box_pack_start(GTK_BOX(pt), applyb, FALSE, FALSE, 0);

    GtkWidget *syncb = gtk_button_new_with_label("Sync Edits to Selected");
    g_signal_connect(syncb, "clicked", G_CALLBACK(on_sync_edits_selected), &g);
    gtk_box_pack_start(GTK_BOX(pt), syncb, FALSE, FALSE, 0);

    GtkWidget *reset_sel_b = gtk_button_new_with_label("Reset Selected Edits");
    gtk_style_context_add_class(gtk_widget_get_style_context(reset_sel_b), "fe-btn-danger");
    g_signal_connect(reset_sel_b, "clicked", G_CALLBACK(on_reset_selected_edits), &g);
    gtk_box_pack_start(GTK_BOX(pt), reset_sel_b, FALSE, FALSE, 0);

    const char *modes[4][2] = {{"Select All", "all"}, {"None", "none"},
                               {"Invert", "invert"}, {"Remove Selected", "remove"}};
    for (int i = 0; i < 4; i++) {
        GtkWidget *b = gtk_button_new_with_label(modes[i][0]);
        g_object_set_data(G_OBJECT(b), "fe-mode", (gpointer)modes[i][1]);
        g_signal_connect(b, "clicked", G_CALLBACK(on_project_select), &g);
        gtk_box_pack_start(GTK_BOX(pt), b, FALSE, FALSE, 0);
    }

    GtkWidget *save_proj_b = gtk_button_new_with_label("Save Project");
    g_signal_connect(save_proj_b, "clicked", G_CALLBACK(on_save_project), &g);
    gtk_box_pack_start(GTK_BOX(pt), save_proj_b, FALSE, FALSE, 0);

    GtkWidget *load_proj_b = gtk_button_new_with_label("Load Project");
    g_signal_connect(load_proj_b, "clicked", G_CALLBACK(on_load_project), &g);
    gtk_box_pack_start(GTK_BOX(pt), load_proj_b, FALSE, FALSE, 0);

    GtkWidget *exportb = gtk_button_new_with_label("Export Selected");
    gtk_style_context_add_class(gtk_widget_get_style_context(exportb), "fe-btn-success");
    g_signal_connect(exportb, "clicked", G_CALLBACK(on_export_selected), &g);
    gtk_box_pack_start(GTK_BOX(pt), exportb, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(project_page), pt, FALSE, FALSE, 0);

    GtkWidget *project_scroll = gtk_scrolled_window_new(NULL, NULL);
    gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(project_scroll),
                                   GTK_POLICY_AUTOMATIC, GTK_POLICY_NEVER);
    gtk_widget_set_size_request(project_scroll, -1, 130);
    g.project_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 6);
    gtk_container_add(GTK_CONTAINER(project_scroll), g.project_box);
    gtk_box_pack_start(GTK_BOX(project_page), project_scroll, TRUE, TRUE, 0);

    GtkWidget *project_placeholder = gtk_label_new(
        "No photos in project yet — press “Add Folder to Project”, “Add Photos to Project”, or “Load Project”.");
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
    g.status = gtk_label_new("Open an image or folder to start");
    gtk_label_set_xalign(GTK_LABEL(g.status), 0.0f);
    gtk_label_set_ellipsize(GTK_LABEL(g.status), PANGO_ELLIPSIZE_END);
    GtkWidget *status_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);
    gtk_style_context_add_class(gtk_widget_get_style_context(status_box), "fe-status-bar");
    gtk_container_set_border_width(GTK_CONTAINER(status_box), 6);
    gtk_box_pack_start(GTK_BOX(status_box), g.status, TRUE, TRUE, 4);

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

    /* Apply active theme & customization preferences */
    gui_apply_theme(&g);
    gui_update_header_toggles(&g);
    gui_update_slider_labels(&g);

    gtk_widget_show_all(g.window);
    gui_update_project_label(&g);
    gtk_widget_grab_focus(g.preview);

    /* Start on the first filter under suppress so startup never writes a sidecar. */
    g.suppress++;
    if (g.nbuttons > 0) gui_select_filter(&g, 0);
    g.suppress--;

    /* If main() already loaded an image from the command line, restore its
       sidecar state and populate its parent folder strip without overwriting
       the sidecar! */
    if (g.app->has_image && g.app->file_path[0]) {
        FeEditState st;
        memset(&st, 0, sizeof(st));
        int has = fe_state_load(g.app->file_path, &st);
        g.suppress++;
        if (has) {
            gui_apply_state(&g, &st);
            g.state = st;
        }
        g.suppress--;
        g.dirty = 0;
        gui_ensure_parent_folder_loaded(&g, g.app->file_path);
        gui_highlight_active_photo(&g);
        gui_schedule_render(&g);
    }

    gtk_main();

    if (g.timeout_id) g_source_remove(g.timeout_id);
    if (g.settle_id) g_source_remove(g.settle_id);
    if (g.save_id) g_source_remove(g.save_id);
    if (g.thumb_idle) g_source_remove(g.thumb_idle);
    gui_thumbs_drain(&g);
    g_queue_free(g.thumb_pending);
    gui_free_folder_paths(&g);

    if (g.theme_css) g_object_unref(g.theme_css);
    if (g.surface) cairo_surface_destroy(g.surface);
    if (g.orig_surface) cairo_surface_destroy(g.orig_surface);
    free(g.rgb);
    free(g.orig_rgb);
    free(g.buttons);
    free(g.expanders);
    fe_project_free(g.project);
    return 0;
}
