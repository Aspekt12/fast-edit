#include "fe_fields.h"

#include <stddef.h>
#include <string.h>

#define FIELD(name) offsetof(FeSettings, name)

/* kind 0 = plain double, 1 = (hue, strength) pair, 2 = mixer object.
   Only kind 0 lives here — the two structured shapes are handled by the
   filter parser in fe_filters.c, which knows their JSON syntax. */
typedef struct {
    const char *name;
    ptrdiff_t offset;
    double floor_val;
} FeField;

#define NO_FLOOR (-1e9)

static const FeField FE_FIELDS[] = {
    /* --- white balance / calibration --------------------------------- */
    {"temperature",           FIELD(temperature),           NO_FLOOR},
    {"tint",                  FIELD(tint),                  NO_FLOOR},
    {"cal_red_hue",           FIELD(cal_red_hue),           NO_FLOOR},
    {"cal_green_hue",         FIELD(cal_green_hue),         NO_FLOOR},
    {"cal_blue_hue",          FIELD(cal_blue_hue),          NO_FLOOR},
    {"cal_red_sat",           FIELD(cal_red_sat),           NO_FLOOR},
    {"cal_green_sat",         FIELD(cal_green_sat),         NO_FLOOR},
    {"cal_blue_sat",          FIELD(cal_blue_sat),          NO_FLOOR},

    /* --- basic tonality ---------------------------------------------- */
    {"exposure",              FIELD(exposure),              NO_FLOOR},
    {"contrast",              FIELD(contrast),              NO_FLOOR},
    {"highlights",            FIELD(highlights),            NO_FLOOR},
    {"shadows",               FIELD(shadows),               NO_FLOOR},
    {"blacks",                FIELD(blacks),                NO_FLOOR},
    {"whites",                FIELD(whites),                NO_FLOOR},
    {"fade",                  FIELD(fade),                  0.0},

    /* --- grading ------------------------------------------------------ */
    {"grade_blending",        FIELD(grade_blending),        NO_FLOOR},

    /* --- colour ------------------------------------------------------- */
    {"saturation",            FIELD(saturation),            NO_FLOOR},
    {"vibrance",              FIELD(vibrance),              NO_FLOOR},

    /* --- presence ------------------------------------------------------ */
    {"texture",               FIELD(texture),               NO_FLOOR},
    {"clarity",               FIELD(clarity),               NO_FLOOR},
    {"dehaze",                FIELD(dehaze),                NO_FLOOR},

    /* --- effects ------------------------------------------------------- */
    {"grain_amount",          FIELD(grain_amount),          0.0},
    {"grain_size",            FIELD(grain_size),            0.6},
    {"grain_color",           FIELD(grain_color),           0.0},
    {"grain_bias",            FIELD(grain_bias),            20.0},
    {"vignette_amount",       FIELD(vignette_amount),       0.0},
    {"bloom_amount",          FIELD(bloom_amount),          0.0},
    {"bloom_radius",          FIELD(bloom_radius),          NO_FLOOR},
    {"bloom_threshold",       FIELD(bloom_threshold),       NO_FLOOR},

    /* --- lens correction ---------------------------------------------- */
    {"lens_distortion",       FIELD(lens_distortion),       NO_FLOOR},
    {"lens_tca_r",            FIELD(lens_tca_r),            NO_FLOOR},
    {"lens_tca_b",            FIELD(lens_tca_b),            NO_FLOOR},
    {"lens_vignette_corr",    FIELD(lens_vignette_corr),    NO_FLOOR},
    {"lens_k1",               FIELD(lens_k1),               NO_FLOOR},
    {"lens_k2",               FIELD(lens_k2),               NO_FLOOR},
    {"lens_k3",               FIELD(lens_k3),               NO_FLOOR},
    {"lens_ca_r",             FIELD(lens_ca_r),             NO_FLOOR},
    {"lens_ca_b",             FIELD(lens_ca_b),             NO_FLOOR},
    {"lens_keystone_h",       FIELD(lens_keystone_h),       NO_FLOOR},
    {"lens_keystone_v",       FIELD(lens_keystone_v),       NO_FLOOR},
    {"lens_rotate",           FIELD(lens_rotate),           NO_FLOOR},

    /* --- sharpening / noise reduction --------------------------------- */
    {"sharpening_amount",     FIELD(sharpening_amount),     0.0},
    {"sharpening_radius",     FIELD(sharpening_radius),     NO_FLOOR},
    {"sharpening_detail",     FIELD(sharpening_detail),     NO_FLOOR},
    {"sharpening_masking",    FIELD(sharpening_masking),    NO_FLOOR},
    {"nr_luminance",          FIELD(nr_luminance),          NO_FLOOR},
    {"nr_color",              FIELD(nr_color),              NO_FLOOR},

    /* --- curves -------------------------------------------------------- */
    {"curve_highlights",      FIELD(curve_highlights),      NO_FLOOR},
    {"curve_lights",          FIELD(curve_lights),          NO_FLOOR},
    {"curve_darks",           FIELD(curve_darks),           NO_FLOOR},
    {"curve_shadows",         FIELD(curve_shadows),         NO_FLOOR},

    /* --- crop / straighten --------------------------------------------- */
    {"crop_x",                FIELD(crop_x),                NO_FLOOR},
    {"crop_y",                FIELD(crop_y),                NO_FLOOR},
    {"crop_w",                FIELD(crop_w),                NO_FLOOR},
    {"crop_h",                FIELD(crop_h),                NO_FLOOR},
    {"straighten_angle",      FIELD(straighten_angle),      NO_FLOOR},

    /* --- pixel art ------------------------------------------------------ */
    {"pixel_size",            FIELD(pixel_size),            0.0},
};

#define FE_FIELD_COUNT ((int)(sizeof(FE_FIELDS) / sizeof(FE_FIELDS[0])))

ptrdiff_t fe_settings_field_offset(const char *name) {
    if (!name || !name[0]) return -1;
    for (int i = 0; i < FE_FIELD_COUNT; i++) {
        if (strcmp(FE_FIELDS[i].name, name) == 0) return FE_FIELDS[i].offset;
    }
    return -1;
}

double fe_settings_field_floor(const char *name) {
    if (!name || !name[0]) return NO_FLOOR;
    for (int i = 0; i < FE_FIELD_COUNT; i++) {
        if (strcmp(FE_FIELDS[i].name, name) == 0) return FE_FIELDS[i].floor_val;
    }
    return NO_FLOOR;
}

int fe_settings_apply_offset(FeSettings *settings, const char *name, double delta) {
    if (!settings || !name || !name[0]) return 0;
    for (int i = 0; i < FE_FIELD_COUNT; i++) {
        if (strcmp(FE_FIELDS[i].name, name) == 0) {
            double *p = (double *)((char *)settings + FE_FIELDS[i].offset);
            double v = *p + delta;
            if (v < FE_FIELDS[i].floor_val) v = FE_FIELDS[i].floor_val;
            *p = v;
            return 1;
        }
    }
    return 0;
}

int fe_settings_field_count(void) { return FE_FIELD_COUNT; }

int fe_settings_field_at(int index, const char **name_out, ptrdiff_t *offset_out) {
    if (index < 0 || index >= FE_FIELD_COUNT) return 0;
    if (name_out) *name_out = FE_FIELDS[index].name;
    if (offset_out) *offset_out = FE_FIELDS[index].offset;
    return 1;
}
