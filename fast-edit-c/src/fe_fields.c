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
} FeField;

static const FeField FE_FIELDS[] = {
    /* --- white balance / calibration --------------------------------- */
    {"temperature",           FIELD(temperature)},
    {"tint",                  FIELD(tint)},
    {"cal_red_hue",           FIELD(cal_red_hue)},
    {"cal_green_hue",         FIELD(cal_green_hue)},
    {"cal_blue_hue",          FIELD(cal_blue_hue)},
    {"cal_red_sat",           FIELD(cal_red_sat)},
    {"cal_green_sat",         FIELD(cal_green_sat)},
    {"cal_blue_sat",          FIELD(cal_blue_sat)},

    /* --- basic tonality ---------------------------------------------- */
    {"exposure",              FIELD(exposure)},
    {"contrast",              FIELD(contrast)},
    {"highlights",            FIELD(highlights)},
    {"shadows",               FIELD(shadows)},
    {"blacks",                FIELD(blacks)},
    {"whites",                FIELD(whites)},
    {"fade",                  FIELD(fade)},

    /* --- grading ------------------------------------------------------ */
    {"grade_blending",        FIELD(grade_blending)},

    /* --- colour ------------------------------------------------------- */
    {"saturation",            FIELD(saturation)},
    {"vibrance",              FIELD(vibrance)},

    /* --- presence ------------------------------------------------------ */
    {"texture",               FIELD(texture)},
    {"clarity",               FIELD(clarity)},
    {"dehaze",                FIELD(dehaze)},

    /* --- effects ------------------------------------------------------- */
    {"grain_amount",          FIELD(grain_amount)},
    {"grain_size",            FIELD(grain_size)},
    {"grain_color",           FIELD(grain_color)},
    {"grain_bias",            FIELD(grain_bias)},
    {"vignette_amount",       FIELD(vignette_amount)},
    {"bloom_amount",          FIELD(bloom_amount)},
    {"bloom_radius",          FIELD(bloom_radius)},
    {"bloom_threshold",       FIELD(bloom_threshold)},

    /* --- lens correction ---------------------------------------------- */
    {"lens_distortion",       FIELD(lens_distortion)},
    {"lens_tca_r",            FIELD(lens_tca_r)},
    {"lens_tca_b",            FIELD(lens_tca_b)},
    {"lens_vignette_corr",    FIELD(lens_vignette_corr)},
    {"lens_k1",               FIELD(lens_k1)},
    {"lens_k2",               FIELD(lens_k2)},
    {"lens_k3",               FIELD(lens_k3)},
    {"lens_ca_r",             FIELD(lens_ca_r)},
    {"lens_ca_b",             FIELD(lens_ca_b)},
    {"lens_keystone_h",       FIELD(lens_keystone_h)},
    {"lens_keystone_v",       FIELD(lens_keystone_v)},
    {"lens_rotate",           FIELD(lens_rotate)},

    /* --- sharpening / noise reduction --------------------------------- */
    {"sharpening_amount",     FIELD(sharpening_amount)},
    {"sharpening_radius",     FIELD(sharpening_radius)},
    {"sharpening_detail",     FIELD(sharpening_detail)},
    {"sharpening_masking",    FIELD(sharpening_masking)},
    {"nr_luminance",          FIELD(nr_luminance)},
    {"nr_color",              FIELD(nr_color)},

    /* --- curves -------------------------------------------------------- */
    {"curve_highlights",      FIELD(curve_highlights)},
    {"curve_lights",          FIELD(curve_lights)},
    {"curve_darks",           FIELD(curve_darks)},
    {"curve_shadows",         FIELD(curve_shadows)},

    /* --- crop / straighten --------------------------------------------- */
    {"crop_x",                FIELD(crop_x)},
    {"crop_y",                FIELD(crop_y)},
    {"crop_w",                FIELD(crop_w)},
    {"crop_h",                FIELD(crop_h)},
    {"straighten_angle",      FIELD(straighten_angle)},

    /* --- pixel art ------------------------------------------------------ */
    {"pixel_size",            FIELD(pixel_size)},
};

#define FE_FIELD_COUNT ((int)(sizeof(FE_FIELDS) / sizeof(FE_FIELDS[0])))

ptrdiff_t fe_settings_field_offset(const char *name) {
    if (!name || !name[0]) return -1;
    for (int i = 0; i < FE_FIELD_COUNT; i++) {
        if (strcmp(FE_FIELDS[i].name, name) == 0) return FE_FIELDS[i].offset;
    }
    return -1;
}

int fe_settings_field_count(void) { return FE_FIELD_COUNT; }

int fe_settings_field_at(int index, const char **name_out, ptrdiff_t *offset_out) {
    if (index < 0 || index >= FE_FIELD_COUNT) return 0;
    if (name_out) *name_out = FE_FIELDS[index].name;
    if (offset_out) *offset_out = FE_FIELDS[index].offset;
    return 1;
}
