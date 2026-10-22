#ifndef FE_FIELDS_H
#define FE_FIELDS_H

/* =============================================================================
   THE NAME OF EVERY SETTING

   One place that says what each field of FeSettings is called in JSON.  It
   is used by two callers:

     * the filter presets   (filters/<category>/<name>.json  -> "settings")
     * the per-image edits  (<photo>.fastedit.json           -> "offsets")

   Sharing the table is what keeps those two formats from drifting apart.
   The GUI's slider table carries the same key in its `key` member, and a
   startup check in gui.c verifies that each slider's key really does name
   the field it writes into.
   ============================================================================= */

#include <stddef.h>

#include "fe_defs.h"

/* Byte offset of `name` inside FeSettings, or -1 when no such field
   exists. */
ptrdiff_t fe_settings_field_offset(const char *name);

/* How many named fields the table holds. */
int fe_settings_field_count(void);

/* Number-th entry of the table; returns 0 when `index` is out of range. */
int fe_settings_field_at(int index, const char **name_out, ptrdiff_t *offset_out);

#endif /* FE_FIELDS_H */
