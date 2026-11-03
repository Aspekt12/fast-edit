#ifndef FE_STATE_H
#define FE_STATE_H

/* =============================================================================
   PER-IMAGE EDIT STATE

   Every photo remembers its own edit: which preset it was developed with and
   how far each slider was moved away from that preset.  The state lives in a
   small JSON sidecar next to the photo:

       <photo>.ARW.fastedit.json

       {
         "version": 1,
         "filter": "02-film/portra-400",
         "offsets": {
           "exposure": 0.25,
           "contrast": -0.1
         }
       }

   "filter" is the preset key, "offsets" are the slider deltas (the slider
   position minus its resting value).  Storing the DELTA rather than the
   final number keeps the file small and makes the intent obvious: it says
   "this photo is portra-400 plus a quarter stop of exposure".  An offset of
   exactly 0 means "the slider sits on the preset" and is left out of the
   file, so what is written is exactly what the user changed.

   The sidecar is written atomically (temp file + rename) so a crash can
   never leave a half-written file behind.
   ============================================================================= */

#include <stddef.h>

#define FE_STATE_MAX_FIELDS 64
#define FE_STATE_FILTER_KEY_MAX 256
#define FE_STATE_NAME_MAX 48

/* one named slider value */
typedef struct {
    char name[FE_STATE_NAME_MAX];
    double value;
} FeStateField;

/* the whole edit state of one photo */
typedef struct {
    char filter_key[FE_STATE_FILTER_KEY_MAX];  /* the preset, e.g. "02-film/portra-400" */
    FeStateField fields[FE_STATE_MAX_FIELDS];  /* slider deltas, keyed by name */
    int nfields;
} FeEditState;

/* Build the sidecar path for an image: "<image path>.fastedit.json". */
void fe_state_path(const char *image_path, char *out, size_t out_len);

/* Read the state of `image_path`.  Returns 1 when a sidecar was found and
   parsed, 0 otherwise (in which case `st` is left untouched). */
int fe_state_load(const char *image_path, FeEditState *st);

/* Write the state of `image_path`.  Returns 1 on success. */
int fe_state_save(const char *image_path, const FeEditState *st);

/* Remove the sidecar of `image_path` (if any).  Returns 1 if removed or not present. */
int fe_state_remove(const char *image_path);

/* Small helpers for the field list. */
int fe_state_get(const FeEditState *st, const char *name, double *out);
void fe_state_set(FeEditState *st, const char *name, double value);

#endif /* FE_STATE_H */
