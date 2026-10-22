#ifndef FE_VISION_H
#define FE_VISION_H

#include "fe_defs.h"

#include <stddef.h>

/* Object recognition / scene analysis for smarter, quality-preserving edits.
   Uses ONNX Runtime when available for modern models; falls back to
   lightweight classical CV (OpenCV) or pure heuristics. */

/* Detected object / region */
typedef struct {
    char label[64];      /* e.g. "person", "face", "sky", "foliage", "water", "building" */
    float x0, y0, x1, y1;/* normalized coords 0..1 */
    float confidence;
    float importance;    /* 0..1 - how much this region should be preserved/enhanced */
} FeRegion;

typedef struct {
    FeRegion *regions;
    int count;
    int capacity;
    /* Global scene hints */
    float sky_ratio;
    float skin_tone_likelihood;
    float has_faces;
    float dynamic_range_estimate;
} FeSceneAnalysis;

/* Initialize/shutdown vision subsystem */
int  fe_vision_init(void);
void fe_vision_shutdown(void);
int  fe_vision_available(void);

/* Analyze a linear RGB image and return detected regions.
   Caller must call fe_scene_analysis_free. */
FeSceneAnalysis *fe_scene_analyze(const FeImage *linear_img);
void fe_scene_analysis_free(FeSceneAnalysis *sa);

/* Apply quality-preserving, region-aware adjustments to the settings.
   "Auto-Enhance" uses object recognition to avoid blowing out skin/sky
   while pulling detail in important areas.  Never sacrifices perceptual
   quality. */
void fe_auto_enhance_from_scene(FeSettings *s, const FeSceneAnalysis *sa,
                                const FeImage *linear_img);

/* Face/skin detection only (fast path) */
int fe_detect_skin_regions(const FeImage *linear_img, FeRegion **out, int *count);

/* Sky detection (fast path) */
int fe_detect_sky_regions(const FeImage *linear_img, FeRegion **out, int *count);

#endif
