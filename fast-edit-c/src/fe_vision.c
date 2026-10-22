#include "fe_vision.h"
#include "fe_math.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef FE_HAVE_ONNX
#include <onnxruntime_c_api.h>
#endif
#ifdef FE_HAVE_OPENCV
#include <opencv2/opencv.hpp>
#endif

static int g_vision_ready = 0;

int fe_vision_init(void) {
    /* Try to initialize backends if available */
#ifdef FE_HAVE_ONNX
    /* ONNX Runtime available - could load models from known locations */
    g_vision_ready = 1;
    printf("Vision: ONNX Runtime available\n");
    return 1;
#endif
#ifdef FE_HAVE_OPENCV
    g_vision_ready = 1;
    printf("Vision: OpenCV available\n");
    return 1;
#else
    g_vision_ready = 1; /* heuristics always available */
    printf("Vision: classical heuristics mode\n");
    return 1;
#endif
}

void fe_vision_shutdown(void) {
    g_vision_ready = 0;
}

int fe_vision_available(void) {
    return g_vision_ready;
}

/* Simple sky detection in HSV/YCbCr-like space */
int fe_detect_sky_regions(const FeImage *img, FeRegion **out, int *count) {
    if (!img || !out || !count) return 0;
    *out = NULL; *count = 0;
    
    int w = img->w, h = img->h;
    int n = w * h;
    
    /* Work at lower resolution for speed */
    int sw = w > 640 ? 640 : w;
    int sh = h > 640 ? (int)(h * 640.0 / w) : h;
    if (sw < 32) sw = w; if (sh < 32) sh = h;
    
    /* Sample pixels - check top portion for sky */
    float sky_score_total = 0;
    int sky_px = 0;
    /* Sample top 40% */
    for (int y = 0; y < sh * 0.4 && y < sh; y++) {
        for (int x = 0; x < sw; x++) {
            /* approximate sampling */
            int sx = x * w / sw;
            int sy = y * h / sh;
            int idx = (sy * w + sx) * 3;
            float r = img->data[idx], g = img->data[idx+1], b = img->data[idx+2];
            /* Sky is typically blueish, low saturation variance, high luminance */
            float lum = 0.2126f*r + 0.7152f*g + 0.0722f*b;
            float maxc = r > g ? r : g; maxc = maxc > b ? maxc : b;
            float minc = r < g ? r : g; minc = minc < b ? minc : b;
            float sat = maxc - minc;
            /* Blue dominates, moderately bright */
            if (b > r * 1.0f && b > g * 0.8f && lum > 0.2f && lum < 0.95f) {
                sky_score_total += 1.0f;
                sky_px++;
            }
        }
    }
    
    float sky_ratio = (sh * 0.4 * sw > 0) ? (sky_px / (float)(sh * 0.4 * sw)) : 0;
    
    if (sky_ratio > 0.15f) {
        FeRegion *r = (FeRegion *)calloc(1, sizeof(FeRegion));
        strncpy(r->label, "sky", 63);
        r->x0 = 0; r->y0 = 0; r->x1 = 1.0f; r->y1 = sky_ratio * 1.5f;
        if (r->y1 > 0.6f) r->y1 = 0.6f;
        r->confidence = sky_ratio;
        r->importance = 0.6f; /* preserve sky, don't over-process */
        *out = r;
        *count = 1;
        return 1;
    }
    return 0;
}

/* Skin tone detection using chroma range */
int fe_detect_skin_regions(const FeImage *img, FeRegion **out, int *count) {
    if (!img || !out || !count) return 0;
    *out = NULL; *count = 0;
    
    int w = img->w, h = img->h;
    int skin_px = 0;
    int total = w * h;
    
    /* Sample every Nth pixel for speed on large images */
    int step = 1;
    if (total > 10 * 1024 * 1024) step = 4;
    else if (total > 5 * 1024 * 1024) step = 2;
    
    for (int i = 0; i < (total / step); i++) {
        int idx = i * step * 3;
        if (idx + 2 >= total * 3) break;
        float r = img->data[idx], g = img->data[idx+1], b = img->data[idx+2];
        /* Skin tone ranges in RGB (works well for most skin tones) */
        if (r > 0.2f && g > 0.1f && b > 0.05f &&
            r > g && g > b &&
            r - g < 0.4f && r - b < 0.6f &&
            (r+g+b) > 0.3f) {
            /* More specific: r/g ratio in skin range */
            if (g > 0.0001f && r/g > 1.0f && r/g < 1.8f) {
                skin_px++;
            }
        }
    }
    
    float skin_ratio = skin_px / (float)(total / step);
    if (skin_ratio > 0.01f) {
        FeRegion *r = (FeRegion *)calloc(1, sizeof(FeRegion));
        strncpy(r->label, "skin", 63);
        r->x0 = 0.1f; r->y0 = 0.1f; r->x1 = 0.9f; r->y1 = 0.9f;
        r->confidence = skin_ratio * 10.0f;
        if (r->confidence > 1.0f) r->confidence = 1.0f;
        r->importance = 0.95f; /* very important to preserve skin tones */
        *out = r;
        *count = 1;
        return 1;
    }
    return 0;
}

FeSceneAnalysis *fe_scene_analyze(const FeImage *linear_img) {
    FeSceneAnalysis *sa = (FeSceneAnalysis *)calloc(1, sizeof(FeSceneAnalysis));
    sa->capacity = 16;
    sa->regions = (FeRegion *)calloc(sa->capacity, sizeof(FeRegion));
    sa->count = 0;
    
    if (!linear_img) return sa;
    
    /* Detect sky */
    FeRegion *sky = NULL; int sky_count = 0;
    if (fe_detect_sky_regions(linear_img, &sky, &sky_count)) {
        if (sa->count < sa->capacity) {
            sa->regions[sa->count++] = sky[0];
            free(sky);
        }
        /* estimate sky ratio */
        sa->sky_ratio = sky[0].y1 - sky[0].y0;
    }
    
    /* Detect skin */
    FeRegion *skin = NULL; int skin_count = 0;
    if (fe_detect_skin_regions(linear_img, &skin, &skin_count)) {
        if (sa->count < sa->capacity) {
            sa->regions[sa->count++] = skin[0];
            free(skin);
        }
        sa->skin_tone_likelihood = skin[0].confidence;
        sa->has_faces = skin_count > 0 ? 1.0f : 0.0f;
    }
    
    /* Estimate dynamic range */
    int n = linear_img->w * linear_img->h;
    float minl = 1.0f, maxl = 0.0f;
    int step = n > 100000 ? n / 10000 : 1;
    for (int i = 0; i < n; i += step) {
        int idx = i * 3;
        float l = 0.2126f * linear_img->data[idx] + 0.7152f * linear_img->data[idx+1] + 0.7152f * linear_img->data[idx+2];
        if (l < minl) minl = l;
        if (l > maxl) maxl = l;
    }
    sa->dynamic_range_estimate = maxl - minl;
    
    return sa;
}

void fe_scene_analysis_free(FeSceneAnalysis *sa) {
    if (!sa) return;
    free(sa->regions);
    free(sa);
}

/* Quality-preserving auto-enhance using scene analysis */
void fe_auto_enhance_from_scene(FeSettings *s, const FeSceneAnalysis *sa,
                                const FeImage *linear_img) {
    if (!s || !sa) return;
    
    /* Start from current settings, adjust conservatively to preserve quality */
    
    /* If we have skin tones, be very gentle on them */
    if (sa->has_faces > 0.5f || sa->skin_tone_likelihood > 0.3f) {
        /* Reduce clarity/texture slightly to avoid over-sharpening skin */
        if (s->clarity > 10.0) s->clarity = 10.0;
        if (s->texture > 15.0) s->texture = 15.0;
        /* Keep saturation reasonable */
        if (s->saturation > 1.15) s->saturation = 1.15;
    }
    
    /* If we have lots of sky, protect it */
    if (sa->sky_ratio > 0.2f) {
        /* Don't blow out highlights too aggressively */
        if (s->highlights < -20.0) s->highlights = -20.0;
        if (s->whites > 30.0) s->whites = 30.0;
    }
    
    /* Adjust based on dynamic range */
    if (sa->dynamic_range_estimate < 0.4f) {
        /* Low contrast scene - gentle lift */
        if (s->contrast < 10.0) s->contrast = 10.0;
        if (s->shadows < 10.0) s->shadows = 10.0;
    } else if (sa->dynamic_range_estimate > 0.7f) {
        /* High dynamic range - use gentle curves */
        if (s->highlights > -15.0) s->highlights = -15.0;
        if (s->shadows > 5.0) s->shadows = 5.0;
    }
    
    /* Never go overboard - quality over everything */
}
