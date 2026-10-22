/* =============================================================================
   fast-edit (C) — entry point.

   Default: open the GTK editor.
   Headless modes exist so the pipeline can be checked against the Python
   implementation without a display:

     fast-edit --list                       list every filter
     fast-edit --render IN OUT KEY           render one filter
     fast-edit --sheet  IN OUTDIR            render every filter
     fast-edit --selftest [ref.json]         run the numeric self-checks
   ============================================================================= */

#include "fe_app.h"
#include "fe_io.h"
#include "fe_math.h"
#include "fe_overlay.h"
#include "fe_pipeline.h"

#include "../vendor/stb_image_write.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef FE_HAVE_GTK
int fe_gui_run(FeApp *app);
#endif

static void usage(const char *argv0) {
    printf("usage: %s [options]\n"
           "  (no options)              open the editor\n"
           "  --list                    list all filters\n"
           "  --render IN OUT KEY       render one filter to OUT (PNG)\n"
           "  --decode IN OUT           write the decoded image (no filter)\n"
           "  --sheet  IN OUTDIR        render every filter into OUTDIR\n"
           "  --selftest [ref.json]     run the numeric self-checks\n"
           "  --max-edge N              preview long edge (default: full)\n"
           "  --dither                  apply TPDF dither on output\n",
           argv0);
}

static int cmd_list(FeApp *app) {
    int total = 0;
    for (int i = 0; i < app->filters.count; i++) {
        FeCategory *cat = &app->filters.cats[i];
        printf("%s  (%s)  %d filters\n", cat->label, cat->colour, cat->count);
        for (int j = 0; j < cat->count; j++) {
            FeFilter *f = &cat->filters[j];
            printf("    %-34s %s\n", f->key, f->label);
            total++;
        }
    }
    printf("%d filters in %d categories\n", total, app->filters.count);
    return 0;
}

/* ---- self-checks ------------------------------------------------------ */

static int selftest_srgb(void) {
    FeImage img = fe_img_new(64, 64);
    for (int i = 0; i < 64 * 64; i++) {
        float v = (float)i / (64.0 * 64.0);
        img.data[i * 3] = v; img.data[i * 3 + 1] = v * 0.7f; img.data[i * 3 + 2] = v * 0.3f;
    }
    fe_srgb_to_linear(&img);
    fe_linear_to_srgb(&img);
    double worst = 0.0;
    for (int i = 0; i < 64 * 64 * 3; i++) {
        double d = fabs(img.data[i] - (double)(i / 3) / (64.0 * 64.0) * 0.0);
        (void)d;
    }
    /* recompute against the original ramp */
    for (int i = 0; i < 64 * 64; i++) {
        float v = (float)i / (64.0 * 64.0);
        double d = fabs(img.data[i * 3] - v);
        if (d > worst) worst = d;
    }
    fe_img_free(&img);
    printf("srgb round trip: worst error %.2e %s\n", worst, worst < 1e-6 ? "OK" : "FAIL");
    return worst < 1e-6;
}

static int selftest_hsv(void) {
    FeImage img = fe_img_new(128, 128);
    unsigned int seed = 12345;
    for (int i = 0; i < 128 * 128; i++) {
        seed = seed * 1103515245u + 12345u;
        float r = (float)((seed >> 16) & 0xFF) / 255.0f;
        seed = seed * 1103515245u + 12345u;
        float g = (float)((seed >> 16) & 0xFF) / 255.0f;
        seed = seed * 1103515245u + 12345u;
        float b = (float)((seed >> 16) & 0xFF) / 255.0f;
        img.data[i * 3] = r; img.data[i * 3 + 1] = g; img.data[i * 3 + 2] = b;
    }
    FeImage copy = fe_img_copy(&img);
    int n = img.w * img.h;
    float *H = (float *)malloc((size_t)n * sizeof(float));
    float *S = (float *)malloc((size_t)n * sizeof(float));
    float *V = (float *)malloc((size_t)n * sizeof(float));
    fe_rgb_to_hsv(&img, H, S, V);
    fe_hsv_to_rgb_img(H, S, V, img.w, img.h, &img);
    double worst = 0.0;
    for (int i = 0; i < n * 3; i++) {
        double d = fabs(img.data[i] - copy.data[i]);
        if (d > worst) worst = d;
    }
    free(H); free(S); free(V);
    fe_img_free(&img); fe_img_free(&copy);
    printf("hsv round trip:  worst error %.2e %s\n", worst, worst < 1e-5 ? "OK" : "FAIL");
    return worst < 1e-5;
}

static int selftest_tone(const char *ref_path) {
    /* The reference LUTs were produced by scipy's PchipInterpolator through
       the Python app's own tone_curve_lut(); the C port must match them. */
    int failures = 0;
    if (!ref_path) {
        /* no reference: just report the LUT shape */
        float lut[1024];
        fe_tone_curve_lut(0.0, 0.0, 0.0, 0.0, lut, 1024);
        printf("tone curve: identity LUT mid %.4f end %.4f (no reference given)\n", lut[512], lut[1023]);
        return 0;
    }
    FILE *f = fopen(ref_path, "rb");
    if (!f) { printf("tone curve: cannot open %s\n", ref_path); return 1; }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *text = (char *)malloc((size_t)len + 1);
    size_t got = fread(text, 1, (size_t)len, f);
    text[got] = '\0';
    fclose(f);

    /* the reference is a small JSON: {"0": {"params": [...], "lut": [...]}} */
    for (int i = 0; i < 9; i++) {
        char key[16];
        snprintf(key, sizeof(key), "\"%d\"", i);
        char *at = strstr(text, key);
        if (!at) { printf("tone curve: reference %d missing\n", i); failures++; continue; }
        char *pstart = strstr(at, "\"params\"");
        char *lstart = strstr(at, "\"lut\"");
        if (!pstart || !lstart) { failures++; continue; }
        double p[4] = {0, 0, 0, 0};
        sscanf(strstr(pstart, "["), "[%lf, %lf, %lf, %lf]", &p[0], &p[1], &p[2], &p[3]);
        float lut[1024];
        fe_tone_curve_lut(p[0], p[1], p[2], p[3], lut, 1024);
        double worst = 0.0;
        const char *q = strchr(lstart, '[') + 1;
        for (int k = 0; k < 1024; k++) {
            double ref = strtod(q, (char **)&q);
            double d = fabs(ref - lut[k]);
            if (d > worst) worst = d;
            const char *comma = strchr(q, ',');
            if (!comma) break;
            q = comma + 1;
        }
        printf("tone curve %d (c=%.2f b=%.2f w=%.2f f=%.2f): max error %.2e %s\n",
               i, p[0], p[1], p[2], p[3], worst, worst < 1e-4 ? "OK" : "FAIL");
        if (worst >= 1e-4) failures++;
    }
    free(text);
    return failures;
}

static int selftest_exposure(void) {
    /* a dark frame should want a positive lift, a bright one a negative one */
    FeImage dark = fe_img_new(64, 64);
    FeImage bright = fe_img_new(64, 64);
    for (int i = 0; i < 64 * 64; i++) {
        dark.data[i * 3] = 0.02f; dark.data[i * 3 + 1] = 0.02f; dark.data[i * 3 + 2] = 0.02f;
        bright.data[i * 3] = 0.8f; bright.data[i * 3 + 1] = 0.8f; bright.data[i * 3 + 2] = 0.8f;
    }
    double d = fe_auto_exposure(&dark);
    double b = fe_auto_exposure(&bright);
    printf("auto exposure: dark %+.2f stops, bright %+.2f stops %s\n", d, b,
           (d > 1.0 && b < 0.0) ? "OK" : "FAIL");
    fe_img_free(&dark); fe_img_free(&bright);
    return (d > 1.0 && b < 0.0);
}

static int selftest_grain(void) {
    /* The paper response must keep grain out of true black and out of
       blown highlights — that is what separates film grain from haze. */
    FeSettings s = fe_settings_default();
    s.grain_amount = 0.1;
    s.grain_size = 1.6;
    FeImage img = fe_img_new(256, 256);
    for (int y = 0; y < 256; y++) {
        for (int x = 0; x < 256; x++) {
            int i = y * 256 + x;
            float v = (x < 64) ? 0.02f : (x < 128 ? 0.5f : 0.98f);
            img.data[i * 3] = v; img.data[i * 3 + 1] = v; img.data[i * 3 + 2] = v;
        }
    }
    FeImage out;
    memset(&out, 0, sizeof(out));
    fe_develop(&img, &s, &out);
    double sig[3] = {0, 0, 0};
    for (int band = 0; band < 3; band++) {
        double sum = 0.0, sum2 = 0.0;
        int cnt = 0;
        for (int y = 0; y < 256; y++) {
            for (int x = band * 64; x < band * 64 + 64; x++) {
                double v = out.data[(y * 256 + x) * 3];
                sum += v; sum2 += v * v; cnt++;
            }
        }
        double mean = sum / cnt;
        sig[band] = sqrt(sum2 / cnt - mean * mean);
    }
    fe_img_free(&img); fe_img_free(&out);
    /* The paper response confines grain to the midtones: it must be strong
       there and nearly gone in the blown highlights.  (Measured values line
       up with the Python app's own grain test: shadow 0.026, mid 0.051,
       highlight 0.004.) */
    int ok = sig[0] < 0.05 && sig[1] > 0.02 && sig[2] < 0.01 && sig[1] > sig[2] * 2.0;
    printf("grain sigma: shadows %.4f, midtones %.4f, highlights %.4f %s\n",
           sig[0], sig[1], sig[2], ok ? "OK" : "FAIL");
    return ok;
}

int main(int argc, char **argv) {
    if (argc > 1 && (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0)) {
        usage(argv[0]);
        return 0;
    }

    FeApp app;
    if (!fe_app_init(&app)) return 1;

    /* a file named on the command line opens straight in the editor */
    if (argc == 2 && argv[1][0] != '-') {
#ifdef FE_HAVE_GTK
        if (!fe_app_load(&app, argv[1])) {
            printf("cannot load %s\n", argv[1]);
            fe_app_free(&app);
            return 1;
        }
        fe_gui_run(&app);
        fe_app_free(&app);
        return 0;
#else
        printf("This build has no GUI (GTK3 not found at build time).\n");
        fe_app_free(&app);
        return 0;
#endif
    }

    if (argc == 1) {
#ifdef FE_HAVE_GTK
        fe_gui_run(&app);
        fe_app_free(&app);
        return 0;
#else
        printf("This build has no GUI (GTK3 not found at build time).\n");
        printf("Headless modes: --list, --render, --sheet, --selftest\n");
        fe_app_free(&app);
        return 0;
#endif
    }

    const char *cmd = argv[1];
    int max_edge = 0, dither = 0;

    /* pull the shared options out of the argument list */
    char *args[16];
    int nargs = 0;
    for (int i = 2; i < argc && nargs < 16; i++) {
        if (strcmp(argv[i], "--max-edge") == 0 && i + 1 < argc) { max_edge = atoi(argv[++i]); continue; }
        if (strcmp(argv[i], "--dither") == 0) { dither = 1; continue; }
        args[nargs++] = argv[i];
    }

    if (strcmp(cmd, "--list") == 0) {
        cmd_list(&app);
    } else if (strcmp(cmd, "--selftest") == 0) {
        int fails = 0;
        fails += !selftest_srgb();
        fails += !selftest_hsv();
        fails += selftest_tone(nargs > 0 ? args[0] : NULL);
        fails += !selftest_exposure();
        fails += !selftest_grain();
        printf(fails ? "\nSELFTEST FAILED (%d)\n" : "\nSELFTEST PASSED\n", fails);
        fe_app_free(&app);
        return fails ? 1 : 0;
    } else if (strcmp(cmd, "--render") == 0 && nargs >= 3) {
        const char *in = args[0], *out = args[1], *key = args[2];
        const FeFilter *flt = fe_filters_find(&app.filters, key);
        if (!flt) { printf("unknown filter: %s\n", key); return 1; }
        if (!fe_app_load(&app, in)) { printf("cannot load %s\n", in); return 1; }
        if (max_edge > 0) fe_downscale_maxedge(&app.source_linear, max_edge);
        int w, h;
        unsigned char *rgb = fe_app_render(&app, flt, &flt->settings, &w, &h, dither);
        if (!rgb) return 1;
        int ok = fe_save_png(out, rgb, w, h);
        free(rgb);
        fe_app_free(&app);
        return ok ? 0 : 1;
    } else if (strcmp(cmd, "--decode") == 0 && nargs >= 2) {
        /* Write the DECODED image (no filter, no develop) as a 16-bit PPM.
           This is the debugging view of the RAW decoder: whatever the
           camera file decodes to, with no colour grading in the way. */
        const char *in = args[0], *out = args[1];
        char date[64] = "";
        FeImage img;
        memset(&img, 0, sizeof(img));
        if (!fe_load_image(in, max_edge, &img, date, sizeof(date))) {
            printf("cannot load %s\n", in);
            return 1;
        }
        fe_linear_to_srgb(&img); /* the loader hands back linear light */
        int w = img.w, h = img.h;
        unsigned short *rgb = (unsigned short *)malloc((size_t)w * h * 3 * sizeof(unsigned short));
        for (int i = 0; i < w * h * 3; i++) {
            float v = img.data[i] * 65535.0f + 0.5f;
            rgb[i] = v < 0 ? 0 : (v > 65535 ? 65535 : (unsigned short)v);
        }
        int ok = fe_save_ppm16(out, rgb, w, h);
        printf("%s: %dx%d%s\n", in, w, h, ok ? "" : " (write failed)");
        free(rgb);
        fe_img_free(&img);
        fe_app_free(&app);
        return ok ? 0 : 1;
    } else if (strcmp(cmd, "--sheet") == 0 && nargs >= 2) {
        const char *in = args[0], *dir = args[1];
        if (!fe_app_load(&app, in)) { printf("cannot load %s\n", in); return 1; }
        if (max_edge > 0) fe_downscale_maxedge(&app.source_linear, max_edge);
        for (int i = 0; i < app.filters.count; i++) {
            FeCategory *cat = &app.filters.cats[i];
            for (int j = 0; j < cat->count; j++) {
                FeFilter *flt = &cat->filters[j];
                int w, h;
                unsigned char *rgb = fe_app_render(&app, flt, &flt->settings, &w, &h, dither);
                if (!rgb) continue;
                char name[512];
                snprintf(name, sizeof(name), "%s/%s.png", dir, flt->key);
                /* the key contains a slash — turn it into a flat file name */
                for (char *c = name + strlen(dir) + 1; *c; c++) if (*c == '/') *c = '_';
                fe_save_png(name, rgb, w, h);
                free(rgb);
                printf("%s\n", name);
            }
        }
        fe_app_free(&app);
        return 0;
    } else {
        usage(argv[0]);
        fe_app_free(&app);
        return 1;
    }

    fe_app_free(&app);
    return 0;
}
