#include "fe_io.h"
#include "fe_math.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <time.h>

#include "../vendor/stb_image.h"
#include "../vendor/stb_image_write.h"

#ifdef FE_HAVE_LIBRAW
#include <libraw/libraw.h>
/* LibRaw 0.20 renamed the processed-image struct; keep the old name working. */
#if LIBRAW_MAJOR_VERSION == 0 && LIBRAW_MINOR_VERSION < 20
typedef libraw_image_t libraw_processed_image_t;
#endif
#endif

/* ------------------------------------------------------------------ */
/* timestamps                                                           */
/* ------------------------------------------------------------------ */

static int fe_is_digit(char c) { return c >= '0' && c <= '9'; }

/* Format "YYYY:MM:DD HH:MM:SS" the way the Python app does:
   "%b. %d %Y" upper-cased, two spaces, then "PM 08:36". */
static void fe_format_vcr_date(const char *exif_stamp, char *out, size_t out_len) {
    if (exif_stamp && strlen(exif_stamp) >= 19 &&
        fe_is_digit(exif_stamp[0]) && fe_is_digit(exif_stamp[1]) &&
        fe_is_digit(exif_stamp[2]) && fe_is_digit(exif_stamp[3]) &&
        exif_stamp[4] == ':' && exif_stamp[7] == ':' &&
        exif_stamp[10] == ' ' && exif_stamp[13] == ':' && exif_stamp[16] == ':') {
        struct tm tm;
        memset(&tm, 0, sizeof(tm));
        int year = (exif_stamp[0] - '0') * 1000 + (exif_stamp[1] - '0') * 100 +
                   (exif_stamp[2] - '0') * 10 + (exif_stamp[3] - '0');
        int mon  = (exif_stamp[5] - '0') * 10 + (exif_stamp[6] - '0');
        int day  = (exif_stamp[8] - '0') * 10 + (exif_stamp[9] - '0');
        int hour = (exif_stamp[11] - '0') * 10 + (exif_stamp[12] - '0');
        int min  = (exif_stamp[14] - '0') * 10 + (exif_stamp[15] - '0');
        tm.tm_year = year - 1900; tm.tm_mon = mon - 1; tm.tm_mday = day;
        tm.tm_hour = hour; tm.tm_min = min; tm.tm_sec = 0;
        tm.tm_isdst = -1;
        mktime(&tm);
        static const char *months[12] = {"JAN","FEB","MAR","APR","MAY","JUN",
                                         "JUL","AUG","SEP","OCT","NOV","DEC"};
        const char *ampm = hour >= 12 ? "PM" : "AM";
        int h12 = hour % 12; if (h12 == 0) h12 = 12;
        snprintf(out, out_len, "%s. %02d %04d  %s %02d:%02d",
                 months[tm.tm_mon % 12], day, year, ampm, h12, min);
        return;
    }
    snprintf(out, out_len, "OCT. 20 2026  PM 11:35");
}

/* Scan the head of the file for the first EXIF timestamp we can find.
   RAW files carry the shooting time in EXIF too, so one code path serves
   both; when the file is a camera RAW we prefer LibRaw's own timestamp. */
void fe_read_exif_date(const char *path, char *out, size_t out_len) {
    snprintf(out, out_len, "OCT. 20 2026  PM 11:35");
    FILE *f = fopen(path, "rb");
    if (!f) return;
    size_t cap = 1 << 18; /* 256 KB is plenty for the EXIF header */
    char *buf = (char *)malloc(cap);
    size_t got = fread(buf, 1, cap, f);
    fclose(f);
    for (size_t i = 0; i + 19 < got; i++) {
        if (!fe_is_digit(buf[i]) || !fe_is_digit(buf[i + 1]) ||
            !fe_is_digit(buf[i + 2]) || !fe_is_digit(buf[i + 3])) continue;
        if (buf[i + 4] != ':' || buf[i + 7] != ':' || buf[i + 10] != ' ' ||
            buf[i + 13] != ':' || buf[i + 16] != ':') continue;
        char stamp[20];
        memcpy(stamp, buf + i, 19);
        stamp[19] = '\0';
        fe_format_vcr_date(stamp, out, out_len);
        break;
    }
    free(buf);
}

/* ------------------------------------------------------------------ */
/* RAW decoding                                                         */
/* ------------------------------------------------------------------ */

#ifdef FE_HAVE_LIBRAW
static int fe_decode_raw_libraw(const char *path, int half, FeImage *out) {
    libraw_data_t *raw = libraw_init(0);
    if (!raw) return 0;
    int ret = libraw_open_file(raw, path);
    if (ret != LIBRAW_SUCCESS) { libraw_close(raw); return 0; }

    /* Same knobs rawpy.postprocess() uses in the Python app, so the two
       editors decode a RAW to the same pixels.  LibRaw's own defaults
       (AHD demosaic, highlight clipping) are what rawpy leaves alone
       too, so they are not touched here. */
    raw->params.use_camera_wb = 1;
    raw->params.half_size = half;
    raw->params.output_bps = 16;
    raw->params.output_color = 1;   /* sRGB */
    raw->params.no_auto_bright = 0;
    raw->params.no_auto_scale = 0;

    /* LibRaw 0.20 and newer split the work: the raw data is unpacked
       first, then processed.  Skipping the unpack makes the process call
       fail with "out of order call" and hand back a black frame. */
    ret = libraw_unpack(raw);
    if (ret != LIBRAW_SUCCESS) { libraw_close(raw); return 0; }

    ret = libraw_dcraw_process(raw);
    if (ret != LIBRAW_SUCCESS) { libraw_close(raw); return 0; }

    int errc = 0;
    libraw_processed_image_t *img = libraw_dcraw_make_mem_image(raw, &errc);
    if (!img) { libraw_close(raw); return 0; }

    int w = img->width, h = img->height, colors = img->colors;
    if (w <= 0 || h <= 0 || colors < 3) {
        libraw_dcraw_clear_mem(img);
        libraw_close(raw);
        return 0;
    }

    /* The samples are 16-bit, host order, packed without a stride. */
    const unsigned short *px = (const unsigned short *)img->data;
    FeImage out_img = fe_img_new(w, h);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            const unsigned short *s = px + ((size_t)y * w + x) * colors;
            float *d = out_img.data + ((size_t)y * w + x) * 3;
            d[0] = s[0] / 65535.0f;
            d[1] = s[1] / 65535.0f;
            d[2] = s[2] / 65535.0f;
        }
    }
    libraw_dcraw_clear_mem(img);
    libraw_close(raw);
    *out = out_img;
    return 1;
}
#endif /* FE_HAVE_LIBRAW */

/* Read a binary PPM (netpbm "P6") from an open stream into an image.
   Two details matter a lot and are easy to get wrong:
     * 16-bit PPM samples are BIG ENDIAN (netpbm spec).  Reading them as
       native shorts byte-swaps every sample, which turns a smooth photo
       into a saw-tooth — the picture keeps its shapes but the colour
       explodes into rainbow static.
     * the sample depth follows the header's maxval: 8-bit data is one
       byte per channel, 16-bit data is two.
   `linear_input` says the samples are linear light (dcraw's -4 flag);
   the samples are then encoded to sRGB so that the common loader step
   (sRGB -> linear) puts the pixels back where they started. */
int fe_read_ppm(FILE *f, FeImage *out, int linear_input) {
    int w = 0, h = 0, maxval = 0;
    char magic[3] = {0, 0, 0};
    if (fscanf(f, "%2s", magic) != 1 || strcmp(magic, "P6") != 0) return 0;
    if (fscanf(f, "%d %d %d", &w, &h, &maxval) != 3) return 0;
    fgetc(f); /* the single whitespace that follows maxval */
    if (w <= 0 || h <= 0 || maxval <= 0) return 0;

    size_t count = (size_t)w * h * 3;
    FeImage img = fe_img_new(w, h);
    int ok = 1;

    if (maxval > 255) {
        /* 16-bit, big endian */
        unsigned char *raw = (unsigned char *)malloc(count * 2);
        if (!raw) { fe_img_free(&img); return 0; }
        size_t got = fread(raw, 1, count * 2, f);
        if (got < count * 2) ok = 0;
        for (size_t i = 0; ok && i < count; i++) {
            unsigned v = ((unsigned)raw[i * 2] << 8) | (unsigned)raw[i * 2 + 1];
            img.data[i] = (float)((double)v / (double)maxval);
        }
        free(raw);
    } else {
        /* 8-bit */
        unsigned char *raw = (unsigned char *)malloc(count);
        if (!raw) { fe_img_free(&img); return 0; }
        size_t got = fread(raw, 1, count, f);
        if (got < count) ok = 0;
        for (size_t i = 0; ok && i < count; i++)
            img.data[i] = (float)((double)raw[i] / (double)maxval);
        free(raw);
    }
    if (!ok) { fe_img_free(&img); return 0; }
    if (linear_input) fe_linear_to_srgb(&img);
    *out = img;
    return 1;
}

/* Fallback: dcraw is present on most systems; ask it for 16-bit PPM on
   stdout and parse that.  (Only used when LibRaw's headers are absent.)
   -4 gives 16-bit LINEAR light, -w uses the camera's white balance and
   -o 1 writes sRGB primaries, matching what rawpy does in the Python
   editor. */
static int fe_decode_raw_dcraw(const char *path, FeImage *out) {
    char cmd[4096];
    snprintf(cmd, sizeof(cmd), "dcraw -c -4 -w -o 1 -q 2 '%s' 2>/dev/null", path);
    FILE *p = popen(cmd, "r");
    if (!p) return 0;
    int ok = fe_read_ppm(p, out, 1 /* dcraw's -4 output is linear light */);
    pclose(p);
    return ok;
}

/* ------------------------------------------------------------------ */
/* public API                                                          */
/* ------------------------------------------------------------------ */

int fe_load_image(const char *path, int max_edge, FeImage *out,
                  char *date_out, size_t date_len) {
    if (date_out && date_len) fe_read_exif_date(path, date_out, date_len);

    int is_raw = 0;
    const char *ext = strrchr(path, '.');
    if (ext && (strcasecmp(ext, ".arw") == 0 || strcasecmp(ext, ".raw") == 0 ||
                strcasecmp(ext, ".cr2") == 0 || strcasecmp(ext, ".nef") == 0 ||
                strcasecmp(ext, ".dng") == 0 || strcasecmp(ext, ".raf") == 0))
        is_raw = 1;

    FeImage img;
    memset(&img, 0, sizeof(img));

    if (is_raw) {
#ifdef FE_HAVE_LIBRAW
        /* LibRaw first; dcraw is the safety net for the odd file LibRaw
           refuses (and for builds without LibRaw at all). */
        if (!fe_decode_raw_libraw(path, max_edge > 0 ? 1 : 0, &img) &&
            !fe_decode_raw_dcraw(path, &img))
            return 0;
#else
        if (!fe_decode_raw_dcraw(path, &img)) return 0;
#endif
    } else {
        int w = 0, h = 0, comp = 0;
        unsigned char *data = stbi_load(path, &w, &h, &comp, 3);
        if (!data) {
            fprintf(stderr, "cannot read %s: %s\n", path, stbi_failure_reason());
            return 0;
        }
        img = fe_img_new(w, h);
        for (int i = 0; i < w * h * 3; i++) img.data[i] = data[i] / 255.0f;
        stbi_image_free(data);
    }

    /* The decoder hands back gamma-encoded pixels; the pipeline works in
       linear light, so undo the sRGB curve once, here. */
    fe_srgb_to_linear(&img);
    if (max_edge > 0) fe_downscale_maxedge(&img, max_edge);
    *out = img;
    return 1;
}

/* ------------------------------------------------------------------ */
/* fast thumbnails for the browser                                     */
/* ------------------------------------------------------------------ */

/* A RAW file's full development pass (demosaic, colour, ...) takes
   hundreds of milliseconds — far too slow to fill a film-strip of
   previews.  Every camera also stores a ready-made JPEG inside the file,
   so the browser reads that instead: a few milliseconds per photo and a
   perfectly recognisable picture.

   Returns 1 on success; *rgb is malloc'd 8-bit RGB and belongs to the
   caller.  `max_edge` > 0 downscales the long edge of the result. */
int fe_load_thumb_rgb(const char *path, int max_edge,
                      unsigned char **rgb, int *w, int *h) {
    if (rgb) *rgb = NULL;
    if (w) *w = 0;
    if (h) *h = 0;
    if (!path || !path[0]) return 0;

    int is_raw = 0;
    const char *ext = strrchr(path, '.');
    if (ext && (strcasecmp(ext, ".arw") == 0 || strcasecmp(ext, ".raw") == 0 ||
                strcasecmp(ext, ".cr2") == 0 || strcasecmp(ext, ".nef") == 0 ||
                strcasecmp(ext, ".dng") == 0 || strcasecmp(ext, ".raf") == 0))
        is_raw = 1;

    unsigned char *decoded = NULL;
    int dw = 0, dh = 0;

#ifdef FE_HAVE_LIBRAW
    if (is_raw) {
        /* Ask LibRaw for the thumbnail only — never for a demosaic pass. */
        libraw_data_t *raw = libraw_init(0);
        libraw_processed_image_t *thumb = NULL;
        if (raw && libraw_open_file(raw, path) == LIBRAW_SUCCESS &&
            libraw_unpack_thumb(raw) == LIBRAW_SUCCESS) {
            int errc = 0;
            thumb = libraw_dcraw_make_mem_thumb(raw, &errc);
        }
        if (raw) libraw_close(raw);

        if (!thumb) return 0;
        if (thumb->type != LIBRAW_IMAGE_JPEG || thumb->data_size == 0) {
            libraw_dcraw_clear_mem(thumb);
            return 0;
        }
        /* stb copies the pixels out of LibRaw's buffer */
        decoded = stbi_load_from_memory(thumb->data, (int)thumb->data_size,
                                        &dw, &dh, NULL, 3);
        libraw_dcraw_clear_mem(thumb);
    } else {
        decoded = (unsigned char *)stbi_load(path, &dw, &dh, NULL, 3);
    }
#else
    /* Without LibRaw a RAW thumbnail would need a full decode, which is
       exactly what this function exists to avoid — so refuse instead. */
    if (is_raw) return 0;
    decoded = (unsigned char *)stbi_load(path, &dw, &dh, NULL, 3);
#endif

    if (!decoded) return 0;
    if (dw <= 0 || dh <= 0) { stbi_image_free(decoded); return 0; }

    /* downscale in place when the caller asked for something small */
    if (max_edge > 0 && (dw > max_edge || dh > max_edge)) {
        double s = (double)max_edge / (dw > dh ? dw : dh);
        int nw = (int)(dw * s + 0.5);
        int nh = (int)(dh * s + 0.5);
        if (nw < 1) nw = 1;
        if (nh < 1) nh = 1;
        unsigned char *small = (unsigned char *)malloc((size_t)nw * nh * 3);
        if (small) {
            for (int y = 0; y < nh; y++) {
                int sy = (int)((double)y * dh / nh);
                if (sy >= dh) sy = dh - 1;
                for (int x = 0; x < nw; x++) {
                    int sx = (int)((double)x * dw / nw);
                    if (sx >= dw) sx = dw - 1;
                    const unsigned char *s = decoded + ((size_t)sy * dw + sx) * 3;
                    unsigned char *d = small + ((size_t)y * nw + x) * 3;
                    d[0] = s[0]; d[1] = s[1]; d[2] = s[2];
                }
            }
            stbi_image_free(decoded);
            decoded = small;
            dw = nw;
            dh = nh;
        }
    }

    *rgb = decoded;
    if (w) *w = dw;
    if (h) *h = dh;
    return 1;
}

int fe_save_png(const char *path, const unsigned char *rgb, int w, int h) {
    if (!stbi_write_png(path, w, h, 3, rgb, w * 3)) {
        fprintf(stderr, "cannot write %s\n", path);
        return 0;
    }
    return 1;
}

int fe_save_jpg(const char *path, const unsigned char *rgb, int w, int h, int quality) {
    if (!stbi_write_jpg(path, w, h, 3, rgb, quality)) {
        fprintf(stderr, "cannot write %s\n", path);
        return 0;
    }
    return 1;
}

/* Write a 16-bit PPM (netpbm P6, big endian).  The PNG writer in stb only
   knows 8 bits per channel, and a RAW preview needs more than that to be
   compared numerically against the Python editor. */
int fe_save_ppm16(const char *path, const unsigned short *rgb, int w, int h) {
    FILE *f = fopen(path, "wb");
    if (!f) return 0;
    fprintf(f, "P6\n%d %d\n65535\n", w, h);
    /* netpbm wants the most significant byte first */
    for (int i = 0; i < w * h * 3; i++) {
        fputc(rgb[i] >> 8, f);
        fputc(rgb[i] & 0xff, f);
    }
    fclose(f);
    return 1;
}
