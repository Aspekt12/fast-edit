#ifndef FE_OPTIMIZE_H
#define FE_OPTIMIZE_H

#include "fe_defs.h"
#include "fe_math.h"

/* Threading / performance utilities.
   All rendering paths are designed to be parallelizable without sacrificing
   quality.  We use OpenMP when available (build flag), falling back to a
   simple row-stride split. */

/* Run a function over horizontal scanlines in parallel.
   fn(ctx, start_y, end_y) must be thread-safe with respect to its own data. */
void fe_parallel_rows(int height, int nthreads,
                      void (*fn)(void *ctx, int y0, int y1), void *ctx);

/* Determine the number of threads to use (honors FE_THREADS env var). */
int fe_thread_count(void);

/* Quick benchmark / capability probe (prints to stdout). */
void fe_opt_probe(void);

/* Tile the image into cache-friendly tiles for band-limited operations. */
typedef struct { int x0, y0, x1, y1; } FeTile;

/* Cache line aligned buffer helpers */
float *fe_aligned_alloc(size_t count);
void   fe_aligned_free(float *p);

#endif
