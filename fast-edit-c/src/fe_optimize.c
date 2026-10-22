#include "fe_optimize.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

int fe_thread_count(void) {
    const char *e = getenv("FE_THREADS");
    if (e && *e) {
        int n = atoi(e);
        if (n > 0) return n;
    }
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

void fe_parallel_rows(int height, int nthreads,
                      void (*fn)(void *ctx, int y0, int y1), void *ctx) {
    if (height <= 0 || !fn) return;
    if (nthreads <= 1) { fn(ctx, 0, height); return; }
#ifdef _OPENMP
    /* dynamic schedule helps with non-uniform work (grain/blurs) */
    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        int nt  = omp_get_num_threads();
        int base = height / nt;
        int rem  = height % nt;
        int y0   = tid * base + (tid < rem ? tid : rem);
        int y1   = y0 + base + (tid < rem ? 1 : 0);
        fn(ctx, y0, y1);
    }
#else
    /* simple round-robin by bands */
    int band = height / nthreads;
    if (band < 1) band = 1;
    for (int t = 0; t < nthreads; t++) {
        int y0 = t * band;
        int y1 = y0 + band;
        if (t == nthreads - 1) y1 = height;
        if (y0 >= height) break;
        fn(ctx, y0, y1);
    }
#endif
}

float *fe_aligned_alloc(size_t count) {
    /* 32-byte aligned for potential SIMD */
    void *p = NULL;
#ifdef _GNU_SOURCE
    if (posix_memalign(&p, 32, count * sizeof(float)) != 0) p = NULL;
#else
    p = malloc(count * sizeof(float) + 32);
    if (p) {
        uintptr_t addr = (uintptr_t)p;
        uintptr_t aligned = (addr + 31) & ~((uintptr_t)31);
        /* store original pointer before aligned one */
        ((void **)aligned)[-1] = p;
        p = (void *)aligned;
    }
#endif
    if (p) memset(p, 0, count * sizeof(float));
    return (float *)p;
}

void fe_aligned_free(float *p) {
    if (!p) return;
#ifndef _GNU_SOURCE
    if (((uintptr_t)p) & 31) { /* shouldn't happen */ }
    void **orig = (void **)p - 1;
    free(*orig);
#else
    free(p);
#endif
}

void fe_opt_probe(void) {
    printf("=== fast-edit optimization probe ===\n");
#ifdef _OPENMP
    printf("OpenMP: enabled, max threads %d\n", omp_get_max_threads());
#else
    printf("OpenMP: disabled\n");
#endif
    printf("fe_thread_count(): %d\n", fe_thread_count());
    printf("Compiler: %s\n", 
#ifdef __GNUC__
        __VERSION__
#else
        "unknown"
#endif
    );
#ifdef __AVX2__
    printf("SIMD: AVX2 available\n");
#elif defined(__SSE4_1__)
    printf("SIMD: SSE4.1 available\n");
#elif defined(__NEON__)
    printf("SIMD: NEON available\n");
#else
    printf("SIMD: scalar only\n");
#endif
    printf("===================================\n");
}
