#ifndef FE_FILTERS_H
#define FE_FILTERS_H

#include "fe_defs.h"

#define FE_PATH_MAX 4096

typedef struct {
    char *dir;         /* folder name, e.g. "02-film"        */
    char *label;       /* display label, e.g. "Film"           */
    char *colour;      /* category colour, e.g. "#7FA05A"     */
    FeFilter *filters; /* the filters in this category        */
    int count;
} FeCategory;

typedef struct {
    FeCategory *cats;
    int count;
} FeFilterSet;

/* Scan <root>/<category>/<filter>.json, exactly like the Python app:
   the folder name gives the order, category.json gives label + colour. */
int fe_filters_load(const char *root, FeFilterSet *out);
void fe_filters_free(FeFilterSet *set);
const FeFilter *fe_filters_find(const FeFilterSet *set, const char *key);

#endif
