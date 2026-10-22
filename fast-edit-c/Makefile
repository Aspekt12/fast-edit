# =============================================================================
# fast-edit (C) — the fast editor
#
#   make            build build/fast-edit
#   make run        build and launch the editor
#   make selftest   build and run the numeric self-checks
#   make sheet IN=scene.png OUTDIR=/tmp/sheet   render every filter headlessly
#
# GTK3 (the GUI) and LibRaw (RAW decoding) are detected with pkg-config:
# build with the GUI when gtk+-3.0 is present, and fall back to the
# dcraw/PPM path for RAW files when LibRaw's headers are missing.
# =============================================================================

CC      ?= cc
CFLAGS  ?= -O3 -march=native -std=c11 -Wall -Wextra -Wno-unused-parameter -D_GNU_SOURCE -fopenmp
INCLUDES = -Isrc -Ivendor

BUILD := build
TARGET := $(BUILD)/fast-edit

# -- optional dependencies ---------------------------------------------------
#
# A local SYSROOT lets the build use dev headers that were unpacked without
# root (e.g. `make SYSROOT=/tmp/opencode/sysroot`).  Leave it empty when the
# system has the -dev packages installed.
#
SYSROOT ?=
ifeq ($(strip $(SYSROOT)),)
PKG_CONFIG_ENV :=
else
PKG_CONFIG_ENV := PKG_CONFIG_PATH=$(SYSROOT)/usr/lib/x86_64-linux-gnu/pkgconfig:$(SYSROOT)/usr/share/pkgconfig:$(SYSROOT)/usr/lib/pkgconfig:$(PKG_CONFIG_PATH) PKG_CONFIG_SYSROOT_DIR=$(SYSROOT)
endif

HAVE_GTK    := $(shell $(PKG_CONFIG_ENV) pkg-config --exists gtk+-3.0 2>/dev/null && echo 1)
HAVE_LIBRAW := $(shell $(PKG_CONFIG_ENV) pkg-config --exists libraw 2>/dev/null && echo 1)
HAVE_ONNX   := $(shell $(PKG_CONFIG_ENV) pkg-config --exists onnxruntime 2>/dev/null && echo 1)
HAVE_OPENCV := $(shell $(PKG_CONFIG_ENV) pkg-config --exists opencv4 2>/dev/null && echo 1)

GTK_CFLAGS  := $(shell $(PKG_CONFIG_ENV) pkg-config --cflags gtk+-3.0 2>/dev/null)
GTK_LIBS    := $(shell $(PKG_CONFIG_ENV) pkg-config --libs gtk+-3.0 2>/dev/null)
CAIRO_CFLAGS := $(shell $(PKG_CONFIG_ENV) pkg-config --cflags cairo 2>/dev/null)
CAIRO_LIBS   := $(shell $(PKG_CONFIG_ENV) pkg-config --libs cairo 2>/dev/null)
LIBRAW_CFLAGS := $(shell $(PKG_CONFIG_ENV) pkg-config --cflags libraw 2>/dev/null)
LIBRAW_LIBS   := $(shell $(PKG_CONFIG_ENV) pkg-config --libs libraw 2>/dev/null)
ONNX_CFLAGS  := $(shell $(PKG_CONFIG_ENV) pkg-config --cflags onnxruntime 2>/dev/null)
ONNX_LIBS    := $(shell $(PKG_CONFIG_ENV) pkg-config --libs onnxruntime 2>/dev/null)
OPENCV_CFLAGS := $(shell $(PKG_CONFIG_ENV) pkg-config --cflags opencv4 2>/dev/null)
OPENCV_LIBS   := $(shell $(PKG_CONFIG_ENV) pkg-config --libs opencv4 2>/dev/null)

SRC := src/fe_defs.c src/fe_math.c src/fe_pipeline.c src/fe_filters.c \
       src/fe_overlay.c src/fe_io.c src/fe_app.c src/main.c src/stb_impl.c \
       src/fe_optimize.c src/fe_vision.c src/fe_project.c src/fe_lens.c src/fe_lens_auto.c \
       src/fe_state.c src/fe_fields.c
VENDOR := vendor/cJSON.c

ifeq ($(HAVE_GTK),1)
SRC += src/gui.c
endif

OBJ := $(patsubst %.c,$(BUILD)/%.o,$(SRC)) $(BUILD)/vendor/cJSON.o

ALL_CFLAGS  = $(CFLAGS) $(INCLUDES) $(CAIRO_CFLAGS) $(GTK_CFLAGS) $(LIBRAW_CFLAGS) $(ONNX_CFLAGS) $(OPENCV_CFLAGS)
ALL_LDLIBS  = $(CAIRO_LIBS) $(GTK_LIBS) $(LIBRAW_LIBS) $(ONNX_LIBS) $(OPENCV_LIBS) -lm -lgomp

ifeq ($(HAVE_GTK),1)
ALL_CFLAGS += -DFE_HAVE_GTK
endif
ifeq ($(HAVE_LIBRAW),1)
ALL_CFLAGS += -DFE_HAVE_LIBRAW
endif
ifeq ($(HAVE_ONNX),1)
ALL_CFLAGS += -DFE_HAVE_ONNX
endif
ifeq ($(HAVE_OPENCV),1)
ALL_CFLAGS += -DFE_HAVE_OPENCV
endif

.PHONY: all run selftest sheet clean deps

all: $(TARGET)

$(TARGET): $(OBJ)
	@mkdir -p $(BUILD)
	$(CC) $(OBJ) -o $@ $(ALL_LDLIBS)
	@echo "built $@  (GTK: $(if $(HAVE_GTK),yes,no)  LibRaw: $(if $(HAVE_LIBRAW),yes,no))"

$(BUILD)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(ALL_CFLAGS) -c $< -o $@

# third-party code: compile without our warning flags
$(BUILD)/vendor/cJSON.o: vendor/cJSON.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -Ivendor -w -c $< -o $@

run: $(TARGET)
	$(TARGET)

selftest: $(TARGET)
	$(TARGET) --selftest

sheet: $(TARGET)
	@test -n "$(IN)" || (echo "usage: make sheet IN=file.png OUTDIR=/tmp/sheet"; exit 1)
	@mkdir -p $(or $(OUTDIR),/tmp/sheet)
	$(TARGET) --sheet $(IN) $(or $(OUTDIR),/tmp/sheet)

deps:
	@echo "sysroot  : $(if $(strip $(SYSROOT)),$(SYSROOT),none)"
	@echo "gtk+-3.0 : $(if $(HAVE_GTK),found,missing)"
	@echo "libraw    : $(if $(HAVE_LIBRAW),found,missing)"
	@echo "cairo     : $(if $(strip $(CAIRO_LIBS)),found,missing)"

clean:
	rm -rf $(BUILD)
