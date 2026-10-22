# fast-edit — one-command setup & run.
#
#   make run     create .venv if missing, install deps, launch the app
#   make clean   delete the venv
#
# The venv is only built once; afterwards `make run` just launches.

VENV  := .venv
PY    := $(VENV)/bin/python
STAMP := $(VENV)/.deps-installed
DEPS  := numpy scipy pillow opencv-python-headless exifread rawpy customtkinter

.PHONY: run clean

run: $(STAMP)
	$(PY) fast-edit-v2.py

# opencv-python-headless instead of opencv-python: the app uses cv2 only for
# number crunching, never for windows, so we skip its Qt/display libraries
# (they can conflict with tkinter).

$(STAMP):
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install --quiet $(DEPS)
	touch $(STAMP)

clean:
	rm -rf $(VENV)
