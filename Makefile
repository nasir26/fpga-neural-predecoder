.PHONY: venv freeze smoke clean

VENV := venv
PY := $(VENV)/bin/python

# Phase 0 -----------------------------------------------------------------

venv:
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

freeze:
	$(VENV)/bin/pip freeze > docs/env.lock

smoke:
	$(PY) scripts/phase0_smoke.py

clean:
	rm -rf $(VENV)

# Later phases add: data, train, quantize, hls-csim, hls-csynth, hls-cosim,
# host, bench, figures, paper. Not defined yet; added when their phase starts.
