.PHONY: venv freeze smoke clean test baseline

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

# Phase 1 -------------------------------------------------------------------

test:
	$(PY) -m pytest tests/ -q

baseline:
	$(PY) bench/run_baseline.py --distance 3 --rounds 3 --basis z --p 0.003 --shots 100000 --timing-shots 1000 --seed 0
	$(PY) bench/run_baseline.py --distance 5 --rounds 5 --basis z --p 0.003 --shots 100000 --timing-shots 1000 --seed 0
	$(PY) bench/run_baseline.py --distance 7 --rounds 7 --basis z --p 0.003 --shots 100000 --timing-shots 1000 --seed 0
	$(PY) bench/run_baseline.py --distance 3 --rounds 3 --basis x --p 0.003 --shots 100000 --timing-shots 1000 --seed 0
	$(PY) bench/run_baseline.py --distance 5 --rounds 5 --basis x --p 0.003 --shots 100000 --timing-shots 1000 --seed 0
	$(PY) bench/run_baseline.py --distance 7 --rounds 7 --basis x --p 0.003 --shots 100000 --timing-shots 1000 --seed 0

# Later phases add: train, quantize, hls-csim, hls-csynth, hls-cosim, host,
# figures, paper. Not defined yet; added when their phase starts.
