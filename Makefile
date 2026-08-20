.PHONY: venv freeze smoke clean test baseline figures paper

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

# Phase 6 ---------------------------------------------------------------

# Regenerates every figure from committed results/metrics/*.json only --
# no network access, no retraining, no re-running HLS/v++. To regenerate
# the underlying sweep data itself (slow, hours), run
# bench/run_phase6_sweep.py directly first.
figures:
	$(PY) scripts/make_figures.py

# Phase 7 ---------------------------------------------------------------

# Rebuilds paper/paper.pdf from paper/paper.tex. Requires a LaTeX
# toolchain (see paper/build.sh's header comment for exact packages / a
# no-root TinyTeX install path).
paper:
	./paper/build.sh
