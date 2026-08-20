#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Nasir Ali, C-DAC Noida
# SPDX-License-Identifier: Apache-2.0
#
# Rebuilds paper/paper.pdf from paper/paper.tex. Two pdflatex passes are
# required (not a bug) so cross-references and the bibliography (a plain
# thebibliography block, not bibtex) resolve; a bare single pass will warn
# "undefined" for every \ref/\cite on the first pass, which is expected.
#
# Requires: pdflatex with the IEEEtran class plus amsmath, algorithmicx,
# booktabs, hyperref, courier/times font packages. If not installed, a
# user-space TeX Live works without root:
#   wget -qO- "https://yihui.org/tinytex/install-bin-unix.sh" | sh
#   export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH"
#   tlmgr install ieeetran algorithms algorithmicx booktabs multirow \
#       xcolor url hyperref caption cite courier helvetic times
set -euo pipefail
cd "$(dirname "$0")"

pdflatex -interaction=nonstopmode -halt-on-error paper.tex
pdflatex -interaction=nonstopmode -halt-on-error paper.tex

echo "wrote paper/paper.pdf"
