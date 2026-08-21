#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Nasir Ali, C-DAC Noida
# SPDX-License-Identifier: Apache-2.0
#
# Rebuilds paper/paper_jsc.pdf (Journal of Supercomputing format, Springer's
# sn-jnl class, sn-basic reference style) from paper/paper_jsc.tex.
#
# Three passes are required, and -halt-on-error CANNOT be used: sn-basic
# makes the class call \bibliographystyle{sn-basic} internally, which
# assumes a real sn-basic.bst file and a bibtex run against a .bib database.
# This project uses a manual thebibliography block instead (no bibtex, same
# approach as paper/paper.tex), so natbib (which sn-jnl loads with the
# `numbers` option for sn-basic) cannot find sn-basic's compatibility
# markers and raises a recoverable "Bibliography not compatible with
# author-year citations -- press <return> to continue in numerical citation
# style" prompt on every single pass, not just the first. This is not a
# bug to silence: the offered fallback (numerical citation style) is
# exactly the style sn-basic already wants, so plain `-interaction=nonstopmode`
# (which auto-continues past this prompt) with stdin redirected from
# /dev/null (so it can never actually block on a real terminal prompt) is
# the correct, permanent way to compile this file -- not a one-time
# workaround.
#
# sn-jnl.cls and sn-mathphys-num.bst are not on CTAN/TeX Live; Springer
# distributes them from their own author-resources page. Copied here from
# a local sibling project that already had them (see docs/DECISIONS.md).
#
# Requires (beyond paper/build.sh's IEEE package list): cuted (via the
# `sttools` bundle), threeparttable, appendix, wrapfig, needspace,
# environ, trimspaces, footmisc, was, mathrsfs (via `jknapltx`+`rsfs`),
# natbib. A no-root TinyTeX install path:
#   wget -qO- "https://yihui.org/tinytex/install-bin-unix.sh" | sh
#   export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH"
#   tlmgr install sttools threeparttable appendix wrapfig needspace \
#       environ trimspaces footmisc was xkeyval mathrsfs jknapltx natbib
set -uo pipefail
cd "$(dirname "$0")"

pdflatex -interaction=nonstopmode paper_jsc.tex < /dev/null
pdflatex -interaction=nonstopmode paper_jsc.tex < /dev/null
pdflatex -interaction=nonstopmode paper_jsc.tex < /dev/null

if [ ! -s paper_jsc.pdf ]; then
    echo "ERROR: paper_jsc.pdf was not produced" >&2
    exit 1
fi

echo "wrote paper/paper_jsc.pdf"
