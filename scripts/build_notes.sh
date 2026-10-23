#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
pandoc docs/MODEL.md --standalone --number-sections --toc \
  --from=markdown+tex_math_dollars --to=latex \
  -V documentclass=article -V papersize=letter -V geometry:margin=0.8in \
  -V fontsize=11pt -V colorlinks=true -V urlcolor=blue \
  -o docs/English_Model_Notes.tex
python scripts/format_latex.py
pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error -output-directory=docs docs/English_Model_Notes.tex
pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error -output-directory=docs docs/English_Model_Notes.tex
pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error -output-directory=docs docs/English_Model_Notes.tex
