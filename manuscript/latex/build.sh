#!/bin/sh
# Builds both variants of the manuscript from the single source main.tex.
#   submission : \submissiontrue  -> main.pdf, delivered as EdgeCloud_ICCA_Revised_IEEEA4.pdf
#   master     : \submissionfalse -> EdgeCloud_master.pdf
# The source is left holding \submissiontrue so an unqualified build is the
# submission. Run scripts/check_submission.py afterwards; its delivered-text
# check is what catches a variant edited but not re-delivered.
set -e
cd "$(dirname "$0")"
build() {  # $1 = submissiontrue|submissionfalse   $2 = output name
  sed "s/^\\\\submissiontrue\$/\\\\$1/" main.tex > _build.tex
  mkdir -p _out && tectonic -X compile _build.tex --outdir _out --keep-logs >/dev/null 2>_err.txt || {
      echo "FAILED ($2):"; tail -25 _err.txt; exit 1; }
  mv _out/_build.pdf "$2"
  echo "$2: $(pdfinfo "$2" | awk '/^Pages/{print $2}') pages"
}
build submissiontrue  main.pdf
build submissionfalse EdgeCloud_master.pdf
rm -rf _build.tex _out _err.txt
cp main.pdf ../EdgeCloud_ICCA_Revised_IEEEA4.pdf
cp EdgeCloud_master.pdf ../EdgeCloud_master.pdf
echo "delivered: manuscript/EdgeCloud_ICCA_Revised_IEEEA4.pdf and manuscript/EdgeCloud_master.pdf"
