#!/bin/sh
# Builds the three variants of the manuscript from the single source main.tex.
#   submission : \submissiontrue                -> main.pdf, delivered as EdgeCloud_ICCA_v2_IEEEA4.pdf
#   master     : \submissionfalse               -> EdgeCloud_master.pdf, delivered as EdgeCloud_master_v2.pdf
#   six-page   : \submissiontrue + \sixpagetrue -> main_6pp.pdf, delivered as EdgeCloud_ICCA_v2_6pp_IEEEA4.pdf
# The source is left holding \submissiontrue and \sixpagefalse so an unqualified build is the
# twelve-page submission. Run scripts/check_submission.py afterwards.
set -e
cd "$(dirname "$0")"
build() {  # $1 = sed expression   $2 = output name
  sed "$1" main.tex > _build.tex
  mkdir -p _out && tectonic -X compile _build.tex --outdir _out --keep-logs >/dev/null 2>_err.txt || {
      echo "FAILED ($2):"; tail -25 _err.txt; exit 1; }
  mv _out/_build.pdf "$2"
  echo "$2: $(pdfinfo "$2" | awk '/^Pages/{print $2}') pages"
}
build 's/^\\submissiontrue$/\\submissiontrue/'   main.pdf
build 's/^\\submissiontrue$/\\submissionfalse/'  EdgeCloud_master.pdf
build 's/^\\sixpagefalse$/\\sixpagetrue/'        main_6pp.pdf
rm -rf _build.tex _out _err.txt
cp main.pdf ../EdgeCloud_ICCA_v2_IEEEA4.pdf
cp EdgeCloud_master.pdf ../EdgeCloud_master_v2.pdf
cp main_6pp.pdf ../EdgeCloud_ICCA_v2_6pp_IEEEA4.pdf
echo "delivered: manuscript/EdgeCloud_ICCA_v2_IEEEA4.pdf, manuscript/EdgeCloud_master_v2.pdf and manuscript/EdgeCloud_ICCA_v2_6pp_IEEEA4.pdf"
