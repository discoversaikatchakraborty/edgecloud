#!/bin/bash
# Assemble the v2.6.18 deposit: the frozen v2.6.17 package plus the three new evidence directories,
# the completed CITATION.cff and a regenerated SHA256SUMS. Nothing in v2.6.17 is modified.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/manuscript/deposit/EdgeCloud_Emergency_AI_Prototype_v2.6.18"
rm -rf "$OUT"; mkdir -p "$OUT"
cp -R "$ROOT/upstream/prototype_v2.3/." "$OUT/"
rm -rf "$OUT/__pycache__" "$OUT/SHA256SUMS"
mkdir -p "$OUT/emulator" "$OUT/reproduction_crisismmd" "$OUT/established_tasks"
cp "$ROOT"/scripts/emulator/*.py "$OUT/emulator/"
cp -R "$ROOT/results/emulator/." "$OUT/emulator/results/"
cp "$ROOT"/scripts/crisismmd_compare_records.py "$ROOT"/scripts/crisismmd_compare_table3.py "$OUT/reproduction_crisismmd/"
cp -R "$ROOT/results/reproduction_crisismmd/." "$OUT/reproduction_crisismmd/results/"
cp "$ROOT/scripts/crisismmd_established_tasks.py" "$OUT/established_tasks/"
cp -R "$ROOT/results/established_tasks/." "$OUT/established_tasks/results/"
cp "$ROOT/manuscript/deposit/CITATION.cff" "$OUT/CITATION.cff"
cp "$ROOT/manuscript/deposit/RELEASE_NOTES_v2.6.18.txt" "$OUT/RELEASE_NOTES.txt"
(cd "$OUT" && find . -type f ! -name SHA256SUMS ! -name .DS_Store | LC_ALL=C sort | xargs shasum -a 256 > SHA256SUMS)
# every v2.6.17 digest must be unchanged, except CITATION.cff, which this release completes on purpose
( cd "$OUT" && while read -r sum path; do [ "$path" = "./CITATION.cff" ] && continue; grep -q "^$sum  $path\$" SHA256SUMS || { echo "DIGEST CHANGED: $path"; exit 1; }; done < "$ROOT/upstream/prototype_v2.3/SHA256SUMS" ) && echo "v2.6.17 digests all present and unchanged (CITATION.cff completed, as intended)"
echo "files: $(wc -l < "$OUT/SHA256SUMS")  size: $(du -sh "$OUT" | cut -f1)  at $OUT"
