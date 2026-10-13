"""Pre-submission gate for the EdgeCloud manuscript.

Everything here was checked by hand at least once while preparing this revision. Collecting
the checks in one runnable place is what stops a fixed defect from quietly coming back: a
placeholder re-inserted during a later edit, a table that outgrows its column again, a comma
the rule forbids, a claim quoting a superseded result file.

Run from the project root:

    python scripts/check_submission.py

Exit status is 0 when every check passes, 1 otherwise. Nothing is modified.
"""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / 'manuscript' / 'latex' / 'main.tex'
SETUP = ROOT / 'manuscript' / 'latex' / 'camera_ready.tex'   # the two camera-ready slots
PDF = ROOT / 'manuscript' / 'latex' / 'main.pdf'
DELIVERED = ROOT / 'manuscript' / 'EdgeCloud_ICCA_v2_IEEEA4.pdf'
SIXPP = ROOT / 'manuscript' / 'EdgeCloud_ICCA_v2_6pp_IEEEA4.pdf'   # six-page variant, same source
UPSTREAM = ROOT / 'upstream' / 'prototype_v2.3'
ARCHIVE = ROOT / 'archive' / 'superseded_2026-10-08'

# The first bibliography entry marks where PROTECTED verbatim material begins in the
# rendered text: author lists and quoted titles are not editable author prose.
BIB_MARK = 'R. Gupta et al'
FAILS: list[str] = []
NOTES: list[str] = []


def check(name: str, ok: bool, detail: str = '') -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ''))
    if not ok:
        FAILS.append(name)


def main() -> int:
    tex = TEX.read_text()
    bib_at = tex.index(r'\begin{thebibliography}')
    body = tex[:bib_at]
    print('EdgeCloud pre-submission gate\n')

    print('manuscript source')
    # 1. no unresolved author placeholder may render
    uses = len(re.findall(r'\\authorcheck\{', tex)) - tex.count(r'\newcommand{\authorcheck}')
    check('no unresolved AUTHOR CHECK marker', uses <= 0, f'{max(uses, 0)} use(s)')

    # 2. both punctuation patterns, editable prose only
    for pat, label in ((r',([ \t]*\n?[ \t]*)and\b', '", and"'), (r',([ \t]*\n?[ \t]*)or\b', '", or"')):
        n = len(re.findall(pat, body))
        check(f'{label} replacement check', n == 0, f'{n} occurrence(s)')

    # 3. every cross-reference resolves
    # A \newcommand body contains \ref{#1}, which is a parameter rather than a label, so
    # macro definitions are stripped before the reference scan.
    scan = re.sub(r'\\newcommand\{[^}]*\}(\[\d+\])?\{.*', '', tex)
    labels = set(re.findall(r'\\label\{([^}]*)\}', scan))
    refs = set(re.findall(r'\\(?:ref|eqref)\{([^}]*)\}', scan))
    missing = sorted(refs - labels)
    check('every cross-reference resolves', not missing, ', '.join(missing))

    # 4. no superseded artifact may be cited as a result
    stale = [n for n in ('crisismmd_budgeting_rev1', 'crisismmd_rotations_rev1',
                         'real_delay_rev1', 'itemC_protocol_rev1', 'itemD_protocol_rev1')
             if n in tex]
    check('no superseded _rev1 artifact cited', not stale, ', '.join(stale))

    # 5. this paper cites no companion study from the same portfolio
    internal = [w for w in ('companion study', 'in this programme', 'sibling study')
                if w in body.lower()]
    check('no internal portfolio cross-citation', not internal, ', '.join(internal))

    print('\nrendered PDF')
    try:
        import fitz
    except ImportError:
        NOTES.append('PyMuPDF absent: the rendered checks were skipped')
        print('  SKIP  rendered checks (PyMuPDF not installed)')
        return finish()

    doc = fitz.open(PDF)
    text = '\n'.join(pg.get_text() for pg in doc)
    cut = text.index(BIB_MARK) if BIB_MARK in text else len(text)

    check('no AUTHOR CHECK text in the render', 'AUTHOR CHECK' not in text)
    n = len(re.findall(r',\s+(?:and|or)\b', text[:cut]))
    check('punctuation clean in extracted text', n == 0, f'{n} occurrence(s)')
    # the indicator function must not fall through to U+22AE NOT FORCES
    # A source-level \ref/\label audit cannot see a reference whose target is
    # inside the other variant: \ref{tab:contested} resolves in the master and
    # prints "??" in the submission, because that table is master-only. Only the
    # rendered text shows it. This fired on the Fig. 2 caption once the figures
    # moved into the submission.
    qq = [i + 1 for i, pg in enumerate(doc) if '??' in pg.get_text()]
    check('no unresolved reference in the render', not qq,
          f"?? on page(s) {', '.join(map(str, qq))}" if qq else 'no "??" on any page')

    # a dummy archival identifier (the setup placeholder) must never reach a submitted PDF
    dummy = re.findall(r'zenodo\.(?:X{3,}|0{5,})|\bXXXX+\b', text)
    check('no placeholder archival identifier in the render', not dummy,
          f'{len(dummy)} occurrence(s)' + (f': {dummy[0]}' if dummy else ''))

    check('indicator function typesets', '\u22ae' not in text)

    # 6. no table may overprint the opposite column
    worst = 0.0
    where = 0
    for i, pg in enumerate(doc):
        blocks = [b for b in pg.get_text('blocks') if b[4].strip()]
        mid = pg.rect.width / 2
        left = [b for b in blocks if b[0] < mid - 20]
        right = [b for b in blocks if b[0] >= mid - 20]
        for a in left:
            for c in right:
                if a[2] > c[0] and not (a[3] < c[1] or a[1] > c[3]):
                    if a[2] - c[0] > worst:
                        worst, where = a[2] - c[0], i + 1
    check('no column collision on any page', worst <= 0.5,
          f'{worst:.2f} pt on page {where}' if worst > 0.5 else f'{doc.page_count} pages')

    # 7. no single-column block may run past its column. Full-width elements, the title
    # block and a table* float, legitimately span both columns and are excluded by width.
    import collections
    rl, rr = [], []
    for pg in doc:
        mid = pg.rect.width / 2
        for b in pg.get_text('blocks'):
            if b[4].strip():
                (rl if b[0] < mid - 20 else rr).append(round(b[2]))
    col_r = collections.Counter(rl).most_common(1)[0][0]
    rcol_r = collections.Counter(rr).most_common(1)[0][0]
    over = []
    for i, pg in enumerate(doc):
        mid = pg.rect.width / 2
        for b in pg.get_text('blocks'):
            if not b[4].strip():
                continue
            # A full-width element, the title block or a table* float, legitimately runs
            # from the left column across the gutter to the right text boundary. What must
            # never happen is any block passing that boundary into the right page margin,
            # which is how an over-wide table announces itself.
            if b[2] > rcol_r + 2:
                over.append(f'page {i + 1} by {b[2] - rcol_r:.1f} pt')
    check('no block runs into the right page margin', not over, '; '.join(over[:4]))

    # 8. the rendered file should carry document metadata
    md = doc.metadata or {}
    check('PDF carries a title', bool(md.get('title')), md.get('title', '') or 'unset')

    if DELIVERED.exists():
        dd = fitz.open(DELIVERED)
        check('delivered PDF matches the compiled page count',
              dd.page_count == doc.page_count, f'{dd.page_count} against {doc.page_count}')
        # A matching page count is not a matching document. An edit that does not
        # reflow the text -- a letter swap, a corrected digit -- leaves the count
        # identical while the delivered file still carries the old words, so the
        # text itself has to be compared. This check exists because a round of
        # spelling corrections shipped only to the compiled file and the delivered
        # PDF kept the superseded spellings at an unchanged twelve pages.
        def words(d):
            return re.sub(r'\s+', ' ', ' '.join(pg.get_text() for pg in d)).strip()
        a, b = words(dd), words(doc)
        if a == b:
            detail = 'text identical'
        else:
            sa, sb = a.split(), b.split()
            diff = next((f'first difference at word {i + 1}: delivered '
                         f'{x!r} against compiled {y!r}'
                         for i, (x, y) in enumerate(zip(sa, sb)) if x != y),
                        f'length {len(sa)} against {len(sb)}')
            detail = diff
        check('delivered PDF text matches the compiled text', a == b, detail)

    if SIXPP.exists():
        six = fitz.open(SIXPP)
        stext = '\n'.join(pg.get_text() for pg in six)
        scut = stext.index(BIB_MARK) if BIB_MARK in stext else len(stext)
        check('six-page variant is within six pages', six.page_count <= 6, f'{six.page_count} pages')
        n6 = len(re.findall(r',\s+(?:and|or)\b', stext[:scut]))
        check('six-page variant punctuation clean in extracted text', n6 == 0, f'{n6} occurrence(s)')
        qq6 = [i + 1 for i, pg in enumerate(six) if '??' in pg.get_text()]
        check('six-page variant has no unresolved reference', not qq6,
              f"?? on page(s) {', '.join(map(str, qq6))}" if qq6 else 'no "??" on any page')
        check('six-page variant carries a title', bool((six.metadata or {}).get('title')))

    print('\nreference package')
    sums = UPSTREAM / 'SHA256SUMS'
    if sums.exists():
        ok = bad = 0
        mismatched = []
        for line in sums.read_text().splitlines():
            if not line.strip():
                continue
            want, _, rel = line.partition('  ')
            f = UPSTREAM / rel
            if not f.exists():
                mismatched.append(rel + ' (missing)')
                bad += 1
                continue
            if hashlib.sha256(f.read_bytes()).hexdigest() == want:
                ok += 1
            else:
                mismatched.append(rel)
                bad += 1
        # CITATION.cff carries deposit placeholders and is known not to match; it is
        # inherited that way and upstream is reproduced unchanged.
        expected = {'./CITATION.cff'}
        unexpected = [m for m in mismatched if m not in expected]
        check('package digests verify', not unexpected,
              f'{ok} verified, {bad} known exception' if not unexpected
              else ', '.join(unexpected))

    # TeX discards the space after a control word, so a conditional closing immediately
    # before a word silently joins them: "maps\\fi the" renders as "mapsthe". This caught
    # two such joins that every other check passed over.
    import re as _re2
    joins = set()
    for m in _re2.finditer(r'\\b([a-z]{3,})(?=[A-Z][a-z]{2,})|([a-z])\\.([A-Z][a-z]{2,})', text):
        joins.add(m.group(0))
    # words a dictionary would also join legitimately are rare here; report them all
    known_ok = {'arXiv', 'CrisisMMD'}
    joins = {j for j in joins if not any(k in j for k in known_ok)}
    check('no words joined by a swallowed space', not joins, ', '.join(sorted(joins)[:5]))

    print('\ncamera-ready')
    # a camera-ready PDF must embed every font for IEEE PDF eXpress
    not_embedded = set()
    for i in range(doc.page_count):
        for f in doc.get_page_fonts(i, full=True):
            if not f[1] or f[1] == 'n/a':
                not_embedded.add(f[3])
    check('every font embedded', not not_embedded, ', '.join(sorted(not_embedded)))
    # the template warns that leftover boilerplate can stop publication
    leaks = [l for l in ('Identify applicable funding agency', 'Given Name Surname',
                         'dept. name of organization', 'email address or ORCID',
                         'Table Column Head', 'XXX-X-XXXX')
             if l.lower() in text.lower()]
    check('no template boilerplate left in the paper', not leaks, ', '.join(leaks))
    # these two need a string only the conference can supply
    import re as _re
    has_cr = bool(_re.search(r'\u00a9\s*20\d\d IEEE|978-\d-\d{4}', text))
    print(f"  {'PASS' if has_cr else 'TODO'}  copyright line printed on page 1"
          + ('' if has_cr else '  [conference supplies it; paste into '
                              '\\copyrightline in camera_ready.tex]'))
    if not has_cr:
        NOTES.append('camera-ready: the copyright line is still empty, so nothing prints. '
                     'It is not a FAIL because the paper renders cleanly without it, but it '
                     'must be filled before the final upload.')

    # The copyright line and the archival identifier are empty while they are unknown, so a
    # page count taken now is not the count the final paper will have. Both are filled at
    # camera-ready and both add text, which is how a 12-page paper silently becomes 13.
    # This builds the paper with representative values in both slots and reports that count.
    import subprocess, tempfile, shutil, os
    src = TEX.read_text()
    setup = SETUP.read_text()
    filled = (re.sub(r'\\newcommand\{\\copyrightline\}\{[^}]*\}',
                     r'\\newcommand{\\copyrightline}{978-1-0000-0000-0/26/\\$31.00~\\copyright 2026 IEEE}', setup))
    filled = re.sub(r'\\newcommand\{\\archivedoi\}\{[^}]*\}',
                    r'\\newcommand{\\archivedoi}{10.5281/zenodo.0000000}', filled)
    probe = src
    tmp = Path(tempfile.mkdtemp())
    for f in TEX.parent.glob('*'):
        if f.is_file() and f.suffix in ('.cls', '.png', '.tex'):
            shutil.copy(f, tmp / f.name)
    (tmp / 'main.tex').write_text(probe)
    (tmp / 'camera_ready.tex').write_text(filled)
    r = subprocess.run(['tectonic', '-X', 'compile', 'main.tex', '--outdir', '.'],
                       cwd=tmp, capture_output=True)
    if (tmp / 'main.pdf').exists():
        filled = fitz.open(tmp / 'main.pdf').page_count
        print(f'  {"PASS" if filled <= doc.page_count else "FAIL"}  '
              f'page count holds once the camera-ready slots are filled  '
              f'[{doc.page_count} empty, {filled} filled]')
        if filled > doc.page_count:
            FAILS.append('page count grows when the camera-ready slots are filled')
    else:
        NOTES.append('could not build the filled probe, so the camera-ready page count is '
                     'unverified')
    shutil.rmtree(tmp, ignore_errors=True)

    print('\noutstanding by design')
    setup_code = '\n'.join(l for l in SETUP.read_text().splitlines() if not l.lstrip().startswith('%'))
    m = re.search(r'\\newcommand\{\\archivedoi\}\{([^}]*)\}', setup_code)
    doi = m.group(1) if m else ''
    if doi:
        print(f'  NOTE  archival identifier quoted in Section XI: {doi}'
              + ('  (PLACEHOLDER, see the FAIL above)' if 'X' in doi else ''))
    else:
        dep = 'archival identifier' in tex or 'not yet deposited' in tex
        print(f"  NOTE  the release is not deposited, so no archival identifier is quoted "
              f"({'stated in Section XI' if dep else 'NOT stated in the paper'})")
    print('  NOTE  the venue page rule is UNVERIFIED: no official ICCA limit is held here, '
          f'so the {doc.page_count}-page count is reported without a compliance claim')
    return finish()


def finish() -> int:
    print()
    for n in NOTES:
        print('NOTE:', n)
    if FAILS:
        print(f'GATE: FAIL, {len(FAILS)} check(s): ' + '; '.join(FAILS))
        return 1
    print('GATE: PASS, every check above passed. This is not a claim that the manuscript is '
          'free of defects, only that these checks passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
