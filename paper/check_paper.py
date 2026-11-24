"""Structural checks on the generated paper.tex, refs.bib and figure set.

There is no LaTeX toolchain on this machine, so the document cannot be compiled here. That is not a
reason to ship it unchecked -- most of what a first compile would catch is checkable without one,
and the rest is named explicitly at the end of this script's output rather than left implied.

    python our_papers/paper1_variance_aware/latex/check_paper.py

Exit status 1 if any FAIL. Warnings do not fail the build; they are decisions, not defects.
"""
import os
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TEX = os.path.join(HERE, "paper.tex")
BIB = os.path.join(HERE, "refs.bib")
# In the working tree the figures are generated into ../draft/figures and copied here; in the
# release bundle only the copy travels. Check whichever exists, so this script runs in both.
FIGS = os.path.join(ROOT, "draft", "figures")
if not os.path.isdir(FIGS):
    FIGS = os.path.join(HERE, "figures")

fails, warns = [], []


def fail(msg):
    fails.append(msg)


def warn(msg):
    warns.append(msg)


tex = open(TEX, encoding="utf-8").read()
bib = open(BIB, encoding="utf-8").read()

# Comments carry instructions to the author, so anything scanning for real commands must see the
# document TeX sees, not the file.
code = re.sub(r"(?<!\\)%.*", "", tex)

# ---------------------------------------------------------------------------- citations
cited = set()
for m in re.finditer(r"\\cite[tp]?\{([^}]*)\}", tex):
    for k in m.group(1).split(","):
        if k.strip():
            cited.add(k.strip())
defined = set(re.findall(r"@\w+\{([^,]+),", bib))

for k in sorted(cited - defined):
    fail(f"cited but not in refs.bib: {k}")
unused = sorted(defined - cited)
if unused:
    warn(f"in refs.bib but never cited ({len(unused)}): {', '.join(unused)}")

# Count only markers that actually sit above an entry. The file's header explains the NEEDS-CHECK
# rule in prose, so a plain string count reported two unverified entries after the last one had
# been filled -- a warning that cries wolf teaches you to skim the warnings.
needs = re.findall(r"^%[^\n]*NEEDS-CHECK[^\n]*\n(?:^%[^\n]*\n)*^@", bib, re.M)
if needs:
    warn(f"{len(needs)} bib entries carry NEEDS-CHECK -- author lists to confirm before submission")

# ---------------------------------------------------------------------------- figures
SEARCH = [os.path.join(HERE, "figures"), HERE]     # must mirror \graphicspath in the preamble
for m in re.finditer(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", code):
    rel = m.group(1)
    # extension-less includes resolve to .pdf, then .png (\DeclareGraphicsExtensions); a PNG must exist
    cands = [rel] if os.path.splitext(rel)[1] else [rel + ".png"]
    if not any(os.path.exists(os.path.join(d, c)) for d in SEARCH for c in cands):
        fail(f"figure referenced but found in none of {SEARCH}: {rel}")
built = {f[:-4] for f in os.listdir(FIGS) if f.endswith(".png")} if os.path.isdir(FIGS) else set()
used = {os.path.splitext(os.path.basename(m))[0]
        for m in re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", code)}
missing_pdf = sorted(f for f in built if not os.path.exists(os.path.join(FIGS, f + ".pdf")))
if missing_pdf:
    warn(f"{len(missing_pdf)} figure(s) have no vector .pdf -- LaTeX will use the raster PNG: {missing_pdf}")
# A figure that exists but was never placed almost always means its anchor subsection was renamed,
# which is silent: the paper simply loses a figure. Fail rather than warn.
for f in sorted(built - used):
    fail(f"figure built but never placed -- its anchor subsection was probably renamed: {f}")

# ---------------------------------------------------------------------------- structure
if tex.count(r"\begin{document}") != 1 or tex.count(r"\end{document}") != 1:
    fail("document environment is not balanced")
for env in ("abstract", "figure", "tabular", "enumerate", "itemize", "quote", "center"):
    b, e = tex.count(rf"\begin{{{env}}}"), tex.count(rf"\end{{{env}}}")
    if b != e:
        fail(f"unbalanced environment {env}: {b} begin, {e} end")

nsec = len(re.findall(r"^\\section\{", tex, re.M))
if nsec != 9:
    fail(f"expected 9 sections, found {nsec}")

# every tabular row must have the column count its preamble declares
def count_columns(spec):
    """Number of columns a tabular preamble declares.

    Everything that is not itself a column has to go first, or its letters are counted as columns:
    the >{...} and <{...} modifiers (which hold \\raggedright, \\arraybackslash and friends) and the
    width argument of p{}, m{} and b{}. What remains is one character per column.
    """
    out, i, depth = [], 0, 0
    while i < len(spec):
        ch = spec[i]
        if ch in "><@!" and i + 1 < len(spec) and spec[i + 1] == "{":
            i += 2
            depth = 1
            while i < len(spec) and depth:
                depth += (spec[i] == "{") - (spec[i] == "}")
                i += 1
            continue
        if ch in "pmb" and i + 1 < len(spec) and spec[i + 1] == "{":
            out.append("p")
            i += 2
            depth = 1
            while i < len(spec) and depth:
                depth += (spec[i] == "{") - (spec[i] == "}")
                i += 1
            continue
        if ch in "lcrpmbX":
            out.append(ch)
        i += 1
    return len(out)


# The tabular preamble may now contain braces, so [^}]* no longer reaches its end.
TABULAR = re.compile(r"\\begin\{tabular\}\{((?:[^{}]|\{[^{}]*\})*)\}(.*?)\\end\{tabular\}", re.S)
for m in TABULAR.finditer(tex):
    ncol = count_columns(m.group(1))
    for row in m.group(2).split(r"\\"):
        row = row.strip()
        if not row or row.startswith("\\toprule") or row.startswith("\\midrule") \
                or row.startswith("\\bottomrule") or row in (r"\toprule", r"\midrule"):
            continue
        body = row.replace(r"\toprule", "").replace(r"\midrule", "").replace(r"\bottomrule", "")
        if not body.strip():
            continue
        # & inside a protected span would be escaped already, so a bare & is a separator
        got = len(re.split(r"(?<!\\)&", body))
        if got != ncol:
            fail(f"table row has {got} cells, preamble declares {ncol}: {body.strip()[:70]}")

# ---------------------------------------------------------------------------- leftovers
LEFTOVER = {
    r"\*\*": "unconverted markdown bold",
    r"(?<![\\$\w])_(?![\w\\])": "bare underscore outside math",
    r"[\u2B1C\u2705\u26A0\U0001F534\U0001F7E1\U0001F7E2\u2B50\U0001F6E1]": "emoji from the drafts",
    r"\[\^": "unconverted footnote mark",
    r"^\s*\|": "unconverted markdown table row",
    r"Notes for revision": "revision notes leaked into the paper",
    r"Draft 1 \u00b7": "draft front-matter leaked into the paper",
}
for pat, what in LEFTOVER.items():
    hits = re.findall(pat, tex, re.M)
    if hits:
        fail(f"{what}: {len(hits)} occurrence(s)")

# ---------------------------------------------------------------------------- placeholders
n_email = len(re.findall(r"\\email EMAIL", code))
if n_email:
    warn(f"{n_email} author email(s) still the placeholder EMAIL -- fill before submission")

# The paper's most quotable claim rests on one architecture. Every place it appears must say so,
# or the paper commits the overclaim it exists to criticise.
for m in re.finditer(r"(SF and AG cannot [^.]*?\.)", code):
    s = m.group(1)
    if not re.search(r"LRRNet|architecture|on one|we test|we ran", s):
        fail(f"unscoped positive-control claim (needs the architecture named): {s[:90]}")

# Withdrawn claims must not survive in the generated text. "Zero of 27" was the headline at three
# seeds and was withdrawn by Run 9 (claim #10); it outlived the withdrawal in a figure caption, which
# lives in build_paper.py rather than in the drafts, so a search of the drafts could not find it.
# An external review then found the paper still quoting the superseded n=3 lead count "8 of 9".
STALE = {
    r"[Zz]ero of 27": "withdrawn headline 'zero of 27' (claim #10, Run 9)",
    r"leading on 8 of 9": "superseded n=3 lead count; at n=8 it is 7 of 9",
    r"every other row \$n=3\$": "figure 3 caption describes the n=3 table, not Run 9",
}
for pat, what in STALE.items():
    for m in re.finditer(pat, code):
        # the withdrawn-claims table and its prose name the old count on purpose
        if re.search(r"withdrawn|At three seeds|was zero|headline itself", code[max(0, m.start() - 200):m.start()]):
            continue
        fail(f"{what}: ...{code[m.start():m.start() + 60]}")

# The title was decided on 2026-09-09 and lived in four different forms before that. paper.tex is
# generated, PAPER.md is hand-maintained, and nothing tied them together -- so they drifted.
TITLE_MAIN = "Ablation Tables as Measuring Instruments"
TITLE_SUB = "Run-to-Run Variability, Confounds and a Positive Control"
m = re.search(r"\\title\{(.*?)\}", code, re.S)
if not m:
    fail("paper.tex has no \\title")
else:
    got = " ".join(m.group(1).replace("\\\\", " ").split())
    for part in (TITLE_MAIN, TITLE_SUB):
        if part not in got:
            fail(f"\\title does not carry the decided title (missing '{part}'): {got[:70]}")
    pm = os.path.join(ROOT, "PAPER.md")
    if os.path.exists(pm) and TITLE_MAIN not in open(pm, encoding="utf-8").read():
        fail(f"PAPER.md no longer states the decided title '{TITLE_MAIN}' -- "
             f"paper.tex and the plan of record have diverged")

if True:
    # paper.tex is the named reading copy; the submission is the anonymised Expert Systems with Applications build
    # that build_paper.py writes to eswa_submission/manuscript/ (2026-11-24; TMLR dropped). Check THAT.
    _sub = os.path.join(os.path.dirname(HERE), "eswa_submission", "manuscript")
    _tex = os.path.join(_sub, "paper_eswa.tex")
    if not os.path.exists(_tex):
        warn("no eswa_submission/manuscript/paper_eswa.tex -- re-run build_paper.py to produce the submission build")
    else:
        _t = open(_tex, encoding="utf-8").read()
        if "{elsarticle}" not in _t or "\\journal{Expert Systems with Applications}" not in _t:
            fail("eswa_submission/manuscript/paper_eswa.tex is not on elsarticle for Expert Systems with Applications")
        if "\\author" in _t or "\\address" in _t or "\\ead" in _t:
            fail("eswa_submission/manuscript/paper_eswa.tex carries an author block -- the review is double-anonymised")
        if "\\begin{keyword}" not in _t:
            fail("eswa_submission/manuscript/paper_eswa.tex has no keywords")
        for f in ("refs.bib",):
            if not os.path.exists(os.path.join(_sub, f)):
                fail(f"eswa_submission/manuscript/ is missing {f}")
        # the source and the zip are uploaded too: neither may name the authors (2026-10-13, kept for ESWA)
        _ids = ("Nada", "Mansoura", "El-Mikaty", "Mikaty", "Elashry", "Hertfordshire", "gmail.com", "mans.edu", "gaf.edu",
                "mostafanada82", "Mostafa", "Abdelgwad", "Dardosh82", "github.com")
        _src_leak = [w for w in _ids if w in _t]
        if _src_leak:
            fail(f"eswa_submission/manuscript/paper_eswa.tex names the authors: {_src_leak}")
        _zip = os.path.join(os.path.dirname(_sub), "manuscript_anonymised.zip")
        if os.path.exists(_zip):
            import zipfile
            with zipfile.ZipFile(_zip) as _z:
                for _n in _z.namelist():
                    if _n.endswith((".tex", ".bib")):
                        _zl = [w for w in _ids if w in _z.read(_n).decode("utf-8", "ignore")]
                        if _zl:
                            fail(f"manuscript_anonymised.zip:{_n} names the authors: {_zl}")
        else:
            warn("eswa_submission/manuscript_anonymised.zip not built -- re-run build_paper.py")
        _pdf = _tex[:-4] + ".pdf"
        if os.path.exists(_pdf):
            try:
                import pymupdf
                _doc = pymupdf.open(_pdf)
                _txt = "".join(p.get_text() for p in _doc)
                _leak = [w for w in _ids if w in _txt]
                if _leak:
                    fail(f"the submission PDF is not anonymous: {_leak}")
                if _doc.metadata.get("author"):
                    fail(f"the submission PDF's metadata names an author: {_doc.metadata.get('author')}")
            except ImportError:
                warn("pymupdf not installed -- the submission PDF's anonymity was not checked")
        else:
            warn("eswa_submission/manuscript/paper_eswa.pdf not built -- run build_pdfs.py eswa")

# The document must not depend on a .sty that is not beside it -- that is the one failure a
# reviewer or a fresh Overleaf project hits first.
for m in re.finditer(r"\\usepackage(?:\[[^\]]*\])?\{([^}]*)\}", code):
    for pkg in m.group(1).split(","):
        pkg = pkg.strip()
        if os.path.exists(os.path.join(HERE, pkg + ".sty")):
            continue
        if pkg in {"graphicx", "booktabs", "amsmath", "amssymb", "inputenc", "fontenc",
                   "microtype", "hyperref", "natbib", "caption", "geometry", "times", "titlesec", "array",
                   "cmap",
                   "enumitem"}:
            continue                      # shipped with every TeX distribution
        fail(f"\\usepackage{{{pkg}}} is neither a standard package nor a file beside paper.tex")

# pdfLaTeX with inputenc utf8 has no mapping for most non-ASCII glyphs, and the first real compile
# died on a single Greek delta that had been sitting in a table header for days. Nothing here could
# see it: it is valid UTF-8, valid Markdown, and survives every structural check above. The compiler
# found it in five seconds, which is the point -- this guard exists so the next one is found here.
#
# The allowed set is deliberately tiny. Anything outside it should be written as a macro, not as a
# character, so the source says what it means.
ALLOWED_NON_ASCII = {
    "\u2014": "em dash",
    "\u2013": "en dash",
    "\u2019": "right single quote",
    "\u201c": "left double quote",
    "\u201d": "right double quote",
    "\u00a0": "non-breaking space",
}
seen = {}
for i, line in enumerate(tex.split("\n"), 1):
    for ch in line:
        if ord(ch) > 127 and ch not in ALLOWED_NON_ASCII:
            seen.setdefault(ch, []).append(i)
for ch, where in sorted(seen.items()):
    fail(f"non-ASCII U+{ord(ch):04X} on line(s) {where[:5]} -- pdfLaTeX cannot typeset it; "
         f"write the LaTeX macro instead")

# plainnat prints the author inline, so "Adebayo et al. \cite{adebayo}" renders the name twice.
# Nine of these reached the first compiled PDF; the LaTeX is valid and the key resolves, so nothing
# here could see it until the rule was written down.
for m in re.finditer(r"([A-Z][A-Za-z'-]+(?:\s+(?:and|&)\s+[A-Z][A-Za-z'-]+)?\s+et al\.)\s*"
                     r"\\cite[tp]?\{([^}]*)\}", code):
    if m.group(0).count("citet") == 0:
        fail(f"author written out before a citation -- natbib will print it twice; use "
             f"\\citet: {m.group(0)[:60]}")

# A bare \cite in author-year mode drops the parentheses.
n_bare = len(re.findall(r"\\cite\{", code))
if n_bare:
    fail(f"{n_bare} bare \\cite{{}} -- plainnat renders these without parentheses; "
         f"use \\citep for parenthetical and \\citet for in-text")

if "\r\n" in open(TEX, "rb").read().decode("utf-8"):
    fail("paper.tex has CRLF line endings; TeX reports these as stray ^^M")

# A footnote that does not end a sentence was cut off. Until 2026-10-06 build_paper.py kept only the first line of a
# multi-line footnote definition, and two footnotes went out ending "appeared" and "zero".
for m in re.finditer(r"\\footnote\{", code):
    depth, k = 1, m.end()
    while depth and k < len(code):
        depth += {"{": 1, "}": -1}.get(code[k], 0)
        k += 1
    text = code[m.end():k - 1].strip()
    if not re.search(r"[.?!)]$", text):
        fail(f"footnote does not end a sentence (cut off?): ...{text[-50:]!r}")

# A typed figure or table number is a claim about layout that nothing verifies. Both of the ones
# this paper carried were wrong -- 'Figure 1' pointed at figure 4 and 'Figure 6' at figure 5 --
# because adding a figure renumbers everything after it. \\ref cannot go stale.
for m in re.finditer(r"(?<!\\)\b(Figure|Table|Section)\s+(\d+)(?![.\d])", code):
    fail(f"typed cross-reference {m.group(0)!r} -- use a \\ref, which cannot go stale "
         f"when a figure is added")

# Every \ref needs a \label to resolve against, or LaTeX prints ??.
labels = set(re.findall(r"\\label\{([^}]*)\}", code))
for m in re.finditer(r"\\ref\{([^}]*)\}", code):
    if m.group(1) not in labels:
        fail(f"\\ref{{{m.group(1)}}} has no matching \\label -- LaTeX will print ??")

# Expert Systems with Applications (2026-11-24): an abstract of at most 250 words, and every table numbered and
# captioned (the twelve tables had neither until the editorial review found it)
_abs = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S)
if _abs:
    _plain = re.sub(r"\\(textbf|emph|citet|citep)\{([^}]*)\}", r"\2", _abs.group(1))
    _words = len(re.sub(r"\\[a-zA-Z]+|[{}$]", " ", _plain).split())
    if _words > 250:
        fail(f"the abstract has {_words} words; Expert Systems with Applications allows 250")
_ntab = code.count("\\begin{tabular}")
_ncap = code.count("\\captionof{table}")
if _ncap != _ntab:
    fail(f"{_ntab} tables but {_ncap} table captions -- every table needs a 'Table: caption {{#tab:label}}' line")
for _lab in re.findall(r"\\captionof\{table\}\{[^\n]*?\}\\label\{([^}]+)\}", code):
    if f"\\ref{{{_lab}}}" not in code:
        fail(f"table {_lab} is never cited in the text")

# the paper states the code is public: https://github.com/Dardosh82/ablation-tables-ivif went live on 2026-11-25
# (public, MIT, 88 files = release/ at commit b71476b, checked through the GitHub API). After a change to release/,
# push the new export there too, so the repository keeps matching what the paper describes.

# ---------------------------------------------------------------------------- report
words = len(re.sub(r"\\[a-zA-Z]+|[{}$&%#_]", " ", tex).split())
print(f"paper.tex   {len(tex.splitlines())} lines, ~{words:,} tokens")
print(f"citations   {len(cited)} distinct keys, {len(defined)} bib entries")
print(f"figures     {len(used)} placed, {len(built)} built")
print(f"sections    {nsec}")
print()
for w in warns:
    print(f"  WARN  {w}")
for f in fails:
    print(f"  FAIL  {f}")
print()
if fails:
    print(f"{len(fails)} FAIL, {len(warns)} WARN")
    sys.exit(1)
print(f"0 FAIL, {len(warns)} WARN -- structurally sound.")
print("NOT CHECKED HERE (needs a LaTeX run): float placement, page count, overfull boxes,")
print("bibliography rendering. Compile on Overleaf before submitting.")

