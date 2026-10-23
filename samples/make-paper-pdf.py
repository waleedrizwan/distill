#!/usr/bin/env python3
"""Write samples/two-column-paper.pdf, the PDF fixture for tests and goldens.

A two-page, two-column "paper" with the furniture real papers have: a title
across both columns, a running header, page numbers, a rotated stamp in the
margin, a footnote in small print, and words hyphenated at line ends. The text
is original (written for this fixture), so the PDF can be committed.

    python3 samples/make-paper-pdf.py

Standard library only; uses the built-in Helvetica fonts, so nothing is embedded.
"""

from pathlib import Path

OUT = Path(__file__).resolve().parent / "two-column-paper.pdf"

TITLE = "Slow Fermentation Improves Crumb Structure in Sourdough Bread"
AUTHORS = "A. Baker and B. Miller, Department of Food Science"

# (heading, paragraphs). "~" marks a word hyphenated across a line break.
BODY = [
    ("Abstract", [
        "We study how fermentation time changes the crumb of sourdough bread. Loaves proofed for "
        "eighteen hours at a cool temperature had larger and more even air cells than loaves proofed "
        "for four hours at room temperature. Slow fermentation also raised acidity, which tasters "
        "preferred. The effect held across three flours and two starter cultures."
    ]),
    ("1 Introduction", [
        "Sourdough bread rises because wild yeast and lactic acid bacteria ferment the sugars in "
        "flour. The yeast makes carbon dioxide, which inflates the dough, and the bacteria make "
        "acids, which give the bread its sour taste. Bakers have long believed that a slow, cool "
        "fermentation gives a better loaf, but the claim is rarely measured.",
        "In this paper we measure it. We baked one hundred and twenty loaves under controlled "
        "conditions and scanned each crumb with a flatbed scanner. Image analysis gave the size and "
        "spacing of every air cell. We then asked a panel of tasters to rate each loaf without "
        "knowing how it was made.",
    ]),
    ("2 Methods", [
        "Each loaf used five hundred grams of flour, three hundred and fifty grams of water, ten "
        "grams of salt and one hundred grams of active starter. Half the loaves were proofed for "
        "four hours at twenty~four degrees. The other half were proofed for eighteen hours at "
        "six degrees in a refrig~erator. All loaves were baked in the same oven for forty minutes.",
        "We used three flours: a white bread flour, a whole wheat flour and a rye blend. Two starter "
        "cultures were kept on a fixed feeding schedule for a month before the study began. The "
        "crumb was scanned at six hundred dots per inch after the loaves had cooled overnight.",
    ]),
    ("3 Results", [
        "Slow fermentation produced larger air cells in every flour. The median cell area rose by a "
        "third for the white flour and by a quarter for the whole wheat flour. Cell size was also "
        "more even: the spread of cell areas fell in every case. The rye blend changed the least, "
        "which we attribute to its weaker gluten network.",
        "Tasters preferred the slowly fermented loaves in seventy percent of paired comparisons. "
        "They most often described them as tangier and more complex. Acidity measurements agreed: "
        "the slow loaves were about half a pH unit more acidic.",
    ]),
    ("4 Conclusion", [
        "A long, cool fermentation gives sourdough a more open and even crumb and a taste that most "
        "people prefer. Home bakers can get the benefit with no extra work by proofing their dough "
        "overnight in the refrigerator. Future work should test more flours and longer proofing times.",
    ]),
]
FOOTNOTE = "* Corresponding author. Supported by a grant from the Bakers Guild."

PAGE_W, PAGE_H = 612, 792
MARGIN, GUTTER = 54, 24
COL_W = (PAGE_W - 2 * MARGIN - GUTTER) / 2
SIZE, LEAD = 10, 12
CHARS = int(COL_W / (SIZE * 0.5))  # Helvetica averages about half an em per character


def esc(s):
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def wrap(text):
    """Greedy wrap to CHARS; "~" in a word forces a hyphenated line break there."""
    lines, cur = [], ""
    for word in text.split():
        if "~" in word:
            head, tail = word.split("~", 1)
            if len((cur + " " + head + "-").strip()) > CHARS:
                lines.append(cur)
                cur = ""
            lines.append((cur + " " + head + "-").strip())
            cur = tail
            continue
        cand = (cur + " " + word).strip()
        if len(cand) <= CHARS:
            cur = cand
            continue
        lines.append(cur)
        cur = word
    if cur:
        lines.append(cur)
    return lines


def layout():
    """Per page, a list of (font, size, x, y, text, rotated)."""
    pages = [[], []]
    top = PAGE_H - MARGIN
    for p in range(2):
        pages[p].append(("F1", 8, MARGIN, PAGE_H - 30, "Journal of Everyday Science, Volume 1", False))
        pages[p].append(("F1", 8, PAGE_W / 2 - 3, 28, str(p + 1), False))
        pages[p].append(("F1", 9, 30, 250, "preprint: not peer reviewed", True))
    pages[0].append(("F2", 16, MARGIN, top - 10, TITLE, False))
    pages[0].append(("F1", 11, MARGIN, top - 30, AUTHORS, False))
    pages[0].append(("F1", 7, MARGIN, 52, FOOTNOTE, False))
    columns = [(0, MARGIN, top - 60, 70), (0, MARGIN + COL_W + GUTTER, top - 60, 70),
               (1, MARGIN, top, 40), (1, MARGIN + COL_W + GUTTER, top, 40)]
    ci = 0
    page, x, y, bottom = columns[ci]
    for heading, paras in BODY:
        items = [("F2", heading)] + [("F1", line) for para in paras for line in wrap(para) + [None]]
        for font, text in items:
            if y < bottom + LEAD:
                ci += 1
                page, x, y, bottom = columns[ci]
            if text is None:
                y -= LEAD / 2  # paragraph gap
                continue
            pages[page].append((font, SIZE, x, y, text, False))
            y -= LEAD
        y -= LEAD / 2
    return pages


def build():
    pages = layout()
    objs = []

    def add(body):
        objs.append(body)
        return len(objs)

    catalog = add(None)
    pages_obj = add(None)
    f1 = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
    f2 = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>")
    kids = []
    for items in pages:
        ops = []
        for font, size, x, y, text, rotated in items:
            m = f"0 1 -1 0 {x:.2f} {y:.2f}" if rotated else f"1 0 0 1 {x:.2f} {y:.2f}"
            ops.append(f"BT /{font} {size} Tf {m} Tm ({esc(text)}) Tj ET")
        stream = "\n".join(ops).encode("latin-1")
        content = add(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
        kids.append(add(
            f"<< /Type /Page /Parent {pages_obj} 0 R /MediaBox [0 0 {PAGE_W} {PAGE_H}] "
            f"/Resources << /Font << /F1 {f1} 0 R /F2 {f2} 0 R >> >> /Contents {content} 0 R >>".encode()
        ))
    objs[catalog - 1] = f"<< /Type /Catalog /Pages {pages_obj} 0 R >>".encode()
    objs[pages_obj - 1] = (
        f"<< /Type /Pages /Kids [{' '.join(f'{k} 0 R' for k in kids)}] /Count {len(kids)} >>".encode()
    )

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root %d 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, catalog, xref)
    OUT.write_bytes(bytes(out))
    print(f"wrote {OUT.name}, {len(out)} bytes")


if __name__ == "__main__":
    build()
