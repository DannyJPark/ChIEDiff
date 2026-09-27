#!/usr/bin/env python3
"""Lay two existing PDF figures side by side as panels (a) and (b), as vector graphics.

    ~/anaconda3/bin/python paper/figure_build/consistency_combined/combine.py [--split 0.40]

Both sources keep their own vector content: each is placed on a new page with PyMuPDF's
show_pdf_page, scaled to the width it is given and aligned at the top, and the panel letters are
drawn as text. Nothing is rasterised, so the result is still a vector PDF for the manuscript.

--split is the fraction of the total width given to panel (a); the rest, minus the gutter, goes to
(b). The combined page is \\textwidth (372 pt) wide, which is what main.tex asks for.
"""
import argparse
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SRC = ROOT / "paper" / "figure_build" / "figures" / "panels"
A_DEFAULT = SRC / "fig2a_rsg_ecdf_vinardo.pdf"
B_DEFAULT = SRC / "fig2c_pose_energy_map_gnina.pdf"
TEXTWIDTH = 372.0          # sn-jnl \textwidth, measured
GUTTER = 8.0
LABEL_H = 15.0             # room above the panels for the (a)/(b) letters and titles
LABEL_SIZE = 10.0
TITLE_SIZE = 9.0           # shrunk per panel if the title would not fit its width
TITLES = ("Vinardo Redocking Score Gap", "GNINA Pose-Energy Consistency Map")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", type=float, default=None,
                    help="fraction of the width given to panel (a); omit to size the two panels "
                         "so they come out the same height")
    ap.add_argument("--out", default=str(HERE / "consistency_combined.pdf"))
    ap.add_argument("--a", default=str(A_DEFAULT), help="panel (a) source PDF")
    ap.add_argument("--b", default=str(B_DEFAULT), help="panel (b) source PDF")
    a = ap.parse_args()

    src_a, src_b = fitz.open(a.a), fitz.open(a.b)
    print(f"(a) {Path(a.a).name}\n(b) {Path(a.b).name}")
    ra, rb = src_a[0].rect, src_b[0].rect
    avail = TEXTWIDTH - GUTTER
    if a.split is None:
        # equal height: w = aspect * h for each, and the two widths must fill the row
        aspect_a, aspect_b = ra.width / ra.height, rb.width / rb.height
        h = avail / (aspect_a + aspect_b)
        wa, wb = aspect_a * h, aspect_b * h
        print(f"equal-height split = {wa / avail:.3f}")
    else:
        wa = avail * a.split
        wb = avail - wa
    ha, hb = ra.height * wa / ra.width, rb.height * wb / rb.width
    page_h = LABEL_H + max(ha, hb)

    out = fitz.open()
    page = out.new_page(width=TEXTWIDTH, height=page_h)
    page.show_pdf_page(fitz.Rect(0, LABEL_H, wa, LABEL_H + ha), src_a, 0)
    page.show_pdf_page(fitz.Rect(wa + GUTTER, LABEL_H, TEXTWIDTH, LABEL_H + hb), src_b, 0)
    # panel letter in bold, then the panel title beside it, shrunk to fit that panel's width
    for x, w, letter, title in ((0, wa, "(a)", TITLES[0]),
                                (wa + GUTTER, wb, "(b)", TITLES[1])):
        page.insert_text((x, LABEL_H - 4.0), letter, fontname="hebo", fontsize=LABEL_SIZE)
        lw = fitz.get_text_length(letter + " ", fontname="hebo", fontsize=LABEL_SIZE)
        size = TITLE_SIZE
        while size > 5.5 and fitz.get_text_length(title, fontname="helv",
                                                  fontsize=size) > w - lw:
            size -= 0.25
        page.insert_text((x + lw, LABEL_H - 4.0), title, fontname="helv", fontsize=size)
        print(f"  {letter} title at {size:.2f} pt "
              f"({fitz.get_text_length(title, fontname='helv', fontsize=size):.0f} of "
              f"{w - lw:.0f} pt)")
    out.save(a.out)
    print(f"{a.out}\n  page {TEXTWIDTH:.0f} x {page_h:.1f} pt "
          f"({page_h / 72:.2f} in tall at \\textwidth)")
    print(f"  (a) {wa:.0f} x {ha:.0f} pt   (b) {wb:.0f} x {hb:.0f} pt")


if __name__ == "__main__":
    main()
