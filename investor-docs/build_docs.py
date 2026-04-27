#!/usr/bin/env python3
"""
build_docs.py — ETFT Investor Document Builder
==============================================
Converts Markdown source files in investor-docs/ into polished
investor-grade PDF and PPTX documents.

Usage
-----
    python3 investor-docs/build_docs.py              # build all documents
    python3 investor-docs/build_docs.py --file A     # build File A only
    python3 investor-docs/build_docs.py --file B     # build File B only (PDF + PPTX)
    python3 investor-docs/build_docs.py --file C     # build File C only
    python3 investor-docs/build_docs.py --no-pptx    # skip PPTX generation

Requirements
------------
    pip install weasyprint python-pptx reportlab markdown

Output
------
    investor-docs/File-A_Executive-Teaser.pdf
    investor-docs/File-B_Strategic-Deck.pdf
    investor-docs/File-B_Strategic-Deck.pptx
    investor-docs/File-C_Technical-Whitepaper-Guide.pdf
"""

import argparse
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).parent.resolve()
TEMPLATES = HERE / "templates"
CSS_FILE = TEMPLATES / "investor.css"

FILES = {
    "A": HERE / "File-A_Executive-Teaser.md",
    "B": HERE / "File-B_Strategic-Deck.md",
    "C": HERE / "File-C_Technical-Whitepaper-Guide.md",
}

# ---------------------------------------------------------------------------
# Colour palette (matches investor.css)
# ---------------------------------------------------------------------------
DARK_BG   = "0D1117"
ACCENT    = "3B6BFF"
LIGHT_BG  = "F7F8FA"
TEXT_MAIN = "1A1F2E"
TEXT_MUTED = "718096"
WHITE     = "FFFFFF"
BORDER    = "E2E8F0"

# ---------------------------------------------------------------------------
# Helper: Markdown → HTML (via markdown library)
# ---------------------------------------------------------------------------
def md_to_html(md_text: str, title: str, subtitle: str = "", confidential: bool = False) -> str:
    """Convert Markdown source to a full HTML document styled with investor.css."""
    import markdown as md_lib

    body_md = md_text

    # Convert Markdown → HTML
    extensions = ["tables", "fenced_code", "nl2br", "sane_lists"]
    body_html = md_lib.markdown(body_md, extensions=extensions)

    # Build optional confidential banner
    banner = ""
    if confidential:
        banner = (
            '<div class="notice confidential" style="margin-bottom:8mm;">'
            '<strong>CONFIDENTIAL — NDA REQUIRED</strong> &nbsp;·&nbsp; '
            'Access to this document is conditional on a signed Non-Disclosure Agreement. '
            'Unauthorised distribution constitutes a breach of contract.'
            '</div>'
        )

    # Cover block
    cover_subtitle = f'<p style="color:#8A94A6;font-size:12pt;margin:0 0 3mm 0;">{subtitle}</p>' if subtitle else ""
    cover = f"""
<div style="background:linear-gradient(145deg,#0D1117 0%,#161B27 60%,#1E2640 100%);
            color:#fff;padding:38mm 28mm 30mm 28mm;margin:-22mm -20mm 14mm -22mm;
            page-break-after:avoid;">
  <p style="color:#3B6BFF;font-size:9pt;font-weight:600;letter-spacing:0.12em;
             text-transform:uppercase;margin:0 0 5mm 0;">ETFT · AutoScientist</p>
  <h1 style="color:#fff;font-size:26pt;font-weight:700;line-height:1.15;
              letter-spacing:-0.02em;margin:0 0 4mm 0;border:none;">{title}</h1>
  {cover_subtitle}
  <p style="color:#8A94A6;font-size:9pt;margin:10mm 0 0 0;">
    Lukas Benda &nbsp;·&nbsp; lukas.benda@boldpivot.cz &nbsp;·&nbsp; April 2026
  </p>
</div>
"""

    # CSS — inline the stylesheet
    css_content = CSS_FILE.read_text(encoding="utf-8") if CSS_FILE.exists() else ""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  <style>
{css_content}
  </style>
</head>
<body>
{cover}
{banner}
{body_html}
</body>
</html>"""
    return html


# ---------------------------------------------------------------------------
# PDF builder (WeasyPrint)
# ---------------------------------------------------------------------------
def build_pdf(md_path: Path, out_path: Path, title: str, subtitle: str = "",
              confidential: bool = False) -> None:
    from weasyprint import HTML, CSS  # type: ignore

    print(f"  → Building PDF: {out_path.name} …")
    md_text = md_path.read_text(encoding="utf-8")
    html_src = md_to_html(md_text, title, subtitle, confidential)

    html_doc = HTML(string=html_src, base_url=str(HERE))
    html_doc.write_pdf(str(out_path))
    size_kb = out_path.stat().st_size // 1024
    print(f"     Done — {size_kb} KB")


# ---------------------------------------------------------------------------
# PPTX builder (File B — Strategic Deck)
# ---------------------------------------------------------------------------
def _hex_rgb(hexstr: str):
    """Return pptx RGBColor from a 6-char hex string."""
    from pptx.util import Pt  # noqa – import guard
    from pptx.dml.color import RGBColor
    h = hexstr.lstrip("#")
    return RGBColor(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _add_text_box(slide, left_pct, top_pct, width_pct, height_pct,
                  text, font_size_pt, bold=False, color_hex=TEXT_MAIN,
                  italic=False, wrap=True, alignment=None):
    from pptx.util import Inches, Pt, Emu
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN

    SLIDE_W = 10.0   # inches (widescreen)
    SLIDE_H = 5.625  # inches

    left   = Inches(SLIDE_W * left_pct)
    top    = Inches(SLIDE_H * top_pct)
    width  = Inches(SLIDE_W * width_pct)
    height = Inches(SLIDE_H * height_pct)

    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = wrap

    p = tf.paragraphs[0]
    p.text = text
    run = p.runs[0]
    run.font.size = Pt(font_size_pt)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = _hex_rgb(color_hex)

    if alignment:
        p.alignment = alignment

    return txBox


def _add_rect(slide, left_pct, top_pct, width_pct, height_pct, fill_hex, alpha=None):
    from pptx.util import Inches
    from pptx.dml.color import RGBColor
    from pptx.util import Pt

    SLIDE_W = 10.0
    SLIDE_H = 5.625

    shape = slide.shapes.add_shape(
        1,  # MSO_SHAPE_TYPE.RECTANGLE
        Inches(SLIDE_W * left_pct),
        Inches(SLIDE_H * top_pct),
        Inches(SLIDE_W * width_pct),
        Inches(SLIDE_H * height_pct),
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = _hex_rgb(fill_hex)
    shape.line.fill.background()
    return shape


def _add_line(slide, x1_pct, y1_pct, x2_pct, y2_pct, color_hex, width_pt=1.5):
    from pptx.util import Inches, Pt
    from pptx.dml.color import RGBColor

    SLIDE_W = 10.0
    SLIDE_H = 5.625

    connector = slide.shapes.add_connector(
        1,  # MSO_CONNECTOR_TYPE.STRAIGHT
        Inches(SLIDE_W * x1_pct), Inches(SLIDE_H * y1_pct),
        Inches(SLIDE_W * x2_pct), Inches(SLIDE_H * y2_pct),
    )
    connector.line.color.rgb = _hex_rgb(color_hex)
    connector.line.width = Pt(width_pt)
    return connector


def _slide_dark_header(slide, label: str, headline: str, slide_num: int, total: int):
    """Render the consistent dark header bar used on every slide."""
    # Full-width dark header band
    _add_rect(slide, 0, 0, 1.0, 0.22, DARK_BG)
    # Accent stripe
    _add_rect(slide, 0, 0.22, 1.0, 0.008, ACCENT)

    # Brand label
    _add_text_box(slide, 0.03, 0.01, 0.5, 0.08,
                  "ETFT · AutoScientist", 8, bold=False, color_hex="3B6BFF")
    # Slide counter
    from pptx.enum.text import PP_ALIGN
    _add_text_box(slide, 0.75, 0.01, 0.22, 0.08,
                  f"{slide_num:02d} / {total:02d}", 8, color_hex="8A94A6",
                  alignment=PP_ALIGN.RIGHT)
    # Headline
    _add_text_box(slide, 0.03, 0.07, 0.94, 0.14,
                  headline, 18, bold=True, color_hex=WHITE)


def _render_slide_content(slide, body_lines: list[str]):
    """Render body lines as bullet points / paragraphs."""
    from pptx.util import Inches, Pt
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN

    SLIDE_W = 10.0
    SLIDE_H = 5.625

    txBox = slide.shapes.add_textbox(
        Inches(SLIDE_W * 0.03),
        Inches(SLIDE_H * 0.28),
        Inches(SLIDE_W * 0.94),
        Inches(SLIDE_H * 0.66),
    )
    tf = txBox.text_frame
    tf.word_wrap = True

    first = True
    for raw_line in body_lines:
        line = raw_line.rstrip()
        if not line:
            continue

        # Strip markdown formatting
        is_h3 = line.startswith("### ")
        is_bullet = line.startswith(("- ", "* ", "+ "))
        is_sub_bullet = line.startswith(("   - ", "   * ", "  - ", "  * "))
        is_table_row = line.startswith("|")
        is_code = line.startswith("    ") or line.startswith("\t")

        if is_h3:
            clean = line.lstrip("#").strip()
            para = tf.paragraphs[0] if first else tf.add_paragraph()
            first = False
            run = para.add_run()
            run.text = clean
            run.font.size = Pt(13)
            run.font.bold = True
            run.font.color.rgb = _hex_rgb(ACCENT)
            para.space_before = Pt(8)
            continue

        if is_sub_bullet:
            clean = re.sub(r"^\s+[-*+]\s+", "", line).strip()
            clean = re.sub(r"\*\*(.+?)\*\*", r"\1", clean)
            clean = re.sub(r"\*(.+?)\*", r"\1", clean)
            para = tf.paragraphs[0] if first else tf.add_paragraph()
            first = False
            run = para.add_run()
            run.text = "    ◦  " + clean
            run.font.size = Pt(9.5)
            run.font.color.rgb = _hex_rgb("4A5568")
            continue

        if is_bullet:
            clean = re.sub(r"^[-*+]\s+", "", line).strip()
            clean = re.sub(r"\*\*(.+?)\*\*", r"\1", clean)
            clean = re.sub(r"\*(.+?)\*", r"\1", clean)
            para = tf.paragraphs[0] if first else tf.add_paragraph()
            first = False
            run = para.add_run()
            run.text = "  •  " + clean
            run.font.size = Pt(10.5)
            run.font.color.rgb = _hex_rgb(TEXT_MAIN)
            continue

        if is_table_row:
            clean = re.sub(r"\|", " · ", line).strip(" ·").strip()
            clean = re.sub(r"\*\*(.+?)\*\*", r"\1", clean)
            # Skip separator rows (---|--- patterns)
            if re.match(r"^[\s·\-:]+$", clean):
                continue
            para = tf.paragraphs[0] if first else tf.add_paragraph()
            first = False
            run = para.add_run()
            run.text = clean
            run.font.size = Pt(9)
            run.font.color.rgb = _hex_rgb("2D3748")
            continue

        if is_code:
            clean = line.strip()
            para = tf.paragraphs[0] if first else tf.add_paragraph()
            first = False
            run = para.add_run()
            run.text = clean
            run.font.size = Pt(8)
            run.font.name = "Courier New"
            run.font.color.rgb = _hex_rgb("C9D1D9")
            continue

        # Normal paragraph — strip markdown bold/italic
        clean = line.strip()
        clean = re.sub(r"\*\*(.+?)\*\*", r"\1", clean)
        clean = re.sub(r"\*(.+?)\*", r"\1", clean)
        clean = re.sub(r"`(.+?)`", r"\1", clean)
        clean = re.sub(r"^>+\s*", "", clean)   # blockquote prefix

        if not clean:
            continue

        para = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        run = para.add_run()
        run.text = clean
        run.font.size = Pt(10.5)
        run.font.color.rgb = _hex_rgb(TEXT_MAIN)


def _parse_slides(md_text: str) -> list[dict]:
    """
    Split the strategic deck Markdown into individual slide dicts:
      { 'label': str, 'headline': str, 'body_lines': [str] }
    """
    slides = []
    current = None

    slide_re = re.compile(r"^##\s+(SLIDE\s+\d+)\s*[—–-]\s*(.+)$", re.IGNORECASE)

    for line in md_text.splitlines():
        m = slide_re.match(line)
        if m:
            if current:
                slides.append(current)
            current = {
                "label": m.group(1).strip(),
                "headline": m.group(2).strip(),
                "body_lines": [],
            }
            continue

        if current is not None:
            # Skip lines that are purely "---" separators or the H1 preamble
            if re.match(r"^-{3,}$", line.strip()):
                continue
            # Skip bold-only key-visual lines (## Key visual:) that repeat the headline
            if re.match(r"^\*\*Key visual.*\*\*$", line.strip()):
                continue
            # Skip "Speaker note / body copy" meta-labels
            if re.match(r"^\*\*Speaker note.*\*\*$", line.strip()):
                continue
            current["body_lines"].append(line)

    if current:
        slides.append(current)

    return slides


def build_pptx(md_path: Path, out_path: Path) -> None:
    from pptx import Presentation
    from pptx.util import Inches, Pt, Emu
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN
    import pptx.oxml.ns as nsmap

    print(f"  → Building PPTX: {out_path.name} …")

    md_text = md_path.read_text(encoding="utf-8")
    slides_data = _parse_slides(md_text)
    total = len(slides_data)

    prs = Presentation()
    # Widescreen 16:9
    prs.slide_width  = Inches(10)
    prs.slide_height = Inches(5.625)

    blank_layout = prs.slide_layouts[6]  # truly blank layout

    # ── Title slide ────────────────────────────────────────────────────────
    title_slide = prs.slides.add_slide(blank_layout)
    _add_rect(title_slide, 0, 0, 1.0, 1.0, DARK_BG)
    _add_rect(title_slide, 0, 0.82, 1.0, 0.02, ACCENT)

    _add_text_box(title_slide, 0.06, 0.12, 0.88, 0.10,
                  "ETFT · AutoScientist", 11, bold=False, color_hex="3B6BFF")
    _add_text_box(title_slide, 0.06, 0.22, 0.88, 0.28,
                  "Strategic Presentation Deck", 30, bold=True, color_hex=WHITE)
    _add_text_box(title_slide, 0.06, 0.55, 0.88, 0.12,
                  "Evolutionary Trajectory Fine-Tuning — Autonomous R&D Acceleration",
                  12, bold=False, color_hex="8A94A6")
    _add_text_box(title_slide, 0.06, 0.85, 0.88, 0.10,
                  "Lukas Benda  ·  lukas.benda@boldpivot.cz  ·  April 2026",
                  9, bold=False, color_hex="8A94A6")

    # ── Content slides ──────────────────────────────────────────────────────
    for idx, s in enumerate(slides_data, start=1):
        sl = prs.slides.add_slide(blank_layout)

        # White background
        _add_rect(sl, 0, 0, 1.0, 1.0, WHITE)

        # Header
        _add_slide_dark_header_pptx(sl, s["label"], s["headline"], idx, total)

        # Body
        _render_slide_content(sl, s["body_lines"])

    # ── Thank-you / contact slide ───────────────────────────────────────────
    end_slide = prs.slides.add_slide(blank_layout)
    _add_rect(end_slide, 0, 0, 1.0, 1.0, DARK_BG)
    _add_rect(end_slide, 0, 0.82, 1.0, 0.02, ACCENT)

    _add_text_box(end_slide, 0.06, 0.25, 0.88, 0.20,
                  "Thank you.", 36, bold=True, color_hex=WHITE)
    _add_text_box(end_slide, 0.06, 0.52, 0.88, 0.12,
                  "Full technical whitepaper available under NDA upon request.",
                  11, bold=False, color_hex="8A94A6")
    _add_text_box(end_slide, 0.06, 0.68, 0.88, 0.10,
                  "lukas.benda@boldpivot.cz  ·  April 2026",
                  10, bold=False, color_hex="3B6BFF")

    prs.save(str(out_path))
    size_kb = out_path.stat().st_size // 1024
    print(f"     Done — {size_kb} KB  ({total} content slides + title + closing)")


def _add_slide_dark_header_pptx(slide, label, headline, slide_num, total):
    """Dark header band for a content slide."""
    _add_rect(slide, 0, 0, 1.0, 0.22, DARK_BG)
    _add_rect(slide, 0, 0.22, 1.0, 0.008, ACCENT)
    _add_text_box(slide, 0.03, 0.01, 0.5, 0.08,
                  "ETFT · AutoScientist", 8, color_hex="3B6BFF")
    from pptx.enum.text import PP_ALIGN
    _add_text_box(slide, 0.75, 0.01, 0.22, 0.08,
                  f"{slide_num:02d} / {total:02d}", 8, color_hex="8A94A6",
                  alignment=PP_ALIGN.RIGHT)
    _add_text_box(slide, 0.03, 0.06, 0.94, 0.15,
                  headline, 16, bold=True, color_hex=WHITE)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Build ETFT investor documents from Markdown sources.")
    parser.add_argument("--file", choices=["A", "B", "C"], help="Build only the specified file (A, B, or C).")
    parser.add_argument("--no-pptx", action="store_true", help="Skip PPTX generation for File B.")
    args = parser.parse_args()

    # Verify dependencies
    missing = []
    for pkg in ("weasyprint", "markdown", "pptx"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg if pkg != "pptx" else "python-pptx")
    if missing:
        print(f"ERROR: Missing packages: {', '.join(missing)}")
        print("Install them with:  pip install weasyprint python-pptx markdown")
        sys.exit(1)

    targets = [args.file] if args.file else ["A", "B", "C"]

    print("=" * 60)
    print("ETFT Investor Document Builder")
    print("=" * 60)

    if "A" in targets:
        print("\n[File A — Executive Teaser]")
        build_pdf(
            FILES["A"],
            HERE / "File-A_Executive-Teaser.pdf",
            title="Executive Teaser",
            subtitle="Evolutionary Trajectory Fine-Tuning · Autonomous SOTA+1 Discovery",
        )

    if "B" in targets:
        print("\n[File B — Strategic Deck]")
        build_pdf(
            FILES["B"],
            HERE / "File-B_Strategic-Deck.pdf",
            title="Strategic Presentation Deck",
            subtitle="8–12 Slide Narrative · Landscape Format",
        )
        if not args.no_pptx:
            build_pptx(
                FILES["B"],
                HERE / "File-B_Strategic-Deck.pptx",
            )

    if "C" in targets:
        print("\n[File C — Technical Whitepaper Package]")
        build_pdf(
            FILES["C"],
            HERE / "File-C_Technical-Whitepaper-Guide.pdf",
            title="Full Technical Whitepaper Package",
            subtitle="Tier C — Deep Due Diligence · NDA Gated",
            confidential=True,
        )

    print("\n" + "=" * 60)
    print("Build complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
