from __future__ import annotations

import math
from io import BytesIO
from pathlib import Path
from typing import Optional, Union

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

from rfp2deck.core.schemas import DeckPlan, DiagramSpec
from rfp2deck.core.logging import get_logger

log = get_logger(__name__)

# Layout constants (legacy defaults, retained for the no-coordinate fallback path)
LEFT_MARGIN_IN = 0.75
RIGHT_MARGIN_IN = 0.75
TOP_MARGIN_IN = 0.45
BOTTOM_MARGIN_IN = 0.45

TITLE_X_IN = LEFT_MARGIN_IN
TITLE_Y_IN = TOP_MARGIN_IN
TITLE_H_IN = 0.70

# Fonts
FONT_NAME = "Calibri"          # body / headings (matches the reference deck)
FONT_NAME_MONO = "Consolas"    # code / diagram captions
FONT_TITLE_PT = 26
FONT_TITLE_MIN_PT = 16
FONT_TITLE_SLIDE_PT = 40
FONT_BODY_START_PT = 16
FONT_BODY_MIN_PT = 11

EMU_PER_INCH = 914400

# ------------------------------------------------------------------
# Brand palette
# ------------------------------------------------------------------
# Modelled on the HCLTech "Credly Modernization" proposal deck: a deep
# navy identity with a bright cyan accent, a green "new/positive" accent,
# and a pale-blue tint used for decorative dot grids on dark slides.
COLOR_PRIMARY = "1E2761"      # deep navy (header bars, dark backgrounds)
COLOR_ACCENT = "4FC3F7"       # cyan accent (rules, bullets, highlights)
COLOR_ACCENT_ALT = "27AE60"   # green accent (positive / "new" emphasis)
COLOR_TINT = "CADCFC"         # pale blue (decorative dot grid on dark slides)
COLOR_WHITE = "FFFFFF"
COLOR_BODY = "333333"         # near-black body text on white
COLOR_BODY_LIGHT = "DCE6F2"   # body text on dark backgrounds
COLOR_HEADER_TEXT = "FFFFFF"  # title text inside the navy header bar
COLOR_FOOTER = "8A93A8"       # muted slate for the footer line

FOOTER_TEXT = "HCLTech  |  Confidential"


def _theme_for(archetype: str) -> dict:
    """Return a per-archetype colour/style theme.

    Dark slides (cover + closing) sit on navy; everything else is a white
    content slide that carries a full-width navy header bar at the top.
    """
    a = (archetype or "").strip().lower()
    if a == "title":
        return {
            "kind": "title",
            "bg": COLOR_PRIMARY,
            "title": COLOR_WHITE,
            "body": COLOR_BODY_LIGHT,
            "accent": COLOR_ACCENT,
        }
    if a == "next steps":
        return {
            "kind": "section",
            "bg": COLOR_PRIMARY,
            "title": COLOR_WHITE,
            "body": COLOR_BODY_LIGHT,
            "accent": COLOR_ACCENT,
        }
    if a == "agenda":
        return {
            "kind": "agenda",
            "bg": COLOR_WHITE,
            "title": COLOR_HEADER_TEXT,
            "body": COLOR_BODY,
            "accent": COLOR_ACCENT,
        }
    return {
        "kind": "content",
        "bg": COLOR_WHITE,
        "title": COLOR_HEADER_TEXT,
        "body": COLOR_BODY,
        "accent": COLOR_ACCENT,
    }


def _set_slide_background(slide, hex_color: str) -> None:
    """Fill the whole slide background with a solid colour."""
    try:
        bg = slide.background
        fill = bg.fill
        fill.solid()
        fill.fore_color.rgb = RGBColor.from_string(hex_color)
    except Exception:
        # Background fill is cosmetic; never fail the render over it.
        pass


def _set_run_font(run, name: str = FONT_NAME) -> None:
    """Force a run's typeface (latin + complex script) so it never falls back."""
    try:
        run.font.name = name
        rPr = run._r.get_or_add_rPr()  # pylint: disable=protected-access
        for tag in ("a:latin", "a:cs"):
            el = rPr.find(qn(tag))
            if el is None:
                el = rPr.makeelement(qn(tag), {})
                rPr.append(el)
            el.set("typeface", name)
    except Exception:
        pass


def _estimate_title_lines(title: str, width_in: float, font_pt: int) -> int:
    """Estimate how many lines a bold title wraps to inside a box of width_in."""
    title = (title or "").strip()
    if not title:
        return 1
    avg_char_pt = max(1.0, font_pt * 0.52)
    chars_per_line = max(1, int((width_in * 72.0) / avg_char_pt))
    return max(1, math.ceil(len(title) / chars_per_line))


def _fit_title_font(title: str, width_in: float, max_lines: int = 2,
                    start_pt: int = FONT_TITLE_PT, min_pt: int = FONT_TITLE_MIN_PT) -> int:
    """Shrink the title font (within bounds) until it fits within max_lines."""
    for pt in range(start_pt, min_pt - 1, -1):
        if _estimate_title_lines(title, width_in, pt) <= max_lines:
            return pt
    return min_pt


def _add_rect(slide, x_in, y_in, w_in, h_in, fill_hex, *, shape=MSO_SHAPE.RECTANGLE,
              line_hex: Optional[str] = None, line_w_pt: float = 0.0):
    """Add a filled (optionally outlined) auto-shape, shadow disabled."""
    try:
        shp = slide.shapes.add_shape(shape, Inches(x_in), Inches(y_in), Inches(w_in), Inches(h_in))
        if fill_hex is None:
            shp.fill.background()
        else:
            shp.fill.solid()
            shp.fill.fore_color.rgb = RGBColor.from_string(fill_hex)
        if line_hex is None:
            shp.line.fill.background()
        else:
            shp.line.color.rgb = RGBColor.from_string(line_hex)
            shp.line.width = Pt(line_w_pt or 1.0)
        shp.shadow.inherit = False
        return shp
    except Exception:
        return None


def _dot_grid(slide, x_in: float, y_in: float, cols: int, rows: int,
              dot_in: float = 0.07, gap_in: float = 0.45, color_hex: str = COLOR_TINT) -> None:
    """Draw a decorative grid of small squares (cover-slide motif)."""
    for r in range(rows):
        for c in range(cols):
            _add_rect(
                slide,
                x_in + c * gap_in,
                y_in + r * gap_in,
                dot_in,
                dot_in,
                color_hex,
            )


# ------------------------------------------------------------------
# Header bar + footer (content slides)
# ------------------------------------------------------------------
def _draw_header(slide, prs: Presentation, title: str, theme: dict) -> float:
    """Draw the full-width navy header bar with the title, return content-top (in)."""
    w_in = float(prs.slide_width) / EMU_PER_INCH
    h_in = float(prs.slide_height) / EMU_PER_INCH

    pad = max(0.4, w_in * 0.04)
    title_w = w_in - 2 * pad

    title_pt = _fit_title_font(title, title_w, max_lines=2,
                               start_pt=FONT_TITLE_PT, min_pt=FONT_TITLE_MIN_PT)
    lines = _estimate_title_lines(title, title_w, title_pt)
    header_h = max(h_in * 0.13, 0.62) if lines <= 1 else max(h_in * 0.19, 0.92)

    # Navy bar across the full top edge.
    _add_rect(slide, 0, 0, w_in, header_h, COLOR_PRIMARY)
    # Cyan baseline rule along the bottom edge of the bar.
    rule_h = 0.05
    _add_rect(slide, 0, header_h, w_in, rule_h, theme["accent"])

    tb = slide.shapes.add_textbox(Inches(pad), Inches(0), Inches(title_w), Inches(header_h))
    tf = tb.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = title
    run.font.size = Pt(title_pt)
    run.font.bold = True
    run.font.color.rgb = RGBColor.from_string(theme["title"])
    _set_run_font(run)
    p.alignment = PP_ALIGN.LEFT

    return header_h + rule_h + h_in * 0.045


def _draw_footer(slide, prs: Presentation, page_no: Optional[int], dark: bool = False) -> float:
    """Draw the footer rule + label; return the content-bottom (in)."""
    w_in = float(prs.slide_width) / EMU_PER_INCH
    h_in = float(prs.slide_height) / EMU_PER_INCH
    pad = max(0.4, w_in * 0.04)

    footer_y = h_in - max(0.34, h_in * 0.065)
    rule_color = COLOR_ACCENT if dark else "E2E7F0"
    text_color = COLOR_BODY_LIGHT if dark else COLOR_FOOTER

    _add_rect(slide, pad, footer_y, w_in - 2 * pad, 0.012, rule_color)

    tb = slide.shapes.add_textbox(
        Inches(pad), Inches(footer_y + 0.02), Inches(w_in - 2 * pad), Inches(0.3)
    )
    tf = tb.text_frame
    tf.clear()
    tf.word_wrap = False
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = FOOTER_TEXT
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor.from_string(text_color)
    _set_run_font(run)
    p.alignment = PP_ALIGN.LEFT

    if page_no is not None:
        # Page number in its own right-aligned box (a tab in the same frame is fragile).
        pb = slide.shapes.add_textbox(
            Inches(w_in - pad - 1.0), Inches(footer_y + 0.02), Inches(1.0), Inches(0.3)
        )
        ptf = pb.text_frame
        ptf.clear()
        pp = ptf.paragraphs[0]
        prun = pp.add_run()
        prun.text = str(page_no)
        prun.font.size = Pt(9)
        prun.font.color.rgb = RGBColor.from_string(text_color)
        _set_run_font(prun)
        pp.alignment = PP_ALIGN.RIGHT

    return footer_y - h_in * 0.02


# ------------------------------------------------------------------
# Content layout (within the header/footer band)
# ------------------------------------------------------------------
def _content_box(prs: Presentation, content_top: float, content_bottom: float) -> dict:
    """Full-width body box between the header and footer."""
    w_in = float(prs.slide_width) / EMU_PER_INCH
    pad = max(0.4, w_in * 0.04)
    return {
        "body": (pad, content_top, w_in - 2 * pad, max(0.6, content_bottom - content_top)),
    }


def _split_box(prs: Presentation, content_top: float, content_bottom: float) -> dict:
    """Two-column body: image LEFT, text RIGHT, between header and footer."""
    w_in = float(prs.slide_width) / EMU_PER_INCH
    pad = max(0.4, w_in * 0.04)
    inner_w = w_in - 2 * pad
    gutter = w_in * 0.03
    img_w = (inner_w - gutter) * 0.46
    body_w = (inner_w - gutter) * 0.54
    h = max(0.6, content_bottom - content_top)
    return {
        "image": (pad, content_top, img_w, h),
        "body": (pad + img_w + gutter, content_top, body_w, h),
    }


def _style_bullet_paragraph(p, color_hex: str, bullet_hex: str, level: int = 0) -> None:
    """Apply consulting-style bullet formatting: bullet glyph, colour, spacing."""
    p.font.color.rgb = RGBColor.from_string(color_hex)
    if level <= 0:
        mar_l, indent, glyph, spc_pts = 0.30, -0.22, "▪", "600"
    else:
        mar_l, indent, glyph, spc_pts = 0.62, -0.20, "–", "300"
    try:
        pPr = p._p.get_or_add_pPr()  # pylint: disable=protected-access
        pPr.set("marL", str(int(Inches(mar_l))))
        pPr.set("indent", str(int(Inches(indent))))

        for tag in ("a:spcBef", "a:buClr", "a:buFont", "a:buChar", "a:buNone", "a:buAutoNum"):
            for el in pPr.findall(qn(tag)):
                pPr.remove(el)

        def_rpr = pPr.find(qn("a:defRPr"))

        def _place(el):
            if def_rpr is not None:
                def_rpr.addprevious(el)
            else:
                pPr.append(el)

        spc_bef = pPr.makeelement(qn("a:spcBef"), {})
        spc_val = pPr.makeelement(qn("a:spcPts"), {"val": spc_pts})
        spc_bef.append(spc_val)
        _place(spc_bef)

        bu_clr = pPr.makeelement(qn("a:buClr"), {})
        srgb = pPr.makeelement(qn("a:srgbClr"), {"val": bullet_hex})
        bu_clr.append(srgb)
        _place(bu_clr)

        bu_font = pPr.makeelement(qn("a:buFont"), {"typeface": "Arial"})
        _place(bu_font)

        bu_char = pPr.makeelement(qn("a:buChar"), {"char": glyph})
        _place(bu_char)
    except Exception:
        pass


def _fit_font_for_box(lines, w_in: float, h_in: float) -> int:
    """Compute a readable font size to avoid overflow."""
    n = max(1, len(lines))
    base = FONT_BODY_START_PT
    est_per_line = 0.26 * (base / 16.0)
    if n * est_per_line <= h_in:
        return base
    size = int(base * (h_in / max(n * est_per_line, 0.001)))
    return max(FONT_BODY_MIN_PT, min(base, size))


def _normalize_body_lines(bullets, detailed_points) -> list[tuple[int, str]]:
    """Flatten slide body into ``(level, text)`` lines."""
    lines: list[tuple[int, str]] = []
    if detailed_points:
        for point in detailed_points:
            text = (getattr(point, "text", "") or "").strip()
            if text:
                lines.append((0, text))
            for sub in getattr(point, "sub_points", []) or []:
                sub_txt = (sub or "").strip()
                if sub_txt:
                    lines.append((1, sub_txt))
        if lines:
            return lines
    for b in bullets or []:
        txt = (b or "").strip()
        if txt:
            lines.append((0, txt))
    return lines


def _add_body(
    slide,
    x_in: float,
    y_in: float,
    w_in: float,
    h_in: float,
    lines: list[tuple[int, str]],
    font_pt: int,
    color_hex: str = COLOR_BODY,
    bullet_hex: str = COLOR_ACCENT,
):
    """Render a list of ``(level, text)`` lines as styled, possibly nested bullets."""
    tb = slide.shapes.add_textbox(Inches(x_in), Inches(y_in), Inches(w_in), Inches(h_in))
    tf = tb.text_frame
    tf.clear()
    tf.word_wrap = True

    first = True
    for level, txt in lines:
        if not txt:
            continue
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        run = p.add_run()
        run.text = txt
        _set_run_font(run)
        p.level = 1 if level >= 1 else 0
        run.font.size = Pt(font_pt if level <= 0 else max(FONT_BODY_MIN_PT, font_pt - 2))
        p.alignment = PP_ALIGN.LEFT
        _style_bullet_paragraph(p, color_hex=color_hex, bullet_hex=bullet_hex, level=level)


def _place_image_contain(
    slide,
    img_source: Union[Path, bytes],
    x_in: float,
    y_in: float,
    w_in: float,
    h_in: float,
    inset_in: float = 0.0,
):
    """Place an image contained within the bounding box (no overflow)."""
    from PIL import Image

    if inset_in > 0:
        x_in = x_in + inset_in
        y_in = y_in + inset_in
        w_in = max(0.01, w_in - (2 * inset_in))
        h_in = max(0.01, h_in - (2 * inset_in))

    if isinstance(img_source, bytes):
        img = Image.open(BytesIO(img_source))
    else:
        img = Image.open(img_source)
    iw, ih = img.size
    img.close()

    box_ratio = w_in / max(h_in, 1e-6)
    img_ratio = iw / max(ih, 1e-6)

    if img_ratio > box_ratio:
        new_w = w_in
        new_h = w_in / img_ratio
    else:
        new_h = h_in
        new_w = h_in * img_ratio

    px = x_in + (w_in - new_w) / 2.0
    py = y_in + (h_in - new_h) / 2.0

    if isinstance(img_source, bytes):
        stream = BytesIO(img_source)
        stream.seek(0)
        slide.shapes.add_picture(
            stream, Inches(px), Inches(py), width=Inches(new_w), height=Inches(new_h)
        )
    else:
        slide.shapes.add_picture(
            str(img_source), Inches(px), Inches(py), width=Inches(new_w), height=Inches(new_h)
        )


def _set_speaker_notes(slide, notes: Optional[str]) -> None:
    """Write speaker notes onto the slide's notes page (best-effort)."""
    text = (notes or "").strip()
    if not text:
        return
    try:
        slide.notes_slide.notes_text_frame.text = text
    except Exception:
        pass


# ------------------------------------------------------------------
# Slide renderers
# ------------------------------------------------------------------
def _render_title_slide(slide, prs: Presentation, title: str, bullets, theme: dict) -> None:
    """Navy cover slide: left accent stripe, dot-grid motif, hero title, footer."""
    w_in = float(prs.slide_width) / EMU_PER_INCH
    h_in = float(prs.slide_height) / EMU_PER_INCH

    # Left cyan stripe + top-right dot-grid decoration (cover motif).
    _add_rect(slide, 0, 0, max(0.16, w_in * 0.018), h_in, theme["accent"])
    grid_cols, grid_rows = 6, 4
    gap = 0.45
    grid_w = grid_cols * gap
    _dot_grid(slide, w_in - grid_w - 0.25, h_in * 0.09, grid_cols, grid_rows, gap_in=gap)

    lm = max(0.7, w_in * 0.06)
    box_w = w_in - lm - max(0.7, w_in * 0.06)

    title_pt = _fit_title_font(title, box_w, max_lines=3,
                               start_pt=FONT_TITLE_SLIDE_PT, min_pt=FONT_TITLE_SLIDE_PT - 12)

    tb = slide.shapes.add_textbox(Inches(lm), Inches(h_in * 0.30), Inches(box_w), Inches(h_in * 0.30))
    tf = tb.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = title
    run.font.size = Pt(title_pt)
    run.font.bold = True
    run.font.color.rgb = RGBColor.from_string(theme["title"])
    _set_run_font(run)
    p.alignment = PP_ALIGN.LEFT

    _add_rect(slide, lm, h_in * 0.62, min(box_w, 2.6), 0.06, theme["accent"])

    subtitle = next((b.strip() for b in (bullets or []) if (b or "").strip()), "")
    if subtitle:
        sb = slide.shapes.add_textbox(Inches(lm), Inches(h_in * 0.66), Inches(box_w), Inches(h_in * 0.16))
        stf = sb.text_frame
        stf.clear()
        stf.word_wrap = True
        sp = stf.paragraphs[0]
        srun = sp.add_run()
        srun.text = subtitle
        srun.font.size = Pt(16)
        srun.font.color.rgb = RGBColor.from_string(theme["body"])
        _set_run_font(srun)
        sp.alignment = PP_ALIGN.LEFT

    # Footer label on the cover.
    fb = slide.shapes.add_textbox(Inches(lm), Inches(h_in - 0.45), Inches(box_w), Inches(0.3))
    ftf = fb.text_frame
    ftf.clear()
    fp = ftf.paragraphs[0]
    frun = fp.add_run()
    frun.text = FOOTER_TEXT
    frun.font.size = Pt(10)
    frun.font.color.rgb = RGBColor.from_string(theme["body"])
    _set_run_font(frun)
    fp.alignment = PP_ALIGN.LEFT


def _render_section_slide(slide, prs: Presentation, title: str, bullets, theme: dict,
                          detailed_points=None, page_no: Optional[int] = None) -> None:
    """Dark closing/section slide: navy bg, dot grid, white title, light bullets."""
    w_in = float(prs.slide_width) / EMU_PER_INCH
    h_in = float(prs.slide_height) / EMU_PER_INCH

    _add_rect(slide, 0, 0, max(0.16, w_in * 0.018), h_in, theme["accent"])
    grid_cols, grid_rows = 5, 3
    gap = 0.45
    _dot_grid(slide, w_in - grid_cols * gap - 0.25, h_in * 0.10, grid_cols, grid_rows, gap_in=gap)

    lm = max(0.7, w_in * 0.06)
    box_w = w_in - 2 * lm

    title_pt = _fit_title_font(title, box_w, max_lines=2, start_pt=32, min_pt=22)
    tb = slide.shapes.add_textbox(Inches(lm), Inches(h_in * 0.16), Inches(box_w), Inches(h_in * 0.22))
    tf = tb.text_frame
    tf.clear()
    tf.word_wrap = True
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = title
    run.font.size = Pt(title_pt)
    run.font.bold = True
    run.font.color.rgb = RGBColor.from_string(theme["title"])
    _set_run_font(run)

    _add_rect(slide, lm, h_in * 0.40, min(box_w, 2.2), 0.05, theme["accent"])

    body_lines = _normalize_body_lines(bullets, detailed_points)
    if body_lines:
        by = h_in * 0.46
        bh = h_in - by - 0.5
        font_pt = _fit_font_for_box(body_lines, box_w, bh)
        _add_body(slide, lm, by, box_w, bh, body_lines, font_pt,
                  color_hex=theme["body"], bullet_hex=theme["accent"])


def _render_agenda_slide(slide, prs: Presentation, title: str, bullets, theme: dict,
                         detailed_points=None, page_no: Optional[int] = None) -> None:
    """Agenda slide: numbered navy chips down the left, item text beside each."""
    content_top = _draw_header(slide, prs, title or "Agenda", theme)
    content_bottom = _draw_footer(slide, prs, page_no)

    w_in = float(prs.slide_width) / EMU_PER_INCH
    pad = max(0.4, w_in * 0.04)

    items = [t for lvl, t in _normalize_body_lines(bullets, detailed_points) if lvl == 0] or \
            [t for _, t in _normalize_body_lines(bullets, detailed_points)]
    items = [t for t in items if t]
    if not items:
        return

    n = len(items)
    avail_h = content_bottom - content_top
    row_h = min(0.62, avail_h / n)
    chip = min(0.45, row_h * 0.8)
    gap = (avail_h - n * row_h) / max(1, n) if n * row_h < avail_h else 0.0
    y = content_top + max(0.0, gap / 2.0)
    font_pt = 16 if row_h > 0.5 else 13

    for i, text in enumerate(items, start=1):
        _add_rect(slide, pad, y, chip, chip, COLOR_PRIMARY, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
        # Number inside the chip.
        nb = slide.shapes.add_textbox(Inches(pad), Inches(y), Inches(chip), Inches(chip))
        ntf = nb.text_frame
        ntf.clear()
        ntf.word_wrap = False
        ntf.vertical_anchor = MSO_ANCHOR.MIDDLE
        npp = ntf.paragraphs[0]
        nrun = npp.add_run()
        nrun.text = f"{i:02d}"
        nrun.font.size = Pt(13)
        nrun.font.bold = True
        nrun.font.color.rgb = RGBColor.from_string(COLOR_WHITE)
        _set_run_font(nrun)
        npp.alignment = PP_ALIGN.CENTER
        # Item label beside the chip.
        lb = slide.shapes.add_textbox(
            Inches(pad + chip + 0.25), Inches(y), Inches(w_in - 2 * pad - chip - 0.25), Inches(chip)
        )
        ltf = lb.text_frame
        ltf.clear()
        ltf.word_wrap = True
        ltf.vertical_anchor = MSO_ANCHOR.MIDDLE
        lpp = ltf.paragraphs[0]
        lrun = lpp.add_run()
        lrun.text = text
        lrun.font.size = Pt(font_pt)
        lrun.font.color.rgb = RGBColor.from_string(theme["body"])
        _set_run_font(lrun)
        lpp.alignment = PP_ALIGN.LEFT
        y += row_h + gap


def _render_consulting_slide(
    slide,
    prs: Presentation,
    title: str,
    bullets,
    diagram=None,
    diagram_bytes: Optional[bytes] = None,
    archetype: str = "Content",
    detailed_points=None,
    page_no: Optional[int] = None,
):
    """Render one slide, dispatching on archetype."""
    theme = _theme_for(archetype)
    _set_slide_background(slide, theme["bg"])

    if theme["kind"] == "title":
        _render_title_slide(slide, prs, title, bullets, theme)
        return
    if theme["kind"] == "section":
        _render_section_slide(slide, prs, title, bullets, theme,
                              detailed_points=detailed_points, page_no=page_no)
        return
    if theme["kind"] == "agenda":
        _render_agenda_slide(slide, prs, title, bullets, theme,
                             detailed_points=detailed_points, page_no=page_no)
        return

    # ---- Standard content slide: navy header bar + body (+ optional diagram) ----
    content_top = _draw_header(slide, prs, title, theme)
    content_bottom = _draw_footer(slide, prs, page_no)

    body_lines = _normalize_body_lines(bullets, detailed_points)

    has_diagram = bool(
        diagram
        and getattr(diagram, "approved", False)
        and (diagram_bytes is not None or getattr(diagram, "image_path", None))
    )
    image_path = getattr(diagram, "image_path", None) if diagram else None

    if has_diagram:
        box = _split_box(prs, content_top, content_bottom)
        ix, iy, iw, ih = box["image"]
        bx, by, bw, bh = box["body"]
        if diagram_bytes is not None:
            _place_image_contain(slide, diagram_bytes, ix, iy, iw, ih, inset_in=0.05)
        elif image_path is not None:
            img = Path(str(image_path))
            if img.exists():
                _place_image_contain(slide, img, ix, iy, iw, ih, inset_in=0.05)
        font_pt = _fit_font_for_box(body_lines, bw, bh)
        _add_body(slide, bx, by, bw, bh, body_lines, font_pt,
                  color_hex=theme["body"], bullet_hex=theme["accent"])
        return

    box = _content_box(prs, content_top, content_bottom)
    bx, by, bw, bh = box["body"]
    font_pt = _fit_font_for_box(body_lines, bw, bh)
    _add_body(slide, bx, by, bw, bh, body_lines, font_pt,
              color_hex=theme["body"], bullet_hex=theme["accent"])


def _find_blank_layout(prs: Presentation):
    """Pick a layout with minimal placeholders."""
    for layout in prs.slide_layouts:
        if len(getattr(layout, "placeholders", [])) == 0:
            return layout
    best = None
    best_n = 10**9
    for layout in prs.slide_layouts:
        n = len(getattr(layout, "placeholders", []))
        if n < best_n:
            best = layout
            best_n = n
    return best or prs.slide_layouts[0]


def render_deck_from_template(
    deck_plan: DeckPlan,
    template_pptx: Union[Path, bytes],
    out_path: Optional[Path] = None,
    diagram_images: Optional[dict[str, bytes]] = None,
) -> Union[Path, bytes]:
    """Render a new PPTX from a template and a deck plan."""
    n_slides = len(deck_plan.slides)
    n_images = len(diagram_images) if diagram_images else 0
    log.info("Rendering deck: %d slides, %d diagram image(s)", n_slides, n_images)
    if isinstance(template_pptx, bytes):
        prs = Presentation(BytesIO(template_pptx))
    else:
        prs = Presentation(str(template_pptx))
    blank_layout = _find_blank_layout(prs)

    # remove all existing slides
    while len(prs.slides) > 0:
        rId = prs.slides._sldIdLst[0].rId  # pylint: disable=protected-access
        prs.part.drop_rel(rId)
        del prs.slides._sldIdLst[0]  # pylint: disable=protected-access

    for idx, slide_spec in enumerate(deck_plan.slides):
        slide = prs.slides.add_slide(blank_layout)

        archetype = getattr(slide_spec, "archetype", "Content") or "Content"
        # No page number on the cover slide.
        page_no = None if (archetype or "").strip().lower() == "title" else idx + 1

        diagram_bytes = None
        if diagram_images is not None:
            diagram_bytes = diagram_images.get(slide_spec.slide_id)
        try:
            _render_consulting_slide(
                slide,
                prs,
                slide_spec.title or "",
                slide_spec.bullets or [],
                diagram=getattr(slide_spec, "diagram", None),
                diagram_bytes=diagram_bytes,
                archetype=archetype,
                detailed_points=getattr(slide_spec, "detailed_points", None),
                page_no=page_no,
            )
        except Exception:
            log.exception(
                "Failed to render slide_id=%s (title=%r)",
                getattr(slide_spec, "slide_id", "?"),
                slide_spec.title,
            )
            raise

        _set_speaker_notes(slide, getattr(slide_spec, "notes", None))

    if out_path is not None:
        prs.save(str(out_path))
        log.info("Deck rendered to %s", out_path)
        return out_path
    stream = BytesIO()
    prs.save(stream)
    data = stream.getvalue()
    log.info("Deck rendered to in-memory PPTX (%d bytes)", len(data))
    return data
