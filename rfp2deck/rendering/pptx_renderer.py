from __future__ import annotations

import math
from io import BytesIO
from pathlib import Path
from typing import Optional, Union

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

from rfp2deck.core.schemas import DeckPlan, DiagramSpec
from rfp2deck.core.logging import get_logger

log = get_logger(__name__)

# Layout constants (legacy defaults)
LEFT_MARGIN_IN = 0.75
RIGHT_MARGIN_IN = 0.75
TOP_MARGIN_IN = 0.45
BOTTOM_MARGIN_IN = 0.45

TITLE_X_IN = LEFT_MARGIN_IN
TITLE_Y_IN = TOP_MARGIN_IN
TITLE_H_IN = 0.70

CONTENT_TOP_Y_IN = 1.25
DIAGRAM_Y_IN = 1.35
DIAGRAM_H_IN = 3.55

BULLETS_Y_IN = 5.05
BULLETS_H_IN = 2.10

# Fonts
FONT_TITLE_PT = 30
FONT_TITLE_MIN_PT = 20
FONT_TITLE_SLIDE_PT = 40
FONT_BODY_START_PT = 18
FONT_BODY_MIN_PT = 12

EMU_PER_INCH = 914400

# --------------------------
# Consulting colour palette
# --------------------------
# A simple, professional "navy + accent" identity. The deck uses a sandwich
# layout: dark title/closing slides, white content slides with navy headlines
# and a thin accent rule under each title.
COLOR_PRIMARY = "1F2A44"   # deep navy (titles, dark backgrounds)
COLOR_ACCENT = "2E75B6"    # accent blue (rules, bullets, section slides)
COLOR_WHITE = "FFFFFF"
COLOR_BODY = "333333"      # near-black body text on white
COLOR_BODY_LIGHT = "DCE6F2"  # body text on dark backgrounds


def _theme_for(archetype: str) -> dict:
    """Return a per-archetype colour/style theme.

    The "sandwich" effect: Title and Next Steps are dark navy slides; every
    other slide is white with a navy headline and accent rule.
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
    return {
        "kind": "content",
        "bg": COLOR_WHITE,
        "title": COLOR_PRIMARY,
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


def _estimate_title_lines(title: str, width_in: float, font_pt: int) -> int:
    """Estimate how many lines a bold title wraps to inside a box of width_in."""
    title = (title or "").strip()
    if not title:
        return 1
    # Bold ~average glyph advance ≈ 0.55 * font size (points). Convert width to pt.
    avg_char_pt = max(1.0, font_pt * 0.55)
    chars_per_line = max(1, int((width_in * 72.0) / avg_char_pt))
    return max(1, math.ceil(len(title) / chars_per_line))


def _fit_title_font(title: str, width_in: float, max_lines: int = 2) -> int:
    """Shrink the title font (within bounds) until it fits within max_lines."""
    for pt in range(FONT_TITLE_PT, FONT_TITLE_MIN_PT - 1, -1):
        if _estimate_title_lines(title, width_in, pt) <= max_lines:
            return pt
    return FONT_TITLE_MIN_PT


def _layout(prs: Presentation, title: str = "") -> dict:
    """Compute non-overlapping layout boxes sized from the actual slide size.

    This prevents text/image overlap when templates deviate slightly from 16:9.
    We keep a consistent hierarchy: title at top, large diagram in middle,
    bullets at bottom. The title height is computed from the *actual* wrapped
    line count so long headlines never crash into the diagram below.
    """
    w_in = float(prs.slide_width) / EMU_PER_INCH
    h_in = float(prs.slide_height) / EMU_PER_INCH

    lm = max(0.55, w_in * 0.06)
    rm = lm
    tm = max(0.35, h_in * 0.06)
    bm = max(0.35, h_in * 0.06)

    title_x = lm
    title_y = tm
    title_w = w_in - lm - rm

    # Line-aware title sizing: fit the font, then size the box to the lines.
    title_pt = _fit_title_font(title, title_w, max_lines=2)
    title_lines = _estimate_title_lines(title, title_w, title_pt)
    line_h_in = (title_pt / 72.0) * 1.25
    title_h = max(0.6, title_lines * line_h_in + 0.12)

    # Thin accent rule sits just under the title block.
    accent_y = title_y + title_h + (h_in * 0.012)
    accent_h = 0.045

    diag_x = lm
    diag_y = accent_y + accent_h + (h_in * 0.025)
    diag_w = title_w
    diag_h = h_in * 0.48  # bigger, consistent diagrams

    bullets_x = lm
    bullets_y = diag_y + diag_h + (h_in * 0.03)
    bullets_w = title_w
    bullets_h = max(h_in - bullets_y - bm, h_in * 0.18)

    return {
        "title": (title_x, title_y, title_w, title_h),
        "title_pt": title_pt,
        "accent": (title_x, accent_y, min(title_w, 2.2), accent_h),
        "diagram": (diag_x, diag_y, diag_w, diag_h),
        "bullets": (bullets_x, bullets_y, bullets_w, bullets_h),
    }


def _layout_split(prs: Presentation, title: str) -> dict:
    """Two-column layout: image on the LEFT, body text on the RIGHT.

    This is the conventional consulting/PowerPoint arrangement and replaces the
    older stacked (image-on-top, text-below) layout for slides that carry a
    diagram. The title spans the full width across the top; the content region
    below splits into an image column and a text column separated by a gutter.
    """
    w_in = float(prs.slide_width) / EMU_PER_INCH
    h_in = float(prs.slide_height) / EMU_PER_INCH

    lm = max(0.55, w_in * 0.06)
    rm = lm
    tm = max(0.35, h_in * 0.06)
    bm = max(0.35, h_in * 0.06)

    title_x = lm
    title_y = tm
    title_w = w_in - lm - rm

    title_pt = _fit_title_font(title, title_w, max_lines=2)
    title_lines = _estimate_title_lines(title, title_w, title_pt)
    line_h_in = (title_pt / 72.0) * 1.25
    title_h = max(0.6, title_lines * line_h_in + 0.12)

    accent_y = title_y + title_h + (h_in * 0.012)
    accent_h = 0.045

    content_y = accent_y + accent_h + (h_in * 0.03)
    content_h = max(h_in - content_y - bm, h_in * 0.3)

    gutter = w_in * 0.035
    # Image column slightly narrower than the text column so copy has room.
    img_w = (title_w - gutter) * 0.46
    body_w = (title_w - gutter) * 0.54

    img_x = lm
    body_x = lm + img_w + gutter

    return {
        "title": (title_x, title_y, title_w, title_h),
        "title_pt": title_pt,
        "accent": (title_x, accent_y, min(title_w, 2.2), accent_h),
        "image": (img_x, content_y, img_w, content_h),
        "body": (body_x, content_y, body_w, content_h),
    }


def _find_blank_layout(prs: Presentation):
    """Pick a layout with minimal placeholders."""
    # Prefer a truly blank layout if present
    for layout in prs.slide_layouts:
        if len(getattr(layout, "placeholders", [])) == 0:
            return layout
    # Otherwise pick the one with the fewest placeholders
    best = None
    best_n = 10**9
    for layout in prs.slide_layouts:
        n = len(getattr(layout, "placeholders", []))
        if n < best_n:
            best = layout
            best_n = n
    return best or prs.slide_layouts[0]


def _clear_text_on_slide(slide):
    """Clear all text frames on a slide (for template-clean rendering)."""
    for shape in slide.shapes:
        try:
            if getattr(shape, "has_text_frame", False):
                shape.text_frame.clear()
        except Exception:
            continue


def _remove_marker_shapes(slide):
    """Remove/clear template marker artifacts so they don't show in edit mode.

    We remove shapes that look like template guidance blocks ({{...}}) or
    PowerPoint placeholder guidance like 'Click to add...'.
    """
    to_remove = []
    for shape in slide.shapes:
        try:
            if getattr(shape, "has_text_frame", False):
                txt = (shape.text_frame.text or "").strip()
                upper = txt.upper()
                if (
                    ("{{" in txt and "}}" in txt)
                    or upper.startswith("TEMPLATE:")
                    or upper.startswith("CLICK TO ADD")
                    or upper in {"TITLE", "SUBTITLE", "CONTENT"}
                ):
                    to_remove.append(shape)
        except Exception:
            continue

    for shape in to_remove:
        try:
            slide.shapes._spTree.remove(shape._element)  # pylint: disable=protected-access
        except Exception:
            try:
                shape.text_frame.clear()
            except Exception:
                pass


def _add_title(
    slide,
    prs: Presentation,
    title: str,
    x_in: float | None = None,
    y_in: float | None = None,
    w_in: float | None = None,
    h_in: float | None = None,
    font_pt: int = FONT_TITLE_PT,
    color_hex: str = COLOR_PRIMARY,
):
    """Add a title box at the top of the slide.

    If coordinates are not provided, we fall back to the legacy constants.
    """
    if x_in is None or y_in is None or w_in is None or h_in is None:
        w = prs.slide_width / 914400.0
        box_w = w - LEFT_MARGIN_IN - RIGHT_MARGIN_IN
        x_in = TITLE_X_IN
        y_in = TITLE_Y_IN
        w_in = box_w
        h_in = TITLE_H_IN

    tb = slide.shapes.add_textbox(Inches(x_in), Inches(y_in), Inches(w_in), Inches(h_in))
    tf = tb.text_frame
    tf.clear()
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(font_pt)
    p.font.bold = True
    p.font.color.rgb = RGBColor.from_string(color_hex)
    p.alignment = PP_ALIGN.LEFT


def _add_accent_bar(slide, x_in: float, y_in: float, w_in: float, h_in: float, color_hex: str) -> None:
    """Draw a thin filled accent rule (a small rectangle) under the title."""
    from pptx.enum.shapes import MSO_SHAPE

    try:
        shp = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, Inches(x_in), Inches(y_in), Inches(w_in), Inches(h_in)
        )
        shp.fill.solid()
        shp.fill.fore_color.rgb = RGBColor.from_string(color_hex)
        shp.line.fill.background()
        shp.shadow.inherit = False
    except Exception:
        pass


def _style_bullet_paragraph(p, color_hex: str, bullet_hex: str, level: int = 0) -> None:
    """Apply consulting-style bullet formatting: bullet glyph, colour, spacing.

    Uses native DrawingML bullet elements (buChar) plus a hanging indent so
    wrapped lines align under the text, not under the bullet.

    `level` 0 = top-level point (filled square glyph). `level` 1 = sub-point
    (smaller en-dash glyph, indented further) used for nested detail.
    """
    p.font.color.rgb = RGBColor.from_string(color_hex)
    # Per-level indentation (in inches) and glyph.
    if level <= 0:
        mar_l, indent, glyph, spc_pts = 0.30, -0.22, "▪", "600"
    else:
        mar_l, indent, glyph, spc_pts = 0.62, -0.20, "–", "300"
    try:
        pPr = p._p.get_or_add_pPr()  # pylint: disable=protected-access
        pPr.set("marL", str(int(Inches(mar_l))))
        pPr.set("indent", str(int(Inches(indent))))

        # Remove any pre-existing bullet/spacing elements we are about to set.
        for tag in ("a:spcBef", "a:buClr", "a:buFont", "a:buChar", "a:buNone", "a:buAutoNum"):
            for el in pPr.findall(qn(tag)):
                pPr.remove(el)

        # CT_TextParagraphProperties enforces child order: spcBef -> buClr ->
        # buFont -> buChar -> ... -> defRPr (last). p.font.* may have already
        # created defRPr, so insert our elements *before* it to stay valid.
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
        # If XML styling fails for any reason, the plain paragraph still renders.
        pass


def _fit_font_for_box(lines, w_in: float, h_in: float) -> int:
    """Compute a readable font size to avoid overflow."""
    # very rough heuristic: more lines => smaller font
    n = max(1, len(lines))
    # scale by available vertical space
    base = FONT_BODY_START_PT
    # approximate: each bullet paragraph consumes ~0.26 inches at 18pt
    est_per_line = 0.26 * (base / 18.0)
    if n * est_per_line <= h_in:
        return base
    # shrink
    size = int(base * (h_in / max(n * est_per_line, 0.001)))
    return max(FONT_BODY_MIN_PT, min(base, size))


def _add_bullets(
    slide,
    x_in: float,
    y_in: float,
    w_in: float,
    h_in: float,
    bullets,
    font_pt: int,
    color_hex: str = COLOR_BODY,
    bullet_hex: str = COLOR_ACCENT,
):
    """Add styled bullets in a bounded box."""
    tb = slide.shapes.add_textbox(Inches(x_in), Inches(y_in), Inches(w_in), Inches(h_in))
    tf = tb.text_frame
    tf.clear()
    tf.word_wrap = True

    first = True
    for b in bullets:
        txt = (b or "").strip()
        if not txt:
            continue
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.text = txt
        p.level = 0
        p.font.size = Pt(font_pt)
        p.alignment = PP_ALIGN.LEFT
        _style_bullet_paragraph(p, color_hex=color_hex, bullet_hex=bullet_hex)


def _normalize_body_lines(bullets, detailed_points) -> list[tuple[int, str]]:
    """Flatten slide body into ``(level, text)`` lines.

    Prefers structured ``detailed_points`` (headline + sub-points). Falls back
    to flat ``bullets`` when no detailed points are present.
    """
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
        p.text = txt
        p.level = 1 if level >= 1 else 0
        # Sub-points render a little smaller for clear visual hierarchy.
        p.font.size = Pt(font_pt if level <= 0 else max(FONT_BODY_MIN_PT, font_pt - 2))
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
        # wider: fit width
        new_w = w_in
        new_h = w_in / img_ratio
    else:
        # taller: fit height
        new_h = h_in
        new_w = h_in * img_ratio

    px = x_in + (w_in - new_w) / 2.0
    py = y_in + (h_in - new_h) / 2.0

    if isinstance(img_source, bytes):
        stream = BytesIO(img_source)
        stream.seek(0)
        slide.shapes.add_picture(
            stream,
            Inches(px),
            Inches(py),
            width=Inches(new_w),
            height=Inches(new_h),
        )
    else:
        slide.shapes.add_picture(
            str(img_source),
            Inches(px),
            Inches(py),
            width=Inches(new_w),
            height=Inches(new_h),
        )


def _set_speaker_notes(slide, notes: Optional[str]) -> None:
    """Write speaker notes onto the slide's notes page (best-effort)."""
    text = (notes or "").strip()
    if not text:
        return
    try:
        slide.notes_slide.notes_text_frame.text = text
    except Exception:
        # Notes are an enhancement; never fail the render over them.
        pass


def _render_title_slide(slide, prs: Presentation, title: str, bullets, theme: dict) -> None:
    """Centered, large hero title on a dark background (deck cover)."""
    w_in = float(prs.slide_width) / EMU_PER_INCH
    h_in = float(prs.slide_height) / EMU_PER_INCH
    lm = max(0.7, w_in * 0.08)
    box_w = w_in - 2 * lm

    title_pt = _fit_title_font(title, box_w, max_lines=3)
    title_pt = max(title_pt, FONT_TITLE_SLIDE_PT - 8)
    title_pt = min(title_pt, FONT_TITLE_SLIDE_PT)

    # Title block, vertically centered in the upper-middle of the slide.
    tb = slide.shapes.add_textbox(Inches(lm), Inches(h_in * 0.32), Inches(box_w), Inches(h_in * 0.30))
    tf = tb.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(title_pt)
    p.font.bold = True
    p.font.color.rgb = RGBColor.from_string(theme["title"])
    p.alignment = PP_ALIGN.LEFT

    # Accent rule beneath the title.
    _add_accent_bar(slide, lm, h_in * 0.64, min(box_w, 2.6), 0.06, theme["accent"])

    # Optional subtitle: first non-empty bullet (e.g., customer / tagline).
    subtitle = next((b.strip() for b in (bullets or []) if (b or "").strip()), "")
    if subtitle:
        sb = slide.shapes.add_textbox(
            Inches(lm), Inches(h_in * 0.68), Inches(box_w), Inches(h_in * 0.16)
        )
        stf = sb.text_frame
        stf.clear()
        stf.word_wrap = True
        sp = stf.paragraphs[0]
        sp.text = subtitle
        sp.font.size = Pt(18)
        sp.font.color.rgb = RGBColor.from_string(theme["body"])
        sp.alignment = PP_ALIGN.LEFT


def _render_consulting_slide(
    slide,
    prs: Presentation,
    title: str,
    bullets,
    diagram=None,
    diagram_bytes: Optional[bytes] = None,
    archetype: str = "Content",
    detailed_points=None,
):
    """Render a content slide.

    When a diagram is present the slide uses a two-column layout — image on the
    LEFT, body text on the RIGHT (the conventional PowerPoint arrangement).
    Without a diagram, the body uses the full content width. Body content is
    drawn from ``detailed_points`` (nested headline + sub-points) when present,
    otherwise from the flat ``bullets`` list.
    """
    # clear existing template artifacts
    _clear_text_on_slide(slide)
    _remove_marker_shapes(slide)

    theme = _theme_for(archetype)
    _set_slide_background(slide, theme["bg"])

    # Dedicated cover layout for the title slide.
    if theme["kind"] == "title":
        _render_title_slide(slide, prs, title, bullets, theme)
        return

    body_lines = _normalize_body_lines(bullets, detailed_points)

    # diagram if approved
    has_diagram = bool(
        diagram
        and getattr(diagram, "approved", False)
        and (diagram_bytes is not None or getattr(diagram, "image_path", None))
    )
    image_path = getattr(diagram, "image_path", None) if diagram else None

    if has_diagram:
        layout = _layout_split(prs, title)
        tx, ty, tw, th = layout["title"]
        ax, ay, aw, ah = layout["accent"]
        ix, iy, iw, ih = layout["image"]
        bx, by, bw, bh = layout["body"]
        title_pt = layout["title_pt"]

        _add_title(
            slide, prs, title, x_in=tx, y_in=ty, w_in=tw, h_in=th,
            font_pt=title_pt, color_hex=theme["title"],
        )
        _add_accent_bar(slide, ax, ay, aw, ah, theme["accent"])

        # Image on the LEFT.
        if diagram_bytes is not None:
            _place_image_contain(slide, diagram_bytes, ix, iy, iw, ih, inset_in=0.05)
        elif image_path is not None:
            img = Path(str(image_path))
            if img.exists():
                _place_image_contain(slide, img, ix, iy, iw, ih, inset_in=0.05)

        # Body text on the RIGHT, vertically using the full content height.
        font_pt = _fit_font_for_box(body_lines, bw, bh)
        _add_body(slide, bx, by, bw, bh, body_lines, font_pt,
                  color_hex=theme["body"], bullet_hex=theme["accent"])
        return

    # No diagram => full-width body under the title.
    layout = _layout(prs, title)
    tx, ty, tw, th = layout["title"]
    ax, ay, aw, ah = layout["accent"]
    dx, dy, dw, dh = layout["diagram"]
    bx, by, bw, bh = layout["bullets"]
    title_pt = layout["title_pt"]

    _add_title(
        slide, prs, title, x_in=tx, y_in=ty, w_in=tw, h_in=th,
        font_pt=title_pt, color_hex=theme["title"],
    )
    _add_accent_bar(slide, ax, ay, aw, ah, theme["accent"])

    body_y = dy
    body_h = (by + bh) - dy
    font_pt = _fit_font_for_box(body_lines, dw, body_h)
    _add_body(slide, dx, body_y, dw, body_h, body_lines, font_pt,
              color_hex=theme["body"], bullet_hex=theme["accent"])


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

    for slide_spec in deck_plan.slides:
        slide = prs.slides.add_slide(blank_layout)

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
                archetype=getattr(slide_spec, "archetype", "Content") or "Content",
                detailed_points=getattr(slide_spec, "detailed_points", None),
            )
        except Exception:
            log.exception(
                "Failed to render slide_id=%s (title=%r)",
                getattr(slide_spec, "slide_id", "?"),
                slide_spec.title,
            )
            raise

        # Speaker notes for the presenter.
        _set_speaker_notes(slide, getattr(slide_spec, "notes", None))

        # Final cleanup: remove any leftover template guidance/markers so edit mode stays clean.
        _remove_marker_shapes(slide)

    if out_path is not None:
        prs.save(str(out_path))
        log.info("Deck rendered to %s", out_path)
        return out_path
    stream = BytesIO()
    prs.save(stream)
    data = stream.getvalue()
    log.info("Deck rendered to in-memory PPTX (%d bytes)", len(data))
    return data
