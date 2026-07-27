from __future__ import annotations

import hashlib
import re
import zipfile
from pathlib import Path

from rfp2deck.core.logging import get_logger

log = get_logger(__name__)

POTX_PRESENTATION_CONTENT_TYPE = (
    b"application/vnd.openxmlformats-officedocument.presentationml.template.main+xml"
)
PPTX_PRESENTATION_CONTENT_TYPE = (
    b"application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"
)


def _safe_stem(path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("._")
    return stem or "template"


def _cache_name_for(source: Path) -> str:
    stat = source.stat()
    fingerprint = hashlib.sha1(
        f"{source.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8")
    ).hexdigest()[:12]
    return f"{_safe_stem(source)}.{fingerprint}.pptx"


def convert_potx_to_pptx(source: Path, cache_dir: Path) -> Path:
    """Convert a POTX package into a PPTX-compatible package for python-pptx.

    The conversion preserves the POTX internals and only changes the package
    content type for /ppt/presentation.xml. That keeps masters, layouts, media,
    examples, theme files, and placeholder definitions intact.
    """
    source = source.expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / _cache_name_for(source)
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    log.info("Converting POTX template to PPTX cache: %s -> %s", source, out_path)
    with zipfile.ZipFile(source, "r") as zin, zipfile.ZipFile(
        out_path, "w", compression=zipfile.ZIP_DEFLATED
    ) as zout:
        for info in zin.infolist():
            data = zin.read(info.filename)
            if info.filename == "[Content_Types].xml":
                data = data.replace(
                    POTX_PRESENTATION_CONTENT_TYPE,
                    PPTX_PRESENTATION_CONTENT_TYPE,
                )
            zout.writestr(info, data)
    return out_path


def resolve_pptx_template(template_path: Path, cache_dir: Path) -> Path:
    """Return a PPTX path ready for python-pptx rendering."""
    template_path = template_path.expanduser()
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    suffix = template_path.suffix.lower()
    if suffix == ".pptx":
        return template_path
    if suffix == ".potx":
        return convert_potx_to_pptx(template_path, cache_dir)
    raise ValueError(
        f"Unsupported template type {template_path.suffix!r}; use .pptx or .potx"
    )
