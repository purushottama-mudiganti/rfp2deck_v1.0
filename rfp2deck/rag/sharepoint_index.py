from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Iterable, Optional

from rfp2deck.core.config import settings
from rfp2deck.ingestion.pptx_parser import parse_pptx
from rfp2deck.rag.indexer import build_faiss_index, chunk_text, save_index
from rfp2deck.rag.sharepoint_client import (
    download_item,
    get_access_token,
    get_auth_config,
    get_drive_id,
    get_site_id,
    walk_drive,
)


def _matches_extension(name: str, extensions: Iterable[str]) -> bool:
    suffix = Path(name).suffix.lower().lstrip(".")
    return suffix in {e.lower().lstrip(".") for e in extensions}


def build_sharepoint_index(
    *,
    site_url: str,
    folder_path: Optional[str],
    out_dir: Path,
    library_name: Optional[str],
    extensions: Iterable[str],
    max_files: Optional[int],
) -> int:
    settings.ensure_dirs()
    config = get_auth_config()
    token = get_access_token(config)
    site_id = get_site_id(site_url, token)
    drive_id = get_drive_id(site_id, token, library_name=library_name)
    items = walk_drive(drive_id, token, folder_path=folder_path)

    texts: list[str] = []
    ingested = 0
    for item in items:
        name = item.get("name", "")
        if not _matches_extension(name, extensions):
            continue
        if max_files is not None and ingested >= max_files:
            break

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / name
            download_item(drive_id, item["id"], token, tmp_path)
            parsed = parse_pptx(tmp_path)
            if not parsed.text:
                continue
            header = f"[source={name}]"
            chunks = chunk_text(parsed.text)
            texts.extend([f"{header}\n{c}" for c in chunks])
            ingested += 1

    if not texts:
        raise RuntimeError("No matching PPTX content found to index.")

    rag = build_faiss_index(texts)
    save_index(rag, out_dir)
    return ingested


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build a SharePoint-backed RAG index (device-code auth).")
    p.add_argument("--site-url", required=True, help="SharePoint site URL")
    p.add_argument("--folder-path", default="", help="Folder path within the library (optional)")
    p.add_argument(
        "--library-name",
        default=None,
        help="Library/drive name (optional; defaults to site drive)",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Output index directory (default: .data/indexes/default_rag)",
    )
    p.add_argument(
        "--extensions",
        default="pptx",
        help="Comma-separated file extensions to include (default: pptx)",
    )
    p.add_argument("--max-files", type=int, default=None, help="Optional cap on files indexed")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else settings.rag_index_dir
    extensions = [e.strip() for e in args.extensions.split(",") if e.strip()]
    count = build_sharepoint_index(
        site_url=args.site_url,
        folder_path=args.folder_path or None,
        out_dir=out_dir,
        library_name=args.library_name,
        extensions=extensions,
        max_files=args.max_files,
    )
    print(f"Indexed {count} file(s) to {out_dir}")


if __name__ == "__main__":
    main()
