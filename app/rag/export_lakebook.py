from __future__ import annotations

import argparse
from pathlib import Path

from app.rag.yuque_lakebook import export_lakebook


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a Yuque .lakebook archive into inspectable Markdown files."
    )
    parser.add_argument("archive", type=Path, help="Path to the .lakebook archive.")
    parser.add_argument("--output", required=True, type=Path, help="Markdown output directory.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail instead of emitting a placeholder when a card cannot be converted.",
    )
    parser.add_argument(
        "--no-image-ocr",
        action="store_true",
        help="Do not append image OCR text to generated Markdown.",
    )
    args = parser.parse_args()

    report = export_lakebook(
        args.archive,
        args.output,
        strict=args.strict,
        include_image_ocr=not args.no_image_ocr,
    )
    card_summary = ", ".join(
        f"{name}={count}" for name, count in report.card_counts.items()
    )
    print(f"Converted {report.document_count} document(s) into {report.output_dir}")
    print(f"Cards: {card_summary or 'none'}")
    print(f"Warnings: {report.warning_count}")
    print(f"Manifest: {report.manifest_path}")
    print(f"Report: {report.report_path}")


if __name__ == "__main__":
    main()
