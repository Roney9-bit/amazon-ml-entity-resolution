#!/usr/bin/env python3
"""Create the challenge's final ZIP package from the working repository."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--team-name", required=True)
    parser.add_argument("--documentation", default="Documentation_template.md")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    output_zip = Path(args.output) if args.output else root / f"{args.team_name}_submission.zip"
    staging = root / f".submission_{args.team_name}"
    if staging.exists():
        shutil.rmtree(staging)

    (staging / "output").mkdir(parents=True)
    (staging / "code" / "business_entity_resolution").mkdir(parents=True)

    required_outputs = [root / "output" / "matching_results.tsv", root / "output" / "candidate_pairs.tsv"]
    for src in required_outputs:
        if not src.exists():
            raise SystemExit(f"Missing required output: {src}. Run scripts/predict.py first.")
        shutil.copy2(src, staging / "output" / src.name)

    src_root = root / "src" / "business_entity_resolution"
    shutil.copytree(src_root, staging / "code" / "business_entity_resolution" / "src")
    shutil.copy2(root / "README.md", staging / "code" / "business_entity_resolution" / "README.md")
    shutil.copy2(root / "requirements.txt", staging / "code" / "business_entity_resolution" / "requirements.txt")

    doc = Path(args.documentation)
    if not doc.is_absolute():
        doc = root / doc
    if not doc.exists():
        fallback = root / "docs" / "methodology.md"
        if fallback.exists():
            doc = fallback
        else:
            raise SystemExit(f"Documentation file not found: {doc}")
    shutil.copy2(doc, staging / "Documentation_template.md")

    if output_zip.exists():
        output_zip.unlink()
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output_zip, "w", ZIP_DEFLATED) as zf:
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(staging))

    shutil.rmtree(staging)
    print(f"Created: {output_zip}")


if __name__ == "__main__":
    main()
