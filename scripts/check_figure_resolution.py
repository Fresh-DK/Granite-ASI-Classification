"""Audit PNG dimensions and embedded DPI metadata under results/."""

from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = RESULTS / "figure_resolution_audit.csv"
MINIMUM_DPI = 990.0  # tolerance for PNG's pixels-per-metre conversion


def main() -> None:
    Image.MAX_IMAGE_PIXELS = None
    rows = []
    for path in sorted(RESULTS.rglob("*.png")):
        with Image.open(path) as image:
            dpi = image.info.get("dpi")
            dpi_x = float(dpi[0]) if dpi else None
            dpi_y = float(dpi[1]) if dpi else None
            width, height = image.size
        passed = (
            dpi_x is not None
            and dpi_y is not None
            and dpi_x >= MINIMUM_DPI
            and dpi_y >= MINIMUM_DPI
        )
        rows.append(
            {
                "relative_path": path.relative_to(ROOT).as_posix(),
                "width_px": width,
                "height_px": height,
                "dpi_x": dpi_x,
                "dpi_y": dpi_y,
                "file_size_bytes": path.stat().st_size,
                "passes_1000_dpi_audit": passed,
            }
        )

    with OUTPUT.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    failures = [row for row in rows if not row["passes_1000_dpi_audit"]]
    print(f"Audited PNG files: {len(rows)}")
    print(f"Files below tolerance: {len(failures)}")
    print(f"Audit table: {OUTPUT}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
