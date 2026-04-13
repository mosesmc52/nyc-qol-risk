#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_INPUT_DIR = Path("./data/raw/nyc/311/by_month")
DEFAULT_CHUNK_SIZE = 250_000


def collect_unique_complaint_types(
    input_dir: str | Path, *, chunk_size: int = DEFAULT_CHUNK_SIZE
) -> list[str]:
    source_dir = Path(input_dir)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {source_dir}")

    complaint_types: set[str] = set()
    csv_files = sorted(source_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {source_dir}")

    for csv_path in csv_files:
        try:
            chunks = pd.read_csv(
                csv_path,
                usecols=["complaint_type"],
                chunksize=chunk_size,
            )
        except ValueError as exc:
            raise KeyError(f"Missing 'complaint_type' column in {csv_path}") from exc

        for chunk in chunks:
            values = chunk["complaint_type"].dropna().astype(str).str.strip()
            complaint_types.update(value for value in values if value)

    return sorted(complaint_types)


def write_complaint_types(complaint_types: list[str], output_path: str | Path) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(complaint_types) + "\n", encoding="utf-8")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect unique complaint_type values across monthly NYC 311 CSVs."
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help="Directory containing monthly 311 CSVs.",
    )
    parser.add_argument(
        "--output-path",
        default="data/processed/311/complaint_types.txt",
        help="Optional file path to write complaint types, one per line.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Number of rows to process per CSV chunk.",
    )
    args = parser.parse_args()

    complaint_types = collect_unique_complaint_types(
        args.input_dir,
        chunk_size=args.chunk_size,
    )

    for complaint_type in complaint_types:
        print(complaint_type)

    print("")
    print(f"Found {len(complaint_types)} unique complaint_type values.")

    if args.output_path:
        saved_path = write_complaint_types(complaint_types, args.output_path)
        print(f"Saved complaint types to {saved_path}")


if __name__ == "__main__":
    main()
