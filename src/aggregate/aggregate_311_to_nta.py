#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import geopandas as gpd
import pandas as pd


SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aggregate.helpers import normalize_category
from features.pluto_nta import spatially_assign_to_nta, standardize_nta_geographies


DEFAULT_INPUT_DIR = Path("./data/raw/nyc/311/by_month")
DEFAULT_NTA_PATH = Path("./data/raw/nyc/geographies/nyc_ntas_2020.geojson")
DEFAULT_OUTPUT_PATH = Path("./data/processed/features/complaints_311_nta_category.parquet")
DEFAULT_CHUNK_SIZE = 250_000
REQUIRED_COLUMNS = ["created_date", "complaint_type", "latitude", "longitude"]


def aggregate_311_to_nta(
    input_dir: str | Path,
    nta_path: str | Path,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> pd.DataFrame:
    source_dir = Path(input_dir)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {source_dir}")

    csv_files = sorted(source_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {source_dir}")

    nta_base = load_nta_geographies(nta_path)
    counts: Counter[tuple[str, str, str, str, str, str, str]] = Counter()

    for csv_path in csv_files:
        try:
            chunks = pd.read_csv(
                csv_path,
                usecols=REQUIRED_COLUMNS,
                chunksize=chunk_size,
                engine="python",
                on_bad_lines="skip",
            )
        except ValueError as exc:
            raise KeyError(
                f"Missing one or more required columns in {csv_path}: {REQUIRED_COLUMNS}"
            ) from exc

        for chunk in chunks:
            chunk_counts = aggregate_chunk(chunk, nta_base)
            counts.update(chunk_counts)

    rows = [
        {
            "complaint_date": complaint_date,
            "time_of_day": time_of_day,
            "nta_id": nta_id,
            "nta_name": nta_name,
            "borough": borough,
            "level_1": level_1,
            "level_2": level_2,
            "complaint_count": complaint_count,
        }
        for (
            complaint_date,
            time_of_day,
            nta_id,
            nta_name,
            borough,
            level_1,
            level_2,
        ), complaint_count in counts.items()
    ]

    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(
            columns=[
                "complaint_date",
                "time_of_day",
                "nta_id",
                "nta_name",
                "borough",
                "level_1",
                "level_2",
                "complaint_count",
            ]
        )

    return result.sort_values(
        ["complaint_date", "time_of_day", "nta_id", "level_1", "level_2"],
        kind="stable",
    ).reset_index(drop=True)


def load_nta_geographies(path: str | Path) -> gpd.GeoDataFrame:
    nta_path = Path(path)
    if not nta_path.is_file():
        raise FileNotFoundError(f"NTA geography file does not exist: {nta_path}")

    nta_gdf = gpd.read_file(nta_path)
    return standardize_nta_geographies(nta_gdf)


def aggregate_chunk(
    chunk: pd.DataFrame,
    nta_base: gpd.GeoDataFrame,
) -> Counter[tuple[str, str, str, str, str, str, str]]:
    prepared = prepare_311_chunk(chunk)
    if prepared.empty:
        return Counter()

    assigned = spatially_assign_to_nta(prepared, nta_base)
    if assigned.empty:
        return Counter()

    categories = pd.DataFrame(
        assigned["complaint_type"].map(normalize_category).tolist(),
        index=assigned.index,
    )
    assigned[["level_1", "level_2"]] = categories[["level_1", "level_2"]]
    grouped = (
        assigned.groupby(
            [
                "complaint_date",
                "time_of_day",
                "nta_id",
                "nta_name",
                "borough",
                "level_1",
                "level_2",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="complaint_count")
    )

    return Counter(
        {
            (
                str(row.complaint_date),
                str(row.time_of_day),
                str(row.nta_id),
                str(row.nta_name),
                str(row.borough),
                str(row.level_1),
                str(row.level_2),
            ): int(row.complaint_count)
            for row in grouped.itertuples(index=False)
        }
    )


def prepare_311_chunk(chunk: pd.DataFrame) -> gpd.GeoDataFrame:
    df = chunk.copy()
    df.columns = [column.strip().lower() for column in df.columns]

    df["created_date"] = pd.to_datetime(df["created_date"], errors="coerce")
    df["complaint_type"] = df["complaint_type"].astype("string").str.strip()
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["complaint_date"] = df["created_date"].dt.strftime("%Y-%m-%d")
    df["time_of_day"] = df["created_date"].apply(classify_time_of_day)

    valid = (
        df["created_date"].notna()
        & df["complaint_type"].notna()
        & (df["complaint_type"] != "")
        & df["latitude"].notna()
        & df["longitude"].notna()
        & (df["latitude"] != 0)
        & (df["longitude"] != 0)
    )
    if not valid.any():
        return gpd.GeoDataFrame(
            df.iloc[0:0].copy(),
            geometry=[],
            crs="EPSG:4326",
        )

    filtered = df.loc[
        valid,
        [
            "complaint_date",
            "time_of_day",
            "complaint_type",
            "latitude",
            "longitude",
        ],
    ].copy()
    return gpd.GeoDataFrame(
        filtered,
        geometry=gpd.points_from_xy(filtered["longitude"], filtered["latitude"]),
        crs="EPSG:4326",
    )


def classify_time_of_day(timestamp: pd.Timestamp) -> str:
    if pd.isna(timestamp):
        return "unknown"

    hour = timestamp.hour
    # Day is 6:00 AM inclusive through 7:59:59 PM. Night is 8:00 PM through 5:59:59 AM.
    if 6 <= hour < 20:
        return "day"
    return "night"


def save_aggregated_311(df: pd.DataFrame, output_path: str | Path) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    suffix = destination.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(destination, index=False)
    else:
        df.to_csv(destination, index=False)

    return destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate NYC 311 complaints into daily NTA-level category counts with day/night labels."
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help="Directory containing monthly 311 CSVs.",
    )
    parser.add_argument(
        "--nta-path",
        default=str(DEFAULT_NTA_PATH),
        help="Path to the NTA geography file.",
    )
    parser.add_argument(
        "--output-path",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path to write the aggregated counts (.parquet or .csv).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Number of rows to process per CSV chunk.",
    )
    args = parser.parse_args()

    aggregated = aggregate_311_to_nta(
        args.input_dir,
        args.nta_path,
        chunk_size=args.chunk_size,
    )
    saved_path = save_aggregated_311(aggregated, args.output_path)

    print(aggregated.head(20).to_string(index=False))
    print("")
    print(f"Aggregated {len(aggregated)} daily NTA/category-level/time-of-day rows.")
    print(f"Saved results to {saved_path}")


if __name__ == "__main__":
    main()
