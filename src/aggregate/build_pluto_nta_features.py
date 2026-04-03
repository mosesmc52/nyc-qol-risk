#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd


SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from features.pluto_nta import aggregate_pluto_to_nta, save_pluto_nta_features


DEFAULT_PLUTO_PATH = Path("./data/raw/nyc/pluto/pluto.csv")
DEFAULT_NTA_PATH = Path("./data/raw/nyc/geographies/nyc_ntas_2020.geojson")
DEFAULT_OUTPUT_PATH = Path("./data/processed/features/pluto_nta_features.parquet")


def resolve_input_path(path: str | Path, patterns: tuple[str, ...]) -> Path:
    input_path = Path(path)
    if input_path.is_file():
        return input_path
    if input_path.is_dir():
        for pattern in patterns:
            matches = sorted(input_path.glob(pattern))
            if matches:
                return matches[0]
    raise FileNotFoundError(f"Could not resolve an input file from {input_path}")


def load_pluto_data(path: str | Path) -> pd.DataFrame:
    data_path = resolve_input_path(path, ("*.parquet", "*.csv"))
    suffix = data_path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(data_path)
    return pd.read_csv(data_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate parcel-level NYC PLUTO data to NTA-level structural features."
    )
    parser.add_argument("--pluto-path", default=str(DEFAULT_PLUTO_PATH))
    parser.add_argument("--nta-path", default=str(DEFAULT_NTA_PATH))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    args = parser.parse_args()

    pluto_df = load_pluto_data(args.pluto_path)
    nta_path = resolve_input_path(args.nta_path, ("*.geojson", "*.gpkg", "*.shp"))
    nta_gdf = gpd.read_file(nta_path)

    features = aggregate_pluto_to_nta(pluto_df, nta_gdf)
    saved_path = save_pluto_nta_features(features, args.output_path)

    print(features.head().to_string(index=False))
    print("")
    print(features.describe(include="all").transpose().to_string())
    print("")
    print(f"Saved {len(features)} NTA rows to {saved_path}")


if __name__ == "__main__":
    main()
