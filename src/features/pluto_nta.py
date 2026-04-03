from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd


NTA_ID_CANDIDATES = ("nta_id", "nta2020", "ntacode", "nta", "geoid")
NTA_NAME_CANDIDATES = ("nta_name", "ntaname", "name")
BOROUGH_CANDIDATES = ("borough", "boro_name", "boroname", "boro")
BOROUGH_CODE_MAP = {
    "BK": "Brooklyn",
    "BX": "Bronx",
    "MN": "Manhattan",
    "QN": "Queens",
    "SI": "Staten Island",
}
PLUTO_NUMERIC_COLUMNS = (
    "lotarea",
    "bldgarea",
    "resarea",
    "comarea",
    "unitsres",
    "unitstotal",
    "numbldgs",
    "yearbuilt",
    "builtfar",
    "latitude",
    "longitude",
)
MIN_VALID_YEAR_BUILT = 1800
MAX_VALID_YEAR_BUILT = datetime.now().year + 1
MAX_VALID_BUILT_FAR = 100.0


def aggregate_pluto_to_nta(
    pluto_df: pd.DataFrame, nta_gdf: gpd.GeoDataFrame
) -> pd.DataFrame:
    """
    Assign PLUTO lots to NTAs and aggregate parcel-level features to one row per NTA.
    """
    nta_base = standardize_nta_geographies(nta_gdf)
    prepared_pluto = prepare_pluto_geometry(pluto_df)
    assigned_pluto = spatially_assign_to_nta(prepared_pluto, nta_base)

    aggregated = (
        assigned_pluto.groupby(["nta_id", "nta_name", "borough"], dropna=False)
        .apply(_aggregate_single_nta)
        .reset_index()
    )

    result = nta_base[["nta_id", "nta_name", "borough"]].drop_duplicates().merge(
        aggregated,
        on=["nta_id", "nta_name", "borough"],
        how="left",
    )

    zero_fill_columns = [
        "lot_count",
        "lot_area_total",
        "bldg_area_total",
        "res_area_total",
        "com_area_total",
        "units_res_total",
        "units_total",
        "num_buildings_total",
    ]
    for column in zero_fill_columns:
        result[column] = result[column].fillna(0)

    ordered_columns = [
        "nta_id",
        "nta_name",
        "borough",
        "lot_count",
        "lot_area_total",
        "bldg_area_total",
        "res_area_total",
        "com_area_total",
        "units_res_total",
        "units_total",
        "num_buildings_total",
        "lot_area_acres",
        "res_units_per_acre",
        "bldg_sqft_per_acre",
        "buildings_per_acre",
        "pct_res_area",
        "pct_com_area",
        "pct_mixed_use_lots",
        "land_use_mode",
        "avg_year_built_weighted",
        "median_year_built",
        "pct_prewar",
        "avg_built_far_weighted",
    ]
    return result[ordered_columns].sort_values("nta_id").reset_index(drop=True)


def prepare_pluto_geometry(pluto_df: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Prepare PLUTO geometries for point-in-polygon assignment.
    """
    if isinstance(pluto_df, gpd.GeoDataFrame) and "geometry" in pluto_df:
        pluto_gdf = pluto_df.copy()
        if pluto_gdf.crs is None:
            pluto_gdf = pluto_gdf.set_crs("EPSG:4326")
        else:
            pluto_gdf = pluto_gdf.to_crs("EPSG:4326")

        geometry_types = set(pluto_gdf.geometry.geom_type.dropna().unique())
        if geometry_types & {"Polygon", "MultiPolygon"}:
            projected = pluto_gdf.to_crs("EPSG:2263")
            pluto_gdf = pluto_gdf.copy()
            pluto_gdf.geometry = projected.centroid.to_crs("EPSG:4326")

        return normalize_pluto_columns(pluto_gdf)

    normalized = normalize_pluto_columns(pluto_df.copy())
    longitude = normalized["longitude"]
    latitude = normalized["latitude"]
    valid_coords = (
        longitude.notna()
        & latitude.notna()
        & (longitude != 0)
        & (latitude != 0)
    )
    if not valid_coords.any():
        raise ValueError(
            "PLUTO data does not include usable geometry or latitude/longitude columns."
        )

    pluto_gdf = gpd.GeoDataFrame(
        normalized.loc[valid_coords].copy(),
        geometry=gpd.points_from_xy(
            normalized.loc[valid_coords, "longitude"],
            normalized.loc[valid_coords, "latitude"],
        ),
        crs="EPSG:4326",
    )
    return pluto_gdf


def spatially_assign_to_nta(
    pluto_gdf: gpd.GeoDataFrame,
    nta_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Spatially join PLUTO lots to NTA polygons using point-in-polygon assignment.
    """
    joined = gpd.sjoin(
        pluto_gdf,
        nta_gdf[["nta_id", "nta_name", "borough", "geometry"]],
        how="inner",
        predicate="within",
    )
    rename_map = {}
    for column in ("nta_id", "nta_name", "borough"):
        if column in joined.columns:
            continue
        for candidate in (f"{column}_right", f"{column}_nta", f"{column}_y"):
            if candidate in joined.columns:
                rename_map[candidate] = column
                break

    joined = joined.rename(columns=rename_map)
    required_columns = {"nta_id", "nta_name", "borough"}
    missing = required_columns - set(joined.columns)
    if missing:
        raise KeyError(
            "Spatial join did not produce the required NTA columns: "
            f"{sorted(missing)}. Available columns: {list(joined.columns)}"
        )

    drop_columns = [column for column in ("index_right",) if column in joined.columns]
    return joined.drop(columns=drop_columns)


def standardize_nta_geographies(nta_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Normalize NTA identifier columns to nta_id, nta_name, and borough.
    """
    if not isinstance(nta_gdf, gpd.GeoDataFrame):
        raise TypeError("nta_gdf must be a GeoDataFrame.")

    gdf = nta_gdf.copy()
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")

    id_column = find_column(gdf.columns, NTA_ID_CANDIDATES)
    name_column = find_column(gdf.columns, NTA_NAME_CANDIDATES)
    borough_column = find_column(gdf.columns, BOROUGH_CANDIDATES, required=False)

    gdf["nta_id"] = gdf[id_column].astype("string").str.strip()
    gdf["nta_name"] = gdf[name_column].astype("string").str.strip()

    if borough_column is not None:
        gdf["borough"] = gdf[borough_column].astype("string").str.strip()
    else:
        gdf["borough"] = gdf["nta_id"].str[:2].map(BOROUGH_CODE_MAP)

    return gdf.dropna(subset=["nta_id", "nta_name"]).copy()


def normalize_pluto_columns(pluto_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column casing and coerce expected analytical columns to numeric.
    """
    df = pluto_df.copy()
    df.columns = [column.strip().lower() for column in df.columns]

    for column in PLUTO_NUMERIC_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if "landuse" not in df.columns:
        df["landuse"] = pd.Series(pd.NA, index=df.index, dtype="object")

    if "yearbuilt" in df.columns:
        df["yearbuilt"] = df["yearbuilt"].where(
            df["yearbuilt"].between(MIN_VALID_YEAR_BUILT, MAX_VALID_YEAR_BUILT)
        )

    if "builtfar" in df.columns:
        df["builtfar"] = df["builtfar"].where(
            df["builtfar"].between(0, MAX_VALID_BUILT_FAR)
        )

    return df


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    """
    Compute a weighted mean after excluding null values and null weights.
    Weights are clipped to a minimum of 1 to avoid zero-weight degenerate cases.
    """
    valid = values.notna() & weights.notna()
    if not valid.any():
        return np.nan

    clean_values = pd.to_numeric(values.loc[valid], errors="coerce")
    clean_weights = pd.to_numeric(weights.loc[valid], errors="coerce").clip(lower=1)
    valid_numeric = clean_values.notna() & clean_weights.notna()
    if not valid_numeric.any():
        return np.nan

    return float(
        np.average(
            clean_values.loc[valid_numeric],
            weights=clean_weights.loc[valid_numeric],
        )
    )


def safe_divide(
    numerator: pd.Series | float, denominator: pd.Series | float
) -> pd.Series | float:
    """
    Divide defensively and return null where the denominator is zero or null.
    """
    if np.isscalar(numerator) and np.isscalar(denominator):
        if pd.isna(denominator) or denominator == 0:
            return np.nan
        return numerator / denominator

    numerator_series = pd.Series(numerator)
    denominator_series = pd.Series(denominator)
    result = numerator_series / denominator_series.replace({0: np.nan})
    return result.where(denominator_series.notna())


def save_pluto_nta_features(df: pd.DataFrame, path: str | Path) -> Path:
    """
    Save NTA features to parquet when supported, otherwise fall back to CSV.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".csv":
        df.to_csv(output_path, index=False)
        return output_path

    try:
        df.to_parquet(output_path, index=False)
        return output_path
    except Exception:
        fallback = output_path.with_suffix(".csv")
        df.to_csv(fallback, index=False)
        return fallback


def find_column(
    columns: Iterable[str],
    candidates: Iterable[str],
    *,
    required: bool = True,
) -> str | None:
    """
    Find a column name case-insensitively from a candidate list.
    """
    lowered = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]

    if required:
        raise KeyError(f"Missing required columns. Tried candidates: {tuple(candidates)}")
    return None


def _aggregate_single_nta(group: pd.DataFrame) -> pd.Series:
    lot_count = float(len(group))
    lot_area_total = group["lotarea"].fillna(0).sum()
    bldg_area_total = group["bldgarea"].fillna(0).sum()
    res_area_total = group["resarea"].fillna(0).sum()
    com_area_total = group["comarea"].fillna(0).sum()
    units_res_total = group["unitsres"].fillna(0).sum()
    units_total = group["unitstotal"].fillna(0).sum()
    num_buildings_total = group["numbldgs"].fillna(0).sum()
    lot_area_acres = safe_divide(lot_area_total, 43560)

    mixed_use_mask = (group["resarea"].fillna(0) > 0) & (group["comarea"].fillna(0) > 0)
    year_built = group["yearbuilt"].where(group["yearbuilt"] > 0)
    built_far = group["builtfar"].where(group["builtfar"] > 0)
    valid_year_built = year_built.dropna()

    return pd.Series(
        {
            "lot_count": int(lot_count),
            "lot_area_total": float(lot_area_total),
            "bldg_area_total": float(bldg_area_total),
            "res_area_total": float(res_area_total),
            "com_area_total": float(com_area_total),
            "units_res_total": float(units_res_total),
            "units_total": float(units_total),
            "num_buildings_total": float(num_buildings_total),
            "lot_area_acres": safe_divide(lot_area_total, 43560),
            "res_units_per_acre": safe_divide(units_res_total, lot_area_acres),
            "bldg_sqft_per_acre": safe_divide(bldg_area_total, lot_area_acres),
            "buildings_per_acre": safe_divide(num_buildings_total, lot_area_acres),
            "pct_res_area": safe_divide(res_area_total, bldg_area_total),
            "pct_com_area": safe_divide(com_area_total, bldg_area_total),
            "pct_mixed_use_lots": safe_divide(float(mixed_use_mask.sum()), lot_count),
            "land_use_mode": mode_or_null(group["landuse"]),
            "avg_year_built_weighted": weighted_mean(year_built, group["bldgarea"]),
            "median_year_built": (
                float(valid_year_built.median()) if not valid_year_built.empty else np.nan
            ),
            "pct_prewar": safe_divide(
                float((year_built < 1940).sum()),
                float(year_built.notna().sum()),
            ),
            "avg_built_far_weighted": weighted_mean(built_far, group["lotarea"]),
        }
    )


def mode_or_null(values: pd.Series) -> str | float:
    """
    Return the most common non-null value with deterministic tie-breaking.
    """
    clean = values.dropna().astype("string").str.strip()
    clean = clean[clean != ""]
    if clean.empty:
        return np.nan

    counts = clean.value_counts(dropna=False)
    top_count = counts.iloc[0]
    top_values = sorted(counts[counts == top_count].index.tolist())
    return str(top_values[0])
