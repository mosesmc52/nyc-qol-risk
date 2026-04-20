import re
import pandas as pd

from pathlib import Path
import arviz as az
import geopandas as gpd
import pandas as pd
from src.aggregate.build_pluto_nta_features import load_pluto_data
from src.features.pluto_nta import (
    prepare_pluto_geometry,
    spatially_assign_to_nta,
    standardize_nta_geographies,
)

def export_idata(idata, out_path: str):
    """
    Save ArviZ InferenceData (PyMC sampling result) to netCDF.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    az.to_netcdf(idata, out_path)
    print(f"✅ Saved idata -> {out_path}")
    return str(out_path)

def load_idata(path: str):
    """
    Load ArviZ InferenceData from netCDF.
    """
    idata = az.from_netcdf(path)
    print(f"✅ Loaded idata <- {path}")
    return idata

def prep_the_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- ensure datetime ---
    df["complaint_date"] = pd.to_datetime(df["complaint_date"], errors="coerce")

    if "category" not in df.columns and "level_2" in df.columns:
        df["category"] = df["level_2"]
    if "category_group" not in df.columns and "level_1" in df.columns:
        df["category_group"] = df["level_1"]

    # --- derive calendar fields ---
    df["dow"] = df["complaint_date"].dt.day_name()
    df["month"] = df["complaint_date"].dt.month_name()


    # --- weekend flag (Saturday/Sunday) ---
    df["is_weekend"] = df["dow"].isin(["Saturday", "Sunday"]).astype("int8")

    # --- month_year label ---
    df["month_year"] = (
        df["month"].astype("string")
        + "__"
        + df["complaint_date"].dt.year.astype("Int64").astype("string")
    )

    # --- build dow_complaint from aggregated descriptor_group ---
    df["dow_complaint"] = (
        df["category"]
        .astype("string")
        .str.upper()
        .str.replace(r"[,_/]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.replace(" ", "_")
        + "__"
        + df["dow"].astype("string")
    )

    return df



PREFERRED_ZONING_COLUMNS = ["zonedist1", "zonedist2", "zonedist3", "zonedist4"]
GROUP_ORDER = [
    "Residential",
    "Commercial",
    "Manufacturing",
    "Special Purpose / Other",
    "Unknown",
]


def classify_zoning_code(code: str) -> str:
    text = str(code).strip().upper()
    if not text or text == "<NA>":
        return "Unknown"
    if text.startswith("R"):
        return "Residential"
    if text.startswith("C"):
        return "Commercial"
    if text.startswith("M"):
        return "Manufacturing"
    return "Special Purpose / Other"




def zoning_groups_for_row(row: pd.Series, zoning_columns: list[str]) -> list[str]:
    groups = set()

    for column in zoning_columns:
        value = row.get(column)
        if pd.isna(value):
            continue

        pieces = re.split(r"[/,;]+", str(value).upper())
        for piece in pieces:
            piece = piece.strip()
            if piece:
                groups.add(classify_zoning_code(piece))

    return sorted(groups) if groups else ["Unknown"]


def prep_zone_data(
    pluto_path: Path | str = Path("../data/raw/nyc/pluto/nyc_pluto.csv"),
    nta_path: Path | str = Path("../data/raw/nyc/geographies/nyc_ntas_2020.geojson"),
    preferred_zoning_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Load PLUTO and NTA data, assign lots to NTAs, derive zoning groups,
    and return both the parcel-level prepared data and an exploded long-form version.

    Returns
    -------
    pluto_with_nta : pd.DataFrame
        Parcel-level data with zoning group allocations.
    zoning_long : pd.DataFrame
        Long-form data with one row per parcel-zoning group assignment.
    available_zoning_columns : list[str]
        Zoning district columns that were found in the PLUTO file.
    """
    preferred_zoning_columns = (
        preferred_zoning_columns or PREFERRED_ZONING_COLUMNS
    )

    pluto_df = load_pluto_data(Path(pluto_path))
    available_zoning_columns = [
        column for column in preferred_zoning_columns if column in pluto_df.columns
    ]

    if not available_zoning_columns:
        raise ValueError(
            "The current PLUTO file does not include any zoning district columns. "
            "Re-run the PLUTO downloader after updating src/queries/nyc_pluto.soql."
        )

    nta_gdf = standardize_nta_geographies(gpd.read_file(nta_path))
    pluto_gdf = prepare_pluto_geometry(pluto_df)
    pluto_with_nta = spatially_assign_to_nta(pluto_gdf, nta_gdf)

    pluto_with_nta = pluto_with_nta[
        ["nta_id", "nta_name", "borough", "lotarea", *available_zoning_columns]
    ].copy()

    pluto_with_nta["lotarea"] = (
        pd.to_numeric(pluto_with_nta["lotarea"], errors="coerce")
        .fillna(0)
    )

    pluto_with_nta["zoning_groups"] = pluto_with_nta.apply(
        zoning_groups_for_row,
        axis=1,
        zoning_columns=available_zoning_columns,
    )
    pluto_with_nta["group_count"] = (
        pluto_with_nta["zoning_groups"].str.len().clip(lower=1)
    )
    pluto_with_nta["allocated_lotarea"] = (
        pluto_with_nta["lotarea"] / pluto_with_nta["group_count"]
    )

    zoning_long = (
        pluto_with_nta[
            ["nta_id", "nta_name", "borough", "allocated_lotarea", "zoning_groups"]
        ]
        .explode("zoning_groups")
        .rename(columns={"zoning_groups": "zoning_group"})
    )

    print(f"Using zoning columns: {available_zoning_columns}")

    return pluto_with_nta, zoning_long, available_zoning_columns
