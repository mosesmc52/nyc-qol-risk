#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from nyc_open_data_api import NYCOpenDataAPI


BASE_URL = "https://data.cityofnewyork.us/resource/64uk-42ks.json"
OUT_DIR = "./data/raw/nyc/pluto"


class NYCPlutoDownloader(NYCOpenDataAPI):
    def __init__(
        self,
        *,
        query_file: str = "./src/queries/nyc_pluto.soql",
        out_dir: str = "data/raw/nyc/pluto",
        limit: int = 1000,
        max_retries: int = 6,
        timeout: int = 120,
        debug: bool = False,
        app_token: str | None = None,
        base_soql: str | None = None,
        base_url: str = BASE_URL,
    ) -> None:
        super().__init__(
            base_url=base_url,
            query_file=query_file,
            out_dir=out_dir,
            limit=limit,
            max_retries=max_retries,
            timeout=timeout,
            debug=debug,
            app_token=app_token,
            base_soql=base_soql,
        )

    def default_output_path(self) -> Path:
        return self.out_dir / "pluto.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--query-file",
        default="./src/queries/nyc_pluto.soql",
        help="Path to SoQL file (no LIMIT/OFFSET needed)",
    )
    ap.add_argument("--out-dir", default=OUT_DIR, help="Base output directory")
    ap.add_argument("--out-file", default=None, help="Optional explicit CSV output path")
    ap.add_argument("--limit", type=int, default=1000, help="Pagination page size")
    ap.add_argument(
        "--max-retries", type=int, default=6, help="Max retries per request"
    )
    ap.add_argument("--timeout", type=int, default=120, help="Request timeout seconds")
    ap.add_argument("--debug", action="store_true", help="Print first-page SoQL")
    args = ap.parse_args()

    load_dotenv(find_dotenv())

    downloader = NYCPlutoDownloader(
        query_file=args.query_file,
        out_dir=args.out_dir,
        limit=args.limit,
        max_retries=args.max_retries,
        timeout=args.timeout,
        debug=args.debug,
    )
    downloader.run(out_file=args.out_file)


if __name__ == "__main__":
    main()
