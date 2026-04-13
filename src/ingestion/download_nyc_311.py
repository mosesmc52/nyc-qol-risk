#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator

from dotenv import find_dotenv, load_dotenv
from src.ingestion.nyc_open_data_api import NYCOpenDataAPI

BASE_URL = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"
OUT_DIR = "./data/raw/nyc/311_noise"

_TAIL_CLAUSE_RE = re.compile(
    r"\b(ORDER\s+BY|SEARCH|GROUP\s+BY|HAVING|LIMIT|OFFSET)\b",
    flags=re.IGNORECASE,
)
_WHERE_RE = re.compile(r"\bWHERE\b", flags=re.IGNORECASE)


@dataclass(frozen=True)
class Window:
    start: datetime
    end: datetime


class NYC311Downloader(NYCOpenDataAPI):
    def __init__(
        self,
        *,
        query_file: str = "./src/queries/nyc_311.soql",
        out_dir: str = "data/raw/nyc/311",
        freq: str = "month",
        limit: int = 1000,
        resume: bool = False,
        max_retries: int = 6,
        timeout: int = 120,
        debug: bool = False,
        base_url: str = BASE_URL,
        app_token: str | None = None,
        base_soql: str | None = None,
    ) -> None:
        if freq not in ("day", "week", "month"):
            raise ValueError("freq must be one of: day, week, month")

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
        self.freq = freq
        self.resume = bool(resume)

    def run(self, start_date: str, end_date: str) -> int:
        base_soql = self._load_base_soql()
        start_dt = self.parse_date_ymd(start_date)
        end_dt = self.parse_date_ymd(end_date)

        part_dir = self.out_dir / f"by_{self.freq}"
        part_dir.mkdir(parents=True, exist_ok=True)

        grand_total = 0
        for window in self.iter_windows(start_dt, end_dt, self.freq):
            label = self.window_label(window, self.freq)
            out_file = part_dir / f"{label}.csv"

            if self.resume and out_file.exists():
                print(f"Skipping (exists): {out_file}")
                continue

            print(
                f"Downloading window [{window.start.date()} -> {window.end.date()}) to {out_file}"
            )
            rows = self.download_partition(
                base_soql=base_soql,
                window=window,
                out_file=out_file,
            )
            print(f"  Completed {out_file.name}: {rows} rows")
            grand_total += rows

        print(f"\nDone. Total rows downloaded across windows: {grand_total}")
        return grand_total

    def download_partition(
        self,
        *,
        base_soql: str,
        window: Window,
        out_file: Path,
    ) -> int:
        predicate = self.soql_time_predicate(window.start, window.end)
        partition_soql = self.inject_time_filter(base_soql, predicate)
        return self.download_to_csv(out_file, base_soql=partition_soql)

    @staticmethod
    def inject_time_filter(base_soql: str, time_predicate: str) -> str:
        query = base_soql.strip().rstrip(";")
        tail_match = _TAIL_CLAUSE_RE.search(query)
        if tail_match:
            head = query[: tail_match.start()].rstrip()
            tail = query[tail_match.start() :].lstrip()
        else:
            head = query
            tail = ""

        if _WHERE_RE.search(head):
            return f"{head} AND {time_predicate} {tail}".strip()
        return f"{head} WHERE {time_predicate} {tail}".strip()

    @staticmethod
    def soql_time_predicate(start: datetime, end: datetime) -> str:
        start_s = start.strftime("%Y-%m-%dT%H:%M:%S")
        end_s = end.strftime("%Y-%m-%dT%H:%M:%S")
        return f"created_date >= '{start_s}' AND created_date < '{end_s}'"

    @staticmethod
    def parse_date_ymd(value: str) -> datetime:
        return datetime.strptime(value, "%Y-%m-%d")

    @staticmethod
    def iter_windows(start: datetime, end: datetime, freq: str) -> Iterator[Window]:
        current = start

        if freq == "day":
            step = timedelta(days=1)
            while current < end:
                nxt = min(current + step, end)
                yield Window(current, nxt)
                current = nxt
            return

        if freq == "week":
            step = timedelta(days=7)
            while current < end:
                nxt = min(current + step, end)
                yield Window(current, nxt)
                current = nxt
            return

        if freq == "month":
            while current < end:
                year, month = current.year, current.month
                if month == 12:
                    nxt = datetime(year + 1, 1, 1)
                else:
                    nxt = datetime(year, month + 1, 1)
                nxt = min(nxt, end)
                yield Window(current, nxt)
                current = nxt
            return

        raise ValueError("freq must be one of: day, week, month")

    @staticmethod
    def window_label(window: Window, freq: str) -> str:
        if freq in ("day", "week"):
            return window.start.strftime("%Y-%m-%d")
        if freq == "month":
            return window.start.strftime("%Y-%m")
        raise ValueError(freq)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--query-file",
        default="./src/queries/nyc_311.soql",
        help="Path to SoQL file (no LIMIT/OFFSET needed)",
    )
    ap.add_argument("--out-dir", default=OUT_DIR, help="Base output directory")
    ap.add_argument("--limit", type=int, default=1000, help="Pagination page size")
    ap.add_argument("--start-date", required=True, help="YYYY-MM-DD inclusive")
    ap.add_argument("--end-date", required=True, help="YYYY-MM-DD exclusive")
    ap.add_argument(
        "--freq",
        choices=["day", "week", "month"],
        default="month",
        help="Partition frequency",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Skip partitions whose output file already exists",
    )
    ap.add_argument(
        "--max-retries", type=int, default=6, help="Max retries per request"
    )
    ap.add_argument("--timeout", type=int, default=120, help="Request timeout seconds")
    ap.add_argument(
        "--debug", action="store_true", help="Print first-page SoQL for each partition"
    )
    args = ap.parse_args()

    load_dotenv(find_dotenv())

    downloader = NYC311Downloader(
        query_file=args.query_file,
        out_dir=args.out_dir,
        freq=args.freq,
        limit=args.limit,
        resume=args.resume,
        max_retries=args.max_retries,
        timeout=args.timeout,
        debug=args.debug,
    )
    downloader.run(args.start_date, args.end_date)


if __name__ == "__main__":
    main()
