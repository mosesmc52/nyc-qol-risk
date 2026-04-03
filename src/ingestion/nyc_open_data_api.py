from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import requests

from ingestion_utils import load_soql, request_with_retry


class NYCOpenDataAPI:
    def __init__(
        self,
        *,
        base_url: str,
        query_file: str,
        out_dir: str,
        limit: int = 1000,
        max_retries: int = 6,
        timeout: int = 120,
        debug: bool = False,
        app_token: str | None = None,
        base_soql: str | None = None,
    ) -> None:
        self.base_url = base_url
        self.query_file = query_file
        self.out_dir = Path(out_dir)
        self.limit = int(limit)
        self.max_retries = int(max_retries)
        self.timeout = int(timeout)
        self.debug = bool(debug)
        self._base_soql = base_soql.strip() if isinstance(base_soql, str) else None
        self.app_token = (
            app_token if app_token is not None else os.getenv("NYC_APP_TOKEN")
        )
        self._session: requests.Session | None = None

    def run(self, *, out_file: str | Path | None = None) -> int:
        return self.download_to_csv(out_file or self.default_output_path())

    def default_output_path(self) -> Path:
        return self.out_dir / "data.csv"

    def download_to_csv(
        self,
        out_file: str | Path,
        *,
        base_soql: str | None = None,
    ) -> int:
        out_path = Path(out_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        query = base_soql.strip() if isinstance(base_soql, str) else self._load_base_soql()
        session = self._get_session()

        offset = 0
        total = 0
        first_write = True

        while True:
            soql = f"{query} LIMIT {self.limit} OFFSET {offset}"

            if self.debug and offset == 0:
                print("SoQL (first page):", soql)

            resp = request_with_retry(
                session,
                "GET",
                self.base_url,
                params={"$query": soql},
                timeout=self.timeout,
                max_retries=self.max_retries,
                backoff_base=1.0,
                backoff_cap=60.0,
            )

            rows = resp.json()
            if not rows:
                break

            df = pd.DataFrame(rows)
            df.to_csv(
                out_path,
                mode="w" if first_write else "a",
                index=False,
                header=first_write,
            )

            batch_size = len(df)
            total += batch_size
            offset += batch_size
            first_write = False
            print(f"Fetched {batch_size} rows (offset={offset}, total={total})")

            if batch_size < self.limit:
                break

        print(f"Completed {out_path}: {total} rows")
        return total

    def _get_session(self) -> requests.Session:
        if self._session is not None:
            return self._session

        session = requests.Session()
        headers = {"Accept": "application/json"}
        if self.app_token:
            headers["X-App-Token"] = self.app_token
        session.headers.update(headers)
        self._session = session
        return session

    def _load_base_soql(self) -> str:
        if self._base_soql:
            return self._base_soql
        soql = load_soql(self.query_file).strip()
        if not soql:
            raise ValueError(f"Empty SoQL loaded from: {self.query_file}")
        return soql
