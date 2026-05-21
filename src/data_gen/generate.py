"""Synthetic warehouse seed data for local DuckDB development."""

from __future__ import annotations

from pathlib import Path

import duckdb
import polars as pl

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
WAREHOUSE_PATH = DATA_DIR / "warehouse.duckdb"


def generate_accounts(n: int = 500, seed: int = 42) -> pl.DataFrame:
    rng = pl.Series("id", range(n))
    domains = [f"account-{i}.example.com" for i in range(n)]
    return pl.DataFrame(
        {
            "account_id": rng,
            "domain": domains,
            "page_views": pl.Series(
                [((i * 17) % 120) for i in range(n)], dtype=pl.Int64
            ),
            "email_opens": pl.Series(
                [((i * 11) % 40) for i in range(n)], dtype=pl.Int64
            ),
        }
    )


def write_warehouse(df: pl.DataFrame, path: Path = WAREHOUSE_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    try:
        con.execute("CREATE OR REPLACE TABLE raw_accounts AS SELECT * FROM df")
    finally:
        con.close()
    return path


def main() -> None:
    df = generate_accounts()
    out = write_warehouse(df)
    print(f"Wrote {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
