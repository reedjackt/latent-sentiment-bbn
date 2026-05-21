#!/usr/bin/env python3
"""
Generate messy synthetic marketing leads with a hidden latent sentiment driver,
then materialize DuckDB at data/raw_leads.duckdb (table: raw_leads).

Run (from repo root):
  uv run python scripts/generate_raw_leads.py
  uv run python scripts/generate_raw_leads.py --rows 50000 --seed 7

The latent column True_Consumer_Sentiment is used only during generation and
is never written to DuckDB.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = REPO_ROOT / "data" / "raw_leads.duckdb"
DEFAULT_ROWS = 50_000
DEFAULT_SEED = 42

# --- Messy string pools (typos, casing, rare tokens) ---
JOB_BASES = [
    "Director of Marketing",
    "VP Sales",
    "Head of Growth",
    "Chief Revenue Officer",
    "Product Marketing Manager",
    "Demand Gen Lead",
    "Marketing Ops Manager",
    "Account Executive",
    "SDR",
    "RevOps Analyst",
]
JOB_TYPO_OPS = [
    lambda s: s,
    lambda s: s.lower(),
    lambda s: s.upper(),
    lambda s: s.replace("Marketing", "Markting"),
    lambda s: s.replace("Director", "Dir."),
    lambda s: s + " (interim)",
    lambda s: "  " + s + "  ",
    lambda s: s.replace(" ", "_"),
    lambda s: s + " / acting",
]

COMPANY_STEMS = [
    "Nimbus",
    "Quill",
    "Vertex",
    "Harbor",
    "Lumen",
    "Atlas",
    "Cobalt",
    "Silverline",
    "Northwind",
    "Bluecanoe",
]
COMPANY_SUFFIXES = ["Labs", "Systems", "AI", "Cloud", "Group", "Holdings", "Inc", "LLC", "Ltd", ""]

CHANNELS = np.array(
    ["paid_search", "organic", "linkedin_ads", "webinar", "partner", "event", "email", "direct", "other"]
)
CHANNEL_PROBS = np.array([0.32, 0.18, 0.14, 0.09, 0.07, 0.06, 0.05, 0.05, 0.04], dtype=np.float64)

CAMPAIGN_TIERS = np.array(["tier1_enterprise", "tier2_growth", "tier3_smb", "pilot", "legacy_2019", "sunset"])
CAMPAIGN_PROBS = np.array([0.08, 0.22, 0.38, 0.12, 0.03, 0.17], dtype=np.float64)

REGIONS = np.array(["NA", "EMEA", "APAC", "LATAM", "unknown"])
REGION_PROBS = np.array([0.48, 0.24, 0.16, 0.07, 0.05], dtype=np.float64)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _messy_job_title(rng: np.random.Generator, n: int) -> list[str | None]:
    bases = rng.choice(JOB_BASES, size=n)
    ops = rng.choice(np.arange(len(JOB_TYPO_OPS)), size=n)
    out: list[str | None] = []
    mask_null = rng.random(n) < 0.012
    for i in range(n):
        if mask_null[i]:
            out.append(None)
            continue
        s = JOB_TYPO_OPS[int(ops[i])](str(bases[i]))
        if rng.random() < 0.04:
            s = s[: max(3, len(s) // 2)]  # truncated paste artifact
        out.append(s)
    return out


def _messy_company_and_domain(
    rng: np.random.Generator, n: int
) -> tuple[list[str | None], list[str | None]]:
    names: list[str | None] = []
    domains: list[str | None] = []
    for _ in range(n):
        stem = str(rng.choice(COMPANY_STEMS))
        suf = str(rng.choice(COMPANY_SUFFIXES))
        legal = f"{stem} {suf}".strip()
        roll = rng.random()
        if roll < 0.08:
            names.append(None)
        elif roll < 0.15:
            names.append(legal.lower())
        elif roll < 0.22:
            names.append(legal.upper())
        elif roll < 0.28:
            names.append(legal.replace(" ", "") + rng.choice(["", ".", "™"]))
        else:
            names.append(legal)

        if rng.random() < 0.11:
            domains.append(None)
            continue
        slug = stem.lower().replace(" ", "")
        tld = rng.choice(["com", "io", "co", "ai", "net"])
        style = rng.random()
        if style < 0.35:
            dom = f"{slug}.{tld}"
        elif style < 0.6:
            dom = f"www.{slug}.{tld}"
        elif style < 0.85:
            dom = f"mail.{slug}.{tld}"  # noisy, not always corporate site
        else:
            dom = f"{slug}{rng.integers(1, 99)}.{tld}"
        if rng.random() < 0.06:
            dom = dom.upper()
        domains.append(dom)
    return names, domains


def _messy_employee_band(rng: np.random.Generator, n: int) -> list[str | None]:
    bands = [
        "1-10",
        "11-50",
        "51-200",
        "201-500",
        "501-1000",
        "1001-5000",
        "5000+",
        "UNKNOWN",
        "n/a",
        "See LinkedIn",
        "~200",
        "50+",
    ]
    out: list[str | None] = []
    for _ in range(n):
        r = rng.random()
        if r < 0.34:
            out.append(None)
        elif r < 0.42:
            out.append(str(rng.choice(bands)))
        elif r < 0.5:
            out.append(str(rng.choice(bands)).replace("-", " to "))
        elif r < 0.58:
            out.append(str(rng.integers(5, 8000)))  # raw number string
        else:
            out.append(str(rng.choice(bands)))
    return out


def build_leads_df(n: int, seed: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Returns (full_debug_df_with_latent, warehouse_df_without_latent).
    """
    rng = np.random.default_rng(seed)

    lead_ids = np.array([f"LD-{i:08d}" for i in range(n)], dtype=object)
    # occasional malformed id
    corrupt = rng.random(n) < 0.003
    for i in np.flatnonzero(corrupt):
        lead_ids[i] = f"ld{rng.integers(1000,9999)}"  # missing zero padding

    # timestamps: skewed recent + some nulls
    base = np.datetime64("2024-01-01")
    offsets_days = rng.exponential(scale=120.0, size=n).astype(np.float64)
    captured = base + offsets_days.astype("timedelta64[D]")
    ts_null = rng.random(n) < 0.021
    captured_dt: np.ndarray = captured.astype("datetime64[ns]")
    captured_dt[ts_null] = np.datetime64("NaT")

    # Latent consumer sentiment: continuous score; drives downstream without being stored
    true_consumer_sentiment = rng.normal(loc=0.0, scale=1.0, size=n)

    # Correlated engagement noise (partially observed funnel)
    engagement_noise = 0.55 * true_consumer_sentiment + rng.normal(0.0, 0.9, size=n)
    web_session_seconds = np.clip(
        rng.lognormal(mean=math.log(180.0), sigma=1.05, size=n)
        * (1.0 + 0.35 * true_consumer_sentiment)
        + engagement_noise * 40.0,
        0.0,
        1.2e6,
    )
    sess_null = rng.random(n) < 0.067
    web_session_seconds = np.where(sess_null, np.nan, web_session_seconds)

    channels = rng.choice(CHANNELS, size=n, p=CHANNEL_PROBS / CHANNEL_PROBS.sum())
    # rare mis-encoding
    for i in range(n):
        if rng.random() < 0.008:
            channels[i] = str(channels[i]).upper()
        if rng.random() < 0.004:
            channels[i] = "paid search"  # space vs underscore inconsistency

    campaigns = rng.choice(CAMPAIGN_TIERS, size=n, p=CAMPAIGN_PROBS / CAMPAIGN_PROBS.sum())
    regions = rng.choice(REGIONS, size=n, p=REGION_PROBS / REGION_PROBS.sum())

    company_names, company_domains = _messy_company_and_domain(rng, n)
    job_titles = _messy_job_title(rng, n)
    employee_bands = _messy_employee_band(rng, n)

    # Outcome-ish: demo request probability increases with latent sentiment
    p_demo = _sigmoid(1.15 * true_consumer_sentiment - 0.25 + rng.normal(0.0, 0.35, size=n))
    demo_requested = rng.binomial(1, np.clip(p_demo, 0.02, 0.98))

    # Lead score: skewed, sentiment-sensitive, heavy tails
    score_base = 42.0 + 18.0 * np.tanh(true_consumer_sentiment) + rng.normal(0.0, 11.0, size=n)
    lead_score = np.clip(np.round(score_base + 0.15 * engagement_noise * 20.0), 0.0, 100.0)
    score_null = rng.random(n) < 0.045
    lead_score = np.where(score_null, np.nan, lead_score)

    # Extra categorical with imbalance + sentiment-tinged "email reply" propensity proxy
    p_reply = _sigmoid(0.75 * true_consumer_sentiment + rng.normal(0.0, 0.25, size=n))
    replied_7d = rng.binomial(1, np.clip(p_reply, 0.03, 0.97))
    replied_label = np.where(
        rng.random(n) < 0.18,
        np.where(replied_7d == 1, "Y", "N"),
        np.where(replied_7d == 1, "yes", "no"),
    )
    replied_null = rng.random(n) < 0.029
    replied_label = np.where(replied_null, None, replied_label)

    full = pl.DataFrame(
        {
            "lead_id": lead_ids,
            "captured_at": captured_dt,
            "job_title_raw": job_titles,
            "company_name_messy": company_names,
            "company_domain_raw": company_domains,
            "employee_band_messy": employee_bands,
            "web_session_seconds": web_session_seconds.astype(np.float64),
            "marketing_channel": channels.astype(object),
            "campaign_tier": campaigns.astype(object),
            "region": regions.astype(object),
            "replied_within_7d": replied_label,
            "demo_requested": pl.Series(demo_requested, dtype=pl.Int8),
            "lead_score": lead_score.astype(np.float64),
            "True_Consumer_Sentiment": true_consumer_sentiment.astype(np.float64),
        }
    )

    warehouse = full.drop("True_Consumer_Sentiment")
    return full, warehouse


def materialize_duckdb(df: pl.DataFrame, db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    con = duckdb.connect(str(db_path))
    try:
        con.register("_leads", df)
        con.execute("CREATE TABLE raw_leads AS SELECT * FROM _leads")
        con.unregister("_leads")
    finally:
        con.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic raw marketing leads into DuckDB.")
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS, help="Number of lead rows to generate.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed for reproducibility.")
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"Output DuckDB file (default: {DEFAULT_DB_PATH})",
    )
    args = parser.parse_args()

    if args.rows < 1:
        raise SystemExit("--rows must be >= 1")

    _, warehouse_df = build_leads_df(args.rows, args.seed)
    materialize_duckdb(warehouse_df, args.db_path.resolve())

    con = duckdb.connect(str(args.db_path))
    try:
        cnt = con.execute("SELECT COUNT(*) FROM raw_leads").fetchone()[0]
        cols = [r[1] for r in con.execute("PRAGMA table_info('raw_leads')").fetchall()]
    finally:
        con.close()

    print(f"Wrote {cnt} rows to {args.db_path.resolve()}")
    print("Columns:", ", ".join(cols))
    if "True_Consumer_Sentiment" in cols:
        raise SystemExit("True_Consumer_Sentiment must not appear in DuckDB")
    if cnt < args.rows:
        raise SystemExit(f"Expected {args.rows} rows, got {cnt}")


if __name__ == "__main__":
    main()
