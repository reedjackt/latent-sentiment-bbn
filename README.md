# Latent Sentiment BBN

End-to-end pipeline for inferring latent customer sentiment from firmographics and engagement signals using a Bayesian belief network (pgmpy), a local DuckDB warehouse, dbt transformations, and a FastAPI serving layer.

## Repository layout

```
├── data/                    # Local warehouse storage (.duckdb files)
├── dbt_pipeline/            # Complete dbt project
│   ├── models/              # dbt SQL transformation models
│   ├── dbt_project.yml      # dbt configuration
│   └── profiles.yml         # Connection profile for DuckDB
├── src/                     # Core application source code
│   ├── __init__.py
│   ├── api/                 # FastAPI application layer
│   │   ├── main.py          # API entry point & routes
│   │   └── services.py      # Mock Clearbit & model loading logic
│   ├── data_gen/            # Synthetic data generation scripts
│   └── models/              # pgmpy training and inference logic
│       ├── train.py         # Structure & parameter learning script
│       └── artifacts/       # Saved serialized model (model.pkl)
├── tests/                   # PyTest suite for API and transformations
├── pyproject.toml           # uv dependencies and metadata
└── README.md
```

## Architecture

```mermaid
flowchart LR
  subgraph ingest
    DG[scripts/generate_raw_leads.py]
    DG --> DW[(data/raw_leads.duckdb)]
  end
  subgraph transform
    DW --> DBT[dbt_pipeline]
    DBT --> MARTS[stg_leads + marts]
  end
  subgraph ml
    MARTS --> TRAIN[models/train.py]
    TRAIN --> PKL[models/artifacts/model.pkl]
  end
  subgraph serve
    PKL --> API[api/main.py]
    API --> CB[mock Clearbit enrich]
    API --> INF[BBN inference]
  end
```

## Structure Learning Constraints

The BBN learns observed-to-observed edges from a whitelist before adding the hidden `latent_sentiment` node and fitting parameters with EM. Exogenous context variables such as channel, campaign tier, region, company size, and job title are allowed to bypass the latent variable and point directly into engagement signals, sentiment proxies, and the conversion outcome. That keeps the model honest about real marketing mechanics: context can change exposure, buying authority, routing, and qualification even when two leads share the same hidden sentiment.

Signal-to-signal edges are only allowed where they match a plausible funnel direction, such as web engagement preceding reply behavior, proxy score, or demo request. The hill-climb search still applies `max_indegree`, so these bypasses expand the candidate graph enough to avoid over-attributing everything to latent sentiment without letting arbitrary observed edges dominate the structure.

## Quick start

```bash
# Install runtime + dev dependencies
uv sync --all-groups

# Seed local DuckDB warehouse
uv run python scripts/generate_raw_leads.py --rows 50000

# Clean leads for model training
uv run python scripts/clean_data.py --skip-inspect

# Run dbt against the local DuckDB warehouse
uv run dbt run --project-dir dbt_pipeline --profiles-dir dbt_pipeline

# Train BBN and write model.pkl
uv run python -m models.train

# Start API
uv run latent-sentiment-api

# Run tests
uv run pytest
```

## Legacy entry point

The original console script remains available:

```bash
uv run latent-sentiment-bbn
```
