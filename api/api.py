"""
Loyalty Intelligence System — API FastAPI
Fase 3: Productizacion

Endpoints:
  GET  /health              — Health check
  GET  /score/{cust_id}     — Score individual client
  POST /score/batch         — Score batch of clients
  GET  /recommend/{cust_id} — Full recommendation (decision engine)
  GET  /segment/{cust_id}   — Cluster assignment
  GET  /stats               — Summary stats

Uso:
  # Con mock data
  uvicorn api:app --host 0.0.0.0 --port 8000 --reload

  # Test
  curl http://localhost:8000/health
  curl http://localhost:8000/score/C0001
  curl http://localhost:8000/recommend/C0100
"""

import os
import sys
import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("api")

BASE_DIR = Path(__file__).resolve().parent.parent

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
except ImportError:
    print("FastAPI not installed. Run: pip install fastapi uvicorn")
    sys.exit(1)

app = FastAPI(
    title="Loyalty Intelligence API",
    description="Scoring y recomendaciones para Loyalty Points",
    version="1.0.0",
)


# ── Data & Models (loaded at startup) ──────────────────────────────────────
_state = {}


def get_data():
    if "df" not in _state:
        load_all()
    return _state


def load_all():
    """Load mock data and train models at startup."""
    log.info("Loading data and training models...")

    import nbformat
    from io import StringIO

    nb_path = BASE_DIR / "notebooks" / "05_decision_engine.ipynb"
    nb = nbformat.read(str(nb_path), as_version=4)

    code = ["import matplotlib; matplotlib.use('Agg')"]
    for cell in nb.cells:
        if cell.cell_type == "code":
            src = cell.source.replace("plt.show()", "plt.close('all')")
            code.append(src)

    script = "\n\n".join(code)
    ns = {}
    old_dir = os.getcwd()
    os.chdir(str(BASE_DIR / "notebooks"))

    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        exec(script, ns)
    finally:
        sys.stdout = old_stdout
        os.chdir(old_dir)

    _state["df"] = ns["df"]
    _state["df_output"] = ns["df_output"]
    log.info(f"Loaded: {_state['df'].shape[0]:,} rows, {_state['df']['cust_id'].nunique()} clients")


# ── Request/Response Models ────────────────────────────────────────────────
class BatchRequest(BaseModel):
    cust_ids: list[str]


class ScoreResponse(BaseModel):
    cust_id: str
    propensity_score: float
    uplift_x: float
    expected_value: float
    prioridad: str
    cluster_name: str


class RecommendResponse(BaseModel):
    cust_id: str
    t0: str
    tier: Optional[str] = None
    funnel_state: Optional[str] = None
    propensity_score: float
    uplift_x: float
    expected_value: float
    prioridad: str
    objetivo: Optional[str] = None
    accion: Optional[str] = None
    retailer_recomendado: Optional[str] = None
    canal: Optional[str] = None
    timing: Optional[str] = None
    cluster_name: Optional[str] = None
    justificacion: Optional[str] = None


# ── Endpoints ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    state = get_data()
    return {
        "status": "ok",
        "clients": state["df"]["cust_id"].nunique(),
        "rows": len(state["df"]),
        "latest_t0": str(state["df"]["t0"].max()),
    }


@app.get("/score/{cust_id}", response_model=ScoreResponse)
def score_client(cust_id: str):
    state = get_data()
    df = state["df"]
    latest_t0 = df["t0"].max()

    client = df[(df["cust_id"] == cust_id.upper()) & (df["t0"] == latest_t0)]
    if len(client) == 0:
        raise HTTPException(status_code=404, detail=f"Client {cust_id} not found")

    row = client.iloc[0]
    return ScoreResponse(
        cust_id=row["cust_id"],
        propensity_score=round(float(row.get("propensity_score", 0)), 4),
        uplift_x=round(float(row.get("uplift_x", 0)), 2),
        expected_value=round(float(row.get("expected_value", 0)), 2),
        prioridad=str(row.get("prioridad", "?")),
        cluster_name=str(row.get("cluster_name", "?")),
    )


@app.post("/score/batch")
def score_batch(request: BatchRequest):
    state = get_data()
    df = state["df"]
    latest_t0 = df["t0"].max()

    results = []
    for cid in request.cust_ids[:1000]:  # Limit to 1000
        client = df[(df["cust_id"] == cid.upper()) & (df["t0"] == latest_t0)]
        if len(client) > 0:
            row = client.iloc[0]
            results.append({
                "cust_id": row["cust_id"],
                "propensity_score": round(float(row.get("propensity_score", 0)), 4),
                "uplift_x": round(float(row.get("uplift_x", 0)), 2),
                "expected_value": round(float(row.get("expected_value", 0)), 2),
                "prioridad": str(row.get("prioridad", "?")),
                "cluster_name": str(row.get("cluster_name", "?")),
            })
    return {"results": results, "n_found": len(results), "n_requested": len(request.cust_ids)}


@app.get("/recommend/{cust_id}", response_model=RecommendResponse)
def recommend_client(cust_id: str):
    state = get_data()
    df_output = state["df_output"]
    df = state["df"]
    latest_t0 = df["t0"].max()

    # Get from decision engine output
    client_out = df_output[(df_output["cust_id"] == cust_id.upper()) & (df_output["t0"] == latest_t0)] if "t0" in df_output.columns else pd.DataFrame()
    client_df = df[(df["cust_id"] == cust_id.upper()) & (df["t0"] == latest_t0)]

    if len(client_df) == 0:
        raise HTTPException(status_code=404, detail=f"Client {cust_id} not found")

    row = client_df.iloc[0]
    out = client_out.iloc[0] if len(client_out) > 0 else row

    return RecommendResponse(
        cust_id=str(row["cust_id"]),
        t0=str(latest_t0),
        tier=str(row.get("tier", "")),
        funnel_state=str(row.get("funnel_state_at_t0", "")),
        propensity_score=round(float(row.get("propensity_score", 0)), 4),
        uplift_x=round(float(row.get("uplift_x", 0)), 2),
        expected_value=round(float(out.get("expected_value", 0)), 2),
        prioridad=str(out.get("prioridad", "?")),
        objetivo=str(out.get("objetivo", "")),
        accion=str(out.get("accion", "")),
        retailer_recomendado=str(out.get("retailer_recomendado", "")),
        canal=str(out.get("canal", "")),
        timing=str(out.get("timing", "")),
        cluster_name=str(out.get("cluster_name", "")),
        justificacion=str(out.get("justificacion", "")),
    )


@app.get("/segment/{cust_id}")
def segment_client(cust_id: str):
    state = get_data()
    df = state["df"]
    latest_t0 = df["t0"].max()

    client = df[(df["cust_id"] == cust_id.upper()) & (df["t0"] == latest_t0)]
    if len(client) == 0:
        raise HTTPException(status_code=404, detail=f"Client {cust_id} not found")

    row = client.iloc[0]
    cluster_feats = ["frequency_monthly_avg", "monetary_monthly_avg", "redeem_rate",
                     "retailer_entropy", "pct_redeem_digital", "earn_velocity_90",
                     "days_since_last_activity", "points_pressure"]
    profile = {f: round(float(row.get(f, 0)), 4) for f in cluster_feats if f in row.index}

    return {
        "cust_id": row["cust_id"],
        "cluster_name": str(row.get("cluster_name", "?")),
        "cluster_id": int(row.get("cluster", -1)) if "cluster" in row.index else -1,
        "profile": profile,
    }


@app.get("/stats")
def stats():
    state = get_data()
    df = state["df"]
    df_output = state["df_output"]
    latest_t0 = df["t0"].max()

    latest = df[df["t0"] == latest_t0]
    latest_out = df_output[df_output["t0"] == latest_t0] if "t0" in df_output.columns else df_output

    result = {
        "latest_t0": str(latest_t0),
        "n_clients": int(latest["cust_id"].nunique()),
        "n_snapshots": int(df["t0"].nunique()),
    }

    if "prioridad" in latest_out.columns:
        result["prioridad"] = latest_out["prioridad"].value_counts().to_dict()

    if "cluster_name" in latest.columns:
        result["clusters"] = latest["cluster_name"].value_counts().to_dict()

    if "propensity_score" in latest.columns:
        result["propensity"] = {
            "mean": round(float(latest["propensity_score"].mean()), 4),
            "median": round(float(latest["propensity_score"].median()), 4),
            "p90": round(float(latest["propensity_score"].quantile(0.9)), 4),
        }

    if "y" in latest.columns:
        y_dist = latest["y"].value_counts(normalize=True).round(4).to_dict()
        result["target_distribution"] = {f"y={k}": v for k, v in y_dist.items()}

    return result
