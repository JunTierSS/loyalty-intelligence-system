#!/usr/bin/env python3
"""
Loyalty Intelligence System — End-to-End Demo

Runs the complete pipeline on 1000 mock customers:
1. Generates mock data (1000 clients × 27 monthly snapshots)
2. Engineers 74 features
3. Trains models (clustering, propensity, uplift)
4. Scores customers
5. Generates recommendations via decision engine
6. Outputs CSV with prioritized action list

Usage:
    python run_demo.py
    python run_demo.py --save-models    # Also save trained models to disk
"""

import sys
import os

# Add pipeline directory to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "pipeline"))

# Ensure output directories exist
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "output"), exist_ok=True)


def main():
    print("=" * 70)
    print("LOYALTY INTELLIGENCE SYSTEM — END-TO-END DEMO")
    print("=" * 70)
    print()

    # Parse args
    save_models = "--save-models" in sys.argv

    # Import and run the scoring pipeline in mock mode
    from scoring_pipeline import load_or_train_models, score_chunk
    import pandas as pd
    import numpy as np
    import nbformat
    from io import StringIO
    import time

    # ── Step 1: Load mock data via notebook ──
    print("[1/4] Loading mock data from notebooks...")
    nb_path = os.path.join(BASE_DIR, "notebooks", "05_decision_engine.ipynb")

    if not os.path.exists(nb_path):
        print(f"ERROR: Notebook not found at {nb_path}")
        print("Make sure all notebooks are in the notebooks/ directory.")
        sys.exit(1)

    nb = nbformat.read(nb_path, as_version=4)

    code = ["import matplotlib; matplotlib.use('Agg')"]
    for cell in nb.cells:
        if cell.cell_type == "code":
            cid = cell.get("id", "")
            if cid in ("imports", "load-data"):
                code.append(cell.source)
            if cid == "load-data":
                break

    script = "\n\n".join(code)
    ns = {}
    old_dir = os.getcwd()
    os.chdir(os.path.join(BASE_DIR, "notebooks"))

    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        exec(script, ns)
    finally:
        sys.stdout = old_stdout
        os.chdir(old_dir)

    df = ns["df"]
    print(f"   Mock data loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"   Clients: {df['cust_id'].nunique()}")
    print(f"   Snapshots: {df['t0'].nunique()} monthly periods")
    print()

    # ── Step 2: Train/test split ──
    print("[2/4] Splitting train/score...")
    t0s = sorted(df["t0"].unique())
    train_t0s = t0s[:-3]
    score_t0 = t0s[-1]

    df_train = df[df["t0"].isin(train_t0s)]
    df_score = df[df["t0"] == score_t0].copy()

    print(f"   Train: {len(df_train):,} rows ({len(train_t0s)} periods)")
    print(f"   Score: {len(df_score):,} rows (t0={score_t0})")
    print()

    # ── Step 3: Train models ──
    print("[3/4] Training models...")
    models = load_or_train_models(df_train, save=save_models)
    print()

    # ── Step 4: Score ──
    print("[4/4] Scoring customers...")
    t_start = time.time()
    df_scored = score_chunk(df_score, models)
    elapsed = time.time() - t_start
    print(f"   Scored {len(df_scored):,} customers in {elapsed:.1f}s")
    print()

    # ── Output ──
    OUTPUT_COLS = [
        "cust_id", "prioridad", "expected_value", "propensity_score", "uplift_x",
        "objetivo", "accion", "retailer_recomendado", "canal", "timing",
        "cluster_name", "funnel_state_at_t0",
    ]
    output_cols = [c for c in OUTPUT_COLS if c in df_scored.columns]
    df_final = df_scored[output_cols].copy()

    prio_order = {"Alta": 0, "Media": 1, "Baja": 2, "No contactar": 3}
    df_final["_sort"] = df_final["prioridad"].map(prio_order)
    df_final = df_final.sort_values(["_sort", "expected_value"], ascending=[True, False]).drop(columns="_sort")

    csv_path = os.path.join(BASE_DIR, "output", "demo_scoring.csv")
    df_final.to_csv(csv_path, index=False)

    # ── Summary ──
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    print("Priority distribution:")
    for prio in ["Alta", "Media", "Baja", "No contactar"]:
        grp = df_final[df_final["prioridad"] == prio]
        n = len(grp)
        pct = n / len(df_final) * 100
        ev = grp["expected_value"].mean() if "expected_value" in grp.columns and len(grp) > 0 else 0
        print(f"  {prio:15s}: {n:4d} ({pct:5.1f}%) | EV avg = ${ev:,.0f}")

    print()
    print("Cluster distribution:")
    for cluster, count in df_final["cluster_name"].value_counts().items():
        print(f"  {cluster:25s}: {count}")

    print()
    print(f"Output saved: {csv_path}")
    print()

    # ── Top 10 recommendations ──
    print("Top 10 recommendations (highest expected value):")
    print("-" * 90)
    top10 = df_final.head(10)
    for _, row in top10.iterrows():
        print(f"  {row['cust_id']} | {row['prioridad']:14s} | EV=${row.get('expected_value',0):>10,.0f} | "
              f"{row.get('accion','')[:30]:30s} | {row.get('canal',''):10s} | {row.get('timing','')[:25]}")

    print()
    print("Demo complete. Next steps:")
    print("  - Launch dashboard:  streamlit run dashboard/app.py")
    print("  - Launch API:        uvicorn api.api:app --reload")
    print("  - View notebooks:    jupyter notebook notebooks/")

    return df_final


if __name__ == "__main__":
    main()
