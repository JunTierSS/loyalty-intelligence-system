"""
Loyalty Intelligence System — Pipeline de Scoring Mensual
Fase 3: Productizacion

Ejecuta scoring mensual sobre los 12M clientes de produccion en chunks.
Pipeline: BigQuery → Chunks 1M → Cascade (4 XGBoost) + Clustering + Uplift → BigQuery

Cascade:
  Paso 1: XGBoost ternary (y=0/1/2) → P(no canje), P(activacion), P(recurrencia)
  Paso 2: XGBoost multiclass → retailer recomendado (canjeadores)
  Paso 3: XGBoost regressor → monto puntos estimado (canjeadores)
  Paso 4: Two-stage XGBoost → estimated revenue (Stage A: P(revenue>0), Stage B: log-regression)

Uso:
  # Mock local
  python scoring_pipeline.py --mock

  # Produccion (requiere credenciales GCP)
  python scoring_pipeline.py --project my-gcp-project --month 2025-03
"""

import argparse
import logging
import sys
import os
import time
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("scoring_pipeline")

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"

# ── Config ──────────────────────────────────────────────────────────────────
CHUNK_SIZE = 1_000_000  # 1M clientes por chunk
DATASET = "loyalty_intelligence"
SCORING_TABLE = "scoring_output"
PROJECT = "my-gcp-project"


def parse_args():
    parser = argparse.ArgumentParser(description="Scoring Pipeline Mensual")
    parser.add_argument("--mock", action="store_true", help="Usar datos mock locales")
    parser.add_argument("--project", type=str, default=PROJECT, help="GCP project ID")
    parser.add_argument("--month", type=str, default=None,
                        help="Mes de scoring (YYYY-MM). Default: mes actual")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--save-models", action="store_true",
                        help="Guardar modelos entrenados a disco")
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════
# PASO 1: CARGAR / ENTRENAR MODELOS
# ══════════════════════════════════════════════════════════════════════════
def load_or_train_models(df_train, save=False):
    """Train all models on training data, or load from disk if available."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    import xgboost as xgb

    models = {}

    # ── Save target columns for cascade training BEFORE dropping post-t0 ──
    CASCADE_TARGET_COLS = ["retailer_post", "monto_redeem_post", "revenue_post_12m"]
    cascade_targets = {}
    for col in CASCADE_TARGET_COLS:
        if col in df_train.columns:
            cascade_targets[col] = df_train[col].copy()

    # ── Drop post-t0 columns to prevent accidental leakage ──
    POST_T0_COLS = ["revenue_post_12m", "spending_post_6m", "txn_count_post_6m",
                    "canjea_post", "n_canjes_post", "retailer_post", "monto_redeem_post"]
    leaked = [c for c in POST_T0_COLS if c in df_train.columns]
    if leaked:
        log.info(f"Dropping post-t0 columns from training data: {leaked}")
        df_train = df_train.drop(columns=leaked)

    # ── Clustering ──
    CLUSTER_FEATURES = [
        "frequency_monthly_avg", "monetary_monthly_avg", "redeem_rate",
        "retailer_entropy", "pct_redeem_digital", "earn_velocity_90",
        "days_since_last_activity", "points_pressure",
    ]
    available = [f for f in CLUSTER_FEATURES if f in df_train.columns]
    X_clust = df_train[available].fillna(0).copy()

    # Contextual fillna
    if "days_since_last_activity" in X_clust.columns:
        X_clust["days_since_last_activity"] = X_clust["days_since_last_activity"].fillna(999)

    # Log transform skewed features (store flags for scoring)
    skew_flags = {}
    for col in X_clust.columns:
        skew_flags[col] = bool(X_clust[col].skew() > 2)
        if skew_flags[col]:
            X_clust[col] = np.log1p(X_clust[col])

    # Winsorize (store bounds for scoring)
    quantile_bounds = {}
    for col in X_clust.columns:
        p1, p99 = X_clust[col].quantile([0.01, 0.99])
        quantile_bounds[col] = (float(p1), float(p99))
        X_clust[col] = X_clust[col].clip(p1, p99)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clust)

    # Find optimal K (range 4-8 to allow 6 archetypes including "En Riesgo")
    from sklearn.metrics import silhouette_score
    best_k, best_sil = 5, -1
    for k in range(4, 9):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels, sample_size=min(10000, len(X_scaled)))
        if sil > best_sil:
            best_k, best_sil = k, sil

    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)

    # Name clusters
    profiles = pd.DataFrame(X_scaled, columns=available)
    profiles["cluster"] = kmeans.labels_
    profile_means = profiles.groupby("cluster").mean()

    # Normalize for naming
    ranges = profile_means.max() - profile_means.min()
    pn = (profile_means - profile_means.min()) / ranges.where(ranges > 0, 1)
    pn = pn.fillna(0)

    ARCHETYPES = {
        "Heavy Users": {"frequency_monthly_avg": 1, "monetary_monthly_avg": 1, "earn_velocity_90": 1},
        "Cazadores de Canje": {"redeem_rate": 2.5, "pct_redeem_digital": 1, "points_pressure": 1},
        "Dormidos": {"days_since_last_activity": 1},
        "Exploradores": {"retailer_entropy": 1},
        "Digitales": {"pct_redeem_digital": 1},
        "En Riesgo": {"days_since_last_activity": 1.5, "points_pressure": 1.5, "redeem_rate": 0.5},
    }

    from scipy.optimize import linear_sum_assignment
    arch_names = list(ARCHETYPES.keys())[:best_k]
    cost = np.zeros((best_k, len(arch_names)))
    for i in range(best_k):
        for j, name in enumerate(arch_names):
            score = sum(pn.loc[i].get(f, 0) * w for f, w in ARCHETYPES[name].items())
            cost[i, j] = -score
    row_ind, col_ind = linear_sum_assignment(cost)
    cluster_names = {}
    for r, c in zip(row_ind, col_ind):
        cluster_names[r] = arch_names[c]

    models["kmeans"] = kmeans
    models["cluster_scaler"] = scaler
    models["cluster_features"] = available
    models["cluster_names"] = cluster_names
    models["cluster_skew_flags"] = skew_flags
    models["cluster_quantile_bounds"] = quantile_bounds
    models["K"] = best_k

    log.info(f"Clustering: K={best_k}, silhouette={best_sil:.4f}")
    for k, v in cluster_names.items():
        log.info(f"  Cluster {k} = {v}")

    # ── Feature setup for cascade (matching notebook 01/02) ──
    CATEGORICAL_FEATURES = ["tier", "gender", "city", "dominant_retailer",
                            "funnel_state_at_t0", "status"]
    BOOLEAN_FEATURES = [
        "cust_active_store_card_flg", "cust_active_deb_flg", "cust_active_omp_flg",
        "contact_email_flg", "contact_phone_flg", "contact_push_flg",
        "redeem_capacity", "is_cyber_month", "is_holiday_month",
    ]
    ID_COLS = ["cust_id", "t0", "fecha_proceso"]
    TARGET_RELATED = ["has_redeemed_before_t0", "canjea_post", "n_canjes_post",
                      "revenue_post_12m", "retailer_post", "monto_redeem_post",
                      "spending_post_6m", "txn_count_post_6m", "treatment"]
    EXCLUDED = set(ID_COLS + ["y"] + TARGET_RELATED + CATEGORICAL_FEATURES + BOOLEAN_FEATURES)
    cat_avail = [f for f in CATEGORICAL_FEATURES if f in df_train.columns]
    bool_avail = [f for f in BOOLEAN_FEATURES if f in df_train.columns]
    num_avail = [c for c in df_train.columns if c not in EXCLUDED and c not in CATEGORICAL_FEATURES and c not in BOOLEAN_FEATURES]
    FEATURE_COLS = cat_avail + bool_avail + num_avail

    # Ordinal encode categoricals (fit on train only)
    ord_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    df_train[cat_avail] = ord_enc.fit_transform(df_train[cat_avail])
    for col in bool_avail:
        df_train[col] = df_train[col].astype(int)

    models["ordinal_encoder"] = ord_enc
    models["cat_features"] = cat_avail
    models["bool_features"] = bool_avail
    models["cascade_features"] = FEATURE_COLS
    log.info(f"Cascade features: {len(FEATURE_COLS)} ({len(cat_avail)} cat, {len(bool_avail)} bool, {len(num_avail)} num)")

    # ── Paso 1: XGBoost ternary classification (y=0/1/2) ──
    y_target = df_train["y"].values.astype(int) if "y" in df_train.columns else np.zeros(len(df_train), dtype=int)
    X_cascade = df_train[FEATURE_COLS].values.astype(np.float32)

    # Class weights (inverse frequency)
    vc = pd.Series(y_target).value_counts()
    n_cls = len(vc)
    weight_map = {cls: len(y_target) / (n_cls * count) for cls, count in vc.items()}
    w_train = np.array([weight_map[y] for y in y_target])

    model1 = xgb.XGBClassifier(
        objective="multi:softprob", num_class=3, tree_method="hist",
        eval_metric="mlogloss", n_estimators=500, max_depth=6,
        learning_rate=0.03, subsample=0.9, colsample_bytree=0.6,
        min_child_weight=5, gamma=0.1, reg_alpha=0.5, reg_lambda=0.01,
        random_state=42, verbosity=0,
    )
    model1.fit(X_cascade, y_target, sample_weight=w_train)
    models["cascade_paso1"] = model1
    log.info(f"Paso 1 (ternary): trained on {len(X_cascade):,} rows, classes={sorted(vc.index.tolist())}")

    # ── Propensity (cross-fitted, for uplift) ──
    PSM_FEATURES = [
        "frequency_monthly_avg", "monetary_monthly_avg", "redeem_rate",
        "retailer_entropy", "pct_redeem_digital", "earn_velocity_90",
        "days_since_last_activity", "points_pressure", "stock_points_at_t0",
        "redeem_count_pre", "frequency_total", "monetary_total", "tenure_months",
    ]
    psm_avail = [f for f in PSM_FEATURES if f in df_train.columns]
    treatment = (df_train["y"] >= 1).astype(int) if "y" in df_train.columns else pd.Series(0, index=df_train.index)

    X_psm = df_train[psm_avail].fillna(0)
    prop_scores = np.zeros(len(df_train))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, val_idx in skf.split(X_psm, treatment):
        fold_model = LogisticRegression(max_iter=1000, random_state=42)
        fold_model.fit(X_psm.iloc[train_idx], treatment.iloc[train_idx])
        prop_scores[val_idx] = fold_model.predict_proba(X_psm.iloc[val_idx])[:, 1]

    prop_model = LogisticRegression(max_iter=1000, random_state=42)
    prop_model.fit(X_psm, treatment)
    models["propensity_model"] = prop_model
    models["psm_features"] = psm_avail
    log.info(f"Propensity model trained (cross-fitted) on {len(psm_avail)} features")

    # ── Uplift (T-Learner with XGBoost) ──
    # NOTE: Uses monetary_total (pre-t0) as outcome proxy to avoid post-t0 leakage.
    # spending_post_6m is available in notebooks for causal analysis but NOT used here
    # because it introduces temporal contamination (treatment and outcome in same window).
    treated_mask = treatment == 1
    if treated_mask.sum() > 50 and (~treated_mask).sum() > 50:
        uplift_features = psm_avail
        X_t = df_train.loc[treated_mask, uplift_features].fillna(0)
        X_c = df_train.loc[~treated_mask, uplift_features].fillna(0)

        # Use monetary_total as outcome (pre-t0 proxy, avoids post-t0 leakage)
        y_col = "monetary_total"
        log.info(f"Uplift outcome: {y_col}")

        y_t = df_train.loc[treated_mask, y_col].fillna(0)
        y_c = df_train.loc[~treated_mask, y_col].fillna(0)

        model_t = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42,
                                    verbosity=0, n_jobs=-1)
        model_c = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42,
                                    verbosity=0, n_jobs=-1)
        model_t.fit(X_t, y_t)
        model_c.fit(X_c, y_c)

        models["uplift_model_t"] = model_t
        models["uplift_model_c"] = model_c
        models["uplift_features"] = uplift_features
        log.info("Uplift T-Learner trained")
    else:
        log.warning("Not enough treated/control for uplift model")

    # ── Paso 2: Retailer prediction (canjeadores only) ──
    RETAILERS_TARGET = ["STOREA", "STOREB", "STOREC", "STORED"]
    if "retailer_post" in cascade_targets:
        retailer_series = cascade_targets["retailer_post"]
        mask_ret = retailer_series.isin(RETAILERS_TARGET)
        if mask_ret.sum() > 20:
            le_ret = LabelEncoder()
            le_ret.fit(RETAILERS_TARGET)
            X_ret = df_train.loc[mask_ret, FEATURE_COLS].values.astype(np.float32)
            y_ret = le_ret.transform(retailer_series[mask_ret])
            # Class weights
            vc_ret = pd.Series(y_ret).value_counts()
            w_map_ret = {c: len(y_ret) / (len(vc_ret) * n) for c, n in vc_ret.items()}
            w_ret = np.array([w_map_ret[y] for y in y_ret])

            model2 = xgb.XGBClassifier(
                objective="multi:softprob", num_class=len(RETAILERS_TARGET),
                tree_method="hist", eval_metric="mlogloss",
                n_estimators=500, max_depth=6, learning_rate=0.03,
                subsample=0.9, colsample_bytree=0.6, min_child_weight=5,
                random_state=42, verbosity=0,
            )
            model2.fit(X_ret, y_ret, sample_weight=w_ret)
            models["cascade_paso2"] = model2
            models["retailer_encoder"] = le_ret
            models["retailers_target"] = RETAILERS_TARGET
            log.info(f"Paso 2 (retailer): trained on {mask_ret.sum():,} canjeadores, classes={RETAILERS_TARGET}")
        else:
            log.warning("Not enough canjeadores for Paso 2 retailer model")
    else:
        log.warning("retailer_post column not available, skipping Paso 2")

    # ── Paso 3: Amount regression (canjeadores with monto > 0) ──
    if "monto_redeem_post" in cascade_targets:
        monto_series = cascade_targets["monto_redeem_post"]
        mask_monto = monto_series.notna() & (monto_series > 0)
        if mask_monto.sum() > 20:
            X_monto = df_train.loc[mask_monto, FEATURE_COLS].values.astype(np.float32)
            y_monto = monto_series[mask_monto].values.astype(np.float32)

            model3 = xgb.XGBRegressor(
                objective="reg:squarederror", tree_method="hist",
                n_estimators=500, max_depth=6, learning_rate=0.03,
                subsample=0.9, colsample_bytree=0.6, min_child_weight=5,
                random_state=42, verbosity=0,
            )
            model3.fit(X_monto, y_monto)
            models["cascade_paso3"] = model3
            log.info(f"Paso 3 (amount): trained on {mask_monto.sum():,} rows, mean={y_monto.mean():,.0f}")
        else:
            log.warning("Not enough data for Paso 3 amount model")
    else:
        log.warning("monto_redeem_post column not available, skipping Paso 3")

    # ── Paso 4: Two-stage revenue (Stage A: binary, Stage B: log-regression) ──
    if "revenue_post_12m" in cascade_targets:
        y_rev = cascade_targets["revenue_post_12m"].fillna(0).values
        y_rev_bin = (y_rev > 0).astype(int)
        mask_rev_pos = y_rev > 0

        if mask_rev_pos.sum() > 20:
            # Stage A: P(revenue > 0)
            X_rev = df_train[FEATURE_COLS].values.astype(np.float32)
            spw = max(1.0, (~mask_rev_pos).sum() / mask_rev_pos.sum())  # scale_pos_weight
            model4a = xgb.XGBClassifier(
                objective="binary:logistic", tree_method="hist", eval_metric="auc",
                n_estimators=400, max_depth=5, learning_rate=0.03,
                subsample=0.9, colsample_bytree=0.6, scale_pos_weight=min(spw, 5.0),
                random_state=42, verbosity=0,
            )
            model4a.fit(X_rev, y_rev_bin)

            # Stage B: log1p(revenue) on positives only
            X_rev_pos = df_train.loc[mask_rev_pos, FEATURE_COLS].values.astype(np.float32)
            y_rev_log = np.log1p(y_rev[mask_rev_pos])
            model4b = xgb.XGBRegressor(
                objective="reg:squarederror", tree_method="hist",
                n_estimators=400, max_depth=5, learning_rate=0.03,
                subsample=0.9, colsample_bytree=0.6,
                random_state=42, verbosity=0,
            )
            model4b.fit(X_rev_pos, y_rev_log)

            models["cascade_paso4a"] = model4a
            models["cascade_paso4b"] = model4b
            pct_pos = mask_rev_pos.mean() * 100
            log.info(f"Paso 4 (revenue): Stage A trained ({pct_pos:.1f}% positive), Stage B on {mask_rev_pos.sum():,} positives")
        else:
            log.warning("Not enough positive-revenue rows for Paso 4")
    else:
        log.warning("revenue_post_12m column not available, skipping Paso 4")

    # ── Compute global priority quantiles on training data ──
    if "uplift_model_t" in models and "uplift_model_c" in models:
        X_uplift = df_train[models["uplift_features"]].fillna(0)
        mu1 = models["uplift_model_t"].predict(X_uplift)
        mu0 = models["uplift_model_c"].predict(X_uplift)
        train_uplift = mu1 - mu0
        pos_uplift = train_uplift[train_uplift > 0]
        if len(pos_uplift) > 0:
            models["priority_q60"] = float(np.quantile(pos_uplift, 0.60))
            models["priority_q80"] = float(np.quantile(pos_uplift, 0.80))
            log.info(f"Global priority thresholds: Q60={models['priority_q60']:.0f}, Q80={models['priority_q80']:.0f}")

    # Save with metadata for versioning
    if save:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        models["_metadata"] = {
            "training_date": datetime.now().isoformat(),
            "n_training_rows": len(df_train),
            "k_clusters": best_k,
            "silhouette_score": round(best_sil, 4),
            "n_features_propensity": len(psm_avail),
            "cascade_models": ["paso1_ternary", "paso2_retailer", "paso3_amount", "paso4_revenue"],
            "version": "2.0",
        }
        with open(MODELS_DIR / "models.pkl", "wb") as f:
            pickle.dump(models, f)
        # Also save metadata as JSON for easy inspection
        with open(MODELS_DIR / "models_metadata.json", "w") as f:
            json.dump(models["_metadata"], f, indent=2)
        log.info(f"Models saved to {MODELS_DIR / 'models.pkl'} with metadata")

    return models


# ══════════════════════════════════════════════════════════════════════════
# PASO 2: SCORING DE UN CHUNK
# ══════════════════════════════════════════════════════════════════════════
def score_chunk(df_chunk, models):
    """Apply all models to a chunk of customers. Only transforms/predicts, never fits."""

    # ── Drop post-t0 columns if present (prevent leakage) ──
    POST_T0_COLS = ["revenue_post_12m", "spending_post_6m", "txn_count_post_6m",
                    "canjea_post", "n_canjes_post"]
    leaked = [c for c in POST_T0_COLS if c in df_chunk.columns]
    if leaked:
        df_chunk = df_chunk.drop(columns=leaked)

    # ── Feature validation ──
    clust_feats = models["cluster_features"]
    missing_clust = [f for f in clust_feats if f not in df_chunk.columns]
    if missing_clust:
        raise ValueError(f"Missing clustering features: {missing_clust}")

    psm_feats = models.get("psm_features", [])
    missing_psm = [f for f in psm_feats if f not in df_chunk.columns]
    if missing_psm:
        log.warning(f"Missing propensity features (will fill 0): {missing_psm}")

    # ── Encode features for cascade models ──
    cat_feats = models.get("cat_features", [])
    bool_feats = models.get("bool_features", [])
    cascade_feats = models.get("cascade_features", [])
    if cat_feats and "ordinal_encoder" in models:
        cat_present = [f for f in cat_feats if f in df_chunk.columns]
        if cat_present:
            df_chunk[cat_present] = models["ordinal_encoder"].transform(df_chunk[cat_present])
        for col in bool_feats:
            if col in df_chunk.columns:
                df_chunk[col] = df_chunk[col].astype(int)

    # ── Clustering (use training-derived preprocessing params) ──
    X_clust = df_chunk[clust_feats].fillna(0).copy()
    if "days_since_last_activity" in X_clust.columns:
        X_clust["days_since_last_activity"] = X_clust["days_since_last_activity"].fillna(999)
    skew_flags = models.get("cluster_skew_flags", {})
    for col in X_clust.columns:
        if skew_flags.get(col, False):
            X_clust[col] = np.log1p(X_clust[col])
    quantile_bounds = models.get("cluster_quantile_bounds", {})
    for col in X_clust.columns:
        if col in quantile_bounds:
            p1, p99 = quantile_bounds[col]
            X_clust[col] = X_clust[col].clip(p1, p99)

    X_scaled = models["cluster_scaler"].transform(X_clust)
    df_chunk["cluster"] = models["kmeans"].predict(X_scaled)
    df_chunk["cluster_name"] = df_chunk["cluster"].map(models["cluster_names"])

    # ── Paso 1: Ternary propensity (XGBoost) ──
    if "cascade_paso1" in models and cascade_feats:
        X_casc = df_chunk[cascade_feats].values.astype(np.float32)
        proba = models["cascade_paso1"].predict_proba(X_casc)
        df_chunk["p_y0"] = proba[:, 0]
        df_chunk["p_y1"] = proba[:, 1]
        df_chunk["p_y2"] = proba[:, 2] if proba.shape[1] > 2 else 0
        df_chunk["propensity_score"] = 1 - proba[:, 0]  # P(canje) = P(y>=1)
    else:
        # Fallback: LogReg binary propensity
        psm_feats_score = models["psm_features"]
        X_psm = df_chunk[psm_feats_score].fillna(0)
        df_chunk["propensity_score"] = models["propensity_model"].predict_proba(X_psm)[:, 1]
        df_chunk["p_y0"] = 1 - df_chunk["propensity_score"]
        df_chunk["p_y1"] = df_chunk["propensity_score"]
        df_chunk["p_y2"] = 0

    # ── Paso 2: Retailer prediction ──
    if "cascade_paso2" in models and cascade_feats:
        X_casc = df_chunk[cascade_feats].values.astype(np.float32)
        ret_proba = models["cascade_paso2"].predict_proba(X_casc)
        ret_pred = models["cascade_paso2"].predict(X_casc)
        le = models["retailer_encoder"]
        df_chunk["retailer_recomendado"] = le.inverse_transform(ret_pred)
        # Top-2 retailers
        top2 = np.argsort(ret_proba, axis=1)[:, -2:]
        df_chunk["retailer_top2"] = [
            f"{le.inverse_transform([t2[1]])[0]},{le.inverse_transform([t2[0]])[0]}"
            for t2 in top2
        ]
        df_chunk["retailer_confidence"] = np.max(ret_proba, axis=1)
    elif "dominant_retailer" in df_chunk.columns:
        df_chunk["retailer_recomendado"] = df_chunk["dominant_retailer"]
        df_chunk.loc[df_chunk["retailer_recomendado"] == "NINGUNO", "retailer_recomendado"] = "STOREA"

    # ── Paso 3: Amount regression (estimated redemption points) ──
    if "cascade_paso3" in models and cascade_feats:
        X_casc = df_chunk[cascade_feats].values.astype(np.float32)
        df_chunk["estimated_points"] = np.maximum(models["cascade_paso3"].predict(X_casc), 0)
        # Weight by P(canje)
        df_chunk["expected_points"] = df_chunk["propensity_score"] * df_chunk["estimated_points"]
    else:
        df_chunk["estimated_points"] = 0
        df_chunk["expected_points"] = 0

    # ── Paso 4: Two-stage revenue ──
    if "cascade_paso4a" in models and "cascade_paso4b" in models and cascade_feats:
        X_casc = df_chunk[cascade_feats].values.astype(np.float32)
        p_rev_pos = models["cascade_paso4a"].predict_proba(X_casc)[:, 1]
        log_rev = models["cascade_paso4b"].predict(X_casc)
        rev_if_pos = np.expm1(np.maximum(log_rev, 0))
        df_chunk["estimated_revenue"] = np.maximum(p_rev_pos * rev_if_pos, 0)
        df_chunk["p_revenue_positive"] = p_rev_pos
    else:
        df_chunk["estimated_revenue"] = 0
        df_chunk["p_revenue_positive"] = 0

    # ── Uplift ──
    if "uplift_model_t" in models:
        uf = models["uplift_features"]
        X_up = df_chunk[uf].fillna(0)
        mu1 = models["uplift_model_t"].predict(X_up)
        mu0 = models["uplift_model_c"].predict(X_up)
        df_chunk["uplift_x"] = mu1 - mu0
    else:
        df_chunk["uplift_x"] = 0

    # ── Expected Value ──
    df_chunk["expected_value"] = df_chunk["propensity_score"] * df_chunk["uplift_x"]

    # ── Prioridad (using global quantiles from training, not per-chunk) ──
    df_chunk["prioridad"] = "Baja"
    mask_neg = df_chunk["uplift_x"] <= 0
    df_chunk.loc[mask_neg, "prioridad"] = "No contactar"

    q60 = models.get("priority_q60")
    q80 = models.get("priority_q80")
    if q60 is None or q80 is None:
        raise ValueError("Global priority quantiles (Q60/Q80) missing from models. Retrain with --save-models.")
    # Boundary: > Q60 (exclusive), <= Q80 (inclusive) — consistent with decision engine
    df_chunk.loc[(~mask_neg) & (df_chunk["uplift_x"] > q80), "prioridad"] = "Alta"
    df_chunk.loc[(~mask_neg) & (df_chunk["uplift_x"] > q60) & (df_chunk["uplift_x"] <= q80), "prioridad"] = "Media"

    # ── Objetivo (from funnel) ──
    OBJETIVO_FUNNEL = {
        "INSCRITO": "Activar primera compra",
        "PARTICIPANTE": "Acelerar acumulacion de puntos",
        "POSIBILIDAD_CANJE": "Empujar primer canje",
        "CANJEADOR": "Generar recurrencia de canje",
        "RECURRENTE": "Retener y aumentar ticket",
        "FUGA": "Reactivar urgente",
    }
    if "funnel_state_at_t0" in df_chunk.columns:
        df_chunk["objetivo"] = df_chunk["funnel_state_at_t0"].map(OBJETIVO_FUNNEL)

    # ── Accion (from cluster) ──
    ACCION_CLUSTER = {
        "Cazadores de Canje": "Descuento personalizado",
        "Exploradores": "Educar sobre beneficios del canje",
        "Heavy Users": "Experiencia premium exclusiva",
        "Dormidos": "Oferta directa de reactivacion",
        "Digitales": "Oferta exclusiva canal digital",
        "En Riesgo": "Retencion preventiva con puntos bonus",
    }
    df_chunk["accion"] = df_chunk["cluster_name"].map(ACCION_CLUSTER).fillna("Contacto general")

    # Override: negative uplift
    df_chunk.loc[mask_neg, "accion"] = "No contactar (uplift negativo)"

    # Override: funnel FUGA
    if "funnel_state_at_t0" in df_chunk.columns:
        mask_fuga = df_chunk["funnel_state_at_t0"] == "FUGA"
        df_chunk.loc[mask_fuga & ~mask_neg, "accion"] = "Reactivacion urgente"
        mask_inscrito = df_chunk["funnel_state_at_t0"] == "INSCRITO"
        df_chunk.loc[mask_inscrito & ~mask_neg, "accion"] = "Activar primera compra"

    # ── Canal ──
    if "pct_redeem_digital" in df_chunk.columns:
        df_chunk["canal"] = np.where(
            df_chunk["pct_redeem_digital"] > 0.5, "Digital",
            np.where(df_chunk["propensity_score"] > 0.3, "Email",
            np.where(df_chunk["propensity_score"] > 0.15, "Presencial", "Push"))
        )
    else:
        df_chunk["canal"] = "Email"

    # ── Timing ──
    df_chunk["timing"] = "Normal"
    if "funnel_state_at_t0" in df_chunk.columns:
        df_chunk.loc[df_chunk["funnel_state_at_t0"] == "FUGA", "timing"] = "Urgente (fuga)"
    if "points_pressure" in df_chunk.columns:
        df_chunk.loc[df_chunk["points_pressure"] > 0.5, "timing"] = "Urgente (puntos por vencer)"
    if "propensity_score" in df_chunk.columns:
        df_chunk.loc[df_chunk["propensity_score"] > 0.4, "timing"] = "Inmediato (alta probabilidad)"
    if "days_since_last_activity" in df_chunk.columns:
        df_chunk.loc[df_chunk["days_since_last_activity"] > 300, "timing"] = "Pronto (estancado)"

    return df_chunk


# ══════════════════════════════════════════════════════════════════════════
# PASO 3: MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════
def run_mock(args):
    """Run pipeline on mock data."""
    log.info("=== SCORING PIPELINE (MOCK) ===")

    # Load mock data via decision engine notebook
    import nbformat
    nb_path = BASE_DIR / "notebooks" / "05_decision_engine.ipynb"
    nb = nbformat.read(str(nb_path), as_version=4)

    # Need imports + load-data cell
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
    os.chdir(str(BASE_DIR / "notebooks"))

    old_stdout = sys.stdout
    from io import StringIO
    sys.stdout = StringIO()
    try:
        exec(script, ns)
    finally:
        sys.stdout = old_stdout
        os.chdir(old_dir)

    df = ns["df"]
    log.info(f"Mock data loaded: {df.shape}")

    # Split train/score
    t0s = sorted(df["t0"].unique())
    train_t0s = t0s[:-3]
    score_t0 = t0s[-1]

    df_train = df[df["t0"].isin(train_t0s)]
    df_score = df[df["t0"] == score_t0].copy()

    log.info(f"Train: {len(df_train):,} rows ({len(train_t0s)} t0s)")
    log.info(f"Score: {len(df_score):,} rows (t0={score_t0})")

    # Train models
    models = load_or_train_models(df_train, save=args.save_models)

    # Score
    t_start = time.time()
    n_chunks = max(1, len(df_score) // args.chunk_size + 1)
    results = []

    for i in range(n_chunks):
        chunk = df_score.iloc[i * args.chunk_size: (i + 1) * args.chunk_size].copy()
        if len(chunk) == 0:
            break
        log.info(f"Scoring chunk {i+1}/{n_chunks}: {len(chunk):,} rows")
        scored = score_chunk(chunk, models)
        results.append(scored)

    df_scored = pd.concat(results, ignore_index=True)
    elapsed = time.time() - t_start
    log.info(f"Scoring complete: {len(df_scored):,} rows in {elapsed:.1f}s")

    # Output columns
    OUTPUT_COLS = [
        "cust_id", "prioridad", "expected_value", "propensity_score",
        "p_y0", "p_y1", "p_y2",
        "uplift_x", "retailer_recomendado", "retailer_confidence",
        "estimated_points", "expected_points",
        "estimated_revenue", "p_revenue_positive",
        "objetivo", "accion", "canal", "timing",
        "cluster_name", "funnel_state_at_t0",
    ]
    output_cols = [c for c in OUTPUT_COLS if c in df_scored.columns]
    df_final = df_scored[output_cols].copy()

    # Sort
    prio_order = {"Alta": 0, "Media": 1, "Baja": 2, "No contactar": 3}
    df_final["_sort"] = df_final["prioridad"].map(prio_order)
    df_final = df_final.sort_values(["_sort", "expected_value"], ascending=[True, False]).drop(columns="_sort")

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    month_str = score_t0.strftime("%Y%m") if hasattr(score_t0, "strftime") else str(score_t0)[:7].replace("-", "")
    csv_path = output_dir / f"scoring_{month_str}.csv"
    df_final.to_csv(csv_path, index=False)
    log.info(f"Output saved: {csv_path}")

    # Summary
    log.info("=== RESUMEN ===")
    for prio, grp in df_final.groupby("prioridad"):
        n = len(grp)
        pct = n / len(df_final) * 100
        ev = grp["expected_value"].mean() if "expected_value" in grp.columns else 0
        log.info(f"  {prio}: {n:,} ({pct:.1f}%) | EV avg=${ev:,.0f}")

    log.info(f"Total: {len(df_final):,} clientes scored")
    return df_final


def run_bigquery(args):
    """Run pipeline on BigQuery production data."""
    log.info("=== SCORING PIPELINE (BIGQUERY) ===")
    log.info(f"Project: {args.project}")
    log.info(f"Month: {args.month}")

    try:
        from google.cloud import bigquery
    except ImportError:
        log.error("google-cloud-bigquery not installed. Run: pip install google-cloud-bigquery")
        sys.exit(1)

    client = bigquery.Client(project=args.project)

    # Get total client count
    count_query = f"""
    SELECT COUNT(DISTINCT cust_id) AS n
    FROM `{args.project}.{DATASET}.customer_snapshot`
    WHERE t0 = '{args.month}-01'
    """
    n_total = client.query(count_query).to_dataframe().iloc[0]["n"]
    log.info(f"Total clients to score: {n_total:,}")

    n_chunks = max(1, int(np.ceil(n_total / args.chunk_size)))
    log.info(f"Chunks: {n_chunks} x {args.chunk_size:,}")

    # Load training data (last 18 months)
    log.info("Loading training data...")
    train_query = f"""
    SELECT *
    FROM `{args.project}.{DATASET}.customer_snapshot`
    WHERE t0 < '{args.month}-01'
    ORDER BY t0
    """
    df_train = client.query(train_query).to_dataframe()
    log.info(f"Training data: {len(df_train):,} rows")

    # Train models
    models = load_or_train_models(df_train, save=args.save_models)

    # Score in chunks
    all_results = []
    t_start = time.time()

    for i in range(n_chunks):
        offset = i * args.chunk_size
        chunk_query = f"""
        SELECT *
        FROM `{args.project}.{DATASET}.customer_snapshot`
        WHERE t0 = '{args.month}-01'
        ORDER BY cust_id
        LIMIT {args.chunk_size} OFFSET {offset}
        """
        log.info(f"Chunk {i+1}/{n_chunks}: loading from BQ...")
        df_chunk = client.query(chunk_query).to_dataframe()

        if len(df_chunk) == 0:
            break

        log.info(f"  Scoring {len(df_chunk):,} rows...")
        scored = score_chunk(df_chunk, models)
        all_results.append(scored)

    df_scored = pd.concat(all_results, ignore_index=True)
    elapsed = time.time() - t_start
    log.info(f"All chunks scored: {len(df_scored):,} rows in {elapsed:.1f}s")

    # Write back to BigQuery
    OUTPUT_COLS = [
        "cust_id", "prioridad", "expected_value", "propensity_score",
        "p_y0", "p_y1", "p_y2",
        "uplift_x", "retailer_recomendado", "retailer_confidence",
        "estimated_points", "expected_points",
        "estimated_revenue", "p_revenue_positive",
        "objetivo", "accion", "canal", "timing",
        "cluster_name", "funnel_state_at_t0",
    ]
    output_cols = [c for c in OUTPUT_COLS if c in df_scored.columns]
    df_final = df_scored[output_cols].copy()
    df_final["scoring_date"] = args.month + "-01"

    table_ref = f"{args.project}.{DATASET}.{SCORING_TABLE}"
    log.info(f"Writing to {table_ref}...")

    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
    )
    job = client.load_table_from_dataframe(df_final, table_ref, job_config=job_config)
    job.result()
    log.info(f"Written {len(df_final):,} rows to {table_ref}")

    # Also save CSV locally
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"scoring_{args.month.replace('-', '')}.csv"
    df_final.to_csv(csv_path, index=False)
    log.info(f"CSV backup: {csv_path}")

    return df_final


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    args = parse_args()

    if args.month is None:
        args.month = datetime.now().strftime("%Y-%m")

    if args.mock:
        run_mock(args)
    else:
        run_bigquery(args)
