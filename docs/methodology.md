# Methodology

**Loyalty Intelligence System — ML & Statistical Approach**

This document explains the key methodological decisions behind the system: why each technique was chosen, how it guards against common pitfalls, and how the components fit together.

---

## 1. Anti-Leakage Framework

Data leakage is the single largest risk in a prediction system built on temporal snapshots. The system enforces multiple layers of protection.

### 1.1 Temporal Design

Every snapshot has a reference date **t0**. Features use only data **before t0**; targets use only data **after t0**.

```
         12 months pre-t0              t0              12 months post-t0
  ──────────────────────────── │ ────────────────────────────
       Features (X)            │         Target (y)
```

27 monthly snapshots (Jan 2023 -- Mar 2025) create a panel where each customer appears at most once per month. The temporal train/val/test split ensures the model never sees future data during training.

### 1.2 Post-t0 Column Guards

The scoring pipeline automatically drops any post-t0 column before training or scoring:

```python
POST_T0_COLS = ["revenue_post_12m", "spending_post_6m", "txn_count_post_6m",
                "canjea_post", "n_canjes_post"]
```

This guard fires in both `load_or_train_models()` and `score_chunk()`, making it impossible for post-period data to reach a model even if the upstream SQL includes it for analysis purposes.

### 1.3 Cross-Fitted Propensity (5-fold)

Propensity scores estimated on the same data they describe are overfit by construction. The system uses **5-fold StratifiedKFold cross-fitting**:

1. Split the training set into 5 folds stratified by treatment
2. For each fold: train LogisticRegression on the other 4, predict on the held-out fold
3. Each customer's propensity score comes from a model that never saw them during fitting
4. A final model trained on all data is stored for scoring new customers

This prevents propensity scores from leaking label information back into downstream models.

### 1.4 Cross-Fitted T-Learner / X-Learner (2-fold)

The same cross-fitting principle applies to CATE estimation:

- **T-Learner**: 2-fold KFold. For each fold, separate GBR models for treated (mu1) and control (mu0) are trained on the other fold. Each customer's counterfactual predictions come from models that never saw their outcome.
- **X-Learner**: Uses the cross-fitted mu1/mu0 residuals as imputed treatment effects (D1, D0), then fits tau models with another round of 2-fold cross-fitting.

### 1.5 Deterministic Preprocessing

During training, the clustering pipeline computes:
- **Skew flags** per feature (skew > 2 triggers log1p transform)
- **Quantile bounds** per feature (1st and 99th percentile for winsorization)

These parameters are stored in the model artifact and applied identically during scoring. Without this, scoring-time data distributions would leak into the transform (e.g., recalculating percentiles per chunk).

### 1.6 Funnel Pre-Computation Limitation

The Markov funnel states are pre-computed monthly over the full date range (2022--2026). Cumulative counters (e.g., total historical redemptions) include transactions up to each month, not just up to t0. The downstream feature snapshot mitigates this by joining only states where `fecha_fin_mes < t0`, but the counters themselves carry a structural limitation. In production, this should be recomputed per t0 or use a temporal CTE.

---

## 2. Cascade Architecture

A single model cannot simultaneously predict whether a customer redeems, where, for how much, and their total revenue. The cascade decomposes this into 4 specialized models, each operating on the appropriate subset.

### 2.1 Step 1: Ternary Propensity

**Model:** XGBoost multiclass (y=0, y=1, y=2)

| y | Meaning |
|---|---------|
| 0 | No redemption in 12 months post-t0 |
| 1 | First-time activation (redeems, but never had before t0) |
| 2 | Recurring redemption (redeems, and had redeemed before) |

**Why ternary, not binary:** Activations (y=1) and recurrences (y=2) require fundamentally different interventions. y=1 customers need education and incentive; y=2 customers need retention and cross-sell. A binary model (redeem yes/no) would conflate these distinct business actions.

P(redemption) = P(y=1) + P(y=2).

### 2.2 Step 2: Retailer Prediction

**Model:** XGBoost multiclass (5 retailers)
**Input:** Only customers with P(redemption) above threshold

Predicts the most likely retailer for the redemption, enabling targeted offers rather than generic campaigns.

### 2.3 Step 3: Amount Regression

**Model:** XGBoost regression (log-transformed target)
**Input:** Customers predicted to redeem

Estimates the expected redemption amount in points. Log transform handles the right-skewed distribution of redemption amounts.

### 2.4 Step 4: Two-Stage Revenue Model

**Problem:** ~70% of customers have zero revenue in the post-12m window, creating a zero-inflated distribution that a single regression handles poorly.

**Solution:** Two-stage approach:
- **Stage A** (Binary): XGBoost classifier predicting P(revenue > 0)
- **Stage B** (Regression): XGBoost regressor predicting log(revenue) among positives only

Expected revenue = P(revenue > 0) x E[revenue | revenue > 0].

### 2.5 Hyperparameter Tuning

Optuna Bayesian optimization tunes each model's hyperparameters (max_depth, learning_rate, n_estimators, subsample, colsample_bytree, min_child_weight, reg_alpha, reg_lambda). The search uses the temporal validation set, never the test set.

---

## 3. Incrementality

The system measures incrementality at two levels: (1) aggregate lift by spending quintile (production methodology), and (2) individual uplift via causal ML models. Both approaches answer the same question: does redemption drive additional spend, or do high-spenders simply redeem more?

### 3.1 Production Methodology (Lift by Spending Quintile)

The baseline approach, inherited from the existing production system:

1. **Classify customers:**
   - **Redeemer**: redeemed in the post-t0 window (y >= 1)
   - **Potential redeemer**: >= 1,000 points, did NOT redeem
   - **Non-potential**: < 1,000 points (excluded from analysis)

2. **Filters:**
   - Survivorship: exclude customers with pre-spend = 0 or post-spend = 0
   - Outliers: exclude top 1% in both pre and post spend

3. **Stratify by pre-t0 spending quintiles** (NTILE(5) over monetary_total, partitioned by t0). This ensures redeemers are compared against non-redeemers with **similar historical spending levels**.

4. **Lift per quintile:**
   ```
   Lift(Q) = (avg_post_spend_redeemer(Q) - avg_post_spend_potential(Q)) / avg_post_spend_potential(Q)
   ```

5. **Global lift**: weighted average across quintiles, weighted by redeemer spend.

This approach is transparent and directly comparable across periods. Its limitation is that it only controls for one variable (spending level) and cannot isolate individual treatment effects.

### 3.2 PSM Enhancement (Within-Quintile Matching)

Propensity Score Matching improves on the production method by matching on multiple covariates **within each spending quintile**:

- **Covariates:** 13 pre-treatment features (frequency, monetary, redeem rate, retailer entropy, digital affinity, earn velocity, inactivity, points pressure, stock points, redeem count, tenure)
- **Matching:** Nearest neighbor within each quintile, caliper = 0.05 SD of logit propensity
- **Balance check:** Standardized Mean Difference (SMD) < 0.1 for all covariates post-matching
- **Same filters** as production (survivorship, outlier removal) for fair comparison

Matching within quintiles ensures that the propensity-matched control has similar spending level AND similar observable characteristics. This corrects the bias from the production method where redeemers and potentials within the same quintile may differ on other dimensions.

### 3.3 T-Learner (Individual CATE)

Two separate GradientBoostingRegressor models estimate individual treatment effects:
- **mu1**: E[Y | X, T=1] -- expected outcome for treated
- **mu0**: E[Y | X, T=0] -- expected outcome for control

CATE(x) = mu1(x) - mu0(x), the individual treatment effect.

Both models are 2-fold cross-fitted: each customer's counterfactual prediction comes from a model that never saw their outcome during training.

### 3.4 X-Learner (Refinement)

The X-Learner improves CATE estimates when treatment and control groups have different sizes:

1. Compute cross-fitted residuals:
   - D1 = Y_treated - mu0(X_treated) (observed vs counterfactual)
   - D0 = mu1(X_control) - Y_control
2. Train tau1 on D1, tau0 on D0 (both 2-fold cross-fitted)
3. Final CATE = propensity-weighted combination of tau1 and tau0

### 3.5 Outcome Variable

For the production lift method, the outcome is **spending_post_6m** (post-t0 spend), which is the natural business metric.

For the uplift models (T-Learner, X-Learner) used in scoring, the outcome is **monetary_total** (pre-t0 total spend). Using a post-t0 outcome as the uplift target in the scoring pipeline would create temporal contamination: the treatment and outcome would overlap in the same window, inflating apparent uplift by ~70x.

### 3.6 ATT Estimation

Average Treatment Effect on the Treated (ATT) is estimated per quintile and globally, with:
- Bootstrap 95% confidence intervals (1000 iterations)
- Permutation test as primary significance test (distribution-free)
- Paired t-test and Wilcoxon as sensitivity checks

### 3.7 Three Levels of Incrementality

| Method | Level | Controls for | Quintiles by |
|--------|-------|-------------|--------------|
| Production (GitLab) | Per quintile | Pre-t0 spending level | Spending pre |
| PSM (within quintile) | Per quintile | Spending + 13 covariates | Spending pre |
| Uplift model (X-Learner) | Individual | All observable features | CATE estimate |

The production method provides the business-facing lift metric. PSM validates whether the lift holds after controlling for confounders. The uplift model enables individual-level targeting for the decision engine.

---

## 4. Behavioral Clustering

Clustering segments customers by behavior patterns, independent of the prediction targets.

### 4.1 Feature Selection and Preprocessing

8 behavioral features: frequency_monthly_avg, monetary_monthly_avg, redeem_rate, retailer_entropy, pct_redeem_digital, earn_velocity_90, days_since_last_activity, points_pressure.

Preprocessing chain:
1. **Log transform**: Applied to features with skew > 2 (log1p)
2. **Winsorization**: Clip to [1st, 99th] percentile
3. **StandardScaler**: Zero mean, unit variance

All transform parameters are stored at training time and reapplied at scoring time (see Section 1.5).

### 4.2 K Selection

KMeans with K in {4, 5, 6, 7, 8}, optimized by silhouette score on a sample of up to 10,000 points. The best K is selected automatically per training run.

### 4.3 Archetype Assignment

Cluster centroids are matched to 6 predefined archetypes using the **Hungarian algorithm** (linear sum assignment) with weighted profile similarity:

| Archetype | Key Traits | Weights |
|-----------|-----------|---------|
| Heavy Users | High frequency, high spend, fast earning | freq=1, monetary=1, earn_vel=1 |
| Cazadores de Canje | High redeem rate, digital, points pressure | redeem_rate=2.5, digital=1, pressure=1 |
| Dormidos | Long inactivity | inactivity=1 |
| Exploradores | High retailer entropy (multi-retailer) | entropy=1 |
| Digitales | Digital-first redemption | digital=1 |
| En Riesgo | Moderate inactivity + expiring points | inactivity=1.5, pressure=1.5, redeem=0.5 |

The Hungarian algorithm ensures a 1:1 mapping between clusters and archetypes, maximizing the total weighted match.

---

## 5. Decision Engine

The decision engine translates model outputs into actionable recommendations through 5 sequential steps.

### 5.1 Step 1: Priority (Uplift Quantiles)

Priority is based on **causal uplift**, not propensity:

| Priority | Rule |
|----------|------|
| Alta (High) | uplift > Q80 |
| Media (Medium) | uplift > Q60 and uplift <= Q80 |
| Baja (Low) | uplift > 0 and uplift <= Q60 |
| No contactar | uplift <= 0 |

**Critical detail:** Q60 and Q80 are computed **once** on the training set and stored globally. They are not recalculated per scoring chunk. This ensures consistent priority assignment across all batches.

**Boundary operators:** The lower bound uses strict inequality (`>`), the upper bound uses inclusive (`<=`). This prevents overlap between tiers and ensures no customer falls into two categories.

### 5.2 Step 2: Objective (Funnel State)

The customer's funnel state determines the campaign objective:

| Funnel State | Objective |
|--------------|-----------|
| INSCRITO | Activate (first purchase) |
| PARTICIPANTE | Accelerate (earn points faster) |
| POSIBILIDAD_CANJE | Push (convert points to redemption) |
| CANJEADOR | Upsell (increase redemption value) |
| RECURRENTE | Retain (maintain engagement) |
| FUGA | Reactivate (win-back) |

### 5.3 Step 3: Action (Cluster + Overrides)

The behavioral cluster drives the default action type:
- Heavy Users: premium catalog, exclusive access
- Cazadores de Canje: points multiplier, limited-time offers
- Dormidos: reactivation bonus, nostalgia campaign
- En Riesgo: expiry reminders, easy-win redemptions

Overrides apply for specific conditions (new customers < 90 days, high churn probability, points expiring within 30 days).

### 5.4 Step 4: Channel

Channel selection combines digital preference and propensity:

- High digital affinity (pct_digital > 0.5): email, app push, in-app
- Low digital affinity: physical mail, in-store, SMS
- Propensity thresholds modulate channel intensity (high propensity = lighter touch; low propensity = stronger nudge)

### 5.5 Step 5: Timing (Urgency)

| Condition | Urgency |
|-----------|---------|
| Points expiring within 30 days | Immediate |
| Churn probability > 0.7 | High |
| Inactivity > 180 days | High |
| Normal engagement | Standard (next campaign cycle) |

---

## 6. Validation Framework

### 6.1 Temporal Split

| Set | Snapshots | Purpose |
|-----|-----------|---------|
| Train | First 60% of t0s | Model fitting + cross-validation |
| Validation | Next 20% | Hyperparameter tuning, early stopping |
| Test | Final 20% | Unbiased evaluation (never used during training) |

The split is strictly temporal: all train t0s precede all val t0s, which precede all test t0s. This prevents any future information from influencing model selection.

### 6.2 Classification Metrics (Step 1)

Per-class metrics for y=0, y=1, y=2:
- **AUC** (one-vs-rest): Discrimination ability per class
- **F1-score**: Balance between precision and recall
- **Precision / Recall**: Per-class detection rates
- **Brier score**: Calibration quality (lower is better)

Minimum thresholds: F1-macro > 0.70, AUC per class > 0.80, Recall per class > 0.60, Brier < 0.18.

### 6.3 Lift Analysis

- **Cumulative capture curve**: Fraction of actual redeemers captured in the top K% by propensity score
- **Decile lift chart**: Observed redemption rate by propensity decile vs baseline rate
- **Target**: Top 20% captures > 40% of redeemers; top 10% lift > 3x

### 6.4 Calibration

Predicted probabilities are compared against observed rates by decile:
- Plot: predicted vs actual redemption rate per decile
- Ratio: actual / predicted -- ideal is 1.0, acceptable range [0.7, 1.3]
- Systematic over/under-prediction flagged for recalibration

### 6.5 Uplift Validation

- **Qini curves**: Cumulative incremental gain by uplift quantile vs random treatment
- **Quintile decomposition**: Mean outcome by uplift quintile for treated vs control
- **Expected pattern**: Top uplift quintile shows the largest treatment-control gap; bottom quintile shows near-zero or negative effect

---

## 7. Production Safeguards

### 7.1 Chunk-Based Scoring

The 12M+ customer base is scored in chunks of 1M. Each chunk goes through:
1. Post-t0 column guard (drop any leaked columns)
2. Feature validation (check required features exist)
3. Preprocessing with stored parameters (not recomputed)
4. Model inference (predict, never fit)
5. Decision engine rules

### 7.2 Global Quantile Thresholds

Priority quantiles (Q60, Q80) are computed on the training set and stored in the model artifact. They are never recalculated per chunk. This ensures that a customer's priority does not depend on which chunk they fall into.

### 7.3 Feature Validation

Before scoring, the pipeline checks that all required features exist in the input data. Missing clustering features raise an error (hard fail). Missing propensity features emit a warning and are filled with 0 (soft fail with degraded accuracy).

### 7.4 Model Metadata and Versioning

Each trained model artifact includes metadata:
- Training date and row count
- Optimal K and silhouette score
- Number of propensity features
- Version tag

Metadata is saved as both a pickled dict and a standalone JSON file for inspection without loading the full model.

---

## Summary of Key Design Choices

| Decision | Choice | Alternative Considered | Why |
|----------|--------|----------------------|-----|
| Target encoding | Ternary (0/1/2) | Binary (0/1) | Activations and recurrences need different strategies |
| Revenue model | Two-stage (binary + regression) | Single regression | 70% zeros distort single-model predictions |
| Priority metric | Causal uplift | Propensity score | Uplift targets persuadable customers, not just likely ones |
| Propensity estimation | 5-fold cross-fitted | Single-fit on full data | Prevents overfitting leak in propensity scores |
| Uplift outcome | Pre-t0 monetary_total | Post-t0 spending | Post-t0 outcome creates temporal contamination |
| Quantile thresholds | Global from training | Per-chunk | Consistent scoring across batches |
| Clustering transforms | Stored from training | Recomputed per chunk | Deterministic, no scoring-time data leakage |
