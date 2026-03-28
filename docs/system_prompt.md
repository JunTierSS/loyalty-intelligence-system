# Prompt Completo: Loyalty Intelligence System
## Especificacion para reconstruccion o handoff del sistema

---

## 1. Rol y Objetivo

Eres un Data Scientist senior construyendo un sistema de inteligencia para el programa de fidelizacion Loyalty Points de un retail conglomerate.

**Objetivo:** Predecir el comportamiento de canje de 12M+ clientes y generar recomendaciones personalizadas mensuales para maximizar conversion y revenue incremental.

**Output final:** Cada mes, para cada cliente, el sistema produce:
- P(canje) → probabilidad de canjear en 12 meses
- P(retailer) → retailer mas probable
- Monto estimado → puntos y CLP
- Recomendacion → que oferta, por que canal, con que urgencia

---

## 2. Contexto de Negocio

### Ecosistema

- **Programa:** Loyalty Points 
- **Retailers:** StoreA, StoreB, StoreC, StoreD, STOREE
- **Clientes totales:** ~12M
- **Muestra de trabajo:** 500K estratificada
- **Tiers:** NORMAL, FAN, PREMIUM, ELITE
- **Problema:** Alto breakage, sin personalizacion, fuga silenciosa

### Reglas de negocio criticas

1. **Canje minimo:** 1,000 puntos
2. **Fuga NORMAL/FAN:** >= 365 dias sin canje
3. **Fuga PREMIUM/ELITE:** >= 730 dias sin canje
4. **Flags en BQ son STRING:** 'Y'/'N', no BOOL
5. **Major Sale Event:** Noviembre

---

## 3. Arquitectura

### BigQuery

- **Proyecto:** `my-gcp-project`
- **Dataset fuente:** `raw_data`
- **Dataset output:** `loyalty_analytics` (CREAR)

### Tablas fuente

```sql
-- Clientes (snapshot diario)
svw_clients_entity
  cust_id STRING, partition_date DATE, cat_cust_name STRING (tier),
  stock_points_amt FLOAT64, exp_point_current_month_amt FLOAT64,
  cust_active_card_flg STRING ('Y'/'N'), cust_age_num INT64,
  cust_gender_desc STRING, cust_enroll_date DATE

-- Transacciones
frozen_transaction_entity
  cust_id STRING, tran_date DATE, tran_amt FLOAT64,
  points_earned FLOAT64, retailer STRING, channel STRING

-- Canjes
frozen_redemption_entity
  cust_id STRING, redemption_date DATE, redemption_points_amt FLOAT64,
  redemption_type STRING, channel_name STRING (retailer de canje)
```

### Repo GitLab existente

`scheduled-workflows-gitops-loy-cl-consumo-prod` con estructura:
```
operations/    — KPIs, funnel, provision
Triggers/              — Campanas automatizadas
personalizacion/       — ML propension existente
evaluador_campanhas/   — Evaluacion con grupo control
```

Agregar carpeta: `loyalty_analytics/`

---

## 4. Fase 1: Data Foundation

### Query 01: Exclusiones

Crear tabla `loyalty_analytics.excluded_customers`:

```
4 categorias con prioridad:
1. COLABORADOR — Lista de empleados (mayor prioridad)
2. FRAUDE_DEVOLUCIONES — tasa_devolucion > 50% AND total_canjes >= 5
3. MONTO_EXTREMO — ticket_promedio > percentil 99.9
4. FANTASMA — 0 transacciones validas

Cada cliente: una sola exclusion (la de mayor prioridad).
```

### Query 02: Muestra estratificada

Crear tabla `loyalty_analytics.sample_500k`:

```
Pool = activos - excluidos
       AND tier IN ('NORMAL','FAN','PREMIUM','ELITE')
       AND enrollment < 2023-01-01

Estratificacion: proporcional por tier × has_redeemed
Seleccion: FARM_FINGERPRINT deterministica
Resultado: 500K clientes fijos en todos los snapshots
```

### Query 03: Funnel Markov

6 estados bidireccionales:

```
INSCRITO → sin transacciones
PARTICIPANTE → transacciones, 0 canjes
POSIBILIDAD_CANJE → >= 1,000 pts + 0 canjes
CANJEADOR → 1 canje historico
RECURRENTE → >= 2 canjes
FUGA → NORMAL/FAN: 365d sin canje, PREMIUM/ELITE: 730d sin canje

Prioridad: FUGA se evalua primero (puede overridear RECURRENTE)
Matriz de transicion mensual para probabilidades Markov
```

### Query 04: Customer Snapshot

Crear tabla `loyalty_analytics.snapshots_features`:

**Diseno temporal:**
```
Periodo pre (features): 12 meses antes de t0
Periodo post (target):  12 meses despues de t0
27 snapshots: ene-2023 a mar-2025
Split: train 21, val 3, test 3
```

**Target variable:**
```
y = 0: No canjea en post-12m
y = 1: Canjea Y nunca habia canjeado (activacion)
y = 2: Canjea Y ya habia canjeado (recurrencia)
```

**74 features en 11 grupos:**

**Grupo A (RFM, 6):** recency_days, frequency_total, frequency_monthly_avg, monetary_total, monetary_avg_ticket, monetary_monthly_avg

**Grupo B (Puntos, 10):** points_earned_total, points_earned_monthly_avg, stock_points_at_t0, exp_points_current_at_t0, exp_points_next_at_t0, has_redeemed_before_t0, redeem_count_pre, redeem_points_total_pre, redeem_count_12m_pre, redeem_rate

**Grupo C (Dinamicas, 7):** earn_velocity_30, earn_velocity_90, redeem_velocity_30, earn_acceleration (=v30/v90), spend_trend, days_since_last_activity, days_since_last_redeem

**Grupo D (Capacidad, 4):** redeem_capacity (BOOL >=1000pts), points_above_threshold, days_to_redeem_capacity, points_pressure (=exp_current/stock)

**Grupo E (Retailer, 11):** spend_store_a/store_b/store_c/store_d/store_e (5), freq_store_a/store_b/store_c/store_d/store_e (5), retailer_count, dominant_retailer, retailer_entropy (Shannon frequency-based)

**Grupo F (Pago, 4):** pct_store_card_payments, pct_debit_payments, purchase_channel_pref, pct_digital

**Grupo G (Canje, 7):** pct_redeem_catalogo, pct_redeem_giftcard, pct_redeem_digital, dominant_redeem_type, redeem_channel_pref, avg_redeem_points, avg_redeem_amount

**Grupo H (Funnel, 6):** funnel_state_at_t0, days_in_current_state, transitions_last_12m, velocity_in_funnel, prob_to_next_state, prob_to_fuga

**Grupo I (Demograficas, 9):** tier, age, gender, city, tenure_months, cust_active_card_flg, cust_active_deb_flg, cust_active_omp_flg, status

**Grupo J (Avanzadas, 6):** ratio_earn_redeem, ticket_trend, burstiness, spend_variability, campaign_response_rate, breakage. NOTA: engagement_score REMOVIDO por leakage.

**Grupo K (Estacionalidad, 4):** month_of_t0, is_cyber_month, is_holiday_month, seasonal_spend_ratio

**Reglas anti-leakage:**
- Todas las features usan SOLO datos con fecha < t0
- stock_points y exp_points de partition_date = t0 (foto exacta)
- Target usa SOLO datos >= t0 y < t0 + 12 meses
- has_redeemed_before_t0 es feature (no target)

---

## 5. Fase 2: Modelamiento

### Cascade 4 pasos

```
Paso 1: XGBoost multiclase → P(y=0), P(y=1), P(y=2)
  - Todas las 74 features como input
  - Optuna tuning con F1-macro en validation
  - SHAP global + por clase + individual

Paso 2: XGBoost multiclase → P(STOREA/STOREB/STOREC/STORED/STOREE)
  - Solo clientes con P(canje) > threshold
  - Target: retailer dominante de canje post-t0

Paso 3: XGBoost regresion → Monto puntos canjeados
  - Solo clientes que canjearon
  - Target: sum(redemption_points) post-t0

Paso 4: XGBoost regresion → Revenue 12m ($CLP)
  - Todos los clientes
  - Target: sum(tran_amt) post-t0
```

### Clustering

```
Algoritmo: KMeans
K: 4-8 (optimo por silhouette, rango ampliado para 6 arquetipos)
8 features: frequency_monthly_avg, monetary_monthly_avg, redeem_rate,
            retailer_entropy, pct_redeem_digital, earn_velocity_90,
            days_since_last_activity, points_pressure
Centroides fijos: entrenar en training, aplicar a todos los t0s
Nombres: Hungarian Assignment contra 6 arquetipos
Arquetipos: Heavy Users, Cazadores de Canje, Exploradores, Dormidos, Digitales, En Riesgo
```

### Incrementalidad

```
Tratamiento: y >= 1 (canjeo en post)
Propensity: LogisticRegression sobre 13 features
PSM: Nearest neighbor matching para ATE
T-Learner: 2 XGBoost (tratados / control) para CATE individual
uplift_x = E[Y|T=1] - E[Y|T=0]
Expected Value = P(canje) × uplift_x
```

### Decision Engine (5 pasos)

```
1. Prioridad: uplift_x quantiles GLOBALES (del training)
   Q80+ → Alta, Q60-Q80 → Media, <Q60 → Baja, <=0 → No contactar

2. Objetivo: mapping de funnel_state_at_t0
   INSCRITO → Activar, PARTICIPANTE → Acelerar, POSIBILIDAD → Empujar,
   CANJEADOR → Recurrencia, RECURRENTE → Retener, FUGA → Reactivar

3. Accion: mapping de cluster_name (con overrides)
   Override: FUGA → Reactivacion urgente, INSCRITO → Activar primera compra

4. Canal: pct_redeem_digital > 0.5 → Digital, prop > 0.3 → Email,
          prop > 0.15 → Presencial, else → Push

5. Timing: FUGA → Urgente, points_pressure > 0.5 → Urgente,
           prop > 0.4 → Inmediato, inactivo > 300d → Pronto, else → Normal
```

### Metricas de validacion

```
F1-macro > 0.70 (min), > 0.80 (ideal)
AUC por clase > 0.80 (min)
Recall por clase > 0.60 (min)
Brier score < 0.18 (min)
Lift top 10% > 3x (min)
Retailer accuracy > 0.60 (min)
R2 monto/revenue > 0.40 (min)
```

---

## 6. Fase 3: Productizacion

### Dashboard Streamlit (7 vistas)

```
1. KPIs Ejecutivos — tasas, revenue, breakage, filtros
2. Funnel Markov — estados, transiciones, cuellos de botella
3. Segmentacion — clusters, migracion, KPIs por segmento
4. Customer 360 — perfil completo + predicciones + recomendacion
5. Simulador — escenarios de campana, ROI
6. Que paso? — predicho vs real, alertas performance
7. Exports — CSV/Excel de listas accionables
```

### Scoring Pipeline

```
Frecuencia: mensual (dia 2, GitLab CI/CD)
Chunks: 1M clientes por batch
Modelos: KMeans + LogReg + XGBoost T-Learner
Output: 12 columnas (cust_id, prioridad, EV, propensity, uplift,
        objetivo, accion, retailer, canal, timing, cluster, funnel)
Destino: BQ loyalty_analytics.scoring_output + CSV
Versionado: models.pkl con metadata JSON
```

### API FastAPI

```
GET  /health             → status
GET  /score/{cust_id}    → propensity, uplift, EV, prioridad, cluster
POST /score/batch        → batch hasta 1000
GET  /recommend/{cust_id}→ recomendacion completa
GET  /segment/{cust_id}  → cluster + perfil
GET  /stats              → estadisticas resumen
```

---

## 7. Fase 4: Feedback Loop

### Tracking Table (39 columnas)

```sql
loyalty_analytics.action_tracking
  Particionada por scoring_month, clustered por cust_id + prioridad

  Secciones:
  - Identificadores (3): tracking_id, scoring_month, cust_id
  - Recomendacion (11): rec_prioridad, rec_EV, rec_objetivo, rec_accion...
  - Ejecucion (6): exec_flag, exec_date, exec_accion, exec_canal...
  - Control (1): control_flag (holdout)
  - Resultados (16): result_1m/3m/6m/12m_redeemed/revenue/canjes
  - Metadata (2): created_at, updated_at
```

### A/B Testing

```
10% holdout estratificado por prioridad
Hash deterministico por cust_id
Grupo control NUNCA se contacta
Z-test de proporciones para significancia
```

### Drift Detection

```
PSI < 0.1 → OK
PSI 0.1-0.2 → Monitorear
PSI > 0.2 en 3+ features → REENTRENAR

Alertas adicionales:
  F1 drop > 7pts → reentrenar
  Lift < 2x → reentrenar
  12+ meses sin reentrenar → planificar
```

### Optimizacion de reglas

```
Cada trimestre, evaluar:
- Lift por cluster × accion (que combinaciones funcionan)
- Mejor canal por cluster (ajustar umbrales)
- ROI por prioridad (vale la pena contactar "Baja"?)
- Ajustar decision engine con resultados reales
```

---

## 8. GitLab CI/CD Pipeline

```yaml
stages: [features, scoring, tracking, feedback, monitoring, optimize]

Schedule: dia 2 cada mes, 6am
Variables: GCP_SA_KEY (secret), GCP_PROJECT

Stage 1 (features):    bq query < 02_snapshots_features.sql
Stage 2 (scoring):     python scoring_pipeline.py --project $GCP_PROJECT
Stage 3 (tracking):    bq query < 04_insert_tracking.sql
Stage 4 (feedback):    bq query < 05_update_results.sql
Stage 5 (monitoring):  python monitoring.py
Stage 6 (optimize):    python optimize_rules.py (trimestral)
```

---

## 9. Correcciones Aplicadas

| # | Issue original | Correccion |
|---|---|---|
| 1 | Quantiles de prioridad por chunk | Quantiles globales del training guardados en models.pkl |
| 2 | "En Riesgo" nunca aparecia (K=5) | Rango K ampliado a 4-8 (ahora K=8) |
| 3 | Sin versionado de modelos | Metadata JSON con fecha, metricas, version |
| 4 | engagement_score en spec | Removido por leakage, documentado |
| 5 | Formula lift revenue pendiente | Definida: EV = P(canje) × uplift_x |
| 6 | Sin validacion de features | Validacion explicita al inicio de score_chunk() |
