-- =============================================================================
-- FASE 1: CUSTOMER SNAPSHOT — QUERY MONOLÍTICA CON CTEs
-- Proyecto: my-gcp-project
-- Dataset destino: loyalty_analytics
-- Tabla destino: loyalty_analytics.customer_snapshot
-- Descripción:
--   Genera 27 snapshots de entrenamiento (ENE 2023 → MAR 2025) × 500K clientes
--   = ~13.5M filas. Para cada snapshot t0 calcula:
--     - ~70 features de los 12 meses pre-t0 (grupos A-K)
--     - Variable target y ∈ {0, 1, 2} de los 12 meses post-t0
--   Pre-requisitos:
--     - loyalty_analytics.sample_customers
--     - loyalty_analytics.funnel_states_monthly
--     - loyalty_analytics.markov_transition_matrix
--     - frozen_transaction_entity, frozen_redemption_entity, svw_clients_entity
-- =============================================================================

CREATE OR REPLACE TABLE `my-gcp-project.loyalty_analytics.customer_snapshot`
PARTITION BY DATE_TRUNC(t0, MONTH)
CLUSTER BY tier, funnel_state_at_t0, y
AS

WITH

-- ===========================================================================
-- T0 LIST: Los 27 puntos de corte mensuales
-- t0 = primer día del mes de corte. Pre = [t0-12m, t0). Post = [t0, t0+12m)
-- ===========================================================================
t0_list AS (
  SELECT t0
  FROM UNNEST(
    GENERATE_DATE_ARRAY('2023-01-01', '2025-03-01', INTERVAL 1 MONTH)
  ) AS t0
),

-- ===========================================================================
-- CLIENTES EN MUESTRA
-- ===========================================================================
muestra AS (
  SELECT cust_id, tier AS tier_actual, enrollment_date, has_redeemed AS has_redeemed_global
  FROM `my-gcp-project.loyalty_analytics.sample_customers`
),

-- ===========================================================================
-- CROSS: 27 t0s × 500K clientes = 13.5M combinaciones base
-- ===========================================================================
base AS (
  SELECT
    t0.t0,
    DATE_SUB(t0.t0, INTERVAL 12 MONTH)   AS pre_start,
    DATE_ADD(t0.t0, INTERVAL 12 MONTH)   AS post_end,
    m.cust_id
  FROM t0_list t0
  CROSS JOIN muestra m
),

-- ===========================================================================
-- GRUPO I: DEMOGRÁFICAS — Estado del cliente en t0 (desde snapshot)
-- ===========================================================================
demograficas AS (
  SELECT
    b.cust_id,
    b.t0,
    -- Usar snapshot más cercano a t0 (último disponible antes de t0)
    UPPER(MAX_BY(c.cat_cust_name,   c.partition_date))  AS tier,
    MAX_BY(c.cust_enroll_date,      c.partition_date)   AS enrollment_date,
    MAX_BY(c.cust_gender_desc,      c.partition_date)   AS gender,
    MAX_BY(c.cust_age_num,          c.partition_date)   AS age,
    MAX_BY(c.city_name,             c.partition_date)   AS city,
    MAX_BY(c.contact_email_flg,     c.partition_date)   AS contact_email_flg,
    MAX_BY(c.contact_phone_flg,     c.partition_date)   AS contact_phone_flg,
    MAX_BY(c.contact_push_flg,      c.partition_date)   AS contact_push_flg,
    MAX_BY(c.cust_active_card_flg,   c.partition_date)   AS cust_active_card_flg,
    MAX_BY(c.cust_active_deb_flg,   c.partition_date)   AS cust_active_deb_flg,
    MAX_BY(c.cust_active_omp_flg,   c.partition_date)   AS cust_active_omp_flg,
    -- Saldo de puntos recalculado históricamente en t0
    MAX_BY(c.cust_stock_point_amt,        c.partition_date) AS stock_points_at_t0,
    MAX_BY(c.exp_point_current_month_amt, c.partition_date) AS exp_points_current_at_t0,
    MAX_BY(c.exp_point_next_month_amt,    c.partition_date) AS exp_points_next_at_t0
  FROM base b
  JOIN `my-gcp-project.raw_data.svw_clients_entity` c
    ON  c.cust_id        = b.cust_id
    AND c.partition_date  < b.t0
    AND c.partition_date >= DATE_SUB(b.t0, INTERVAL 2 MONTH)  -- cota inferior: evita escanear toda la historia
  GROUP BY b.cust_id, b.t0
),

-- ===========================================================================
-- TRANSACCIONES PRE-T0: features calculadas en ventana [t0-12m, t0)
-- ===========================================================================
trx_pre AS (
  SELECT
    b.cust_id,
    b.t0,

    -- GRUPO A: RFM base
    DATE_DIFF(b.t0, MAX(t.tran_date), DAY)                  AS recency_days,
    COUNT(DISTINCT t.tran_id)                                AS frequency_total,
    SAFE_DIVIDE(COUNT(DISTINCT t.tran_id), 12.0)             AS frequency_monthly_avg,
    SUM(t.tran_amt)                                          AS monetary_total,
    SAFE_DIVIDE(SUM(t.tran_amt), COUNT(DISTINCT t.tran_id))  AS monetary_avg_ticket,
    SAFE_DIVIDE(SUM(t.tran_amt), 12.0)                       AS monetary_monthly_avg,

    -- GRUPO A: Puntos acumulados
    SUM(t.point_amt)                                         AS points_earned_total,
    SAFE_DIVIDE(SUM(t.point_amt), 12.0)                      AS points_earned_monthly_avg,

    -- GRUPO C: Velocidades (últimos 30 días pre-t0)
    SUM(CASE WHEN t.tran_date >= DATE_SUB(b.t0, INTERVAL 30 DAY)
             THEN t.point_amt ELSE 0 END)                    AS earn_velocity_30,
    SUM(CASE WHEN t.tran_date >= DATE_SUB(b.t0, INTERVAL 90 DAY)
             THEN t.point_amt ELSE 0 END)                    AS earn_velocity_90,

    -- Gasto últimos 3 meses vs 3 meses anteriores (para tendencia)
    SUM(CASE WHEN t.tran_date >= DATE_SUB(b.t0, INTERVAL 3 MONTH)
             THEN t.tran_amt ELSE 0 END)                     AS spend_last_3m,
    SUM(CASE WHEN t.tran_date  < DATE_SUB(b.t0, INTERVAL 3 MONTH)
             AND t.tran_date  >= DATE_SUB(b.t0, INTERVAL 6 MONTH)
             THEN t.tran_amt ELSE 0 END)                     AS spend_prev_3m,

    -- Frecuencia por sub-periodo (para ticket_trend deployable)
    COUNT(DISTINCT CASE WHEN t.tran_date >= DATE_SUB(b.t0, INTERVAL 3 MONTH)
                        THEN t.tran_id END)                  AS freq_last_3m,
    COUNT(DISTINCT CASE WHEN t.tran_date  < DATE_SUB(b.t0, INTERVAL 3 MONTH)
                        AND t.tran_date  >= DATE_SUB(b.t0, INTERVAL 6 MONTH)
                        THEN t.tran_id END)                  AS freq_prev_3m,

    -- GRUPO E: Por retailer
    SUM(CASE WHEN t.channel_name = 'STOREA' THEN t.tran_amt ELSE 0 END) AS spend_store_a,
    SUM(CASE WHEN t.channel_name = 'STOREB'   THEN t.tran_amt ELSE 0 END) AS spend_store_b,
    SUM(CASE WHEN t.channel_name = 'STOREC'    THEN t.tran_amt ELSE 0 END) AS spend_store_c,
    SUM(CASE WHEN t.channel_name = 'STORED'      THEN t.tran_amt ELSE 0 END) AS spend_store_d,
    SUM(CASE WHEN t.channel_name = 'STOREE'      THEN t.tran_amt ELSE 0 END) AS spend_store_e,
    COUNT(DISTINCT CASE WHEN t.channel_name = 'STOREA' THEN t.tran_id END) AS freq_store_a,
    COUNT(DISTINCT CASE WHEN t.channel_name = 'STOREB'   THEN t.tran_id END) AS freq_store_b,
    COUNT(DISTINCT CASE WHEN t.channel_name = 'STOREC'    THEN t.tran_id END) AS freq_store_c,
    COUNT(DISTINCT CASE WHEN t.channel_name = 'STORED'      THEN t.tran_id END) AS freq_store_d,
    COUNT(DISTINCT CASE WHEN t.channel_name = 'STOREE'      THEN t.tran_id END) AS freq_store_e,
    COUNT(DISTINCT t.channel_name)                           AS retailer_count,

    -- GRUPO F: Medio de pago
    SAFE_DIVIDE(
      COUNT(DISTINCT CASE WHEN UPPER(t.payment_method_name) LIKE '%STORE_CARD%' THEN t.tran_id END),
      COUNT(DISTINCT t.tran_id)
    )                                                        AS pct_store_card_payments,
    SAFE_DIVIDE(
      COUNT(DISTINCT CASE WHEN UPPER(t.payment_method_name) LIKE '%DEB%'
                            OR UPPER(t.payment_method_name) LIKE '%DEBITO%' THEN t.tran_id END),
      COUNT(DISTINCT t.tran_id)
    )                                                        AS pct_debit_payments,

    -- GRUPO J: Dispersión (burstiness y variabilidad)
    STDDEV(t.tran_amt)                                       AS std_tran_amt,
    AVG(t.tran_amt)                                          AS avg_tran_amt,

    -- Días entre compras (para calcular burstiness fuera del CTE)
    MIN(t.tran_date)                                         AS primera_compra_pre,
    MAX(t.tran_date)                                         AS ultima_compra_pre

  FROM base b
  JOIN `my-gcp-project.operations.frozen_transaction_entity` t
    ON  t.cust_id         = b.cust_id
    AND t.tran_date       >= b.pre_start
    AND t.tran_date        < b.t0
    AND t.tran_type        = 'COMPRA'
    AND t.tran_amt         > 0
    AND t.tran_valid_flg   = 1
  WHERE t.mes >= '2022-01-01' AND t.mes <= '2025-03-01'     -- partition pruning
  GROUP BY b.cust_id, b.t0
),

-- ===========================================================================
-- CANJES PRE-T0: Toda la historia de canjes hasta t0
-- ===========================================================================
canjes_pre AS (
  SELECT
    b.cust_id,
    b.t0,

    -- GRUPO B: Canjes históricos
    COUNT(*)                                                 AS redeem_count_pre,
    SUM(r.redemption_points_amt)                             AS redeem_points_total_pre,
    MAX(r.redemption_date)                                   AS last_redeem_date_pre,
    DATE_DIFF(b.t0, MAX(r.redemption_date), DAY)             AS days_since_last_redeem,

    -- Canjes en ventana 12m pre-t0
    COUNTIF(r.redemption_date >= b.pre_start)                AS redeem_count_12m_pre,
    SUM(CASE WHEN r.redemption_date >= b.pre_start
             THEN r.redemption_points_amt ELSE 0 END)        AS redeem_points_12m_pre,

    -- GRUPO C: Velocidad de canje (últimos 30 días)
    SUM(CASE WHEN r.redemption_date >= DATE_SUB(b.t0, INTERVAL 30 DAY)
             THEN r.redemption_points_amt ELSE 0 END)        AS redeem_velocity_30,

    -- GRUPO G: Detalle tipo de canje
    SAFE_DIVIDE(
      COUNTIF(LOWER(r.price_type_desc) LIKE '%catalogo%'
           OR LOWER(r.award_family_type_name) LIKE '%catalogo%'),
      COUNT(*)
    )                                                        AS pct_redeem_catalogo,
    SAFE_DIVIDE(
      COUNTIF(LOWER(r.price_type_desc) LIKE '%giftcard%'
           OR LOWER(r.award_family_type_name) LIKE '%giftcard%'),
      COUNT(*)
    )                                                        AS pct_redeem_giftcard,
    SAFE_DIVIDE(
      COUNTIF(LOWER(r.tipo_pos) = 'digital'),
      COUNT(*)
    )                                                        AS pct_redeem_digital,
    AVG(r.redemption_points_amt)                             AS avg_redeem_points,

    -- GRUPO J: Campaign response
    SAFE_DIVIDE(
      COUNTIF(r.redemption_campaign_desc IS NOT NULL
           AND r.redemption_campaign_desc != ''),
      COUNT(*)
    )                                                        AS campaign_response_rate

  FROM base b
  JOIN `my-gcp-project.operations.frozen_redemption_entity` r
    ON  r.cust_id         = b.cust_id
    AND r.redemption_date  < b.t0                           -- TODA la historia pre-t0
    AND r.return_flag      IS FALSE
  WHERE r.mes <= '2025-03-01'     -- partition pruning (sin cota inferior para features cumulativos)
  GROUP BY b.cust_id, b.t0
),

-- ===========================================================================
-- GRUPO H: FUNNEL STATE EN T0
-- Tomado de la tabla precalculada funnel_states_monthly
-- Incluye: estado actual, días en estado, transiciones y velocidad en funnel
-- ===========================================================================
funnel_with_lag AS (
  SELECT
    fs.cust_id,
    fs.fecha_fin_mes,
    fs.funnel_state,
    fs.total_canjes,
    fs.total_compras,
    LAG(fs.funnel_state) OVER (
      PARTITION BY fs.cust_id ORDER BY fs.fecha_fin_mes
    ) AS prev_funnel_state
  FROM `my-gcp-project.loyalty_analytics.funnel_states_monthly` fs
),

funnel_pre AS (
  SELECT
    b.cust_id,
    b.t0,
    -- Estado en el último día del mes anterior a t0
    MAX_BY(fl.funnel_state,        fl.fecha_fin_mes)  AS funnel_state_at_t0,
    MAX_BY(fl.total_canjes,        fl.fecha_fin_mes)  AS total_canjes_at_t0,
    MAX_BY(fl.total_compras,       fl.fecha_fin_mes)  AS total_compras_at_t0,
    -- Meses en estado actual (conteo consecutivo del estado más reciente)
    COUNT(DISTINCT CASE WHEN fl.funnel_state = MAX_BY(fl.funnel_state, fl.fecha_fin_mes)
                        THEN fl.fecha_fin_mes END)    AS months_in_current_state,
    -- Transiciones en últimos 12m (cambios de estado mes a mes)
    COUNTIF(fl.funnel_state != fl.prev_funnel_state
        AND fl.prev_funnel_state IS NOT NULL)         AS transitions_last_12m,
    -- Velocidad en funnel: meses promedio entre cambios de estado
    SAFE_DIVIDE(
      COUNT(DISTINCT fl.fecha_fin_mes),
      NULLIF(COUNTIF(fl.funnel_state != fl.prev_funnel_state
                 AND fl.prev_funnel_state IS NOT NULL), 0)
    )                                                 AS velocity_in_funnel
  FROM base b
  JOIN funnel_with_lag fl
    ON  fl.cust_id       = b.cust_id
    AND fl.fecha_fin_mes  < b.t0
    AND fl.fecha_fin_mes >= DATE_SUB(b.t0, INTERVAL 13 MONTH)
  GROUP BY b.cust_id, b.t0
),

-- ===========================================================================
-- GRUPO H: PROBABILIDADES MARKOV (join con matriz de transición)
-- ===========================================================================
markov_probs AS (
  SELECT
    fp.cust_id,
    fp.t0,
    fp.funnel_state_at_t0,
    -- Probabilidad de avanzar al siguiente estado positivo
    COALESCE(MAX(CASE WHEN mt.estado_destino IN ('CANJEADOR', 'RECURRENTE', 'POSIBILIDAD_CANJE')
                      THEN mt.prob_transicion END), 0) AS prob_to_next_state,
    -- Probabilidad de caer a fuga
    COALESCE(MAX(CASE WHEN mt.estado_destino = 'FUGA'
                      THEN mt.prob_transicion END), 0) AS prob_to_fuga
  FROM funnel_pre fp
  LEFT JOIN demograficas d_tier
    ON  d_tier.cust_id = fp.cust_id AND d_tier.t0 = fp.t0
  LEFT JOIN `my-gcp-project.loyalty_analytics.markov_transition_matrix` mt
    ON  mt.estado_origen  = fp.funnel_state_at_t0
    AND mt.tier_group     = CASE
                              WHEN UPPER(d_tier.tier) IN ('ELITE', 'PREMIUM') THEN 'ALTO'
                              ELSE 'BASE'
                            END
  GROUP BY fp.cust_id, fp.t0, fp.funnel_state_at_t0
),

-- ===========================================================================
-- TARGET: Variable y en [t0, t0+12m)
-- y=0: No canjea
-- y=1: Primer canje (nunca había canjeado antes de t0)
-- y=2: Recurrencia (ya había canjeado antes de t0)
-- ===========================================================================
-- TARGET: Separado en 3 CTEs para evitar producto cartesiano
target_has_redeemed AS (
  SELECT
    b.cust_id,
    b.t0,
    TRUE AS has_redeemed_before_t0
  FROM base b
  WHERE EXISTS (
    SELECT 1
    FROM `my-gcp-project.operations.frozen_redemption_entity` r
    WHERE r.cust_id         = b.cust_id
      AND r.redemption_date < b.t0
      AND r.return_flag     IS FALSE
      -- Sin cota inferior: necesitamos TODA la historia pre-t0 para
      -- distinguir correctamente y=1 (primer canje) vs y=2 (recurrente)
      AND r.mes <= '2025-03-01'  -- partition pruning (solo cota superior)
  )
),

target_post_canjes AS (
  SELECT
    b.cust_id,
    b.t0,
    COUNT(*)                               AS n_canjes_post
  FROM base b
  JOIN `my-gcp-project.operations.frozen_redemption_entity` r
    ON  r.cust_id         = b.cust_id
    AND r.redemption_date >= b.t0
    AND r.redemption_date  < b.post_end
    AND r.return_flag      IS FALSE
  WHERE r.mes >= '2022-01-01' AND r.mes <= '2026-03-01'    -- partition pruning
  GROUP BY b.cust_id, b.t0
),

target_revenue AS (
  SELECT
    b.cust_id,
    b.t0,
    SUM(t.tran_amt)                        AS revenue_post_12m
  FROM base b
  JOIN `my-gcp-project.operations.frozen_transaction_entity` t
    ON  t.cust_id         = b.cust_id
    AND t.tran_date       >= b.t0
    AND t.tran_date        < b.post_end
    AND t.tran_type        = 'COMPRA'
    AND t.tran_amt         > 0
    AND t.tran_valid_flg   = 1
  WHERE t.mes >= '2022-01-01' AND t.mes <= '2026-03-01'    -- partition pruning
  GROUP BY b.cust_id, b.t0
),

target AS (
  SELECT
    b.cust_id,
    b.t0,
    COALESCE(thr.has_redeemed_before_t0, FALSE) AS has_redeemed_before_t0,
    COALESCE(tpc.n_canjes_post, 0) > 0          AS canjea_post,
    COALESCE(tpc.n_canjes_post, 0)              AS n_canjes_post,
    COALESCE(trv.revenue_post_12m, 0)           AS revenue_post_12m
  FROM base b
  LEFT JOIN target_has_redeemed thr ON thr.cust_id = b.cust_id AND thr.t0 = b.t0
  LEFT JOIN target_post_canjes  tpc ON tpc.cust_id = b.cust_id AND tpc.t0 = b.t0
  LEFT JOIN target_revenue      trv ON trv.cust_id = b.cust_id AND trv.t0 = b.t0
),

-- ===========================================================================
-- ENSAMBLE FINAL: Todas las features + target
-- ===========================================================================
snapshot_final AS (
  SELECT
    -- Identificadores
    b.cust_id,
    b.t0,

    -- -----------------------------------------------------------------------
    -- GRUPO I: DEMOGRÁFICAS
    -- -----------------------------------------------------------------------
    UPPER(COALESCE(d.tier, mu.tier_actual)) AS tier,
    DATE_DIFF(b.t0, COALESCE(d.enrollment_date, mu.enrollment_date), MONTH) AS tenure_months,
    d.gender,
    d.age,
    d.city,
    d.cust_active_card_flg,
    d.cust_active_deb_flg,
    d.cust_active_omp_flg,
    COALESCE(d.contact_email_flg, FALSE)   AS contact_email_flg,
    COALESCE(d.contact_phone_flg, FALSE)   AS contact_phone_flg,
    COALESCE(d.contact_push_flg, FALSE)    AS contact_push_flg,

    -- -----------------------------------------------------------------------
    -- GRUPO A: RFM
    -- -----------------------------------------------------------------------
    COALESCE(trx.recency_days,             999)               AS recency_days,
    COALESCE(trx.frequency_total,          0)                 AS frequency_total,
    COALESCE(trx.frequency_monthly_avg,    0)                 AS frequency_monthly_avg,
    COALESCE(trx.monetary_total,           0)                 AS monetary_total,
    COALESCE(trx.monetary_avg_ticket,      0)                 AS monetary_avg_ticket,
    COALESCE(trx.monetary_monthly_avg,     0)                 AS monetary_monthly_avg,

    -- -----------------------------------------------------------------------
    -- GRUPO B: PUNTOS Y CANJE
    -- -----------------------------------------------------------------------
    COALESCE(trx.points_earned_total,      0)                 AS points_earned_total,
    COALESCE(trx.points_earned_monthly_avg,0)                 AS points_earned_monthly_avg,
    COALESCE(d.stock_points_at_t0,         0)                 AS stock_points_at_t0,
    COALESCE(d.exp_points_current_at_t0,   0)                 AS exp_points_current_at_t0,
    COALESCE(d.exp_points_next_at_t0,      0)                 AS exp_points_next_at_t0,
    COALESCE(cj.redeem_count_pre,          0)                 AS redeem_count_pre,
    COALESCE(cj.redeem_points_total_pre,   0)                 AS redeem_points_total_pre,
    COALESCE(cj.redeem_count_12m_pre,      0)                 AS redeem_count_12m_pre,
    COALESCE(cj.redeem_points_12m_pre,     0)                 AS redeem_points_12m_pre,
    SAFE_DIVIDE(
      COALESCE(cj.redeem_points_total_pre, 0),
      NULLIF(COALESCE(trx.points_earned_total, 0), 0)
    )                                                         AS redeem_rate,
    COALESCE(1 - SAFE_DIVIDE(
      COALESCE(cj.redeem_points_total_pre, 0),
      NULLIF(COALESCE(trx.points_earned_total, 0), 0)
    ), 1)                                                     AS breakage,

    -- -----------------------------------------------------------------------
    -- GRUPO C: DINÁMICAS (velocidad, aceleración, tendencia)
    -- -----------------------------------------------------------------------
    COALESCE(trx.earn_velocity_30,         0)                 AS earn_velocity_30,
    COALESCE(trx.earn_velocity_90,         0)                 AS earn_velocity_90,
    COALESCE(cj.redeem_velocity_30,        0)                 AS redeem_velocity_30,
    COALESCE(SAFE_DIVIDE(
      COALESCE(trx.spend_last_3m, 0),
      NULLIF(COALESCE(trx.spend_prev_3m, 0), 0)
    ), 0)                                                     AS spend_trend,
    CASE
      WHEN trx.ultima_compra_pre IS NOT NULL
        THEN DATE_DIFF(b.t0, trx.ultima_compra_pre, DAY)
      ELSE 999
    END                                                       AS days_since_last_activity,
    COALESCE(cj.days_since_last_redeem,    999)               AS days_since_last_redeem,

    -- -----------------------------------------------------------------------
    -- GRUPO D: CAPACIDAD DE CANJE
    -- -----------------------------------------------------------------------
    CASE WHEN COALESCE(d.stock_points_at_t0, 0) >= 1000 THEN 1 ELSE 0 END  AS redeem_capacity,
    GREATEST(COALESCE(d.stock_points_at_t0, 0) - 1000, 0)   AS points_above_threshold,
    COALESCE(SAFE_DIVIDE(
      GREATEST(1000 - COALESCE(d.stock_points_at_t0, 0), 0),
      NULLIF(COALESCE(trx.earn_velocity_30, 0), 0)
    ) * 30, 999)                                              AS days_to_redeem_capacity,
    SAFE_DIVIDE(
      COALESCE(d.exp_points_current_at_t0, 0),
      NULLIF(COALESCE(d.stock_points_at_t0, 0), 0)
    )                                                         AS points_pressure,

    -- -----------------------------------------------------------------------
    -- GRUPO E: POR RETAILER
    -- -----------------------------------------------------------------------
    COALESCE(trx.spend_store_a,          0)                 AS spend_store_a,
    COALESCE(trx.spend_store_b,            0)                 AS spend_store_b,
    COALESCE(trx.spend_store_c,             0)                 AS spend_store_c,
    COALESCE(trx.spend_store_d,               0)                 AS spend_store_d,
    COALESCE(trx.spend_store_e,               0)                 AS spend_store_e,
    COALESCE(trx.freq_store_a,           0)                 AS freq_store_a,
    COALESCE(trx.freq_store_b,             0)                 AS freq_store_b,
    COALESCE(trx.freq_store_c,              0)                 AS freq_store_c,
    COALESCE(trx.freq_store_d,                0)                 AS freq_store_d,
    COALESCE(trx.freq_store_e,                0)                 AS freq_store_e,
    COALESCE(trx.retailer_count,           0)                 AS retailer_count,
    -- Retailer dominante (mayor gasto, incluye los 5 retailers)
    CASE
      WHEN GREATEST(
             COALESCE(trx.spend_store_a, 0), COALESCE(trx.spend_store_b, 0),
             COALESCE(trx.spend_store_c, 0), COALESCE(trx.spend_store_d, 0),
             COALESCE(trx.spend_store_e, 0)
           ) = COALESCE(trx.spend_store_c, 0)    AND COALESCE(trx.spend_store_c, 0)    > 0 THEN 'STOREC'
      WHEN GREATEST(
             COALESCE(trx.spend_store_a, 0), COALESCE(trx.spend_store_b, 0),
             COALESCE(trx.spend_store_c, 0), COALESCE(trx.spend_store_d, 0),
             COALESCE(trx.spend_store_e, 0)
           ) = COALESCE(trx.spend_store_a, 0) AND COALESCE(trx.spend_store_a, 0) > 0 THEN 'STOREA'
      WHEN GREATEST(
             COALESCE(trx.spend_store_a, 0), COALESCE(trx.spend_store_b, 0),
             COALESCE(trx.spend_store_c, 0), COALESCE(trx.spend_store_d, 0),
             COALESCE(trx.spend_store_e, 0)
           ) = COALESCE(trx.spend_store_b, 0)   AND COALESCE(trx.spend_store_b, 0)   > 0 THEN 'STOREB'
      WHEN GREATEST(
             COALESCE(trx.spend_store_a, 0), COALESCE(trx.spend_store_b, 0),
             COALESCE(trx.spend_store_c, 0), COALESCE(trx.spend_store_d, 0),
             COALESCE(trx.spend_store_e, 0)
           ) = COALESCE(trx.spend_store_d, 0)      AND COALESCE(trx.spend_store_d, 0)      > 0 THEN 'STORED'
      WHEN GREATEST(
             COALESCE(trx.spend_store_a, 0), COALESCE(trx.spend_store_b, 0),
             COALESCE(trx.spend_store_c, 0), COALESCE(trx.spend_store_d, 0),
             COALESCE(trx.spend_store_e, 0)
           ) = COALESCE(trx.spend_store_e, 0)      AND COALESCE(trx.spend_store_e, 0)      > 0 THEN 'STOREE'
      ELSE 'NINGUNO'
    END                                                       AS dominant_retailer,
    -- Diversidad de retailer (entropía de Shannon, 5 retailers)
    -- Diversidad de retailer (entropía de Shannon, 5 retailers)
    -- COALESCE por término para evitar NULL cuando un retailer tiene freq=0
    CASE WHEN COALESCE(trx.frequency_total, 0) > 0 THEN
      -(
        COALESCE(SAFE_DIVIDE(COALESCE(trx.freq_store_a, 0), trx.frequency_total)
          * LOG(NULLIF(SAFE_DIVIDE(COALESCE(trx.freq_store_a, 0), trx.frequency_total), 0)), 0)
        + COALESCE(SAFE_DIVIDE(COALESCE(trx.freq_store_b, 0), trx.frequency_total)
          * LOG(NULLIF(SAFE_DIVIDE(COALESCE(trx.freq_store_b, 0), trx.frequency_total), 0)), 0)
        + COALESCE(SAFE_DIVIDE(COALESCE(trx.freq_store_c, 0), trx.frequency_total)
          * LOG(NULLIF(SAFE_DIVIDE(COALESCE(trx.freq_store_c, 0), trx.frequency_total), 0)), 0)
        + COALESCE(SAFE_DIVIDE(COALESCE(trx.freq_store_d, 0), trx.frequency_total)
          * LOG(NULLIF(SAFE_DIVIDE(COALESCE(trx.freq_store_d, 0), trx.frequency_total), 0)), 0)
        + COALESCE(SAFE_DIVIDE(COALESCE(trx.freq_store_e, 0), trx.frequency_total)
          * LOG(NULLIF(SAFE_DIVIDE(COALESCE(trx.freq_store_e, 0), trx.frequency_total), 0)), 0)
      )
    ELSE 0 END                                                AS retailer_entropy,

    -- -----------------------------------------------------------------------
    -- GRUPO F: MEDIO DE PAGO
    -- -----------------------------------------------------------------------
    COALESCE(trx.pct_store_card_payments,         0)                 AS pct_store_card_payments,
    COALESCE(trx.pct_debit_payments,       0)                 AS pct_debit_payments,

    -- -----------------------------------------------------------------------
    -- GRUPO G: DETALLE CANJE
    -- -----------------------------------------------------------------------
    COALESCE(cj.pct_redeem_catalogo,       0)                 AS pct_redeem_catalogo,
    COALESCE(cj.pct_redeem_giftcard,       0)                 AS pct_redeem_giftcard,
    COALESCE(cj.pct_redeem_digital,        0)                 AS pct_redeem_digital,
    COALESCE(cj.avg_redeem_points,         0)                 AS avg_redeem_points,

    -- -----------------------------------------------------------------------
    -- GRUPO H: FUNNEL Y MARKOV
    -- -----------------------------------------------------------------------
    COALESCE(fp.funnel_state_at_t0, 'INSCRITO')               AS funnel_state_at_t0,
    COALESCE(fp.months_in_current_state,   0) * 30            AS days_in_current_state,
    COALESCE(fp.transitions_last_12m,      0)                 AS transitions_last_12m,
    COALESCE(fp.velocity_in_funnel,        0)                 AS velocity_in_funnel,
    COALESCE(mp.prob_to_next_state,        0)                 AS prob_to_next_state,
    COALESCE(mp.prob_to_fuga,              0)                 AS prob_to_fuga,

    -- -----------------------------------------------------------------------
    -- GRUPO J: FEATURES AVANZADAS
    -- -----------------------------------------------------------------------
    SAFE_DIVIDE(
      COALESCE(trx.points_earned_total, 0),
      NULLIF(COALESCE(cj.redeem_points_total_pre, 0), 0)
    )                                                         AS ratio_earn_redeem,
    -- Ticket trend: avg ticket últimos 3m vs 3m anteriores (deployable, sin LAG entre t0s)
    COALESCE(SAFE_DIVIDE(
      SAFE_DIVIDE(COALESCE(trx.spend_last_3m, 0), NULLIF(trx.freq_last_3m, 0)),
      NULLIF(SAFE_DIVIDE(COALESCE(trx.spend_prev_3m, 0), NULLIF(trx.freq_prev_3m, 0)), 0)
    ), 0)                                                     AS ticket_trend,
    -- Burstiness: std/mean días entre compras
    SAFE_DIVIDE(
      COALESCE(trx.std_tran_amt, 0),
      NULLIF(COALESCE(trx.avg_tran_amt, 0), 0)
    )                                                         AS spend_variability,
    COALESCE(cj.campaign_response_rate,    0)                 AS campaign_response_rate,

    -- -----------------------------------------------------------------------
    -- GRUPO K: ESTACIONALIDAD
    -- -----------------------------------------------------------------------
    EXTRACT(MONTH FROM b.t0)                                  AS month_of_t0,
    CASE WHEN EXTRACT(MONTH FROM b.t0) IN (11)   THEN 1 ELSE 0 END AS is_cyber_month,
    CASE WHEN EXTRACT(MONTH FROM b.t0) IN (12, 1) THEN 1 ELSE 0 END AS is_holiday_month,

    -- -----------------------------------------------------------------------
    -- STATUS (activo/inactivo)
    -- -----------------------------------------------------------------------
    CASE
      WHEN CASE
             WHEN trx.ultima_compra_pre IS NOT NULL
               THEN DATE_DIFF(b.t0, trx.ultima_compra_pre, DAY)
             ELSE 999
           END <= 365
        OR (cj.last_redeem_date_pre IS NOT NULL
            AND DATE_DIFF(b.t0, cj.last_redeem_date_pre, DAY) <= 365)
      THEN 'ACTIVO'
      ELSE 'INACTIVO'
    END                                                       AS status,

    -- -----------------------------------------------------------------------
    -- TARGET VARIABLE
    -- -----------------------------------------------------------------------
    tgt.has_redeemed_before_t0,
    tgt.canjea_post,
    tgt.n_canjes_post,
    tgt.revenue_post_12m,
    CASE
      WHEN tgt.canjea_post = FALSE THEN 0        -- No canjea
      WHEN tgt.has_redeemed_before_t0 = FALSE THEN 1  -- Primer canje (activación)
      ELSE 2                                     -- Recurrencia
    END                                                       AS y,

    CURRENT_DATE()                                            AS fecha_proceso

  FROM base b
  LEFT JOIN muestra              mu  ON mu.cust_id  = b.cust_id
  LEFT JOIN demograficas          d  ON d.cust_id   = b.cust_id AND d.t0 = b.t0
  LEFT JOIN trx_pre              trx ON trx.cust_id = b.cust_id AND trx.t0 = b.t0
  LEFT JOIN canjes_pre            cj ON cj.cust_id  = b.cust_id AND cj.t0 = b.t0
  LEFT JOIN funnel_pre            fp ON fp.cust_id  = b.cust_id AND fp.t0 = b.t0
  LEFT JOIN markov_probs          mp ON mp.cust_id  = b.cust_id AND mp.t0 = b.t0
  LEFT JOIN target               tgt ON tgt.cust_id = b.cust_id AND tgt.t0 = b.t0
)

SELECT
  sf.*,

  -- -----------------------------------------------------------------------
  -- QUINTILES (calculados por t0 para comparabilidad temporal)
  -- -----------------------------------------------------------------------
  NTILE(5) OVER (PARTITION BY sf.t0 ORDER BY sf.monetary_total)         AS quintil_gasto,
  NTILE(5) OVER (PARTITION BY sf.t0 ORDER BY sf.stock_points_at_t0)     AS quintil_puntos,
  NTILE(5) OVER (PARTITION BY sf.t0 ORDER BY sf.frequency_total)        AS quintil_frecuencia,
  NTILE(5) OVER (PARTITION BY sf.t0 ORDER BY sf.monetary_monthly_avg)   AS quintil_gasto_mensual

FROM snapshot_final sf;


-- ===========================================================================
-- VERIFICACIONES POST-CREACIÓN
-- ===========================================================================
/*
-- Conteo por t0 y distribución del target
SELECT
  t0,
  y,
  COUNT(*) AS n,
  ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY t0), 2) AS pct
FROM `my-gcp-project.loyalty_analytics.customer_snapshot`
GROUP BY t0, y
ORDER BY t0, y;

-- Total filas
SELECT COUNT(*) AS total_filas, COUNT(DISTINCT cust_id) AS clientes_distintos,
       COUNT(DISTINCT t0) AS n_snapshots
FROM `my-gcp-project.loyalty_analytics.customer_snapshot`;

-- Distribución por funnel state en t0
SELECT
  funnel_state_at_t0,
  COUNT(*) AS n,
  AVG(CASE WHEN y > 0 THEN 1.0 ELSE 0 END) AS tasa_canje
FROM `my-gcp-project.loyalty_analytics.customer_snapshot`
GROUP BY 1
ORDER BY n DESC;

-- Verificar no-leakage: target debe estar vacío para t0 > 2025-03
SELECT t0, COUNT(*) FROM `my-gcp-project.loyalty_analytics.customer_snapshot`
WHERE t0 > '2025-03-01' GROUP BY 1;
*/
