-- =============================================================================
-- FASE 1: FUNNEL MARKOV — 6 ESTADOS + MATRIZ DE TRANSICIÓN MENSUAL
-- Proyecto: my-gcp-project
-- Dataset destino: loyalty_intelligence
-- Tablas destino:
--   loyalty_intelligence.funnel_states_monthly  (estado mensual por cliente)
--   loyalty_intelligence.markov_transition_matrix (probabilidades de transición)
-- Descripción:
--   6 estados del funnel de canje:
--     1. INSCRITO          — Sin transacciones históricas
--     2. PARTICIPANTE      — Tiene compras, nunca canjeó
--     3. POSIBILIDAD_CANJE — Participante con stock >= 1,000 puntos y nunca canjeó
--     4. CANJEADOR         — 1 canje histórico válido
--     5. RECURRENTE        — >= 2 canjes históricos válidos
--     6. FUGA              — Dejó de canjear: NORMAL/FAN >= 12m / PREMIUM/ELITE >= 24m
--
--   Nota: Se reutiliza clients_funnel existente como base para flags de acumulador/canjeador.
--   Este script calcula el estado en el último día de cada mes (estado a fin de mes).
-- =============================================================================

-- ===========================================================================
-- PASO 1: ESTADO MENSUAL POR CLIENTE
-- Captura el estado de cada cliente en el último día de cada mes
-- Rango: Enero 2022 - Marzo 2026 (para cubrir pre y post de los 27 snapshots)
--
-- ⚠ DATA LEAKAGE NOTE: This table is pre-computed over the ENTIRE date range
-- (2022-01 to 2026-03), including months that fall after t0 for some snapshots.
-- The downstream 04_customer_snapshot.sql JOIN filters funnel states to
-- fecha_fin_mes < t0, which prevents direct leakage. However, the cumulative
-- counters used to determine funnel states (e.g., n_canjes_historicos) are
-- computed using the full transaction history up to each month, not just up to
-- t0. This is a structural limitation inherent in pre-computing a monthly table.
-- In production, this should be recomputed per t0 or use a temporal CTE.
-- ===========================================================================

CREATE OR REPLACE TABLE `my-gcp-project.loyalty_intelligence.funnel_states_monthly` AS

WITH

-- -----------------------------------------------------------------------
-- FECHAS FIN DE MES a calcular (último día de cada mes)
-- -----------------------------------------------------------------------
meses AS (
  SELECT
    DATE_SUB(DATE_ADD(DATE_TRUNC(mes, MONTH), INTERVAL 1 MONTH), INTERVAL 1 DAY) AS fecha_fin_mes,
    DATE_TRUNC(mes, MONTH)                                                         AS mes_inicio
  FROM UNNEST(
    GENERATE_DATE_ARRAY('2022-01-01', '2026-03-01', INTERVAL 1 MONTH)
  ) AS mes
),

-- -----------------------------------------------------------------------
-- CLIENTES EN MUESTRA
-- -----------------------------------------------------------------------
muestra AS (
  SELECT cust_id, enrollment_date
  FROM `my-gcp-project.loyalty_intelligence.sample_customers`
),

-- -----------------------------------------------------------------------
-- TOTAL CANJES HISTÓRICOS acumulado hasta cada fecha fin de mes
-- -----------------------------------------------------------------------
canjes_acum AS (
  SELECT
    r.cust_id,
    m.fecha_fin_mes,
    COUNT(*)                               AS total_canjes_historicos,
    MAX(r.redemption_date)                 AS ultimo_canje
  FROM `my-gcp-project.operations.frozen_redemption_entity` r
  INNER JOIN meses m
    ON r.redemption_date <= m.fecha_fin_mes
  WHERE r.return_flag IS FALSE
    AND r.mes >= '2022-01-01' AND r.mes <= '2026-03-01'   -- partition pruning
    AND r.cust_id IN (SELECT cust_id FROM muestra)
  GROUP BY r.cust_id, m.fecha_fin_mes
),

-- -----------------------------------------------------------------------
-- TOTAL COMPRAS HISTÓRICAS acumulado hasta cada fecha fin de mes
-- -----------------------------------------------------------------------
compras_acum AS (
  SELECT
    t.cust_id,
    m.fecha_fin_mes,
    COUNT(DISTINCT t.tran_id)              AS total_compras_historicas,
    MAX(t.tran_date)                       AS ultima_compra
  FROM `my-gcp-project.operations.frozen_transaction_entity` t
  INNER JOIN meses m
    ON t.tran_date <= m.fecha_fin_mes
  WHERE t.tran_type     = 'COMPRA'
    AND t.tran_amt      > 0
    AND t.tran_valid_flg = 1
    AND t.mes           >= '2022-01-01' AND t.mes <= '2026-03-01'  -- partition pruning
    AND t.cust_id       IN (SELECT cust_id FROM muestra)
  GROUP BY t.cust_id, m.fecha_fin_mes
),

-- -----------------------------------------------------------------------
-- SALDO DE PUNTOS al fin de cada mes
-- Usamos snapshot de clients_entity más cercano al fin de mes
-- -----------------------------------------------------------------------
-- Snapshot más cercano al fin de mes (dentro de 7 días antes)
stock_puntos_ranked AS (
  SELECT
    c.cust_id,
    m.fecha_fin_mes,
    c.cust_stock_point_amt                 AS stock_points,
    UPPER(c.cat_cust_name)                 AS tier,
    ROW_NUMBER() OVER (
      PARTITION BY c.cust_id, m.fecha_fin_mes
      ORDER BY c.partition_date DESC
    )                                      AS rn
  FROM `my-gcp-project.raw_data.svw_clients_entity` c
  INNER JOIN meses m
    ON c.partition_date BETWEEN DATE_SUB(m.fecha_fin_mes, INTERVAL 7 DAY)
                            AND m.fecha_fin_mes
  WHERE c.cust_id IN (SELECT cust_id FROM muestra)
),
stock_puntos AS (
  SELECT cust_id, fecha_fin_mes, stock_points, tier
  FROM stock_puntos_ranked
  WHERE rn = 1
),

-- -----------------------------------------------------------------------
-- JUNTAR TODO Y CALCULAR ESTADO DEL FUNNEL
-- -----------------------------------------------------------------------
estado_raw AS (
  SELECT
    m.cust_id,
    mes.fecha_fin_mes,
    mes.mes_inicio,
    COALESCE(sp.tier, 'NORMAL')            AS tier,
    COALESCE(ca.total_compras_historicas, 0) AS total_compras,
    ca.ultima_compra,
    COALESCE(cj.total_canjes_historicos, 0) AS total_canjes,
    cj.ultimo_canje,
    COALESCE(sp.stock_points, 0)           AS stock_points
  FROM muestra m
  -- Solo meses donde el cliente ya estaba inscrito en el programa
  INNER JOIN meses mes ON mes.fecha_fin_mes >= m.enrollment_date
  LEFT JOIN compras_acum ca  ON ca.cust_id = m.cust_id AND ca.fecha_fin_mes = mes.fecha_fin_mes
  LEFT JOIN canjes_acum  cj  ON cj.cust_id = m.cust_id AND cj.fecha_fin_mes = mes.fecha_fin_mes
  LEFT JOIN stock_puntos sp  ON sp.cust_id = m.cust_id AND sp.fecha_fin_mes = mes.fecha_fin_mes
),

estado_calculado AS (
  SELECT
    *,
    -- Días desde último canje (para detectar fuga)
    DATE_DIFF(fecha_fin_mes, ultimo_canje, DAY) AS dias_desde_ultimo_canje,
    -- Umbral de fuga según tier (diferenciado)
    CASE
      WHEN UPPER(tier) IN ('ELITE', 'PREMIUM') THEN 730  -- 24 meses
      ELSE 365                                            -- 12 meses
    END                                        AS umbral_fuga_dias,

    -- ESTADO DEL FUNNEL (6 estados)
    CASE
      -- FUGA: Ha canjeado alguna vez pero superó ventana sin canje
      WHEN total_canjes >= 1
        AND ultimo_canje IS NOT NULL
        AND DATE_DIFF(fecha_fin_mes, ultimo_canje, DAY) >
            CASE WHEN UPPER(tier) IN ('ELITE', 'PREMIUM') THEN 730 ELSE 365 END
      THEN 'FUGA'

      -- RECURRENTE: 2 o más canjes históricos válidos (sin fuga activa)
      WHEN total_canjes >= 2
      THEN 'RECURRENTE'

      -- CANJEADOR: Exactamente 1 canje histórico (sin fuga activa)
      WHEN total_canjes = 1
      THEN 'CANJEADOR'

      -- POSIBILIDAD_CANJE: Participa (compra) + tiene >= 1000 puntos + nunca canjeó
      WHEN total_compras >= 1
        AND total_canjes  = 0
        AND stock_points  >= 1000
      THEN 'POSIBILIDAD_CANJE'

      -- PARTICIPANTE: Tiene compras pero nunca canjeó ni tiene suficientes puntos
      WHEN total_compras >= 1
        AND total_canjes  = 0
      THEN 'PARTICIPANTE'

      -- INSCRITO: Sin ninguna transacción histórica
      ELSE 'INSCRITO'
    END                                        AS funnel_state
  FROM estado_raw
)

SELECT
  cust_id,
  fecha_fin_mes,
  mes_inicio,
  tier,
  total_compras,
  ultima_compra,
  total_canjes,
  ultimo_canje,
  stock_points,
  dias_desde_ultimo_canje,
  umbral_fuga_dias,
  funnel_state,
  CURRENT_DATE()                           AS fecha_proceso
FROM estado_calculado;


-- ===========================================================================
-- PASO 2: MATRIZ DE TRANSICIÓN MARKOV (mensual, por segmento)
-- Calcula P(estado_siguiente | estado_actual, tier_group, retailer_dominante)
-- ===========================================================================

CREATE OR REPLACE TABLE `my-gcp-project.loyalty_intelligence.markov_transition_matrix` AS

WITH

-- -----------------------------------------------------------------------
-- PARES DE ESTADOS CONSECUTIVOS (mes t → mes t+1)
-- -----------------------------------------------------------------------
transiciones AS (
  SELECT
    curr.cust_id,
    curr.fecha_fin_mes                     AS fecha_origen,
    DATE_ADD(curr.fecha_fin_mes, INTERVAL 1 MONTH)
                                           AS fecha_destino,
    curr.funnel_state                      AS estado_origen,
    next_m.funnel_state                    AS estado_destino,
    -- Segmentación por grupo de tier
    CASE
      WHEN UPPER(curr.tier) IN ('ELITE', 'PREMIUM') THEN 'ALTO'
      ELSE 'BASE'
    END                                    AS tier_group
  FROM `my-gcp-project.loyalty_intelligence.funnel_states_monthly` curr
  INNER JOIN `my-gcp-project.loyalty_intelligence.funnel_states_monthly` next_m
    ON  curr.cust_id      = next_m.cust_id
    AND DATE_ADD(curr.fecha_fin_mes, INTERVAL 1 MONTH) = next_m.fecha_fin_mes
  -- Solo periodo de entrenamiento (hasta último snapshot del train set)
  WHERE curr.fecha_fin_mes BETWEEN '2022-01-31' AND '2025-03-31'
),

-- -----------------------------------------------------------------------
-- CONTEO DE TRANSICIONES POR SEGMENTO
-- -----------------------------------------------------------------------
conteo_transiciones AS (
  SELECT
    tier_group,
    estado_origen,
    estado_destino,
    COUNT(*)                               AS n_transiciones
  FROM transiciones
  WHERE estado_destino IS NOT NULL
  GROUP BY tier_group, estado_origen, estado_destino
),

-- -----------------------------------------------------------------------
-- TOTAL POR ESTADO ORIGEN (para calcular probabilidades)
-- -----------------------------------------------------------------------
total_por_origen AS (
  SELECT
    tier_group,
    estado_origen,
    SUM(n_transiciones)                    AS total_desde_origen
  FROM conteo_transiciones
  GROUP BY tier_group, estado_origen
),

-- -----------------------------------------------------------------------
-- PROBABILIDADES DE TRANSICIÓN
-- -----------------------------------------------------------------------
probabilidades AS (
  SELECT
    ct.tier_group,
    ct.estado_origen,
    ct.estado_destino,
    ct.n_transiciones,
    tp.total_desde_origen,
    SAFE_DIVIDE(ct.n_transiciones, tp.total_desde_origen) AS prob_transicion
  FROM conteo_transiciones ct
  JOIN total_por_origen tp
    ON ct.tier_group   = tp.tier_group
    AND ct.estado_origen = tp.estado_origen
)

SELECT
  tier_group,
  estado_origen,
  estado_destino,
  n_transiciones,
  total_desde_origen,
  ROUND(prob_transicion, 4)                AS prob_transicion,
  CURRENT_DATE()                           AS fecha_proceso
FROM probabilidades
ORDER BY tier_group, estado_origen, prob_transicion DESC;


-- ===========================================================================
-- VERIFICACIONES
-- ===========================================================================
/*
-- Distribución de estados por mes (evolución del funnel)
SELECT
  mes_inicio,
  funnel_state,
  COUNT(*) AS n_clientes,
  ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY mes_inicio), 2) AS pct
FROM `my-gcp-project.loyalty_intelligence.funnel_states_monthly`
GROUP BY 1, 2
ORDER BY 1, 2;

-- Matriz de transición completa
SELECT *
FROM `my-gcp-project.loyalty_intelligence.markov_transition_matrix`
ORDER BY tier_group, estado_origen, prob_transicion DESC;

-- Verificar que probabilidades suman 1 por estado origen
SELECT
  tier_group,
  estado_origen,
  SUM(prob_transicion) AS suma_prob
FROM `my-gcp-project.loyalty_intelligence.markov_transition_matrix`
GROUP BY 1, 2;
*/
