-- =============================================================================
-- FASE 1: MUESTRA ESTRATIFICADA 500K CLIENTES
-- Proyecto: my-gcp-project
-- Dataset destino: loyalty_analytics
-- Tabla destino: loyalty_analytics.sample_customers
-- Descripción: Selecciona 500,000 clientes representativos del universo válido,
--   estratificados por tier × has_redeemed (nunca canjeó / sí canjeó).
--   Estos 500K clientes son FIJOS para los 27 snapshots de entrenamiento.
-- Pre-requisito: loyalty_analytics.excluded_customers debe existir.
-- =============================================================================

CREATE OR REPLACE TABLE `my-gcp-project.loyalty_analytics.sample_customers` AS

WITH

-- -----------------------------------------------------------------------
-- UNIVERSO VÁLIDO: Clientes con snapshot al último mes disponible,
--   excluyendo outliers identificados en paso anterior
-- -----------------------------------------------------------------------
universo_valido AS (
  SELECT
    c.cust_id,
    c.cat_cust_name                        AS tier,
    c.cust_enroll_date                     AS enrollment_date
  FROM `my-gcp-project.raw_data.svw_clients_entity` c
  -- Último snapshot disponible
  WHERE c.partition_date = (
    SELECT MAX(partition_date)
    FROM `my-gcp-project.raw_data.svw_clients_entity`
  )
  -- Excluir outliers
  AND c.cust_id NOT IN (
    SELECT cust_id
    FROM `my-gcp-project.loyalty_analytics.excluded_customers`
  )
  -- Solo tiers del programa
  AND UPPER(c.cat_cust_name) IN ('NORMAL', 'FAN', 'PREMIUM', 'ELITE')
  -- Solo clientes que existían antes del primer t0 (ene 2023)
  -- Así garantizamos que los 500K están presentes en los 27 snapshots
  AND c.cust_enroll_date < '2023-01-01'
),

-- -----------------------------------------------------------------------
-- FLAG HAS_REDEEMED: ¿Ha canjeado ANTES del primer t0 (2023-01-01)?
-- Usar solo datos pre-primer-t0 para evitar leak futuro en estratificación
-- -----------------------------------------------------------------------
redemption_historica AS (
  SELECT DISTINCT cust_id
  FROM `my-gcp-project.operations.frozen_redemption_entity`
  WHERE return_flag IS FALSE
    AND redemption_date < '2023-01-01'
),

-- -----------------------------------------------------------------------
-- ENRIQUECER CON FLAG Y ESTRATO
-- Estrato = tier × has_redeemed (8 combinaciones)
-- -----------------------------------------------------------------------
universo_enriquecido AS (
  SELECT
    uv.cust_id,
    UPPER(uv.tier)                                             AS tier,
    uv.enrollment_date,
    CASE WHEN r.cust_id IS NOT NULL THEN 1 ELSE 0 END  AS has_redeemed,
    CONCAT(UPPER(uv.tier), '_',
           CASE WHEN r.cust_id IS NOT NULL THEN 'CANJEADOR' ELSE 'NO_CANJEADOR' END
    )                                                   AS estrato
  FROM universo_valido uv
  LEFT JOIN redemption_historica r ON uv.cust_id = r.cust_id
),

-- -----------------------------------------------------------------------
-- CALCULAR TAMAÑO DE CADA ESTRATO (proporcional, total = 500K)
-- -----------------------------------------------------------------------
conteo_estratos AS (
  SELECT
    estrato,
    tier,
    has_redeemed,
    COUNT(*)                               AS n_universo,
    SUM(COUNT(*)) OVER ()                  AS n_total_universo
  FROM universo_enriquecido
  GROUP BY estrato, tier, has_redeemed
),

tamanio_estratos AS (
  SELECT
    estrato,
    tier,
    has_redeemed,
    n_universo,
    -- Mínimo 100 por estrato para garantizar representatividad
    GREATEST(
      100,
      CAST(ROUND(500000 * n_universo / n_total_universo) AS INT64)
    )                                      AS n_muestra_objetivo
  FROM conteo_estratos
),

-- -----------------------------------------------------------------------
-- ASIGNAR NÚMERO ALEATORIO DENTRO DE CADA ESTRATO Y SELECCIONAR
-- FARM_FINGERPRINT con seed fijo para reproducibilidad
-- -----------------------------------------------------------------------
universo_con_rank AS (
  SELECT
    ue.*,
    ts.n_muestra_objetivo,
    ROW_NUMBER() OVER (
      PARTITION BY ue.estrato
      ORDER BY FARM_FINGERPRINT(CONCAT(ue.cust_id, '_loyalty_seed_v1'))
    )                                      AS rn_estrato
  FROM universo_enriquecido ue
  JOIN tamanio_estratos ts USING (estrato)
),

muestra_seleccionada AS (
  SELECT
    cust_id,
    UPPER(tier)                            AS tier,
    enrollment_date,
    has_redeemed,
    estrato,
    CURRENT_DATE()                         AS fecha_proceso
  FROM universo_con_rank
  WHERE rn_estrato <= n_muestra_objetivo
)

SELECT * FROM muestra_seleccionada;


-- -----------------------------------------------------------------------
-- VERIFICACIÓN POST-CREACIÓN
-- -----------------------------------------------------------------------
/*
-- Total y distribución por estrato
SELECT
  estrato,
  tier,
  has_redeemed,
  COUNT(*)  AS n_muestra,
  ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct
FROM `my-gcp-project.loyalty_analytics.sample_customers`
GROUP BY 1, 2, 3
ORDER BY tier, has_redeemed;

-- Total general
SELECT COUNT(*) AS total_muestra
FROM `my-gcp-project.loyalty_analytics.sample_customers`;
*/
