-- =============================================================================
-- FASE 1: EXCLUSIÓN DE OUTLIERS
-- Proyecto: my-gcp-project
-- Dataset destino: loyalty_analytics
-- Tabla destino: loyalty_analytics.excluded_customers
-- Descripción: Identifica clientes a excluir del modelo por ser:
--   1. Fantasmas (sin ninguna transacción histórica)
--   2. Montos extremos (ticket promedio > P99.9)
--   3. Fraude/devoluciones masivas (tasa devolución canjes > umbral)
--   4. IDs manuales (corporativos/empleados en tabla separada)
-- =============================================================================

-- Crear dataset si no existe (ejecutar una sola vez)
-- CREATE SCHEMA IF NOT EXISTS `my-gcp-project.loyalty_analytics`;

-- Crear tabla de exclusiones
CREATE OR REPLACE TABLE `my-gcp-project.loyalty_analytics.excluded_customers` AS

WITH

-- -----------------------------------------------------------------------
-- BASE: Universo de clientes registrados en el programa
-- -----------------------------------------------------------------------
clientes_universo AS (
  SELECT DISTINCT
    cust_id,
    cat_cust_name                          AS tier,
    cust_enroll_date                       AS enrollment_date
  FROM `my-gcp-project.raw_data.svw_clients_entity`
  WHERE partition_date = (
    SELECT MAX(partition_date)
    FROM `my-gcp-project.raw_data.svw_clients_entity`
  )
),

-- -----------------------------------------------------------------------
-- TRANSACCIONES VALIDAS: 2022-01-01 hasta 2026-03-31
-- -----------------------------------------------------------------------
trx_validas AS (
  SELECT
    cust_id,
    COUNT(DISTINCT tran_id)                AS total_transacciones,
    SUM(tran_amt)                          AS total_monto,
    SAFE_DIVIDE(SUM(tran_amt), COUNT(DISTINCT tran_id))  AS ticket_promedio
  FROM `my-gcp-project.operations.frozen_transaction_entity`
  WHERE tran_type    = 'COMPRA'
    AND tran_amt     > 0
    AND tran_valid_flg = 1
    AND mes BETWEEN '2022-01-01' AND '2026-03-31'
  GROUP BY cust_id
),

-- -----------------------------------------------------------------------
-- PERCENTIL 99.9 de ticket promedio (para excluir montos extremos)
-- -----------------------------------------------------------------------
percentiles AS (
  SELECT
    APPROX_QUANTILES(ticket_promedio, 1000)[OFFSET(999)] AS p99_9_ticket
  FROM trx_validas
  WHERE ticket_promedio IS NOT NULL
),

-- -----------------------------------------------------------------------
-- CANJES TOTALES Y DEVOLUCIONES (para detectar fraude)
-- -----------------------------------------------------------------------
canje_stats AS (
  SELECT
    cust_id,
    COUNT(*)                               AS total_canjes,
    COUNTIF(return_flag IS TRUE)           AS total_devoluciones,
    SAFE_DIVIDE(
      COUNTIF(return_flag IS TRUE),
      COUNT(*)
    )                                      AS tasa_devolucion
  FROM `my-gcp-project.operations.frozen_redemption_entity`
  WHERE mes BETWEEN '2022-01-01' AND '2026-03-31'
  GROUP BY cust_id
),

-- -----------------------------------------------------------------------
-- CRITERIO 1: FANTASMAS — Sin ninguna transacción en todo el periodo
-- -----------------------------------------------------------------------
exclusion_fantasmas AS (
  SELECT
    cu.cust_id,
    'FANTASMA'                             AS motivo_exclusion,
    'Sin transacciones 2022-2026'          AS descripcion
  FROM clientes_universo cu
  LEFT JOIN trx_validas t ON cu.cust_id = t.cust_id
  WHERE t.cust_id IS NULL
),

-- -----------------------------------------------------------------------
-- CRITERIO 2: MONTOS EXTREMOS — Ticket promedio > P99.9
-- -----------------------------------------------------------------------
exclusion_extremos AS (
  SELECT
    t.cust_id,
    'MONTO_EXTREMO'                        AS motivo_exclusion,
    CONCAT('Ticket promedio: ', CAST(ROUND(t.ticket_promedio, 0) AS STRING),
           ' > P99.9: ', CAST(ROUND(p.p99_9_ticket, 0) AS STRING)) AS descripcion
  FROM trx_validas t
  CROSS JOIN percentiles p
  WHERE t.ticket_promedio > p.p99_9_ticket
),

-- -----------------------------------------------------------------------
-- CRITERIO 3: FRAUDE/DEVOLUCIONES — Tasa devolución > 50% con al menos 5 canjes
-- -----------------------------------------------------------------------
exclusion_fraude AS (
  SELECT
    cust_id,
    'FRAUDE_DEVOLUCIONES'                  AS motivo_exclusion,
    CONCAT('Tasa devolución: ',
           CAST(ROUND(tasa_devolucion * 100, 1) AS STRING), '% (',
           CAST(total_devoluciones AS STRING), '/',
           CAST(total_canjes AS STRING), ' canjes)')  AS descripcion
  FROM canje_stats
  WHERE tasa_devolucion > 0.5
    AND total_canjes    >= 5
),

-- -----------------------------------------------------------------------
-- CRITERIO 4: IDs MANUALES (corporativos / empleados)
-- Asumir que existe tabla con exclusiones manuales
-- Si no existe, comentar este CTE y el UNION ALL correspondiente
-- -----------------------------------------------------------------------
exclusion_colaboradores AS (
  SELECT
    cust_id,
    'COLABORADOR'                          AS motivo_exclusion,
    'Empleado del retail conglomerate'         AS descripcion
  FROM `my-gcp-discovery.operations.base_colaboradores`
  -- TODO: Confirmar dataset exacto dentro de my-gcp-discovery
),

-- -----------------------------------------------------------------------
-- UNION DE TODOS LOS CRITERIOS (deduplicado por cust_id, prioridad manual > fraude > extremo > fantasma)
-- -----------------------------------------------------------------------
todas_exclusiones AS (
  SELECT cust_id, motivo_exclusion, descripcion, 1 AS prioridad FROM exclusion_colaboradores
  UNION ALL
  SELECT cust_id, motivo_exclusion, descripcion, 2 AS prioridad FROM exclusion_fraude
  UNION ALL
  SELECT cust_id, motivo_exclusion, descripcion, 3 AS prioridad FROM exclusion_extremos
  UNION ALL
  SELECT cust_id, motivo_exclusion, descripcion, 4 AS prioridad FROM exclusion_fantasmas
),

exclusiones_dedup AS (
  SELECT
    cust_id,
    motivo_exclusion,
    descripcion,
    CURRENT_DATE()                         AS fecha_proceso
  FROM (
    SELECT *,
      ROW_NUMBER() OVER (PARTITION BY cust_id ORDER BY prioridad ASC) AS rn
    FROM todas_exclusiones
  )
  WHERE rn = 1
)

SELECT * FROM exclusiones_dedup;


-- -----------------------------------------------------------------------
-- VERIFICACIÓN POST-CREACIÓN
-- -----------------------------------------------------------------------
/*
SELECT
  motivo_exclusion,
  COUNT(*) AS total_clientes
FROM `my-gcp-project.loyalty_analytics.excluded_customers`
GROUP BY 1
ORDER BY 2 DESC;
*/
