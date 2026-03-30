-- =============================================================================
-- MAQUETA: TABLAS DE JUGUETE PARA TEST END-TO-END (1000 CLIENTES)
-- Proyecto: my-gcp-project
-- Dataset: loyalty_analytics
-- Descripción:
--   Genera 1000 clientes sintéticos programáticamente con distribución realista:
--   - ~50 FANTASMA (sin transacciones → exclusión)
--   - ~10 MONTO_EXTREMO (ticket altísimo → exclusión)
--   - ~10 FRAUDE_DEVOLUCIONES (>50% devoluciones → exclusión)
--   - ~10 COLABORADOR (empleado → exclusión)
--   - ~920 clientes válidos con distribución de funnel:
--       ~250 PARTICIPANTE/INSCRITO (compran, nunca canjean)
--       ~100 POSIBILIDAD_CANJE (compran, >=1000pts, nunca canjean)
--       ~200 CANJEADOR (1 canje)
--       ~250 RECURRENTE (>=2 canjes)
--       ~120 FUGA (canjeó pero dejó de hacerlo)
--   - Tiers: ~500 NORMAL, ~250 FAN, ~150 PREMIUM, ~100 ELITE
--
--   Ejecutar ANTES de 00_run_mock.sql
-- =============================================================================


-- ===========================================================================
-- 1. MOCK: svw_clients_entity (snapshots mensuales de clientes)
-- ===========================================================================
CREATE OR REPLACE TABLE `my-gcp-project.loyalty_analytics.mock_clients_entity` AS

WITH
-- Generar 1000 IDs: C0001 a C1000
client_ids AS (
  SELECT
    CONCAT('C', LPAD(CAST(n AS STRING), 4, '0')) AS cust_id,
    n AS client_num
  FROM UNNEST(GENERATE_ARRAY(1, 1000)) AS n
),

-- Asignar tier según rango
-- 1-500: NORMAL, 501-750: FAN, 751-900: PREMIUM, 901-1000: ELITE
client_tiers AS (
  SELECT
    cust_id,
    client_num,
    CASE
      WHEN client_num <= 500 THEN 'NORMAL'
      WHEN client_num <= 750 THEN 'FAN'
      WHEN client_num <= 900 THEN 'PREMIUM'
      ELSE 'ELITE'
    END AS cat_cust_name,
    -- Enrollment date: distribuido entre 2018 y 2022
    DATE_ADD(
      DATE '2018-01-01',
      INTERVAL CAST(MOD(ABS(FARM_FINGERPRINT(CONCAT('enroll_', cust_id))), 1461) AS INT64) DAY
    ) AS cust_enroll_date,
    -- Género
    CASE WHEN MOD(client_num, 2) = 0 THEN 'F' ELSE 'M' END AS cust_gender_desc,
    -- Edad: 20-65
    20 + CAST(MOD(ABS(FARM_FINGERPRINT(CONCAT('age_', cust_id))), 46) AS INT64) AS cust_age_num,
    -- Ciudad
    CASE MOD(ABS(CAST(FARM_FINGERPRINT(CONCAT('city_', cust_id)) AS INT64)), 6)
      WHEN 0 THEN 'SANTIAGO'
      WHEN 1 THEN 'VALPARAISO'
      WHEN 2 THEN 'CONCEPCION'
      WHEN 3 THEN 'TEMUCO'
      WHEN 4 THEN 'ANTOFAGASTA'
      ELSE 'RANCAGUA'
    END AS city_name
  FROM client_ids
),

-- Meses de snapshot
months AS (
  SELECT d AS partition_date
  FROM UNNEST(GENERATE_DATE_ARRAY('2022-01-31', '2026-03-31', INTERVAL 1 MONTH)) AS d
),

-- Perfil de comportamiento (qué tipo de cliente es para el mock)
-- Grupos:
--   1-50: FANTASMA (sin txns) → exclusión
--   51-60: MONTO_EXTREMO → exclusión
--   61-70: FRAUDE → exclusión
--   71-80: COLABORADOR → exclusión
--   81-330: PARTICIPANTE/INSCRITO (compra, nunca canjea, <1000 pts a veces)
--   331-430: POSIBILIDAD_CANJE (compra, >=1000 pts, nunca canjea)
--   431-630: CANJEADOR (1 canje)
--   631-880: RECURRENTE (>=2 canjes)
--   881-1000: FUGA (canjeó pero dejó)
client_profiles AS (
  SELECT
    ct.*,
    CASE
      WHEN client_num <= 50 THEN 'FANTASMA'
      WHEN client_num <= 60 THEN 'MONTO_EXTREMO'
      WHEN client_num <= 70 THEN 'FRAUDE'
      WHEN client_num <= 80 THEN 'COLABORADOR'
      WHEN client_num <= 330 THEN 'PARTICIPANTE'
      WHEN client_num <= 430 THEN 'POSIBILIDAD_CANJE'
      WHEN client_num <= 630 THEN 'CANJEADOR'
      WHEN client_num <= 880 THEN 'RECURRENTE'
      ELSE 'FUGA'
    END AS perfil
  FROM client_tiers ct
),

-- Stock de puntos base (varía por perfil)
stock_base AS (
  SELECT
    cp.*,
    CASE
      WHEN perfil = 'FANTASMA'          THEN 0
      WHEN perfil = 'MONTO_EXTREMO'     THEN 200
      WHEN perfil = 'FRAUDE'            THEN 1500
      WHEN perfil = 'COLABORADOR'       THEN 300
      WHEN perfil = 'PARTICIPANTE'      THEN 100 + MOD(ABS(FARM_FINGERPRINT(CONCAT('stock_', cust_id))), 800)   -- 100-899
      WHEN perfil = 'POSIBILIDAD_CANJE' THEN 1000 + MOD(ABS(FARM_FINGERPRINT(CONCAT('stock_', cust_id))), 4000) -- 1000-4999
      WHEN perfil = 'CANJEADOR'         THEN 200 + MOD(ABS(FARM_FINGERPRINT(CONCAT('stock_', cust_id))), 2000)  -- 200-2199
      WHEN perfil = 'RECURRENTE'        THEN 500 + MOD(ABS(FARM_FINGERPRINT(CONCAT('stock_', cust_id))), 5000)  -- 500-5499
      WHEN perfil = 'FUGA'             THEN 100 + MOD(ABS(FARM_FINGERPRINT(CONCAT('stock_', cust_id))), 1500)   -- 100-1599
    END AS stock_base_points,
    -- Crecimiento mensual de puntos
    CASE
      WHEN perfil IN ('FANTASMA') THEN 0
      WHEN perfil = 'MONTO_EXTREMO' THEN 100
      WHEN perfil = 'POSIBILIDAD_CANJE' THEN 40 + MOD(ABS(FARM_FINGERPRINT(CONCAT('grow_', cust_id))), 60)
      WHEN perfil = 'RECURRENTE' THEN 30 + MOD(ABS(FARM_FINGERPRINT(CONCAT('grow_', cust_id))), 70)
      WHEN perfil = 'FUGA' THEN 5 + MOD(ABS(FARM_FINGERPRINT(CONCAT('grow_', cust_id))), 20)
      ELSE 10 + MOD(ABS(FARM_FINGERPRINT(CONCAT('grow_', cust_id))), 40)
    END AS stock_growth_per_month
  FROM client_profiles cp
)

SELECT
  sb.cust_id,
  m.partition_date,
  sb.cat_cust_name,
  sb.cust_enroll_date,
  sb.cust_gender_desc,
  sb.cust_age_num,
  sb.city_name,
  -- Contactabilidad
  CASE WHEN MOD(sb.client_num, 5) != 0 THEN TRUE ELSE FALSE END AS contact_email_flg,
  CASE WHEN MOD(sb.client_num, 3) != 0 THEN TRUE ELSE FALSE END AS contact_phone_flg,
  CASE WHEN MOD(sb.client_num, 7) = 0 THEN TRUE ELSE FALSE END AS contact_push_flg,
  -- Productos activos
  CASE WHEN sb.cat_cust_name IN ('PREMIUM', 'ELITE') THEN TRUE
       WHEN MOD(sb.client_num, 3) = 0 THEN TRUE ELSE FALSE END AS cust_active_store_card_flg,
  CASE WHEN MOD(sb.client_num, 4) = 0 THEN TRUE ELSE FALSE END AS cust_active_deb_flg,
  CASE WHEN MOD(sb.client_num, 10) = 0 THEN TRUE ELSE FALSE END AS cust_active_omp_flg,
  -- Stock de puntos: crece con el tiempo
  GREATEST(
    sb.stock_base_points + CAST(DATE_DIFF(m.partition_date, DATE '2022-01-01', MONTH) * sb.stock_growth_per_month AS INT64),
    0
  ) AS cust_stock_point_amt,
  -- Puntos por expirar
  CASE WHEN sb.cat_cust_name IN ('PREMIUM', 'ELITE') THEN 300 + MOD(sb.client_num, 500)
       ELSE 50 + MOD(sb.client_num, 200) END AS exp_point_current_month_amt,
  CASE WHEN sb.cat_cust_name IN ('PREMIUM', 'ELITE') THEN 200 + MOD(sb.client_num, 300)
       ELSE 30 + MOD(sb.client_num, 100) END AS exp_point_next_month_amt
FROM stock_base sb
CROSS JOIN months m
WHERE m.partition_date >= sb.cust_enroll_date;


-- ===========================================================================
-- 2. MOCK: frozen_transaction_entity (transacciones)
-- Genera transacciones programáticamente según perfil de cliente
-- ===========================================================================
CREATE OR REPLACE TABLE `my-gcp-project.loyalty_analytics.mock_transaction_entity` AS

WITH
client_ids AS (
  SELECT
    CONCAT('C', LPAD(CAST(n AS STRING), 4, '0')) AS cust_id,
    n AS client_num
  FROM UNNEST(GENERATE_ARRAY(1, 1000)) AS n
),
client_profiles AS (
  SELECT
    cust_id, client_num,
    CASE
      WHEN client_num <= 50 THEN 'FANTASMA'
      WHEN client_num <= 60 THEN 'MONTO_EXTREMO'
      WHEN client_num <= 70 THEN 'FRAUDE'
      WHEN client_num <= 80 THEN 'COLABORADOR'
      WHEN client_num <= 330 THEN 'PARTICIPANTE'
      WHEN client_num <= 430 THEN 'POSIBILIDAD_CANJE'
      WHEN client_num <= 630 THEN 'CANJEADOR'
      WHEN client_num <= 880 THEN 'RECURRENTE'
      ELSE 'FUGA'
    END AS perfil,
    -- Número de transacciones a generar por cliente
    CASE
      WHEN client_num <= 50 THEN 0                  -- FANTASMA: sin txns
      WHEN client_num <= 60 THEN 2                   -- MONTO_EXTREMO: pocas txns altísimas
      WHEN client_num <= 70 THEN 3                   -- FRAUDE: pocas txns normales
      WHEN client_num <= 80 THEN 2                   -- COLABORADOR
      WHEN client_num <= 330 THEN 3 + MOD(ABS(FARM_FINGERPRINT(CONCAT('ntxn_', CAST(client_num AS STRING)))), 8)  -- 3-10
      WHEN client_num <= 430 THEN 4 + MOD(ABS(FARM_FINGERPRINT(CONCAT('ntxn_', CAST(client_num AS STRING)))), 8)  -- 4-11
      WHEN client_num <= 630 THEN 3 + MOD(ABS(FARM_FINGERPRINT(CONCAT('ntxn_', CAST(client_num AS STRING)))), 10) -- 3-12
      WHEN client_num <= 880 THEN 5 + MOD(ABS(FARM_FINGERPRINT(CONCAT('ntxn_', CAST(client_num AS STRING)))), 12) -- 5-16
      ELSE 4 + MOD(ABS(FARM_FINGERPRINT(CONCAT('ntxn_', CAST(client_num AS STRING)))), 8)                         -- FUGA: 4-11
    END AS n_txns,
    -- Monto base por transacción
    CASE
      WHEN client_num <= 60 THEN 5000000.0  -- MONTO_EXTREMO: 5M+
      ELSE 10000.0 + MOD(ABS(FARM_FINGERPRINT(CONCAT('amt_', CAST(client_num AS STRING)))), 90000) * 1.0  -- 10K-100K
    END AS base_amt
  FROM client_ids
),
-- Generar filas de transacciones (hasta 16 por cliente)
txn_slots AS (
  SELECT cp.*, slot
  FROM client_profiles cp
  CROSS JOIN UNNEST(GENERATE_ARRAY(1, 16)) AS slot
  WHERE slot <= cp.n_txns
),
-- Asignar fechas y atributos
txns_generated AS (
  SELECT
    ts.cust_id,
    CONCAT(ts.cust_id, '_T', LPAD(CAST(ts.slot AS STRING), 2, '0')) AS tran_id,
    -- Fecha distribuida entre 2022-01 y 2025-02
    DATE_ADD(
      DATE '2022-01-15',
      INTERVAL CAST(
        MOD(ABS(FARM_FINGERPRINT(CONCAT('tdate_', ts.cust_id, '_', CAST(ts.slot AS STRING)))), 1140) -- ~38 meses en días
        AS INT64
      ) DAY
    ) AS tran_date,
    -- Monto con variación
    ROUND(
      ts.base_amt * (0.5 + MOD(ABS(FARM_FINGERPRINT(CONCAT('tamt_', ts.cust_id, '_', CAST(ts.slot AS STRING)))), 100) / 100.0),
      0
    ) AS tran_amt,
    -- Retailer
    CASE MOD(ABS(CAST(FARM_FINGERPRINT(CONCAT('ret_', ts.cust_id, '_', CAST(ts.slot AS STRING))) AS INT64)), 10)
      WHEN 0 THEN 'STOREB'
      WHEN 1 THEN 'STOREC'
      WHEN 2 THEN 'STORED'
      WHEN 3 THEN 'STOREE'
      ELSE 'STOREA'  -- 60% StoreA
    END AS channel_name,
    -- Medio de pago
    CASE WHEN MOD(ABS(CAST(FARM_FINGERPRINT(CONCAT('pay_', ts.cust_id, '_', CAST(ts.slot AS STRING))) AS INT64)), 3) = 0
      THEN 'DEBITO' ELSE 'STORE_CARD'
    END AS payment_method_name,
    ts.perfil,
    ts.client_num
  FROM txn_slots ts
)

SELECT
  cust_id,
  tran_id,
  tran_date,
  DATE_TRUNC(tran_date, MONTH) AS mes,
  tran_amt,
  ROUND(tran_amt * 0.015, 0) AS point_amt,  -- ~1.5% puntos
  'COMPRA' AS tran_type,
  1 AS tran_valid_flg,
  channel_name,
  payment_method_name
FROM txns_generated
WHERE perfil != 'FANTASMA';


-- ===========================================================================
-- 3. MOCK: frozen_redemption_entity (canjes)
-- Genera canjes según perfil de cliente
-- ===========================================================================
CREATE OR REPLACE TABLE `my-gcp-project.loyalty_analytics.mock_redemption_entity` AS

WITH
client_ids AS (
  SELECT
    CONCAT('C', LPAD(CAST(n AS STRING), 4, '0')) AS cust_id,
    n AS client_num
  FROM UNNEST(GENERATE_ARRAY(1, 1000)) AS n
),
client_profiles AS (
  SELECT
    cust_id, client_num,
    CASE
      WHEN client_num <= 50 THEN 'FANTASMA'
      WHEN client_num <= 60 THEN 'MONTO_EXTREMO'
      WHEN client_num <= 70 THEN 'FRAUDE'
      WHEN client_num <= 80 THEN 'COLABORADOR'
      WHEN client_num <= 330 THEN 'PARTICIPANTE'
      WHEN client_num <= 430 THEN 'POSIBILIDAD_CANJE'
      WHEN client_num <= 630 THEN 'CANJEADOR'
      WHEN client_num <= 880 THEN 'RECURRENTE'
      ELSE 'FUGA'
    END AS perfil,
    -- Canjes válidos a generar
    CASE
      WHEN client_num <= 430 THEN 0                  -- FANTASMA/EXTREMO/COLABORADOR/PARTICIPANTE/POSIBILIDAD: 0
      WHEN client_num <= 630 THEN 1                   -- CANJEADOR: 1
      WHEN client_num <= 880 THEN 2 + MOD(ABS(FARM_FINGERPRINT(CONCAT('nred_', CAST(client_num AS STRING)))), 4)  -- RECURRENTE: 2-5
      ELSE 1 + MOD(ABS(FARM_FINGERPRINT(CONCAT('nred_', CAST(client_num AS STRING)))), 3)                         -- FUGA: 1-3
    END AS n_canjes,
    -- FRAUDE: genera 10 canjes (8 devueltos)
    CASE WHEN client_num BETWEEN 61 AND 70 THEN 10 ELSE 0 END AS n_canjes_fraude
  FROM client_ids
),

-- Slots para canjes válidos (hasta 5)
redeem_slots_valid AS (
  SELECT cp.*, slot,
    -- Fecha de canje distribuida
    CASE
      WHEN cp.perfil = 'FUGA' THEN
        -- FUGA: canjes tempranos (2022), para que queden >12m/24m sin canjear
        DATE_ADD(DATE '2022-01-15', INTERVAL CAST(MOD(ABS(FARM_FINGERPRINT(CONCAT('rdate_', cp.cust_id, '_', CAST(slot AS STRING)))), 300) AS INT64) DAY)
      WHEN cp.perfil = 'CANJEADOR' THEN
        -- CANJEADOR: 1 canje entre 2023-01 y 2025-09 (cubre todos los t0s)
        DATE_ADD(DATE '2023-01-15', INTERVAL CAST(MOD(ABS(FARM_FINGERPRINT(CONCAT('rdate_', cp.cust_id, '_', CAST(slot AS STRING)))), 990) AS INT64) DAY)
      ELSE
        -- RECURRENTE: canjes distribuidos 2022-2026
        DATE_ADD(DATE '2022-03-01', INTERVAL CAST(MOD(ABS(FARM_FINGERPRINT(CONCAT('rdate_', cp.cust_id, '_', CAST(slot AS STRING)))), 1460) AS INT64) DAY)
    END AS redemption_date,
    CAST(1000 + MOD(ABS(FARM_FINGERPRINT(CONCAT('rpts_', cp.cust_id, '_', CAST(slot AS STRING)))), 4000) AS INT64) AS redemption_points_amt,
    FALSE AS return_flag
  FROM client_profiles cp
  CROSS JOIN UNNEST(GENERATE_ARRAY(1, 5)) AS slot
  WHERE slot <= cp.n_canjes AND cp.n_canjes > 0
),

-- Slots para canjes de FRAUDE (10 canjes, 8 devueltos)
redeem_slots_fraude AS (
  SELECT cp.cust_id, cp.client_num, cp.perfil, slot,
    DATE_ADD(DATE '2022-03-01', INTERVAL CAST(slot * 30 AS INT64) DAY) AS redemption_date,
    CAST(800 + MOD(ABS(FARM_FINGERPRINT(CONCAT('frpts_', cp.cust_id, '_', CAST(slot AS STRING)))), 1000) AS INT64) AS redemption_points_amt,
    -- 8 de 10 son devueltos
    CASE WHEN slot <= 8 THEN TRUE ELSE FALSE END AS return_flag
  FROM client_profiles cp
  CROSS JOIN UNNEST(GENERATE_ARRAY(1, 10)) AS slot
  WHERE cp.perfil = 'FRAUDE'
),

-- Unir todos los canjes
all_redeems AS (
  SELECT cust_id, client_num, perfil, slot, redemption_date, redemption_points_amt, return_flag
  FROM redeem_slots_valid
  UNION ALL
  SELECT cust_id, client_num, perfil, slot, redemption_date, redemption_points_amt, return_flag
  FROM redeem_slots_fraude
)

SELECT
  cust_id,
  CONCAT(cust_id, '_R', LPAD(CAST(slot AS STRING), 2, '0')) AS redemption_id,
  redemption_date,
  DATE_TRUNC(redemption_date, MONTH) AS mes,
  redemption_points_amt,
  return_flag,
  -- Tipo de canje
  CASE MOD(ABS(CAST(FARM_FINGERPRINT(CONCAT('rtype_', cust_id, '_', CAST(slot AS STRING))) AS INT64)), 3)
    WHEN 0 THEN 'CATALOGO'
    WHEN 1 THEN 'GIFTCARD'
    ELSE 'CATALOGO'
  END AS price_type_desc,
  CASE MOD(ABS(CAST(FARM_FINGERPRINT(CONCAT('rtype_', cust_id, '_', CAST(slot AS STRING))) AS INT64)), 3)
    WHEN 0 THEN 'CATALOGO'
    WHEN 1 THEN 'GIFTCARD'
    ELSE 'CATALOGO'
  END AS award_family_type_name,
  CASE WHEN MOD(ABS(CAST(FARM_FINGERPRINT(CONCAT('rpos_', cust_id, '_', CAST(slot AS STRING))) AS INT64)), 2) = 0
    THEN 'DIGITAL' ELSE 'TIENDA'
  END AS tipo_pos,
  CASE WHEN MOD(ABS(CAST(FARM_FINGERPRINT(CONCAT('rcamp_', cust_id, '_', CAST(slot AS STRING))) AS INT64)), 4) = 0
    THEN CONCAT('CAMPAIGN_', CAST(EXTRACT(YEAR FROM redemption_date) AS STRING))
    ELSE NULL
  END AS redemption_campaign_desc
FROM all_redeems;


-- ===========================================================================
-- 4. MOCK: base_colaboradores (10 clientes empleados)
-- ===========================================================================
CREATE OR REPLACE TABLE `my-gcp-project.loyalty_analytics.mock_colaboradores` AS
SELECT CONCAT('C', LPAD(CAST(n AS STRING), 4, '0')) AS cust_id
FROM UNNEST(GENERATE_ARRAY(71, 80)) AS n;


-- ===========================================================================
-- VERIFICACIÓN: Resumen de datos mock
-- ===========================================================================
SELECT 'clients'      AS tabla, COUNT(*) AS rows, COUNT(DISTINCT cust_id) AS clientes FROM `my-gcp-project.loyalty_analytics.mock_clients_entity`
UNION ALL
SELECT 'transactions', COUNT(*), COUNT(DISTINCT cust_id) FROM `my-gcp-project.loyalty_analytics.mock_transaction_entity`
UNION ALL
SELECT 'redemptions',  COUNT(*), COUNT(DISTINCT cust_id) FROM `my-gcp-project.loyalty_analytics.mock_redemption_entity`
UNION ALL
SELECT 'colaboradores', COUNT(*), COUNT(DISTINCT cust_id) FROM `my-gcp-project.loyalty_analytics.mock_colaboradores`;

-- Distribución por perfil (verificar cantidades)
SELECT
  CASE
    WHEN client_num <= 50 THEN 'FANTASMA'
    WHEN client_num <= 60 THEN 'MONTO_EXTREMO'
    WHEN client_num <= 70 THEN 'FRAUDE'
    WHEN client_num <= 80 THEN 'COLABORADOR'
    WHEN client_num <= 330 THEN 'PARTICIPANTE'
    WHEN client_num <= 430 THEN 'POSIBILIDAD_CANJE'
    WHEN client_num <= 630 THEN 'CANJEADOR'
    WHEN client_num <= 880 THEN 'RECURRENTE'
    ELSE 'FUGA'
  END AS perfil,
  COUNT(*) AS n_clientes
FROM UNNEST(GENERATE_ARRAY(1, 1000)) AS client_num
GROUP BY 1 ORDER BY 2 DESC;
