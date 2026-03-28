"""
Test local end-to-end de Fase 1 con 1000 clientes mock.
Usa DuckDB como motor SQL (compatible ~95% con BigQuery).
"""
import duckdb
import pandas as pd
import hashlib

con = duckdb.connect(':memory:')

# =====================================================================
# Funciones auxiliares para simular FARM_FINGERPRINT
# =====================================================================
con.execute("""
CREATE OR REPLACE MACRO farm_fingerprint(s) AS
  abs(hash(s))
""")

print("=" * 70)
print("FASE 1 — TEST LOCAL CON 1000 CLIENTES MOCK")
print("=" * 70)

# =====================================================================
# TABLA 1: mock_clients_entity
# =====================================================================
print("\n[1/4] Generando mock_clients_entity...")

con.execute("""
CREATE TABLE mock_clients_entity AS
WITH
client_ids AS (
  SELECT
    'C' || LPAD(CAST(n AS VARCHAR), 4, '0') AS cust_id,
    n AS client_num
  FROM generate_series(1, 1000) AS t(n)
),
client_tiers AS (
  SELECT
    cust_id, client_num,
    CASE
      WHEN client_num <= 500 THEN 'NORMAL'
      WHEN client_num <= 750 THEN 'FAN'
      WHEN client_num <= 900 THEN 'PREMIUM'
      ELSE 'ELITE'
    END AS cat_cust_name,
    DATE '2018-01-01' + INTERVAL (abs(hash('enroll_' || cust_id)) % 1461) DAY AS cust_enroll_date,
    CASE WHEN client_num % 2 = 0 THEN 'F' ELSE 'M' END AS cust_gender_desc,
    20 + (abs(hash('age_' || cust_id)) % 46) AS cust_age_num,
    CASE abs(hash('city_' || cust_id)) % 6
      WHEN 0 THEN 'SANTIAGO' WHEN 1 THEN 'VALPARAISO' WHEN 2 THEN 'CONCEPCION'
      WHEN 3 THEN 'TEMUCO' WHEN 4 THEN 'ANTOFAGASTA' ELSE 'RANCAGUA'
    END AS city_name,
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
  FROM client_ids
),
months AS (
  SELECT CAST(d AS DATE) AS partition_date
  FROM generate_series(DATE '2022-01-31', DATE '2026-03-31', INTERVAL 1 MONTH) AS t(d)
),
stock_params AS (
  SELECT ct.*,
    CASE
      WHEN perfil = 'FANTASMA' THEN 0
      WHEN perfil = 'MONTO_EXTREMO' THEN 200
      WHEN perfil = 'FRAUDE' THEN 1500
      WHEN perfil = 'COLABORADOR' THEN 300
      WHEN perfil = 'PARTICIPANTE' THEN 100 + abs(hash('stock_' || cust_id)) % 800
      WHEN perfil = 'POSIBILIDAD_CANJE' THEN 1000 + abs(hash('stock_' || cust_id)) % 4000
      WHEN perfil = 'CANJEADOR' THEN 200 + abs(hash('stock_' || cust_id)) % 2000
      WHEN perfil = 'RECURRENTE' THEN 500 + abs(hash('stock_' || cust_id)) % 5000
      WHEN perfil = 'FUGA' THEN 100 + abs(hash('stock_' || cust_id)) % 1500
    END AS stock_base,
    CASE
      WHEN perfil = 'FANTASMA' THEN 0
      WHEN perfil = 'MONTO_EXTREMO' THEN 100
      WHEN perfil = 'POSIBILIDAD_CANJE' THEN 40 + abs(hash('grow_' || cust_id)) % 60
      WHEN perfil = 'RECURRENTE' THEN 30 + abs(hash('grow_' || cust_id)) % 70
      WHEN perfil = 'FUGA' THEN 5 + abs(hash('grow_' || cust_id)) % 20
      ELSE 10 + abs(hash('grow_' || cust_id)) % 40
    END AS stock_growth
  FROM client_tiers ct
)
SELECT
  sp.cust_id, m.partition_date, sp.cat_cust_name, CAST(sp.cust_enroll_date AS DATE) AS cust_enroll_date,
  sp.cust_gender_desc, CAST(sp.cust_age_num AS INT) AS cust_age_num, sp.city_name,
  CASE WHEN sp.client_num % 5 != 0 THEN TRUE ELSE FALSE END AS contact_email_flg,
  CASE WHEN sp.client_num % 3 != 0 THEN TRUE ELSE FALSE END AS contact_phone_flg,
  CASE WHEN sp.client_num % 7 = 0 THEN TRUE ELSE FALSE END AS contact_push_flg,
  CASE WHEN sp.cat_cust_name IN ('PREMIUM','ELITE') THEN TRUE
       WHEN sp.client_num % 3 = 0 THEN TRUE ELSE FALSE END AS cust_active_card_flg,
  CASE WHEN sp.client_num % 4 = 0 THEN TRUE ELSE FALSE END AS cust_active_deb_flg,
  CASE WHEN sp.client_num % 10 = 0 THEN TRUE ELSE FALSE END AS cust_active_omp_flg,
  GREATEST(sp.stock_base + CAST(
    DATE_DIFF('month', DATE '2022-01-01', m.partition_date) * sp.stock_growth AS INT
  ), 0) AS cust_stock_point_amt,
  CASE WHEN sp.cat_cust_name IN ('PREMIUM','ELITE') THEN 300 + sp.client_num % 500
       ELSE 50 + sp.client_num % 200 END AS exp_point_current_month_amt,
  CASE WHEN sp.cat_cust_name IN ('PREMIUM','ELITE') THEN 200 + sp.client_num % 300
       ELSE 30 + sp.client_num % 100 END AS exp_point_next_month_amt
FROM stock_params sp
CROSS JOIN months m
WHERE m.partition_date >= CAST(sp.cust_enroll_date AS DATE)
""")

cnt = con.execute("SELECT COUNT(*) AS rows, COUNT(DISTINCT cust_id) AS clients FROM mock_clients_entity").fetchone()
print(f"   -> {cnt[0]:,} filas, {cnt[1]} clientes")


# =====================================================================
# TABLA 2: mock_transaction_entity
# =====================================================================
print("\n[2/4] Generando mock_transaction_entity...")

con.execute("""
CREATE TABLE mock_transaction_entity AS
WITH
client_ids AS (
  SELECT 'C' || LPAD(CAST(n AS VARCHAR), 4, '0') AS cust_id, n AS client_num
  FROM generate_series(1, 1000) AS t(n)
),
client_profiles AS (
  SELECT cust_id, client_num,
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
    CASE
      WHEN client_num <= 50 THEN 0
      WHEN client_num <= 60 THEN 2
      WHEN client_num <= 70 THEN 3
      WHEN client_num <= 80 THEN 2
      WHEN client_num <= 330 THEN 3 + abs(hash('ntxn_' || CAST(client_num AS VARCHAR))) % 8
      WHEN client_num <= 430 THEN 4 + abs(hash('ntxn_' || CAST(client_num AS VARCHAR))) % 8
      WHEN client_num <= 630 THEN 3 + abs(hash('ntxn_' || CAST(client_num AS VARCHAR))) % 10
      WHEN client_num <= 880 THEN 5 + abs(hash('ntxn_' || CAST(client_num AS VARCHAR))) % 12
      ELSE 4 + abs(hash('ntxn_' || CAST(client_num AS VARCHAR))) % 8
    END AS n_txns,
    CASE WHEN client_num <= 60 THEN 5000000.0
         ELSE 10000.0 + (abs(hash('amt_' || CAST(client_num AS VARCHAR))) % 90000) * 1.0
    END AS base_amt
  FROM client_ids
),
txn_slots AS (
  SELECT cp.*, s AS slot
  FROM client_profiles cp
  CROSS JOIN generate_series(1, 16) AS t(s)
  WHERE t.s <= cp.n_txns AND cp.n_txns > 0
),
txns_gen AS (
  SELECT
    ts.cust_id,
    ts.cust_id || '_T' || LPAD(CAST(ts.slot AS VARCHAR), 2, '0') AS tran_id,
    CAST(DATE '2022-01-15' + INTERVAL (abs(hash('tdate_' || ts.cust_id || '_' || CAST(ts.slot AS VARCHAR))) % 1140) DAY AS DATE) AS tran_date,
    ROUND(ts.base_amt * (0.5 + (abs(hash('tamt_' || ts.cust_id || '_' || CAST(ts.slot AS VARCHAR))) % 100) / 100.0), 0) AS tran_amt,
    CASE abs(hash('ret_' || ts.cust_id || '_' || CAST(ts.slot AS VARCHAR))) % 10
      WHEN 0 THEN 'STOREB' WHEN 1 THEN 'STOREC' WHEN 2 THEN 'STORED' WHEN 3 THEN 'STOREE'
      ELSE 'STOREA'
    END AS channel_name,
    CASE WHEN abs(hash('pay_' || ts.cust_id || '_' || CAST(ts.slot AS VARCHAR))) % 3 = 0
      THEN 'DEBITO' ELSE 'STORE_CARD' END AS payment_method_name
  FROM txn_slots ts
  WHERE ts.perfil != 'FANTASMA'
)
SELECT cust_id, tran_id, tran_date,
  DATE_TRUNC('month', tran_date) AS mes,
  tran_amt,
  ROUND(tran_amt * 0.015, 0) AS point_amt,
  'COMPRA' AS tran_type,
  1 AS tran_valid_flg,
  channel_name, payment_method_name
FROM txns_gen
""")

cnt = con.execute("SELECT COUNT(*) AS rows, COUNT(DISTINCT cust_id) AS clients FROM mock_transaction_entity").fetchone()
print(f"   -> {cnt[0]:,} filas, {cnt[1]} clientes con transacciones")


# =====================================================================
# TABLA 3: mock_redemption_entity
# =====================================================================
print("\n[3/4] Generando mock_redemption_entity...")

con.execute("""
CREATE TABLE mock_redemption_entity AS
WITH
client_ids AS (
  SELECT 'C' || LPAD(CAST(n AS VARCHAR), 4, '0') AS cust_id, n AS client_num
  FROM generate_series(1, 1000) AS t(n)
),
client_profiles AS (
  SELECT cust_id, client_num,
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
    CASE
      WHEN client_num <= 430 THEN 0
      WHEN client_num <= 630 THEN 1
      WHEN client_num <= 880 THEN 2 + abs(hash('nred_' || CAST(client_num AS VARCHAR))) % 4
      ELSE 1 + abs(hash('nred_' || CAST(client_num AS VARCHAR))) % 3
    END AS n_canjes
  FROM client_ids
),
-- Canjes validos
redeem_valid AS (
  SELECT cp.cust_id, cp.client_num, cp.perfil, s AS slot,
    CASE
      WHEN cp.perfil = 'FUGA' THEN
        CAST(DATE '2022-01-15' + INTERVAL (abs(hash('rdate_' || cp.cust_id || '_' || CAST(s AS VARCHAR))) % 300) DAY AS DATE)
      WHEN cp.perfil = 'CANJEADOR' THEN
        CAST(DATE '2023-01-15' + INTERVAL (abs(hash('rdate_' || cp.cust_id || '_' || CAST(s AS VARCHAR))) % 990) DAY AS DATE)
      ELSE
        CAST(DATE '2022-03-01' + INTERVAL (abs(hash('rdate_' || cp.cust_id || '_' || CAST(s AS VARCHAR))) % 1460) DAY AS DATE)
    END AS redemption_date,
    -- Monto correlacionado con client_num (proxy de stock_points): clientes altos canjean mas
    CAST(500 + (cp.client_num * 3) + abs(hash('rpts_' || cp.cust_id || '_' || CAST(s AS VARCHAR))) % 2000 AS INT) AS redemption_points_amt,
    FALSE AS return_flag
  FROM client_profiles cp
  CROSS JOIN generate_series(1, 5) AS t(s)
  WHERE t.s <= cp.n_canjes AND cp.n_canjes > 0
),
-- Canjes fraude (10 canjes, 8 devueltos)
redeem_fraude AS (
  SELECT cp.cust_id, cp.client_num, cp.perfil, s AS slot,
    CAST(DATE '2022-03-01' + INTERVAL (s * 30) DAY AS DATE) AS redemption_date,
    CAST(800 + abs(hash('frpts_' || cp.cust_id || '_' || CAST(s AS VARCHAR))) % 1000 AS INT) AS redemption_points_amt,
    CASE WHEN s <= 8 THEN TRUE ELSE FALSE END AS return_flag
  FROM client_profiles cp
  CROSS JOIN generate_series(1, 10) AS t(s)
  WHERE cp.perfil = 'FRAUDE'
),
all_redeems AS (
  SELECT * FROM redeem_valid
  UNION ALL
  SELECT * FROM redeem_fraude
)
SELECT
  cust_id,
  cust_id || '_R' || LPAD(CAST(slot AS VARCHAR), 2, '0') AS redemption_id,
  redemption_date,
  DATE_TRUNC('month', redemption_date) AS mes,
  redemption_points_amt,
  return_flag,
  CASE abs(hash('rtype_' || cust_id || '_' || CAST(slot AS VARCHAR))) % 3
    WHEN 1 THEN 'GIFTCARD' ELSE 'CATALOGO' END AS price_type_desc,
  CASE abs(hash('rtype_' || cust_id || '_' || CAST(slot AS VARCHAR))) % 3
    WHEN 1 THEN 'GIFTCARD' ELSE 'CATALOGO' END AS award_family_type_name,
  CASE WHEN abs(hash('rpos_' || cust_id || '_' || CAST(slot AS VARCHAR))) % 2 = 0
    THEN 'DIGITAL' ELSE 'TIENDA' END AS tipo_pos,
  CASE WHEN abs(hash('rcamp_' || cust_id || '_' || CAST(slot AS VARCHAR))) % 4 = 0
    THEN 'CAMPAIGN_' || CAST(EXTRACT(YEAR FROM redemption_date) AS VARCHAR)
    ELSE NULL END AS redemption_campaign_desc,
  -- Retailer correlacionado con patron de compras del cliente:
  -- 60% canjea en su retailer dominante (basado en client_num), 40% random
  CASE WHEN abs(hash('rretbias_' || cust_id || '_' || CAST(slot AS VARCHAR))) % 10 < 6
    THEN  -- retailer dominante (misma logica que transacciones: hash('ret_' || cust_id) % 10)
      CASE abs(hash('ret_' || cust_id || '_01')) % 10
        WHEN 0 THEN 'STOREB' WHEN 1 THEN 'STOREC' WHEN 2 THEN 'STORED' WHEN 3 THEN 'STOREE'
        ELSE 'STOREA'
      END
    ELSE  -- random
      CASE abs(hash('rretailer_' || cust_id || '_' || CAST(slot AS VARCHAR))) % 5
        WHEN 0 THEN 'STOREA' WHEN 1 THEN 'STOREB' WHEN 2 THEN 'STOREC'
        WHEN 3 THEN 'STORED' ELSE 'STOREE'
      END
  END AS channel_name
FROM all_redeems
""")

cnt = con.execute("SELECT COUNT(*) AS rows, COUNT(DISTINCT cust_id) AS clients FROM mock_redemption_entity").fetchone()
print(f"   -> {cnt[0]:,} filas, {cnt[1]} clientes con canjes")


# =====================================================================
# TABLA 4: mock_colaboradores
# =====================================================================
con.execute("""
CREATE TABLE mock_colaboradores AS
SELECT 'C' || LPAD(CAST(n AS VARCHAR), 4, '0') AS cust_id
FROM generate_series(71, 80) AS t(n)
""")
print("\n[4/4] mock_colaboradores: 10 empleados (C0071-C0080)")


# =====================================================================
# RESUMEN DATOS MOCK
# =====================================================================
print("\n" + "=" * 70)
print("RESUMEN TABLAS MOCK")
print("=" * 70)
df = con.execute("""
SELECT 'clients' AS tabla, COUNT(*) AS rows, COUNT(DISTINCT cust_id) AS clientes FROM mock_clients_entity
UNION ALL
SELECT 'transactions', COUNT(*), COUNT(DISTINCT cust_id) FROM mock_transaction_entity
UNION ALL
SELECT 'redemptions', COUNT(*), COUNT(DISTINCT cust_id) FROM mock_redemption_entity
UNION ALL
SELECT 'colaboradores', COUNT(*), COUNT(DISTINCT cust_id) FROM mock_colaboradores
""").df()
print(df.to_string(index=False))


# =====================================================================
# QUERY 01: EXCLUSIONES
# =====================================================================
print("\n" + "=" * 70)
print("QUERY 01: EXCLUSIONES")
print("=" * 70)

con.execute("""
CREATE TABLE excluded_customers AS
WITH
clientes_universo AS (
  SELECT DISTINCT cust_id, cat_cust_name AS tier, cust_enroll_date AS enrollment_date
  FROM mock_clients_entity
  WHERE partition_date = (SELECT MAX(partition_date) FROM mock_clients_entity)
),
trx_validas AS (
  SELECT cust_id,
    COUNT(DISTINCT tran_id) AS total_transacciones,
    SUM(tran_amt) AS total_monto,
    SUM(tran_amt) / NULLIF(COUNT(DISTINCT tran_id), 0) AS ticket_promedio
  FROM mock_transaction_entity
  WHERE tran_type = 'COMPRA' AND tran_amt > 0 AND tran_valid_flg = 1
  GROUP BY cust_id
),
percentiles AS (
  SELECT PERCENTILE_CONT(0.999) WITHIN GROUP (ORDER BY ticket_promedio) AS p99_9_ticket
  FROM trx_validas WHERE ticket_promedio IS NOT NULL
),
canje_stats AS (
  SELECT cust_id,
    COUNT(*) AS total_canjes,
    COUNT(CASE WHEN return_flag = TRUE THEN 1 END) AS total_devoluciones,
    CAST(COUNT(CASE WHEN return_flag = TRUE THEN 1 END) AS DOUBLE) / COUNT(*) AS tasa_devolucion
  FROM mock_redemption_entity
  GROUP BY cust_id
),
exclusion_fantasmas AS (
  SELECT cu.cust_id, 'FANTASMA' AS motivo_exclusion
  FROM clientes_universo cu LEFT JOIN trx_validas t ON cu.cust_id = t.cust_id
  WHERE t.cust_id IS NULL
),
exclusion_extremos AS (
  SELECT t.cust_id, 'MONTO_EXTREMO' AS motivo_exclusion
  FROM trx_validas t CROSS JOIN percentiles p WHERE t.ticket_promedio > p.p99_9_ticket
),
exclusion_fraude AS (
  SELECT cust_id, 'FRAUDE_DEVOLUCIONES' AS motivo_exclusion
  FROM canje_stats WHERE tasa_devolucion > 0.5 AND total_canjes >= 5
),
exclusion_colaboradores AS (
  SELECT cust_id, 'COLABORADOR' AS motivo_exclusion FROM mock_colaboradores
),
todas AS (
  SELECT cust_id, motivo_exclusion, 1 AS pri FROM exclusion_colaboradores
  UNION ALL SELECT cust_id, motivo_exclusion, 2 FROM exclusion_fraude
  UNION ALL SELECT cust_id, motivo_exclusion, 3 FROM exclusion_extremos
  UNION ALL SELECT cust_id, motivo_exclusion, 4 FROM exclusion_fantasmas
),
dedup AS (
  SELECT cust_id, motivo_exclusion,
    ROW_NUMBER() OVER (PARTITION BY cust_id ORDER BY pri) AS rn
  FROM todas
)
SELECT cust_id, motivo_exclusion, CURRENT_DATE AS fecha_proceso
FROM dedup WHERE rn = 1
""")

df = con.execute("""
SELECT motivo_exclusion, COUNT(*) AS n
FROM excluded_customers GROUP BY 1 ORDER BY 2 DESC
""").df()
print(df.to_string(index=False))
total_excl = con.execute("SELECT COUNT(*) FROM excluded_customers").fetchone()[0]
print(f"\nTotal excluidos: {total_excl}")


# =====================================================================
# QUERY 02: MUESTRA (toma todos los no excluidos)
# =====================================================================
print("\n" + "=" * 70)
print("QUERY 02: MUESTRA")
print("=" * 70)

con.execute("""
CREATE TABLE sample_customers AS
WITH
universo_valido AS (
  SELECT c.cust_id, c.cat_cust_name AS tier, c.cust_enroll_date AS enrollment_date
  FROM mock_clients_entity c
  WHERE c.partition_date = (SELECT MAX(partition_date) FROM mock_clients_entity)
    AND c.cust_id NOT IN (SELECT cust_id FROM excluded_customers)
    AND UPPER(c.cat_cust_name) IN ('NORMAL','FAN','PREMIUM','ELITE')
    AND c.cust_enroll_date < DATE '2023-01-01'
),
redemption_historica AS (
  SELECT DISTINCT cust_id
  FROM mock_redemption_entity
  WHERE return_flag = FALSE
    AND redemption_date < DATE '2023-01-01'
)
SELECT
  uv.cust_id,
  UPPER(uv.tier) AS tier,
  CAST(uv.enrollment_date AS DATE) AS enrollment_date,
  CASE WHEN r.cust_id IS NOT NULL THEN 1 ELSE 0 END AS has_redeemed,
  UPPER(uv.tier) || '_' ||
    CASE WHEN r.cust_id IS NOT NULL THEN 'CANJEADOR' ELSE 'NO_CANJEADOR' END AS estrato,
  CURRENT_DATE AS fecha_proceso
FROM universo_valido uv
LEFT JOIN redemption_historica r ON uv.cust_id = r.cust_id
""")

df = con.execute("""
SELECT estrato, tier, has_redeemed, COUNT(*) AS n
FROM sample_customers GROUP BY 1,2,3 ORDER BY 1
""").df()
print(df.to_string(index=False))
total_sample = con.execute("SELECT COUNT(*) FROM sample_customers").fetchone()[0]
print(f"\nTotal en muestra: {total_sample}")


# =====================================================================
# QUERY 03: FUNNEL STATES MONTHLY
# =====================================================================
print("\n" + "=" * 70)
print("QUERY 03a: FUNNEL STATES MONTHLY")
print("=" * 70)

con.execute("""
CREATE TABLE funnel_states_monthly AS
WITH
meses AS (
  SELECT
    CAST(d AS DATE) - INTERVAL 1 DAY + INTERVAL 1 MONTH AS fecha_fin_mes_raw,
    LAST_DAY(CAST(d AS DATE)) AS fecha_fin_mes,
    CAST(d AS DATE) AS mes_inicio
  FROM generate_series(DATE '2022-01-01', DATE '2026-03-01', INTERVAL 1 MONTH) AS t(d)
),
muestra AS (
  SELECT cust_id, enrollment_date FROM sample_customers
),
canjes_acum AS (
  SELECT r.cust_id, m.fecha_fin_mes,
    COUNT(*) AS total_canjes_historicos, MAX(r.redemption_date) AS ultimo_canje
  FROM mock_redemption_entity r
  INNER JOIN meses m ON r.redemption_date <= m.fecha_fin_mes
  WHERE r.return_flag = FALSE
    AND r.cust_id IN (SELECT cust_id FROM muestra)
  GROUP BY r.cust_id, m.fecha_fin_mes
),
compras_acum AS (
  SELECT t.cust_id, m.fecha_fin_mes,
    COUNT(DISTINCT t.tran_id) AS total_compras_historicas, MAX(t.tran_date) AS ultima_compra
  FROM mock_transaction_entity t
  INNER JOIN meses m ON t.tran_date <= m.fecha_fin_mes
  WHERE t.tran_type = 'COMPRA' AND t.tran_amt > 0 AND t.tran_valid_flg = 1
    AND t.cust_id IN (SELECT cust_id FROM muestra)
  GROUP BY t.cust_id, m.fecha_fin_mes
),
stock_puntos_ranked AS (
  SELECT c.cust_id, m.fecha_fin_mes, c.cust_stock_point_amt AS stock_points,
    UPPER(c.cat_cust_name) AS tier,
    ROW_NUMBER() OVER (PARTITION BY c.cust_id, m.fecha_fin_mes ORDER BY c.partition_date DESC) AS rn
  FROM mock_clients_entity c
  INNER JOIN meses m ON c.partition_date BETWEEN m.fecha_fin_mes - INTERVAL 7 DAY AND m.fecha_fin_mes
  WHERE c.cust_id IN (SELECT cust_id FROM muestra)
),
stock_puntos AS (
  SELECT cust_id, fecha_fin_mes, stock_points, tier FROM stock_puntos_ranked WHERE rn = 1
),
estado_raw AS (
  SELECT m.cust_id, mes.fecha_fin_mes, mes.mes_inicio,
    COALESCE(sp.tier, 'NORMAL') AS tier,
    COALESCE(ca.total_compras_historicas, 0) AS total_compras,
    ca.ultima_compra,
    COALESCE(cj.total_canjes_historicos, 0) AS total_canjes,
    cj.ultimo_canje,
    COALESCE(sp.stock_points, 0) AS stock_points
  FROM muestra m
  INNER JOIN meses mes ON mes.fecha_fin_mes >= m.enrollment_date
  LEFT JOIN compras_acum ca ON ca.cust_id = m.cust_id AND ca.fecha_fin_mes = mes.fecha_fin_mes
  LEFT JOIN canjes_acum cj  ON cj.cust_id = m.cust_id AND cj.fecha_fin_mes = mes.fecha_fin_mes
  LEFT JOIN stock_puntos sp ON sp.cust_id = m.cust_id AND sp.fecha_fin_mes = mes.fecha_fin_mes
),
estado_calculado AS (
  SELECT *,
    DATE_DIFF('day', ultimo_canje, fecha_fin_mes) AS dias_desde_ultimo_canje,
    CASE WHEN UPPER(tier) IN ('ELITE','PREMIUM') THEN 730 ELSE 365 END AS umbral_fuga_dias,
    CASE
      WHEN total_canjes >= 1 AND ultimo_canje IS NOT NULL
        AND DATE_DIFF('day', ultimo_canje, fecha_fin_mes) >
            CASE WHEN UPPER(tier) IN ('ELITE','PREMIUM') THEN 730 ELSE 365 END
      THEN 'FUGA'
      WHEN total_canjes >= 2 THEN 'RECURRENTE'
      WHEN total_canjes = 1  THEN 'CANJEADOR'
      WHEN total_compras >= 1 AND total_canjes = 0 AND stock_points >= 1000 THEN 'POSIBILIDAD_CANJE'
      WHEN total_compras >= 1 AND total_canjes = 0 THEN 'PARTICIPANTE'
      ELSE 'INSCRITO'
    END AS funnel_state
  FROM estado_raw
)
SELECT cust_id, fecha_fin_mes, mes_inicio, tier, total_compras, ultima_compra,
  total_canjes, ultimo_canje, stock_points, dias_desde_ultimo_canje, umbral_fuga_dias,
  funnel_state, CURRENT_DATE AS fecha_proceso
FROM estado_calculado
""")

df = con.execute("""
SELECT mes_inicio, funnel_state, COUNT(*) AS n
FROM funnel_states_monthly
WHERE mes_inicio IN ('2022-06-01','2023-01-01','2023-06-01','2024-01-01','2024-06-01','2025-01-01')
GROUP BY 1, 2 ORDER BY 1, 2
""").df()
print(df.to_string(index=False))


# =====================================================================
# QUERY 03b: MARKOV TRANSITION MATRIX
# =====================================================================
print("\n" + "=" * 70)
print("QUERY 03b: MARKOV TRANSITION MATRIX")
print("=" * 70)

con.execute("""
CREATE TABLE markov_transition_matrix AS
WITH
transiciones AS (
  SELECT curr.cust_id,
    curr.funnel_state AS estado_origen, next_m.funnel_state AS estado_destino,
    CASE WHEN UPPER(curr.tier) IN ('ELITE','PREMIUM') THEN 'ALTO' ELSE 'BASE' END AS tier_group
  FROM funnel_states_monthly curr
  INNER JOIN funnel_states_monthly next_m
    ON curr.cust_id = next_m.cust_id
    AND curr.fecha_fin_mes + INTERVAL 1 MONTH = next_m.fecha_fin_mes
  WHERE curr.fecha_fin_mes BETWEEN DATE '2022-01-31' AND DATE '2025-03-31'
),
conteo AS (
  SELECT tier_group, estado_origen, estado_destino, COUNT(*) AS n_transiciones
  FROM transiciones WHERE estado_destino IS NOT NULL
  GROUP BY 1, 2, 3
),
total_origen AS (
  SELECT tier_group, estado_origen, SUM(n_transiciones) AS total_desde_origen
  FROM conteo GROUP BY 1, 2
)
SELECT c.tier_group, c.estado_origen, c.estado_destino, c.n_transiciones,
  t.total_desde_origen,
  ROUND(CAST(c.n_transiciones AS DOUBLE) / t.total_desde_origen, 4) AS prob_transicion,
  CURRENT_DATE AS fecha_proceso
FROM conteo c JOIN total_origen t ON c.tier_group = t.tier_group AND c.estado_origen = t.estado_origen
ORDER BY c.tier_group, c.estado_origen, prob_transicion DESC
""")

df = con.execute("""
SELECT tier_group, estado_origen, estado_destino, prob_transicion
FROM markov_transition_matrix
ORDER BY 1, 2, 4 DESC
""").df()
print(df.to_string(index=False))

# Verificar sumas = 1
print("\nVerificacion: probabilidades suman 1 por estado origen?")
df_sum = con.execute("""
SELECT tier_group, estado_origen, ROUND(SUM(prob_transicion), 4) AS suma
FROM markov_transition_matrix GROUP BY 1, 2 ORDER BY 1, 2
""").df()
print(df_sum.to_string(index=False))


# =====================================================================
# QUERY 04: CUSTOMER SNAPSHOT
# =====================================================================
print("\n" + "=" * 70)
print("QUERY 04: CUSTOMER SNAPSHOT")
print("=" * 70)

con.execute("""
CREATE TABLE customer_snapshot AS
WITH
t0_list AS (
  SELECT CAST(t0 AS DATE) AS t0
  FROM generate_series(DATE '2023-01-01', DATE '2025-03-01', INTERVAL 1 MONTH) AS t(t0)
),
muestra AS (
  SELECT cust_id, tier AS tier_actual, enrollment_date, has_redeemed AS has_redeemed_global
  FROM sample_customers
),
base AS (
  SELECT t0.t0, t0.t0 - INTERVAL 12 MONTH AS pre_start,
    t0.t0 + INTERVAL 12 MONTH AS post_end, m.cust_id
  FROM t0_list t0 CROSS JOIN muestra m
),
demograficas AS (
  SELECT b.cust_id, b.t0,
    UPPER(LAST(c.cat_cust_name ORDER BY c.partition_date)) AS tier,
    LAST(c.cust_enroll_date ORDER BY c.partition_date) AS enrollment_date,
    LAST(c.cust_gender_desc ORDER BY c.partition_date) AS gender,
    LAST(c.cust_age_num ORDER BY c.partition_date) AS age,
    LAST(c.city_name ORDER BY c.partition_date) AS city,
    LAST(c.contact_email_flg ORDER BY c.partition_date) AS contact_email_flg,
    LAST(c.contact_phone_flg ORDER BY c.partition_date) AS contact_phone_flg,
    LAST(c.contact_push_flg ORDER BY c.partition_date) AS contact_push_flg,
    LAST(c.cust_active_card_flg ORDER BY c.partition_date) AS cust_active_card_flg,
    LAST(c.cust_active_deb_flg ORDER BY c.partition_date) AS cust_active_deb_flg,
    LAST(c.cust_active_omp_flg ORDER BY c.partition_date) AS cust_active_omp_flg,
    LAST(c.cust_stock_point_amt ORDER BY c.partition_date) AS stock_points_at_t0,
    LAST(c.exp_point_current_month_amt ORDER BY c.partition_date) AS exp_points_current_at_t0,
    LAST(c.exp_point_next_month_amt ORDER BY c.partition_date) AS exp_points_next_at_t0
  FROM base b
  JOIN mock_clients_entity c
    ON c.cust_id = b.cust_id AND c.partition_date < CAST(b.t0 AS DATE)
    AND c.partition_date >= CAST(b.t0 AS DATE) - INTERVAL 2 MONTH
  GROUP BY b.cust_id, b.t0
),
trx_pre AS (
  SELECT b.cust_id, b.t0,
    DATE_DIFF('day', MAX(t.tran_date), CAST(b.t0 AS DATE)) AS recency_days,
    COUNT(DISTINCT t.tran_id) AS frequency_total,
    COUNT(DISTINCT t.tran_id) / 12.0 AS frequency_monthly_avg,
    SUM(t.tran_amt) AS monetary_total,
    SUM(t.tran_amt) / NULLIF(COUNT(DISTINCT t.tran_id), 0) AS monetary_avg_ticket,
    SUM(t.tran_amt) / 12.0 AS monetary_monthly_avg,
    SUM(t.point_amt) AS points_earned_total,
    SUM(t.point_amt) / 12.0 AS points_earned_monthly_avg,
    SUM(CASE WHEN t.tran_date >= CAST(b.t0 AS DATE) - INTERVAL 30 DAY THEN t.point_amt ELSE 0 END) AS earn_velocity_30,
    SUM(CASE WHEN t.tran_date >= CAST(b.t0 AS DATE) - INTERVAL 90 DAY THEN t.point_amt ELSE 0 END) AS earn_velocity_90,
    SUM(CASE WHEN t.tran_date >= CAST(b.t0 AS DATE) - INTERVAL 3 MONTH THEN t.tran_amt ELSE 0 END) AS spend_last_3m,
    SUM(CASE WHEN t.tran_date < CAST(b.t0 AS DATE) - INTERVAL 3 MONTH AND t.tran_date >= CAST(b.t0 AS DATE) - INTERVAL 6 MONTH THEN t.tran_amt ELSE 0 END) AS spend_prev_3m,
    COUNT(DISTINCT CASE WHEN t.tran_date >= CAST(b.t0 AS DATE) - INTERVAL 3 MONTH THEN t.tran_id END) AS freq_last_3m,
    COUNT(DISTINCT CASE WHEN t.tran_date < CAST(b.t0 AS DATE) - INTERVAL 3 MONTH AND t.tran_date >= CAST(b.t0 AS DATE) - INTERVAL 6 MONTH THEN t.tran_id END) AS freq_prev_3m,
    SUM(CASE WHEN t.channel_name='STOREA' THEN t.tran_amt ELSE 0 END) AS spend_store_a,
    SUM(CASE WHEN t.channel_name='STOREB' THEN t.tran_amt ELSE 0 END) AS spend_store_b,
    SUM(CASE WHEN t.channel_name='STOREC' THEN t.tran_amt ELSE 0 END) AS spend_store_c,
    SUM(CASE WHEN t.channel_name='STORED' THEN t.tran_amt ELSE 0 END) AS spend_store_d,
    SUM(CASE WHEN t.channel_name='STOREE' THEN t.tran_amt ELSE 0 END) AS spend_store_e,
    COUNT(DISTINCT CASE WHEN t.channel_name='STOREA' THEN t.tran_id END) AS freq_store_a,
    COUNT(DISTINCT CASE WHEN t.channel_name='STOREB' THEN t.tran_id END) AS freq_store_b,
    COUNT(DISTINCT CASE WHEN t.channel_name='STOREC' THEN t.tran_id END) AS freq_store_c,
    COUNT(DISTINCT CASE WHEN t.channel_name='STORED' THEN t.tran_id END) AS freq_store_d,
    COUNT(DISTINCT CASE WHEN t.channel_name='STOREE' THEN t.tran_id END) AS freq_store_e,
    COUNT(DISTINCT t.channel_name) AS retailer_count,
    CAST(COUNT(DISTINCT CASE WHEN UPPER(t.payment_method_name) LIKE '%STORE_CARD%' THEN t.tran_id END) AS DOUBLE) / NULLIF(COUNT(DISTINCT t.tran_id), 0) AS pct_store_card_payments,
    CAST(COUNT(DISTINCT CASE WHEN UPPER(t.payment_method_name) LIKE '%DEB%' THEN t.tran_id END) AS DOUBLE) / NULLIF(COUNT(DISTINCT t.tran_id), 0) AS pct_debit_payments,
    STDDEV(t.tran_amt) AS std_tran_amt,
    AVG(t.tran_amt) AS avg_tran_amt,
    MIN(t.tran_date) AS primera_compra_pre,
    MAX(t.tran_date) AS ultima_compra_pre
  FROM base b
  JOIN mock_transaction_entity t
    ON t.cust_id = b.cust_id AND t.tran_date >= CAST(b.pre_start AS DATE) AND t.tran_date < CAST(b.t0 AS DATE)
    AND t.tran_type = 'COMPRA' AND t.tran_amt > 0 AND t.tran_valid_flg = 1
  GROUP BY b.cust_id, b.t0
),
canjes_pre AS (
  SELECT b.cust_id, b.t0,
    COUNT(*) AS redeem_count_pre,
    SUM(r.redemption_points_amt) AS redeem_points_total_pre,
    MAX(r.redemption_date) AS last_redeem_date_pre,
    DATE_DIFF('day', MAX(r.redemption_date), CAST(b.t0 AS DATE)) AS days_since_last_redeem,
    COUNT(CASE WHEN r.redemption_date >= CAST(b.pre_start AS DATE) THEN 1 END) AS redeem_count_12m_pre,
    SUM(CASE WHEN r.redemption_date >= CAST(b.pre_start AS DATE) THEN r.redemption_points_amt ELSE 0 END) AS redeem_points_12m_pre,
    SUM(CASE WHEN r.redemption_date >= CAST(b.t0 AS DATE) - INTERVAL 30 DAY THEN r.redemption_points_amt ELSE 0 END) AS redeem_velocity_30,
    CAST(COUNT(CASE WHEN LOWER(r.price_type_desc) LIKE '%catalogo%' THEN 1 END) AS DOUBLE) / COUNT(*) AS pct_redeem_catalogo,
    CAST(COUNT(CASE WHEN LOWER(r.price_type_desc) LIKE '%giftcard%' THEN 1 END) AS DOUBLE) / COUNT(*) AS pct_redeem_giftcard,
    CAST(COUNT(CASE WHEN LOWER(r.tipo_pos) = 'digital' THEN 1 END) AS DOUBLE) / COUNT(*) AS pct_redeem_digital,
    AVG(r.redemption_points_amt) AS avg_redeem_points,
    CAST(COUNT(CASE WHEN r.redemption_campaign_desc IS NOT NULL AND r.redemption_campaign_desc != '' THEN 1 END) AS DOUBLE) / COUNT(*) AS campaign_response_rate
  FROM base b
  JOIN mock_redemption_entity r
    ON r.cust_id = b.cust_id AND r.redemption_date < CAST(b.t0 AS DATE) AND r.return_flag = FALSE
  GROUP BY b.cust_id, b.t0
),
funnel_with_lag AS (
  SELECT fs.cust_id, fs.fecha_fin_mes, fs.funnel_state, fs.total_canjes, fs.total_compras,
    LAG(fs.funnel_state) OVER (PARTITION BY fs.cust_id ORDER BY fs.fecha_fin_mes) AS prev_funnel_state
  FROM funnel_states_monthly fs
),
funnel_agg AS (
  SELECT b.cust_id, b.t0,
    LAST(fl.funnel_state ORDER BY fl.fecha_fin_mes) AS funnel_state_at_t0,
    LAST(fl.total_canjes ORDER BY fl.fecha_fin_mes) AS total_canjes_at_t0,
    LAST(fl.total_compras ORDER BY fl.fecha_fin_mes) AS total_compras_at_t0,
    COUNT(CASE WHEN fl.funnel_state != fl.prev_funnel_state AND fl.prev_funnel_state IS NOT NULL THEN 1 END) AS transitions_last_12m,
    CAST(COUNT(DISTINCT fl.fecha_fin_mes) AS DOUBLE) /
      NULLIF(COUNT(CASE WHEN fl.funnel_state != fl.prev_funnel_state AND fl.prev_funnel_state IS NOT NULL THEN 1 END), 0) AS velocity_in_funnel
  FROM base b
  JOIN funnel_with_lag fl ON fl.cust_id = b.cust_id
    AND fl.fecha_fin_mes < CAST(b.t0 AS DATE) AND fl.fecha_fin_mes >= CAST(b.t0 AS DATE) - INTERVAL 13 MONTH
  GROUP BY b.cust_id, b.t0
),
funnel_pre AS (
  SELECT fa.cust_id, fa.t0, fa.funnel_state_at_t0, fa.total_canjes_at_t0, fa.total_compras_at_t0,
    fa.transitions_last_12m, fa.velocity_in_funnel,
    (SELECT COUNT(DISTINCT fl2.fecha_fin_mes)
     FROM funnel_with_lag fl2
     WHERE fl2.cust_id = fa.cust_id
       AND fl2.fecha_fin_mes < CAST(fa.t0 AS DATE)
       AND fl2.fecha_fin_mes >= CAST(fa.t0 AS DATE) - INTERVAL 13 MONTH
       AND fl2.funnel_state = fa.funnel_state_at_t0
    ) AS months_in_current_state
  FROM funnel_agg fa
),
markov_probs AS (
  SELECT fp.cust_id, fp.t0, fp.funnel_state_at_t0,
    COALESCE(MAX(CASE WHEN mt.estado_destino IN ('CANJEADOR','RECURRENTE','POSIBILIDAD_CANJE') THEN mt.prob_transicion END), 0) AS prob_to_next_state,
    COALESCE(MAX(CASE WHEN mt.estado_destino = 'FUGA' THEN mt.prob_transicion END), 0) AS prob_to_fuga
  FROM funnel_pre fp
  LEFT JOIN demograficas d_tier ON d_tier.cust_id = fp.cust_id AND d_tier.t0 = fp.t0
  LEFT JOIN markov_transition_matrix mt
    ON mt.estado_origen = fp.funnel_state_at_t0
    AND mt.tier_group = CASE WHEN UPPER(d_tier.tier) IN ('ELITE','PREMIUM') THEN 'ALTO' ELSE 'BASE' END
  GROUP BY fp.cust_id, fp.t0, fp.funnel_state_at_t0
),
target_has_redeemed AS (
  SELECT b.cust_id, b.t0, TRUE AS has_redeemed_before_t0
  FROM base b
  WHERE EXISTS (
    SELECT 1 FROM mock_redemption_entity r
    WHERE r.cust_id = b.cust_id AND r.redemption_date < CAST(b.t0 AS DATE) AND r.return_flag = FALSE
  )
),
target_post_canjes AS (
  SELECT b.cust_id, b.t0, COUNT(*) AS n_canjes_post
  FROM base b
  JOIN mock_redemption_entity r
    ON r.cust_id = b.cust_id AND r.redemption_date >= CAST(b.t0 AS DATE) AND r.redemption_date < CAST(b.post_end AS DATE)
    AND r.return_flag = FALSE
  GROUP BY b.cust_id, b.t0
),
target_revenue AS (
  SELECT b.cust_id, b.t0, SUM(t.tran_amt) AS revenue_post_12m
  FROM base b
  JOIN mock_transaction_entity t
    ON t.cust_id = b.cust_id AND t.tran_date >= CAST(b.t0 AS DATE) AND t.tran_date < CAST(b.post_end AS DATE)
    AND t.tran_type = 'COMPRA' AND t.tran_amt > 0 AND t.tran_valid_flg = 1
  GROUP BY b.cust_id, b.t0
),
target AS (
  SELECT b.cust_id, b.t0,
    COALESCE(thr.has_redeemed_before_t0, FALSE) AS has_redeemed_before_t0,
    COALESCE(tpc.n_canjes_post, 0) > 0 AS canjea_post,
    COALESCE(tpc.n_canjes_post, 0) AS n_canjes_post,
    COALESCE(trv.revenue_post_12m, 0) AS revenue_post_12m
  FROM base b
  LEFT JOIN target_has_redeemed thr ON thr.cust_id = b.cust_id AND thr.t0 = b.t0
  LEFT JOIN target_post_canjes  tpc ON tpc.cust_id = b.cust_id AND tpc.t0 = b.t0
  LEFT JOIN target_revenue      trv ON trv.cust_id = b.cust_id AND trv.t0 = b.t0
),
snapshot_final AS (
  SELECT
    b.cust_id, b.t0,
    UPPER(COALESCE(d.tier, mu.tier_actual)) AS tier,
    DATE_DIFF('month', COALESCE(CAST(d.enrollment_date AS DATE), mu.enrollment_date), CAST(b.t0 AS DATE)) AS tenure_months,
    d.gender, d.age, d.city,
    d.cust_active_card_flg, d.cust_active_deb_flg, d.cust_active_omp_flg,
    COALESCE(d.contact_email_flg, FALSE) AS contact_email_flg,
    COALESCE(d.contact_phone_flg, FALSE) AS contact_phone_flg,
    COALESCE(d.contact_push_flg, FALSE) AS contact_push_flg,
    COALESCE(trx.recency_days, 999) AS recency_days,
    COALESCE(trx.frequency_total, 0) AS frequency_total,
    COALESCE(trx.frequency_monthly_avg, 0) AS frequency_monthly_avg,
    COALESCE(trx.monetary_total, 0) AS monetary_total,
    COALESCE(trx.monetary_avg_ticket, 0) AS monetary_avg_ticket,
    COALESCE(trx.monetary_monthly_avg, 0) AS monetary_monthly_avg,
    COALESCE(trx.points_earned_total, 0) AS points_earned_total,
    COALESCE(trx.points_earned_monthly_avg, 0) AS points_earned_monthly_avg,
    COALESCE(d.stock_points_at_t0, 0) AS stock_points_at_t0,
    COALESCE(d.exp_points_current_at_t0, 0) AS exp_points_current_at_t0,
    COALESCE(d.exp_points_next_at_t0, 0) AS exp_points_next_at_t0,
    COALESCE(cj.redeem_count_pre, 0) AS redeem_count_pre,
    COALESCE(cj.redeem_points_total_pre, 0) AS redeem_points_total_pre,
    COALESCE(cj.redeem_count_12m_pre, 0) AS redeem_count_12m_pre,
    COALESCE(cj.redeem_points_12m_pre, 0) AS redeem_points_12m_pre,
    CAST(COALESCE(cj.redeem_points_total_pre, 0) AS DOUBLE) / NULLIF(COALESCE(trx.points_earned_total, 0), 0) AS redeem_rate,
    COALESCE(1.0 - CAST(COALESCE(cj.redeem_points_total_pre, 0) AS DOUBLE) / NULLIF(COALESCE(trx.points_earned_total, 0), 0), 1) AS breakage,
    COALESCE(trx.earn_velocity_30, 0) AS earn_velocity_30,
    COALESCE(trx.earn_velocity_90, 0) AS earn_velocity_90,
    COALESCE(cj.redeem_velocity_30, 0) AS redeem_velocity_30,
    COALESCE(CAST(COALESCE(trx.spend_last_3m, 0) AS DOUBLE) / NULLIF(COALESCE(trx.spend_prev_3m, 0), 0), 0) AS spend_trend,
    CASE WHEN trx.ultima_compra_pre IS NOT NULL THEN DATE_DIFF('day', trx.ultima_compra_pre, CAST(b.t0 AS DATE)) ELSE 999 END AS days_since_last_activity,
    COALESCE(cj.days_since_last_redeem, 999) AS days_since_last_redeem,
    CASE WHEN COALESCE(d.stock_points_at_t0, 0) >= 1000 THEN 1 ELSE 0 END AS redeem_capacity,
    GREATEST(COALESCE(d.stock_points_at_t0, 0) - 1000, 0) AS points_above_threshold,
    COALESCE(CAST(GREATEST(1000 - COALESCE(d.stock_points_at_t0, 0), 0) AS DOUBLE) / NULLIF(COALESCE(trx.earn_velocity_30, 0), 0) * 30, 999) AS days_to_redeem_capacity,
    CAST(COALESCE(d.exp_points_current_at_t0, 0) AS DOUBLE) / NULLIF(COALESCE(d.stock_points_at_t0, 0), 0) AS points_pressure,
    COALESCE(trx.spend_store_a, 0) AS spend_store_a,
    COALESCE(trx.spend_store_b, 0) AS spend_store_b,
    COALESCE(trx.spend_store_c, 0) AS spend_store_c,
    COALESCE(trx.spend_store_d, 0) AS spend_store_d,
    COALESCE(trx.spend_store_e, 0) AS spend_store_e,
    COALESCE(trx.freq_store_a, 0) AS freq_store_a,
    COALESCE(trx.freq_store_b, 0) AS freq_store_b,
    COALESCE(trx.freq_store_c, 0) AS freq_store_c,
    COALESCE(trx.freq_store_d, 0) AS freq_store_d,
    COALESCE(trx.freq_store_e, 0) AS freq_store_e,
    COALESCE(trx.retailer_count, 0) AS retailer_count,
    CASE
      WHEN GREATEST(COALESCE(trx.spend_store_a,0),COALESCE(trx.spend_store_b,0),COALESCE(trx.spend_store_c,0),COALESCE(trx.spend_store_d,0),COALESCE(trx.spend_store_e,0)) = COALESCE(trx.spend_store_c,0) AND COALESCE(trx.spend_store_c,0) > 0 THEN 'STOREC'
      WHEN GREATEST(COALESCE(trx.spend_store_a,0),COALESCE(trx.spend_store_b,0),COALESCE(trx.spend_store_c,0),COALESCE(trx.spend_store_d,0),COALESCE(trx.spend_store_e,0)) = COALESCE(trx.spend_store_a,0) AND COALESCE(trx.spend_store_a,0) > 0 THEN 'STOREA'
      WHEN GREATEST(COALESCE(trx.spend_store_a,0),COALESCE(trx.spend_store_b,0),COALESCE(trx.spend_store_c,0),COALESCE(trx.spend_store_d,0),COALESCE(trx.spend_store_e,0)) = COALESCE(trx.spend_store_b,0) AND COALESCE(trx.spend_store_b,0) > 0 THEN 'STOREB'
      WHEN GREATEST(COALESCE(trx.spend_store_a,0),COALESCE(trx.spend_store_b,0),COALESCE(trx.spend_store_c,0),COALESCE(trx.spend_store_d,0),COALESCE(trx.spend_store_e,0)) = COALESCE(trx.spend_store_d,0) AND COALESCE(trx.spend_store_d,0) > 0 THEN 'STORED'
      WHEN GREATEST(COALESCE(trx.spend_store_a,0),COALESCE(trx.spend_store_b,0),COALESCE(trx.spend_store_c,0),COALESCE(trx.spend_store_d,0),COALESCE(trx.spend_store_e,0)) = COALESCE(trx.spend_store_e,0) AND COALESCE(trx.spend_store_e,0) > 0 THEN 'STOREE'
      ELSE 'NINGUNO'
    END AS dominant_retailer,
    -- retailer_entropy (Shannon entropy, frequency-based — matching production SQL)
    CASE WHEN COALESCE(trx.frequency_total, 0) > 0 THEN
      -(
        COALESCE(CAST(COALESCE(trx.freq_store_a, 0) AS DOUBLE) / trx.frequency_total
          * LN(NULLIF(CAST(COALESCE(trx.freq_store_a, 0) AS DOUBLE) / trx.frequency_total, 0)), 0)
        + COALESCE(CAST(COALESCE(trx.freq_store_b, 0) AS DOUBLE) / trx.frequency_total
          * LN(NULLIF(CAST(COALESCE(trx.freq_store_b, 0) AS DOUBLE) / trx.frequency_total, 0)), 0)
        + COALESCE(CAST(COALESCE(trx.freq_store_c, 0) AS DOUBLE) / trx.frequency_total
          * LN(NULLIF(CAST(COALESCE(trx.freq_store_c, 0) AS DOUBLE) / trx.frequency_total, 0)), 0)
        + COALESCE(CAST(COALESCE(trx.freq_store_d, 0) AS DOUBLE) / trx.frequency_total
          * LN(NULLIF(CAST(COALESCE(trx.freq_store_d, 0) AS DOUBLE) / trx.frequency_total, 0)), 0)
        + COALESCE(CAST(COALESCE(trx.freq_store_e, 0) AS DOUBLE) / trx.frequency_total
          * LN(NULLIF(CAST(COALESCE(trx.freq_store_e, 0) AS DOUBLE) / trx.frequency_total, 0)), 0)
      )
    ELSE 0 END AS retailer_entropy,
    COALESCE(trx.pct_store_card_payments, 0) AS pct_store_card_payments,
    COALESCE(trx.pct_debit_payments, 0) AS pct_debit_payments,
    COALESCE(cj.pct_redeem_catalogo, 0) AS pct_redeem_catalogo,
    COALESCE(cj.pct_redeem_giftcard, 0) AS pct_redeem_giftcard,
    COALESCE(cj.pct_redeem_digital, 0) AS pct_redeem_digital,
    COALESCE(cj.avg_redeem_points, 0) AS avg_redeem_points,
    COALESCE(fp.funnel_state_at_t0, 'INSCRITO') AS funnel_state_at_t0,
    COALESCE(fp.months_in_current_state, 0) * 30 AS days_in_current_state,
    COALESCE(fp.transitions_last_12m, 0) AS transitions_last_12m,
    COALESCE(fp.velocity_in_funnel, 0) AS velocity_in_funnel,
    COALESCE(mp.prob_to_next_state, 0) AS prob_to_next_state,
    COALESCE(mp.prob_to_fuga, 0) AS prob_to_fuga,
    CAST(COALESCE(trx.points_earned_total, 0) AS DOUBLE) / NULLIF(COALESCE(cj.redeem_points_total_pre, 0), 0) AS ratio_earn_redeem,
    COALESCE(
      (CAST(COALESCE(trx.spend_last_3m, 0) AS DOUBLE) / NULLIF(trx.freq_last_3m, 0)) /
      NULLIF(CAST(COALESCE(trx.spend_prev_3m, 0) AS DOUBLE) / NULLIF(trx.freq_prev_3m, 0), 0)
    , 0) AS ticket_trend,
    COALESCE(trx.std_tran_amt, 0) / NULLIF(COALESCE(trx.avg_tran_amt, 0), 0) AS spend_variability,
    COALESCE(cj.campaign_response_rate, 0) AS campaign_response_rate,
    EXTRACT(MONTH FROM CAST(b.t0 AS DATE)) AS month_of_t0,
    CASE WHEN EXTRACT(MONTH FROM CAST(b.t0 AS DATE)) IN (11) THEN 1 ELSE 0 END AS is_cyber_month,
    CASE WHEN EXTRACT(MONTH FROM CAST(b.t0 AS DATE)) IN (12,1) THEN 1 ELSE 0 END AS is_holiday_month,
    CASE
      WHEN CASE WHEN trx.ultima_compra_pre IS NOT NULL THEN DATE_DIFF('day', trx.ultima_compra_pre, CAST(b.t0 AS DATE)) ELSE 999 END <= 365
        OR (cj.last_redeem_date_pre IS NOT NULL AND DATE_DIFF('day', cj.last_redeem_date_pre, CAST(b.t0 AS DATE)) <= 365)
      THEN 'ACTIVO' ELSE 'INACTIVO'
    END AS status,
    tgt.has_redeemed_before_t0,
    tgt.canjea_post,
    tgt.n_canjes_post,
    tgt.revenue_post_12m,
    CASE
      WHEN tgt.canjea_post = FALSE THEN 0
      WHEN tgt.has_redeemed_before_t0 = FALSE THEN 1
      ELSE 2
    END AS y,
    CURRENT_DATE AS fecha_proceso
  FROM base b
  LEFT JOIN muestra mu ON mu.cust_id = b.cust_id
  LEFT JOIN demograficas d ON d.cust_id = b.cust_id AND d.t0 = b.t0
  LEFT JOIN trx_pre trx ON trx.cust_id = b.cust_id AND trx.t0 = b.t0
  LEFT JOIN canjes_pre cj ON cj.cust_id = b.cust_id AND cj.t0 = b.t0
  LEFT JOIN funnel_pre fp ON fp.cust_id = b.cust_id AND fp.t0 = b.t0
  LEFT JOIN markov_probs mp ON mp.cust_id = b.cust_id AND mp.t0 = b.t0
  LEFT JOIN target tgt ON tgt.cust_id = b.cust_id AND tgt.t0 = b.t0
)
SELECT
  sf.*,
  NTILE(5) OVER (PARTITION BY sf.t0 ORDER BY sf.monetary_total)       AS quintil_gasto,
  NTILE(5) OVER (PARTITION BY sf.t0 ORDER BY sf.stock_points_at_t0)   AS quintil_puntos,
  NTILE(5) OVER (PARTITION BY sf.t0 ORDER BY sf.frequency_total)      AS quintil_frecuencia,
  NTILE(5) OVER (PARTITION BY sf.t0 ORDER BY sf.monetary_monthly_avg) AS quintil_gasto_mensual
FROM snapshot_final sf
""")

total_snap = con.execute("SELECT COUNT(*) FROM customer_snapshot").fetchone()[0]
n_cols = con.execute("SELECT COUNT(*) FROM information_schema.columns WHERE table_name='customer_snapshot'").fetchone()[0]
print(f"customer_snapshot: {total_snap:,} filas, {n_cols} columnas")

# Distribución del target por t0
print("\nDistribucion del target y por t0:")
df = con.execute("""
SELECT t0, y, COUNT(*) AS n,
  ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY t0), 1) AS pct
FROM customer_snapshot
GROUP BY 1, 2 ORDER BY 1, 2
""").df()
print(df.to_string(index=False))

# Funnel state por t0
print("\nFunnel state por t0:")
df = con.execute("""
SELECT t0, funnel_state_at_t0, COUNT(*) AS n
FROM customer_snapshot
GROUP BY 1, 2 ORDER BY 1, 2
""").df()
print(df.to_string(index=False))

# Tier distribution
print("\nDistribucion por tier:")
df = con.execute("""
SELECT tier, COUNT(*) AS n, COUNT(DISTINCT cust_id) AS clientes_unicos
FROM customer_snapshot GROUP BY 1 ORDER BY 2 DESC
""").df()
print(df.to_string(index=False))

# Dominant retailer
print("\nDominant retailer:")
df = con.execute("""
SELECT dominant_retailer, COUNT(*) AS n
FROM customer_snapshot GROUP BY 1 ORDER BY 2 DESC
""").df()
print(df.to_string(index=False))

# Status
print("\nStatus (ACTIVO/INACTIVO):")
df = con.execute("""
SELECT status, COUNT(*) AS n,
  ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) AS pct
FROM customer_snapshot GROUP BY 1
""").df()
print(df.to_string(index=False))

# Sample de filas
print("\nMuestra de 10 filas:")
df = con.execute("""
SELECT cust_id, t0, tier, funnel_state_at_t0, status,
  frequency_total, monetary_total, stock_points_at_t0,
  redeem_count_pre, dominant_retailer,
  ROUND(prob_to_next_state, 3) AS p_next,
  ROUND(prob_to_fuga, 3) AS p_fuga,
  has_redeemed_before_t0, n_canjes_post, y
FROM customer_snapshot
WHERE cust_id IN ('C0081','C0100','C0350','C0450','C0650','C0900','C0950')
  AND t0 = DATE '2024-01-01'
ORDER BY cust_id
""").df()
print(df.to_string(index=False))

# NULLs check
print("\nColumnas con NULLs (en snapshot):")
df = con.execute("""
SELECT
  COUNT(*) - COUNT(tier) AS nulls_tier,
  COUNT(*) - COUNT(tenure_months) AS nulls_tenure,
  COUNT(*) - COUNT(funnel_state_at_t0) AS nulls_funnel,
  COUNT(*) - COUNT(y) AS nulls_y,
  COUNT(*) - COUNT(dominant_retailer) AS nulls_retailer,
  COUNT(*) - COUNT(status) AS nulls_status
FROM customer_snapshot
""").df()
print(df.to_string(index=False))

print("\n" + "=" * 70)
print("FASE 1 COMPLETA - PIPELINE END-TO-END OK")
print("=" * 70)

con.close()
