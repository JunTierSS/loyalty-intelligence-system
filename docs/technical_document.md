# Loyalty Intelligence System
## Documento Tecnico Completo

**Version:** 1.0
**Fecha:** 28 de Marzo de 2026
**Confidencialidad:** Uso interno — Loyalty Points / StoreA
**Estado:** Listo para revision de gerencia

---

## 1. Resumen Ejecutivo

El **Loyalty Intelligence System** es un sistema de machine learning que predice el comportamiento de canje de los 12M+ clientes de Loyalty Points y genera recomendaciones personalizadas para maximizar la conversion y el revenue incremental.

### Cifras clave

| Metrica | Valor |
|---|---|
| Clientes en scope | 12M+ (500K muestra de entrenamiento) |
| Features calculadas | 74 (11 grupos A-K) |
| Snapshots temporales | 27 (ene-2023 a mar-2025) |
| Modelos ML | 4 en cascada (canje, retailer, monto, revenue) |
| Clusters conductuales | 6-8 arquetipos |
| Dashboard | 7 vistas interactivas |
| Frecuencia scoring | Mensual automatizado |

### Componentes del sistema

```
Fase 1: Data Foundation     → Query SQL monolitica, 74 features, target y=0/1/2
Fase 2: Modelamiento        → XGBoost cascada 4 pasos + KMeans clustering + Uplift
Fase 3: Productizacion      → Dashboard Streamlit + Scoring pipeline + API FastAPI
Fase 4: Feedback Loop       → A/B testing + tracking + drift detection + reentrenamiento
```

### Estado actual

- Implementacion completa sobre **datos sinteticos (mock)**
- Todas las fases funcionales y probadas end-to-end
- **Pendiente:** Conexion a BigQuery real y validacion con datos de produccion

---

## 2. Contexto de Negocio

### Ecosistema Loyalty Points

Loyalty Points es el programa de fidelizacion de un retail conglomerate, con presencia en 5 retailers:

| Retailer | Tipo | Relevancia |
|---|---|---|
| StoreA | Tienda por departamento | Principal generador de puntos |
| StoreB | Mejoramiento del hogar | Segunda fuente |
| StoreC | Supermercado | Alta frecuencia, bajo ticket |
| StoreD | E-commerce | Canal digital creciente |
| STOREE | Hogar y decoracion | Incorporacion reciente |

### Problema de negocio

1. **Breakage alto**: Porcentaje significativo de clientes nunca canjea sus puntos
2. **Sin personalizacion**: Las campanas son masivas, no individualizadas
3. **Sin medicion de incrementalidad**: No se sabe si las acciones generan valor real vs base
4. **Fuga silenciosa**: Clientes se van sin ser detectados a tiempo

### Objetivo del sistema

Responder 4 preguntas por cada cliente, cada mes:

1. **¿Canjea?** → Probabilidad de canje en los proximos 12 meses
2. **¿Donde?** → Retailer mas probable para el canje
3. **¿Cuanto?** → Monto estimado de puntos a canjear
4. **¿Revenue?** → Gasto total estimado en los proximos 12 meses

Y traducir las respuestas en **acciones concretas**: que oferta, por que canal, con que urgencia.

---

## 3. Arquitectura Tecnica

### Pipeline de produccion

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   BigQuery   │ →  │   Features   │ →  │   Modelos    │ →  │   Scoring    │ →  │  Dashboard   │
│   (ETL)      │    │  (SQL CTEs)  │    │  (Python)    │    │  (chunks)    │    │  (Streamlit) │
└─────────────┘    └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

### Infraestructura

| Componente | Tecnologia | Ubicacion |
|---|---|---|
| Almacenamiento | BigQuery | Proyecto `my-gcp-project` |
| Feature engineering | SQL (query monolitica con CTEs) | GitLab repo |
| Modelos | Python (XGBoost, KMeans, Logistic) | Colab / GitLab CI |
| Scoring | Python (chunks de 1M) | GitLab CI scheduled |
| Dashboard | Streamlit | Cloud Run |
| API | FastAPI | Cloud Run |
| Orquestacion | GitLab CI/CD | Schedule mensual |

### Datasets BigQuery

| Dataset | Contenido |
|---|---|
| `raw_data` | Vistas a tablas fuente |
| `operations` | KPIs, funnel, incrementalidad |
| `funnel_clientes` | Funnel diario de clientes |
| `acc_loy_cl_prd` | Comportamiento de clientes |
| `loyalty_analytics` | **NUEVO** — tablas de salida del sistema |

### Tablas fuente principales

| Tabla | Descripcion | Campos clave |
|---|---|---|
| `svw_clients_entity` | Clientes con tier, puntos, flags | cust_id, tier, stock_points_amt, cust_active_card_flg |
| `frozen_transaction_entity` | Transacciones historicas | cust_id, tran_date, tran_amt, retailer, points_earned |
| `frozen_redemption_entity` | Canjes historicos | cust_id, redemption_date, points_redeemed, channel |

---

## 4. Fase 1: Data Foundation

### 4.1 Query 01 — Exclusion de Outliers

**Objetivo:** Eliminar clientes que distorsionarian los modelos.

**4 categorias de exclusion con prioridad:**

| # | Categoria | Condicion | Supuesto |
|---|---|---|---|
| 1 | COLABORADOR | Lista explicita de empleados | Empleados tienen patrones atipicos (descuentos internos) |
| 2 | FRAUDE_DEVOLUCIONES | tasa_devolucion > 50% AND total_canjes >= 5 | Canjes sistematicos con devolucion son fraude |
| 3 | MONTO_EXTREMO | ticket_promedio > percentil 99.9 | Outliers extremos de monto son corporativos no identificados |
| 4 | FANTASMA | 0 transacciones validas | Cuentas inactivas sin utilidad predictiva |

**Supuestos:**
- Cada cliente recibe **una sola exclusion** (la de mayor prioridad)
- La exclusion se aplica una vez y persiste para todos los snapshots
- Los umbrales (50% devolucion, p99.9) fueron validados con el equipo de negocio

### 4.2 Query 02 — Muestra Estratificada 500K

**Objetivo:** Seleccionar una muestra fija y representativa para entrenamiento.

**Logica:**
```
Pool elegible = Clientes activos - Excluidos
                AND tier IN ('NORMAL','FAN','PREMIUM','ELITE')
                AND enrollment_date < 2023-01-01
```

**Estratificacion:** Proporcional por `tier × has_redeemed`

| Estrato | Ejemplo distribucion |
|---|---|
| NORMAL × no canjeo | ~55% |
| NORMAL × si canjeo | ~20% |
| FAN × no canjeo | ~10% |
| FAN × si canjeo | ~5% |
| PREMIUM/ELITE | ~10% |

**Supuestos:**
- 500K es suficiente para representatividad estadistica (de 12M)
- La estratificacion por tier × canje preserva la proporcion real
- El enrollment cutoff (< 2023-01-01) garantiza al menos 12 meses de historia pre-t0
- **Los mismos 500K clientes aparecen en los 27 snapshots** para trazabilidad temporal
- Se usa `FARM_FINGERPRINT` para seleccion deterministica y reproducible

### 4.3 Query 03 — Funnel Markov (6 estados)

**Objetivo:** Clasificar a cada cliente en un estado del funnel de canje cada mes.

**6 estados bidireccionales:**

| Estado | Criterio | Supuesto |
|---|---|---|
| INSCRITO | Sin transacciones | Estado inicial tras enrollment |
| PARTICIPANTE | Tiene transacciones, 0 canjes | Compra pero no canjea |
| POSIBILIDAD_CANJE | >= 1,000 puntos + 0 canjes | Tiene capacidad pero no canjea |
| CANJEADOR | Exactamente 1 canje historico | Primer canje realizado |
| RECURRENTE | >= 2 canjes historicos | Patron de canje establecido |
| FUGA | Inactivo segun tier | Riesgo de abandono |

**Regla de FUGA diferenciada por tier:**

| Tier | Umbral de inactividad | Supuesto |
|---|---|---|
| NORMAL / FAN | >= 365 dias sin canje | Ciclo natural mas corto |
| PREMIUM / ELITE | >= 730 dias sin canje | Mayor tolerancia por engagement historico |

**Supuestos:**
- Los estados son **mutuamente excluyentes** y se evaluan en orden de prioridad (FUGA primero)
- FUGA se evalua sobre ultimo canje, no ultima compra
- El umbral de 1,000 puntos para POSIBILIDAD_CANJE viene de la regla de negocio de canje minimo
- Un cliente puede **retroceder** en el funnel (bidireccional) — ej: RECURRENTE → FUGA
- La **matriz de transicion** se calcula mensualmente para el modelo Markov

### 4.4 Query 04 — Customer Snapshot (74 features)

**Objetivo:** Calcular todas las features para cada cliente en cada t0.

#### Diseno de snapshots temporales

```
ENE 2022          ENE 2023                              MAR 2025          MAR 2026
   |<--- pre 12m --->|<------- 27 t0s de entrenamiento ------->|<--- post 12m --->|
```

| Parametro | Valor | Supuesto |
|---|---|---|
| Periodo pre (features) | 12 meses fijo | Un ciclo anual completo captura estacionalidad |
| Periodo post (target) | 12 meses fijo | Ventana suficiente para observar comportamiento |
| Total snapshots | 27 (ene-2023 a mar-2025) | Cubren todos los meses al menos 2 veces |
| Split temporal | Train 21, Val 3, Test 3 | Simula deployment real (nunca se ve futuro) |

#### Prevencion de leakage

| Regla | Implementacion | Supuesto |
|---|---|---|
| Features solo pre-t0 | `WHERE fecha < t0` en todas las CTEs | Simula informacion disponible al momento de scoring |
| Stock points en t0 | `partition_date = t0` de svw_clients_entity | Foto exacta al momento |
| Tier en t0 | `partition_date = t0` | Tier vigente, no futuro |
| Target solo post-t0 | `WHERE fecha >= t0 AND fecha < t0 + 12m` | Ventana de observacion fija |

#### Variable target (y)

| y | Significado | Condicion |
|---|---|---|
| 0 | No canjea | 0 canjes en 12 meses post-t0 |
| 1 | Primer canje | Canjea en post Y nunca canjeo antes de t0 |
| 2 | Recurrencia | Canjea en post Y ya habia canjeado antes |

**Supuesto critico:** La distincion entre y=1 (activacion) y y=2 (recurrencia) tiene alto valor de negocio porque requieren estrategias diferentes.

#### Features por grupo (74 total)

##### Grupo A: RFM (6 features)

| Feature | Formula | Supuesto |
|---|---|---|
| recency_days | DATEDIFF(t0, ultima_compra) | DEFAULT 999 si nunca compro |
| frequency_total | COUNT(transacciones pre-t0) | Solo transacciones validas (monto > 0) |
| frequency_monthly_avg | frequency_total / meses_activo | Normalizada por tiempo |
| monetary_total | SUM(tran_amt) pre-t0 | En CLP, sin ajuste inflacion |
| monetary_avg_ticket | monetary_total / frequency_total | Ticket promedio historico |
| monetary_monthly_avg | monetary_total / meses_activo | Normalizada por tiempo |

**Supuesto:** No se ajusta por inflacion. En produccion real, considerar deflactar por IPC si los periodos son > 2 anios.

##### Grupo B: Puntos y Canje (10 features)

| Feature | Formula | Supuesto |
|---|---|---|
| points_earned_total | SUM(points_earned) pre-t0 | Acumulacion lifetime |
| points_earned_monthly_avg | points_earned_total / meses | Velocidad de acumulacion |
| stock_points_at_t0 | Snapshot de svw_clients_entity en t0 | Foto exacta, no calculada |
| exp_points_current_at_t0 | Puntos que vencen este mes | De partition_date = t0 |
| exp_points_next_at_t0 | Puntos que vencen proximo mes | Ventana corta de urgencia |
| has_redeemed_before_t0 | BOOL: >= 1 canje antes de t0 | Feature (no target) |
| redeem_count_pre | COUNT(canjes) pre-t0 | Total historico |
| redeem_points_total_pre | SUM(puntos canjeados) pre-t0 | Volumen de canje |
| redeem_count_12m_pre | COUNT(canjes) en 12m pre-t0 | Actividad reciente |
| redeem_rate | puntos_canjeados / puntos_ganados | Proxy de engagement, capped [0,1] |

**Supuesto:** `redeem_rate` se capea en 1.0 porque en edge cases los puntos canjeados pueden exceder los ganados (por transferencias o ajustes).

##### Grupo C: Dinamicas (7 features)

| Feature | Formula | Supuesto |
|---|---|---|
| earn_velocity_30 | Puntos ganados en ultimos 30 dias | Momentum a corto plazo |
| earn_velocity_90 | Puntos ganados en ultimos 90 dias | Tendencia trimestral |
| redeem_velocity_30 | Puntos canjeados en ultimos 30 dias | Actividad de canje reciente |
| earn_acceleration | earn_velocity_30 / earn_velocity_90 | > 1 = acelerando, < 1 = desacelerando |
| spend_trend | gasto_3m / gasto_prev_3m | Tendencia de gasto |
| days_since_last_activity | MIN(days_since_purchase, days_since_redeem) | Recencia general |
| days_since_last_redeem | DATEDIFF(t0, ultimo_canje) | DEFAULT 999 si nunca canjeo |

**Supuesto:** `earn_acceleration` es la feature mas novedosa respecto al modelo existente de propension. Captura momentum que features estaticas no capturan.

##### Grupo D: Capacidad de Canje (4 features)

| Feature | Formula | Supuesto |
|---|---|---|
| redeem_capacity | BOOL: stock_points >= 1000 | Umbral minimo de canje STORE_CARD |
| points_above_threshold | MAX(stock_points - 1000, 0) | Puntos "disponibles" para canje |
| days_to_redeem_capacity | Dias estimados para llegar a 1000 pts | Basado en velocidad de earn |
| points_pressure | exp_points_current / stock_points | Urgencia: que % del stock vence |

**Supuesto:** El umbral de 1,000 puntos es el minimo de canje definido por Loyalty Points. Si cambia, se debe actualizar.

##### Grupo E: Por Retailer (11 features)

| Feature | Formula | Supuesto |
|---|---|---|
| spend_store_a/sodimac/tottus/fcom/ikea | SUM(tran_amt) por retailer | 5 features de gasto |
| freq_store_a/sodimac/tottus/fcom/ikea | COUNT(txns) por retailer | 5 features de frecuencia |
| retailer_count | COUNT(DISTINCT retailer) | Diversificacion |
| dominant_retailer | Retailer con mayor gasto | Para recomendacion de oferta |
| retailer_entropy | Shannon entropy sobre frecuencias | H = -SUM(p * log(p)), frecuency-based |

**Supuesto:** STOREE fue incorporado como quinto retailer (antes eran 4). El modelo debe manejar valores 0 para STOREE en clientes historicos.

##### Grupo F: Medio de Pago (4 features)

| Feature | Formula | Supuesto |
|---|---|---|
| pct_store_card_payments | % transacciones con tarjeta STORE_CARD | Indica vinculacion con STORE_CARD |
| pct_debit_payments | % transacciones con debito | Canal de pago alternativo |
| purchase_channel_pref | Canal preferido (fisico/digital) | Para recomendacion de canal |
| pct_digital | % compras en canal digital | Proxy de afinidad digital |

##### Grupo G: Detalle de Canje (7 features)

| Feature | Formula | Supuesto |
|---|---|---|
| pct_redeem_catalogo | % canjes en catalogo | Tipo de canje preferido |
| pct_redeem_giftcard | % canjes en giftcard | Tipo de canje preferido |
| pct_redeem_digital | % canjes en canal digital | Afinidad digital en canje |
| dominant_redeem_type | Tipo de canje mas frecuente | Para recomendacion |
| redeem_channel_pref | Canal de canje preferido | Para recomendacion |
| avg_redeem_points | Promedio de puntos por canje | Ticket de canje |
| avg_redeem_amount | Monto promedio canjeado | Valor monetario del canje |

**Supuesto:** Si un cliente nunca canjeo, todos los porcentajes son 0 y el dominante es "NINGUNO".

##### Grupo H: Funnel y Markov (6 features)

| Feature | Formula | Supuesto |
|---|---|---|
| funnel_state_at_t0 | Estado del funnel al cierre de t0 | Input al modelo (no target) |
| days_in_current_state | Meses en estado actual × 30 | Estancamiento |
| transitions_last_12m | Cambios de estado en 12 meses | Movilidad en funnel |
| velocity_in_funnel | transitions / tiempo | Velocidad de progresion |
| prob_to_next_state | P(transicion a estado activo) | De la matriz Markov |
| prob_to_fuga | P(transicion a FUGA) | Riesgo de abandono |

**Supuesto:** Las probabilidades Markov se calculan sobre la matriz de transicion mensual global (no por cliente individual).

##### Grupo I: Demograficas (9 features)

| Feature | Formula | Supuesto |
|---|---|---|
| tier | NORMAL/FAN/PREMIUM/ELITE | Vigente en t0 |
| age | Edad al momento de t0 | De cust_age_num |
| gender | Genero | De cust_gender_desc |
| city | Ciudad (region) | Proxy geografico |
| tenure_months | Meses desde enrollment hasta t0 | Antiguedad |
| cust_active_card_flg | Tiene STORE_CARD credito activo | FLAG STRING Y/N |
| cust_active_deb_flg | Tiene debito activo | FLAG STRING Y/N |
| cust_active_omp_flg | Tiene otro medio de pago | FLAG STRING Y/N |
| status | ACTIVO/INACTIVO | Actividad en ultimos 365 dias |

**Supuesto:** Los flags son STRING ('Y'/'N') en BigQuery, no BOOL. Se convierten en binario en el modelo.

##### Grupo J: Avanzadas (6 features)

| Feature | Formula | Supuesto |
|---|---|---|
| ratio_earn_redeem | points_earned / points_redeemed | Balance acumulacion/uso |
| ticket_trend | ticket_3m / ticket_prev_3m | Tendencia de ticket |
| burstiness | Variabilidad inter-compra | Irregularidad temporal |
| spend_variability | STD(tran_amt) / AVG(tran_amt) | Coeficiente de variacion |
| campaign_response_rate | Tasa historica de respuesta | Sensibilidad a campanas |
| breakage | 1 - redeem_rate | Puntos no usados / ganados |

**Nota:** `engagement_score` fue removido del spec original por riesgo de leakage (normalizacion min-max sobre datos completos). Las 6 features restantes no tienen este problema.

##### Grupo K: Estacionalidad (4 features)

| Feature | Formula | Supuesto |
|---|---|---|
| month_of_t0 | Mes del snapshot | Captura estacionalidad |
| is_cyber_month | Noviembre (Major Sale Event) | Pico de gasto |
| is_holiday_month | Diciembre/Enero | Navidad + vacaciones |
| seasonal_spend_ratio | Gasto en cyber months / gasto total | Sensibilidad a eventos |

**Supuesto:** Major Sale Event  es en noviembre. Si cambia, actualizar la feature.

---

## 5. Fase 2: Modelamiento

### 5.1 Modelo Cascada — 4 Pasos

```
PASO 1: ¿Canjea? → XGBoost multiclase → P(y=0), P(y=1), P(y=2)
    |
    +-- P(canje) = P(y=1) + P(y=2)
    |
PASO 2: ¿Donde? → XGBoost multiclase → P(STOREA), P(STOREB), P(STOREC), P(STORED), P(STOREE)
    |
PASO 3: ¿Cuanto? → XGBoost regresion → Monto estimado en puntos
    |
PASO 4: ¿Revenue? → XGBoost regresion → Revenue estimado $CLP (12 meses)
```

**Supuestos del modelado:**

| Supuesto | Justificacion |
|---|---|
| XGBoost como algoritmo | Mejor balance precision/interpretabilidad para datos tabulares |
| Split temporal (no random) | Simula deployment real, previene leakage temporal |
| Optuna para hyperparameters | Optimizacion bayesiana sobre F1-macro en validation |
| SHAP para explicabilidad | Permite explicar predicciones individuales a negocio |
| Paso 1 es multiclase (3 clases) | Distinguir activacion (y=1) de recurrencia (y=2) tiene valor de negocio distinto |

### 5.2 Clustering — Segmentacion Conductual

**8 features de clustering:**

```
frequency_monthly_avg, monetary_monthly_avg, redeem_rate, retailer_entropy,
pct_redeem_digital, earn_velocity_90, days_since_last_activity, points_pressure
```

**Metodo:** KMeans con centroides fijos (entrenados en primer t0, aplicados a todos)

**Arquetipos (6-8 clusters):**

| Arquetipo | Perfil | Accion recomendada |
|---|---|---|
| Heavy Users | Alta frecuencia + alto gasto | Experiencia premium exclusiva |
| Cazadores de Canje | Alta tasa canje + digital | Descuento personalizado |
| Exploradores | Alta diversidad retailer | Educar sobre beneficios |
| Dormidos | Alta inactividad | Oferta directa de reactivacion |
| Digitales | Alto % canje digital | Oferta exclusiva canal digital |
| En Riesgo | Inactividad + puntos por vencer | Retencion preventiva |

**Supuestos:**
- 8 features conductuales (no las 74 del modelo predictivo) para interpretabilidad
- StandardScaler fit SOLO en training, transform en todo (consistencia temporal)
- K optimo determinado por silhouette score (rango 4-8)
- Nombres asignados automaticamente por Hungarian Assignment
- Centroides fijos permiten comparar clusters entre periodos

### 5.3 Incrementalidad — Uplift Modeling

**Framework:**

| Componente | Metodo | Proposito |
|---|---|---|
| Propensity Score | Logistic Regression | Balancear tratamiento vs control |
| PSM | Nearest Neighbor matching | Estimar ATE (efecto promedio) |
| T-Learner | 2 XGBoost separados | Estimar CATE individual (uplift_x) |
| Expected Value | P(canje) x uplift_x | Priorizar por valor esperado |

**Formula de Expected Value:**
```
EV = P(canje) × uplift_x
donde:
  P(canje) = probabilidad de canje del modelo propensity
  uplift_x = E[Revenue|tratado] - E[Revenue|no tratado] (del T-Learner)
```

**13 features para propensity/uplift:**
```
frequency_monthly_avg, monetary_monthly_avg, redeem_rate, retailer_entropy,
pct_redeem_digital, earn_velocity_90, days_since_last_activity, points_pressure,
stock_points_at_t0, redeem_count_pre, frequency_total, monetary_total, tenure_months
```

**Supuestos:**
- El tratamiento se define como "canjeo en el post-period" (observacional, no experimental)
- El T-Learner asume que las diferencias entre tratados/no tratados capturan el efecto causal
- Limitacion: sin datos de campanas reales, el uplift es una estimacion conservadora
- En produccion, el feedback loop con A/B testing reemplazara esta estimacion

### 5.4 Decision Engine — 5 Pasos

| Paso | Input | Output | Logica |
|---|---|---|---|
| 1. Prioridad | uplift_x | Alta/Media/Baja/No contactar | Quantiles globales Q60/Q80 del training |
| 2. Objetivo | funnel_state_at_t0 | Activar/Acelerar/Empujar/Recurrencia/Retener/Reactivar | Mapping directo del funnel |
| 3. Accion | cluster_name | Oferta especifica por arquetipo | Override: FUGA → urgente, INSCRITO → activar |
| 4. Canal | pct_redeem_digital + propensity | Digital/Email/Presencial/Push | Umbrales: >0.5 digital, >0.3 email, >0.15 presencial |
| 5. Timing | multiple signals | Normal/Urgente/Inmediato/Pronto | FUGA=urgente, puntos_vencer=urgente, prop>0.4=inmediato |

**Supuestos:**
- Los quantiles de prioridad se calculan **una vez en entrenamiento** (globales), no por chunk
- Los umbrales de canal (0.5, 0.3, 0.15) son iniciales y se ajustan via feedback loop
- Override de FUGA tiene prioridad sobre regla de cluster

---

## 6. Fase 3: Productizacion

### 6.1 Dashboard Streamlit (7 vistas)

| Vista | Contenido | Audiencia |
|---|---|---|
| 1. KPIs Ejecutivos | Tasas de canje, revenue, breakage, filtros | Gerencia |
| 2. Funnel Markov | 6 estados, transiciones, cuellos de botella | Estrategia |
| 3. Segmentacion | Clusters, migracion, KPIs por segmento | Marketing |
| 4. Customer 360 | Perfil individual + predicciones + recomendacion | CRM |
| 5. Simulador | Escenarios de campana, ROI | Planificacion |
| 6. Que paso? | Predicho vs real, alertas de performance | Data team |
| 7. Exports | CSV/Excel de listas accionables | Ejecucion |

### 6.2 Scoring Pipeline

- **Frecuencia:** Mensual (dia 2 de cada mes, automatizado via GitLab CI)
- **Volumen:** 500K clientes (muestra) o 12M (full) en chunks de 1M
- **Modelos:** KMeans clustering + Logistic propensity + XGBoost T-Learner uplift
- **Output:** 12 columnas por cliente (cust_id, prioridad, EV, propensity, uplift, objetivo, accion, retailer, canal, timing, cluster, funnel_state)
- **Destino:** BigQuery `loyalty_analytics.scoring_output` + CSV backup

### 6.3 API FastAPI (6 endpoints)

| Endpoint | Metodo | Uso |
|---|---|---|
| /health | GET | Status check |
| /score/{cust_id} | GET | Score individual (P(canje), uplift, EV) |
| /score/batch | POST | Score batch (hasta 1000 clientes) |
| /recommend/{cust_id} | GET | Recomendacion completa del decision engine |
| /segment/{cust_id} | GET | Cluster + perfil de features |
| /stats | GET | Estadisticas resumen |

---

## 7. Fase 4: Feedback Loop

### 7.1 Tracking Table

Tabla central `loyalty_analytics.action_tracking` (39 columnas):

| Seccion | Columnas | Descripcion |
|---|---|---|
| Identificadores | 3 | tracking_id, scoring_month, cust_id |
| Recomendacion | 11 | Lo que el modelo recomendo |
| Ejecucion | 6 | Lo que marketing ejecuto |
| Control | 1 | Flag de grupo holdout |
| Resultados 1m | 3 | Canjes y revenue a 1 mes |
| Resultados 3m | 4 | Canjes y revenue a 3 meses |
| Resultados 6m | 4 | Canjes y revenue a 6 meses |
| Resultados 12m | 5 | Validacion completa a 12 meses |
| Metadata | 2 | Timestamps |

### 7.2 A/B Testing

- **10% holdout** estratificado por prioridad
- **Hash deterministico** por cust_id (mismo grupo siempre)
- Grupo control **nunca se contacta**
- Comparacion: tasa de canje y revenue entre tratamiento vs control
- **Test estadistico:** Z-test de proporciones para significancia

### 7.3 Drift Detection

| Metrica | Umbral amarillo | Umbral rojo (reentrenar) |
|---|---|---|
| PSI features | > 0.1 | > 0.2 en 3+ features |
| F1 drop | > 3 puntos | > 7 puntos |
| Tasa canje diff | > 10% | > 25% |
| Lift top 10% | < 2.5x | < 2x |

### 7.4 Triggers de Reentrenamiento

1. PSI > 0.2 en 3+ features → **REENTRENAR**
2. Lift no significativo (p > 0.05) → **REENTRENAR**
3. 12+ meses sin reentrenar → **PLANIFICAR**
4. Tasa canje real < predicha 25%+ → **RECALIBRAR**

---

## 8. Metricas de Validacion

### Pre-deployment (test set ene-mar 2025)

| Metrica | Minimo aceptable | Ideal |
|---|---|---|
| F1-macro (Paso 1) | > 0.70 | > 0.80 |
| AUC por clase | > 0.80 | > 0.90 |
| Recall por clase | > 0.60 | > 0.75 |
| Precision por clase | > 0.55 | > 0.70 |
| Brier score | < 0.18 | < 0.10 |
| Lift top 10% | > 3x | > 5x |
| Accuracy retailer (Paso 2) | > 0.60 | > 0.75 |
| R2 monto (Paso 3) | > 0.40 | > 0.60 |
| R2 revenue (Paso 4) | > 0.40 | > 0.60 |

### Validacion progresiva en produccion

| Tiempo post-scoring | Validacion |
|---|---|
| 1 mes | Lift temprano por decil |
| 3 meses | Tasa canje acumulada |
| 6 meses | Revenue parcial |
| 12 meses | Validacion completa |

---

## 9. Inconsistencias Encontradas y Corregidas

### Issue 1: Quantiles de prioridad por chunk — CORREGIDO

**Problema:** Los quantiles Q60/Q80 para asignar prioridad se calculaban por chunk, causando clasificaciones inconsistentes entre batches.

**Correccion:** Los quantiles ahora se calculan **una vez sobre datos de entrenamiento** y se guardan en `models.pkl`. Se aplican identicos a todos los chunks.

### Issue 2: Arquetipo "En Riesgo" nunca aparecia — CORREGIDO

**Problema:** Se definieron 6 arquetipos pero K=5, por lo que "En Riesgo" nunca se asignaba.

**Correccion:** El rango de K ahora es 4-8 (antes 4-6). Con mock data, K=8 y "En Riesgo" aparece correctamente.

### Issue 3: Sin versionado de modelos — CORREGIDO

**Problema:** `models.pkl` no tenia metadata (fecha, hash, metricas).

**Correccion:** Se agrega metadata al pickle y se genera `models_metadata.json` con:
```json
{
  "training_date": "2026-03-28T03:42:15",
  "n_training_rows": 22296,
  "k_clusters": 8,
  "silhouette_score": 0.3367,
  "n_features_propensity": 13,
  "version": "1.0"
}
```

### Issue 4: engagement_score removido — DOCUMENTADO

**Problema:** El spec original incluia `engagement_score` como feature avanzada.

**Resolucion:** Se removio porque su calculo requeria normalizacion min-max sobre datos completos (leakage). Las 6 features restantes del Grupo J no tienen este problema.

### Issue 5: Formula lift revenue definida — CORREGIDO

**Problema:** El spec decia "definicion exacta pendiente".

**Resolucion:** Se definio como:
```
EV = P(canje) × uplift_x
uplift_x = E[Revenue|tratado] - E[Revenue|no tratado]
```
Donde el T-Learner estima cada termino por separado con XGBoost.

### Issue 6: Feature validation al inicio del scoring — CORREGIDO

**Problema:** score_chunk no validaba que las features requeridas existieran en los datos.

**Correccion:** Se agrego validacion explicita al inicio de `score_chunk()`:
```python
missing = [f for f in required if f not in df_chunk.columns]
if missing: raise ValueError(f"Missing features: {missing}")
```

---

## 10. Plan de Deployment a Produccion

### Timeline (8 semanas)

| Semana | Actividad | Responsable |
|---|---|---|
| 1-2 | Crear dataset `loyalty_analytics` en BQ, adaptar SQL de features | Data Engineer |
| 3 | Ejecutar features SQL en BQ, validar output vs mock | Data Scientist |
| 4 | Reentrenar modelos con datos reales, evaluar metricas | Data Scientist |
| 5 | Configurar GitLab CI/CD (scoring + tracking + monitoring) | Data Engineer |
| 6 | Deploy dashboard a Cloud Run | DevOps |
| 7 | Primer scoring real + review manual de resultados | Equipo completo |
| 8 | Go-live + primer ciclo de feedback | Equipo completo |

### Recursos necesarios

| Recurso | Costo estimado mensual |
|---|---|
| BigQuery (queries + storage) | $500-1,000 USD |
| Cloud Run (dashboard + API) | $50-100 USD |
| GitLab CI runners | Incluido en plan existente |
| Colab Pro (reentrenamiento) | $12 USD |
| **Total infraestructura** | **~$600-1,200 USD/mes** |

---

## 11. Riesgos y Mitigaciones

| Riesgo | Probabilidad | Impacto | Mitigacion |
|---|---|---|---|
| Datos reales difieren del mock | Alta | Alto | Validar distribuciones antes de entrenar |
| Metricas no alcanzan minimos | Media | Alto | Iterar features y hyperparameters |
| Drift en datos post-deployment | Media | Medio | Monitoring PSI mensual + alertas |
| Leakage no detectado | Baja | Critico | Assertions explicitos + review |
| Resistencia de marketing a usar listas | Media | Alto | Dashboard facil de usar + capacitacion |
| Costo BQ excede presupuesto | Baja | Bajo | Queries optimizadas con particiones |

---

## 12. Solicitud de Aprobacion

### Se solicita aprobacion para:

1. **Crear dataset `loyalty_analytics` en BigQuery** y ejecutar las 4 queries de feature engineering sobre datos reales

2. **Asignar 8 semanas de equipo** (1 DS + 1 DE parcial) para validacion con datos reales y deployment

3. **Presupuesto de infraestructura** de ~$1,000 USD/mes para BigQuery + Cloud Run

### Criterios de exito

| Criterio | Umbral |
|---|---|
| F1-macro en datos reales | > 0.70 |
| Lift top 10% | > 3x |
| Dashboard operativo | Si |
| Primer scoring mensual completo | Semana 7 |
| Feedback loop activo | Semana 8 |

---

**Firma de aprobacion:**

```
Gerente: _________________________    Fecha: ___/___/______

Data Lead: _________________________    Fecha: ___/___/______
```
