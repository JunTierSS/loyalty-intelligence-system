"""
Loyalty Intelligence System — Dashboard Streamlit (6 Vistas)
Fase 3: Productizacion
Fuente: Ejecuta pipeline completo (Fase 1 mock + Fase 2 modelos)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys, os, warnings
warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Loyalty Intelligence System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Run full pipeline (cached) ─────────────────────────────────────────────
@st.cache_data(ttl=3600)
def run_pipeline():
    """Execute Phase 1 mock + Phase 2 models, return all outputs."""
    import nbformat

    # 1. Run Phase 2e notebook (it loads mock + runs clustering + uplift + decision engine)
    nb_path = os.path.join(BASE_DIR, "notebooks", "05_decision_engine.ipynb")
    nb = nbformat.read(nb_path, as_version=4)

    code_cells = []
    for cell in nb.cells:
        if cell.cell_type == "code":
            # Skip matplotlib show calls for headless
            src = cell.source.replace("plt.show()", "plt.close('all')")
            src = src.replace("plt.tight_layout()\nplt.show()", "plt.close('all')")
            code_cells.append(src)

    # Prepend matplotlib backend
    script = "import matplotlib\nmatplotlib.use('Agg')\n\n" + "\n\n".join(code_cells)

    # Execute in isolated namespace
    ns = {"__name__": "__main__"}
    old_dir = os.getcwd()
    os.chdir(os.path.join(BASE_DIR, "notebooks"))

    old_stdout = sys.stdout
    from io import StringIO
    sys.stdout = StringIO()
    try:
        exec(script, ns)
    finally:
        sys.stdout = old_stdout
        os.chdir(old_dir)

    # Extract key dataframes from namespace
    df = ns["df"]           # Full dataset with all features + model outputs
    df_output = ns["df_output"]  # Decision engine output table

    # Also run fase2a to get XGBoost predictions
    nb_2a_path = os.path.join(BASE_DIR, "notebooks", "01_eda_modelo.ipynb")
    nb_2a = nbformat.read(nb_2a_path, as_version=4)
    code_2a = ["import matplotlib\nmatplotlib.use('Agg')"]
    for cell in nb_2a.cells:
        if cell.cell_type == "code":
            src = cell.source.replace("plt.show()", "plt.close('all')")
            src = src.replace("plt.tight_layout()\nplt.show()", "plt.close('all')")
            code_2a.append(src)
    script_2a = "\n\n".join(code_2a)
    ns_2a = {"__name__": "__main__"}
    os.chdir(os.path.join(BASE_DIR, "notebooks"))
    sys.stdout = StringIO()
    try:
        exec(script_2a, ns_2a)
    finally:
        sys.stdout = old_stdout
        os.chdir(old_dir)

    # Extract XGBoost results
    xgb_results = {
        "model": ns_2a.get("best_model"),
        "feature_names": ns_2a.get("FEATURE_COLS", []),
        "y_test_pred_proba": ns_2a.get("y_test_pred_proba"),
        "y_test": ns_2a.get("y_test"),
        "shap_values": ns_2a.get("shap_values"),
        "shap_feature_names": ns_2a.get("FEATURE_COLS", []),
    }

    # Get SHAP importances
    if "shap_values" in ns_2a and ns_2a["shap_values"] is not None:
        sv = ns_2a["shap_values"]
        if hasattr(sv, "values"):
            imp = np.abs(sv.values).mean(axis=0)
            if imp.ndim > 1:
                imp = imp.mean(axis=1)
            feat_names = ns_2a.get("FEATURE_COLS", [f"f{i}" for i in range(len(imp))])
            shap_df = pd.DataFrame({"feature": feat_names[:len(imp)], "importance": imp}).sort_values("importance", ascending=False)
        else:
            shap_df = pd.DataFrame({"feature": ["N/A"], "importance": [0]})
    else:
        shap_df = pd.DataFrame({"feature": ["N/A"], "importance": [0]})

    return {
        "df": df,
        "df_output": df_output,
        "shap_df": shap_df,
        "xgb_results": xgb_results,
    }


# ── Load ───────────────────────────────────────────────────────────────────
with st.spinner("Ejecutando pipeline completo (Fase 1 + Fase 2)... ~3 min primera vez"):
    pipeline = run_pipeline()

df = pipeline["df"]
df_output = pipeline["df_output"]
shap_df = pipeline["shap_df"]

# Ensure t0 is datetime
df["t0"] = pd.to_datetime(df["t0"])
if "t0" in df_output.columns:
    df_output["t0"] = pd.to_datetime(df_output["t0"])

# ── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.title("Loyalty Intelligence")
st.sidebar.markdown("**Loyalty Intelligence System**")
st.sidebar.markdown("---")

vista = st.sidebar.radio(
    "Vista",
    [
        "1. KPIs Ejecutivos",
        "2. Funnel Markov",
        "3. Segmentacion",
        "4. Customer 360",
        "5. Simulador",
        "6. Que paso?",
        "7. Exports",
    ],
)

st.sidebar.markdown("---")
n_clients = df["cust_id"].nunique()
n_t0s = df["t0"].nunique()
latest_t0 = df["t0"].max()
st.sidebar.caption(f"Clientes: {n_clients:,} | Snapshots: {n_t0s}")
st.sidebar.caption(f"Ultimo t0: {latest_t0.strftime('%Y-%m')}")
st.sidebar.caption(f"Total filas: {len(df):,}")

# Filters
st.sidebar.markdown("---")
tier_filter = st.sidebar.multiselect(
    "Filtrar por Tier",
    options=sorted(df["tier"].unique()),
    default=sorted(df["tier"].unique()),
)
df_filtered = df[df["tier"].isin(tier_filter)]
df_out_filtered = df_output[df_output["cust_id"].isin(df_filtered["cust_id"])] if "cust_id" in df_output.columns else df_output


# ══════════════════════════════════════════════════════════════════════════
# VISTA 1: KPIs EJECUTIVOS
# ══════════════════════════════════════════════════════════════════════════
if vista == "1. KPIs Ejecutivos":
    st.title("KPIs Ejecutivos")

    dff = df_filtered.copy()

    # Monthly KPIs
    kpis = dff.groupby("t0").agg(
        clientes=("cust_id", "nunique"),
        freq_avg=("frequency_total", "mean"),
        monetary_avg=("monetary_total", "mean"),
        stock_pts_avg=("stock_points_at_t0", "mean"),
        redeem_count_avg=("redeem_count_pre", "mean") if "redeem_count_pre" in dff.columns else ("cust_id", "count"),
        tasa_canje=("y", lambda x: (x > 0).mean() * 100),
        pct_y1=("y", lambda x: (x == 1).mean() * 100),
        pct_y2=("y", lambda x: (x == 2).mean() * 100),
    ).reset_index()

    # Big numbers (latest)
    latest = kpis.iloc[-1]
    prev = kpis.iloc[-2] if len(kpis) > 1 else latest

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Clientes", f"{int(latest['clientes']):,}")
    c2.metric("Monetary Avg", f"${int(latest['monetary_avg']):,}",
              delta=f"${int(latest['monetary_avg'] - prev['monetary_avg']):+,}")
    c3.metric("Tasa Canje", f"{latest['tasa_canje']:.1f}%",
              delta=f"{latest['tasa_canje'] - prev['tasa_canje']:.1f}pp")
    c4.metric("Stock Pts Avg", f"{int(latest['stock_pts_avg']):,}",
              delta=f"{int(latest['stock_pts_avg'] - prev['stock_pts_avg']):+,}")
    c5.metric("% Activacion (y=1)", f"{latest['pct_y1']:.1f}%",
              delta=f"{latest['pct_y1'] - prev['pct_y1']:.1f}pp")

    st.markdown("---")

    # Probabilidad de canje (from model)
    st.subheader("Probabilidad de Canje (modelo XGBoost)")
    if "propensity_score" in dff.columns:
        latest_df = dff[dff["t0"] == latest_t0]

        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(
                latest_df, x="propensity_score", nbins=40,
                title=f"Distribucion P(canje) — {latest_t0.strftime('%Y-%m')}",
                labels={"propensity_score": "P(canje)", "count": "Clientes"},
                color_discrete_sequence=["#636EFA"],
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # P(canje) by tier
            tier_p = latest_df.groupby("tier")["propensity_score"].agg(["mean", "median", "std"]).round(3).reset_index()
            tier_p.columns = ["Tier", "P(canje) Media", "P(canje) Mediana", "Std"]
            fig = px.bar(tier_p, x="Tier", y="P(canje) Media",
                        title="P(canje) Promedio por Tier",
                        error_y="Std",
                        color="Tier")
            fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Columna propensity_score no disponible.")

    # Tasa canje temporal
    col3, col4 = st.columns(2)
    with col3:
        fig = px.line(kpis, x="t0", y="tasa_canje",
                      title="Tasa de Canje por Snapshot (%)",
                      labels={"tasa_canje": "%", "t0": ""})
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        fig = px.area(kpis, x="t0", y=["pct_y1", "pct_y2"],
                      title="% Activacion (y=1) vs Recurrencia (y=2)",
                      labels={"value": "%", "t0": "", "variable": "Target"})
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    # Breakage (puntos no canjeados / acumulados)
    st.subheader("Breakage y Presion de Puntos")
    col5, col6 = st.columns(2)
    with col5:
        if "points_pressure" in dff.columns:
            pp_by_t0 = dff.groupby("t0")["points_pressure"].mean().reset_index()
            fig = px.line(pp_by_t0, x="t0", y="points_pressure",
                         title="Points Pressure Promedio (temporal)",
                         labels={"points_pressure": "Pressure", "t0": ""})
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

    with col6:
        if "stock_points_at_t0" in dff.columns:
            sp_by_tier = dff[dff["t0"] == latest_t0].groupby("tier")["stock_points_at_t0"].agg(["mean", "sum"]).round(0).reset_index()
            sp_by_tier.columns = ["Tier", "Avg Stock", "Total Stock"]
            fig = px.bar(sp_by_tier, x="Tier", y="Total Stock",
                        title=f"Stock de Puntos Total por Tier — {latest_t0.strftime('%Y-%m')}",
                        color="Tier")
            fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    # Retailer
    st.subheader("Revenue por Retailer")
    if "dominant_retailer" in dff.columns:
        ret = dff[dff["t0"] == latest_t0].groupby("dominant_retailer").agg(
            clientes=("cust_id", "nunique"),
            monetary_avg=("monetary_total", "mean"),
        ).reset_index().sort_values("monetary_avg", ascending=False)
        fig = px.bar(ret, x="dominant_retailer", y="monetary_avg",
                    color="dominant_retailer",
                    title="Monetary Promedio por Retailer Dominante",
                    labels={"monetary_avg": "Monetary Avg (CLP)", "dominant_retailer": ""})
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # SHAP top features
    st.subheader("Top Features del Modelo (SHAP)")
    top_shap = shap_df.head(15)
    fig = px.bar(top_shap, x="importance", y="feature", orientation="h",
                title="Top 15 Features por SHAP Importance",
                labels={"importance": "Mean |SHAP|", "feature": ""})
    fig.update_layout(height=450, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# VISTA 2: FUNNEL MARKOV
# ══════════════════════════════════════════════════════════════════════════
elif vista == "2. Funnel Markov":
    st.title("Funnel Markov — 6 Estados")
    st.markdown("INSCRITO → PARTICIPANTE → POSIBILIDAD_CANJE → CANJEADOR → RECURRENTE | FUGA")

    dff = df_filtered.copy()
    funnel_col = "funnel_state_at_t0"

    # Funnel over time
    funnel_time = dff.groupby(["t0", funnel_col]).size().reset_index(name="n")
    state_order = ["INSCRITO", "PARTICIPANTE", "POSIBILIDAD_CANJE", "CANJEADOR", "RECURRENTE", "FUGA"]
    color_map = {
        "INSCRITO": "#636EFA", "PARTICIPANTE": "#EF553B",
        "POSIBILIDAD_CANJE": "#00CC96", "CANJEADOR": "#AB63FA",
        "RECURRENTE": "#FFA15A", "FUGA": "#FF6692",
    }

    fig = px.area(funnel_time, x="t0", y="n", color=funnel_col,
                  title="Distribucion de Estados del Funnel (temporal)",
                  labels={"n": "Clientes", "t0": ""},
                  color_discrete_map=color_map,
                  category_orders={funnel_col: state_order})
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

    # Latest snapshot
    latest_df = dff[dff["t0"] == latest_t0]
    funnel_latest = latest_df[funnel_col].value_counts().reset_index()
    funnel_latest.columns = ["Estado", "N"]
    funnel_latest["Estado"] = pd.Categorical(funnel_latest["Estado"], categories=state_order, ordered=True)
    funnel_latest = funnel_latest.sort_values("Estado")
    total = funnel_latest["N"].sum()
    funnel_latest["Pct"] = (funnel_latest["N"] / total * 100).round(1)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"Snapshot: {latest_t0.strftime('%Y-%m')}")
        fig = px.funnel(funnel_latest, x="N", y="Estado",
                       title="Funnel Actual", color="Estado",
                       color_discrete_map=color_map)
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Distribucion")
        fig = px.pie(funnel_latest, values="N", names="Estado",
                    color="Estado", color_discrete_map=color_map)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Transition probabilities (from Markov)
    st.subheader("Probabilidades de Transicion")
    if "prob_to_fuga" in dff.columns and "p_next" in dff.columns:
        markov = latest_df.groupby(funnel_col).agg(
            n=("cust_id", "nunique"),
            p_avance_avg=("p_next", "mean"),
            p_fuga_avg=("prob_to_fuga", "mean"),
            tasa_canje=("y", lambda x: (x > 0).mean() * 100),
        ).round(3).reset_index()
        markov.columns = ["Estado", "N Clientes", "P(avance)", "P(fuga)", "Tasa Canje Real %"]
        st.dataframe(markov, use_container_width=True)

    # Cuellos de botella
    st.subheader("Cuellos de Botella")
    st.markdown("""
    - **POSIBILIDAD_CANJE** = mayor cuello — clientes que compran pero no canjean
    - **FUGA** = creciente con el tiempo — clientes que abandonan
    - Estados con **P(avance) baja** y **alto volumen** necesitan intervencion
    """)

    # Growth by state
    first_t0 = dff["t0"].min()
    first_df = dff[dff["t0"] == first_t0]
    growth = []
    for state in state_order:
        n_first = len(first_df[first_df[funnel_col] == state])
        n_last = len(latest_df[latest_df[funnel_col] == state])
        pct = ((n_last - n_first) / max(n_first, 1)) * 100
        growth.append({"Estado": state, f"N ({first_t0.strftime('%Y-%m')})": n_first,
                       f"N ({latest_t0.strftime('%Y-%m')})": n_last, "Cambio %": round(pct, 1)})
    st.dataframe(pd.DataFrame(growth), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# VISTA 3: SEGMENTACION
# ══════════════════════════════════════════════════════════════════════════
elif vista == "3. Segmentacion":
    st.title("Segmentacion Conductual")

    dff = df_filtered.copy()

    if "cluster_name" not in dff.columns:
        st.warning("Clustering no disponible en los datos.")
    else:
        latest_df = dff[dff["t0"] == latest_t0]

        # Cluster summary
        CLUSTER_FEATURES = ["frequency_monthly_avg", "monetary_monthly_avg", "redeem_rate",
                           "retailer_entropy", "pct_redeem_digital", "earn_velocity_90",
                           "days_since_last_activity", "points_pressure"]
        available_feats = [f for f in CLUSTER_FEATURES if f in dff.columns]

        summary = latest_df.groupby("cluster_name").agg(
            n=("cust_id", "nunique"),
            **{f: (f, "mean") for f in available_feats},
            tasa_canje=("y", lambda x: (x > 0).mean() * 100),
            p_canje_avg=("propensity_score", "mean") if "propensity_score" in latest_df.columns else ("y", "mean"),
        ).round(3)
        summary["pct"] = (summary["n"] / summary["n"].sum() * 100).round(1)

        # Big numbers
        clusters = summary.index.tolist()
        cols = st.columns(len(clusters))
        for i, name in enumerate(clusters):
            row = summary.loc[name]
            with cols[i]:
                st.metric(name, f"{int(row['n']):,}", f"{row['pct']:.1f}%")
                st.caption(f"Tasa canje: {row['tasa_canje']:.1f}%")

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            fig = px.pie(summary.reset_index(), values="n", names="cluster_name",
                        title="Distribucion de Segmentos")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Radar chart
            if len(available_feats) >= 3:
                # Normalize profiles 0-1
                profiles = summary[available_feats]
                ranges = profiles.max() - profiles.min()
                pn = (profiles - profiles.min()) / ranges.where(ranges > 0, 1)
                pn = pn.fillna(0)

                short_names = [f.replace("_monthly_avg", "").replace("_at_t0", "")
                              .replace("days_since_last_", "d_") for f in available_feats]
                fig = go.Figure()
                for name in pn.index:
                    vals = pn.loc[name].tolist()
                    fig.add_trace(go.Scatterpolar(
                        r=vals + [vals[0]], theta=short_names + [short_names[0]], name=name
                    ))
                fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1.1])),
                                 title="Perfil por Segmento", height=400)
                st.plotly_chart(fig, use_container_width=True)

        # Profile table
        st.subheader("Perfiles Detallados")
        st.dataframe(summary, use_container_width=True)

        # Migration: cluster distribution over time
        st.subheader("Migracion Temporal de Clusters")
        clust_time = dff.groupby(["t0", "cluster_name"]).size().reset_index(name="n")
        clust_pct = clust_time.copy()
        totals = clust_pct.groupby("t0")["n"].transform("sum")
        clust_pct["pct"] = (clust_pct["n"] / totals * 100).round(1)

        fig = px.area(clust_pct, x="t0", y="pct", color="cluster_name",
                     title="Distribucion de Clusters por Snapshot (%)",
                     labels={"pct": "%", "t0": "", "cluster_name": "Cluster"})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Cluster x Tier heatmap
        st.subheader("Segmento x Tier")
        ct = pd.crosstab(latest_df["cluster_name"], latest_df["tier"], normalize="index").round(3) * 100
        fig = px.imshow(ct, text_auto=".1f", title="% de cada Tier por Segmento",
                       labels={"color": "%"}, aspect="auto")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

        # Cluster x Target
        st.subheader("Segmento x Target")
        ct2 = pd.crosstab(latest_df["cluster_name"], latest_df["y"], normalize="index").round(3) * 100
        ct2.columns = ["No canje (0)", "Activacion (1)", "Recurrencia (2)"]
        fig = px.imshow(ct2, text_auto=".1f", title="% de cada Target por Segmento",
                       labels={"color": "%"}, aspect="auto")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# VISTA 4: CUSTOMER 360
# ══════════════════════════════════════════════════════════════════════════
elif vista == "4. Customer 360":
    st.title("Customer 360")

    # Search
    all_ids = sorted(df["cust_id"].unique())
    search = st.selectbox("Seleccionar cliente", all_ids, index=0)

    client_df = df[df["cust_id"] == search].sort_values("t0")
    client_out = df_output[df_output["cust_id"] == search].sort_values("t0") if "cust_id" in df_output.columns else pd.DataFrame()

    if len(client_df) == 0:
        st.warning("Cliente no encontrado.")
    else:
        latest_row = client_df.iloc[-1]
        latest_out = client_out.iloc[-1] if len(client_out) > 0 else None

        # Header
        tier = latest_row.get("tier", "?")
        funnel = latest_row.get("funnel_state_at_t0", "?")
        status = latest_row.get("status", "?")
        st.markdown(f"### {search} — Tier: **{tier}** | Funnel: **{funnel}** | Status: **{status}**")

        # KPIs row
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Frecuencia", f"{int(latest_row.get('frequency_total', 0))}")
        c2.metric("Monetary", f"${int(latest_row.get('monetary_total', 0)):,}")
        c3.metric("Stock Puntos", f"{int(latest_row.get('stock_points_at_t0', 0)):,}")
        c4.metric("Canjes Pre", f"{int(latest_row.get('redeem_count_pre', 0))}")
        c5.metric("Days Since", f"{int(latest_row.get('days_since_last_activity', 999))}")
        pp = latest_row.get("points_pressure", 0)
        c6.metric("Points Pressure", f"{pp:.2f}")

        st.markdown("---")

        # Predictions
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Predicciones del Modelo")
            p_score = latest_row.get("propensity_score", None)
            uplift = latest_row.get("uplift_x", None)
            ev = latest_row.get("expected_value", None) if latest_out is not None else None

            if p_score is not None:
                st.write(f"- **P(canje):** {p_score:.3f}")
            if uplift is not None:
                st.write(f"- **Uplift (X-Learner):** ${uplift:,.0f}")
            if ev is not None:
                st.write(f"- **Expected Value:** ${ev:,.0f}")

            y_val = int(latest_row.get("y", 0))
            y_label = {0: "No canjea", 1: "Activacion (primer canje)", 2: "Recurrencia"}
            st.write(f"- **Target real (y):** {y_val} — {y_label.get(y_val, '?')}")

            if "cluster_name" in latest_row.index:
                st.write(f"- **Segmento:** {latest_row['cluster_name']}")

        with col_b:
            st.subheader("Recomendacion Decision Engine")
            if latest_out is not None:
                prioridad = latest_out.get("prioridad", "?")
                color_map = {"Alta": "red", "Media": "orange", "Baja": "blue", "No contactar": "gray"}
                st.markdown(f"**Prioridad:** :{color_map.get(prioridad, 'black')}[{prioridad}]")
                st.write(f"- **Objetivo:** {latest_out.get('objetivo', '?')}")
                st.write(f"- **Accion:** {latest_out.get('accion', '?')}")
                st.write(f"- **Retailer:** {latest_out.get('retailer_recomendado', '?')}")
                st.write(f"- **Canal:** {latest_out.get('canal', '?')}")
                st.write(f"- **Timing:** {latest_out.get('timing', '?')}")
                if "justificacion" in latest_out.index:
                    st.caption(f"Justificacion: {latest_out['justificacion']}")
            else:
                st.info("Decision engine output no disponible para este cliente.")

        # Historical evolution
        st.markdown("---")
        st.subheader("Evolucion Temporal")

        hist_cols = ["frequency_total", "monetary_total", "stock_points_at_t0", "redeem_count_pre"]
        available_hist = [c for c in hist_cols if c in client_df.columns]

        if available_hist:
            fig = make_subplots(rows=2, cols=2, subplot_titles=available_hist)
            for i, col in enumerate(available_hist[:4]):
                row, c = divmod(i, 2)
                fig.add_trace(
                    go.Scatter(x=client_df["t0"], y=client_df[col], mode="lines+markers", name=col),
                    row=row+1, col=c+1,
                )
            fig.update_layout(height=500, showlegend=False, title_text=f"Evolucion de {search}")
            st.plotly_chart(fig, use_container_width=True)

        # Funnel evolution
        if "funnel_state_at_t0" in client_df.columns:
            st.write("**Trayectoria en el Funnel:**")
            trajectory = client_df[["t0", "funnel_state_at_t0"]].copy()
            trajectory["t0"] = trajectory["t0"].dt.strftime("%Y-%m")
            st.dataframe(trajectory.set_index("t0").T, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# VISTA 5: SIMULADOR
# ══════════════════════════════════════════════════════════════════════════
elif vista == "5. Simulador":
    st.title("Simulador de Campana")
    st.markdown("Simula el impacto de una campana usando los datos reales del modelo.")

    dff = df_filtered[df_filtered["t0"] == latest_t0].copy()

    col1, col2 = st.columns(2)
    with col1:
        target_segment = st.selectbox("Segmento objetivo", [
            "Todos los contactables",
            "Solo prioridad Alta",
            "Alta + Media",
            "Dormidos",
            "Cazadores de Canje",
            "Heavy Users",
            "Exploradores",
            "Funnel: POSIBILIDAD_CANJE",
            "Funnel: FUGA",
        ])
        campaign_budget = st.number_input("Presupuesto (CLP)", value=10_000_000, step=1_000_000)
        cost_per_contact = st.number_input("Costo por contacto (CLP)", value=500, step=100)

    with col2:
        avg_redeem_value = st.number_input("Valor promedio por canje (CLP)", value=25_000, step=5000)
        uplift_mult = st.slider("Multiplicador de uplift", 0.5, 3.0, 1.0, 0.1,
                               help="1.0 = usar uplift del modelo directamente")

    # Filter segment
    if "prioridad" in dff.columns and "cluster_name" in dff.columns:
        if target_segment == "Solo prioridad Alta":
            df_seg = dff[dff["prioridad"] == "Alta"]
        elif target_segment == "Alta + Media":
            df_seg = dff[dff["prioridad"].isin(["Alta", "Media"])]
        elif target_segment == "Dormidos":
            df_seg = dff[dff["cluster_name"] == "Dormidos"]
        elif target_segment == "Cazadores de Canje":
            df_seg = dff[dff["cluster_name"] == "Cazadores de Canje"]
        elif target_segment == "Heavy Users":
            df_seg = dff[dff["cluster_name"] == "Heavy Users"]
        elif target_segment == "Exploradores":
            df_seg = dff[dff["cluster_name"] == "Exploradores"]
        elif target_segment == "Funnel: POSIBILIDAD_CANJE":
            df_seg = dff[dff["funnel_state_at_t0"] == "POSIBILIDAD_CANJE"]
        elif target_segment == "Funnel: FUGA":
            df_seg = dff[dff["funnel_state_at_t0"] == "FUGA"]
        else:
            df_seg = dff[dff["prioridad"] != "No contactar"]
    else:
        df_seg = dff

    n_target = len(df_seg)
    n_contactable = min(n_target, int(campaign_budget / max(cost_per_contact, 1)))

    # Use model P(canje) and uplift
    if "propensity_score" in df_seg.columns and "uplift_x" in df_seg.columns:
        # Sort by expected value (or uplift) and take top n_contactable
        df_seg = df_seg.sort_values("expected_value" if "expected_value" in df_seg.columns else "uplift_x",
                                    ascending=False).head(n_contactable)
        avg_p_canje = df_seg["propensity_score"].mean()
        avg_uplift = df_seg["uplift_x"].mean() * uplift_mult
        n_converted = int(n_contactable * avg_p_canje)
        revenue_incremental = avg_uplift * n_contactable
    else:
        avg_p_canje = 0.15
        avg_uplift = 5000
        n_converted = int(n_contactable * avg_p_canje)
        revenue_incremental = avg_uplift * n_contactable

    cost_total = n_contactable * cost_per_contact
    roi = (revenue_incremental - cost_total) / max(cost_total, 1) * 100

    st.markdown("---")
    st.subheader("Resultados")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Clientes Segmento", f"{n_target:,}")
    c2.metric("Contactables", f"{n_contactable:,}")
    c3.metric("P(canje) Promedio", f"{avg_p_canje:.1%}")
    c4.metric("Conversiones Esperadas", f"{n_converted:,}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Uplift Avg", f"${avg_uplift:,.0f}")
    c6.metric("Revenue Incremental", f"${revenue_incremental:,.0f}")
    c7.metric("Costo", f"${cost_total:,.0f}")
    c8.metric("ROI", f"{roi:.1f}%", delta="Positivo" if roi > 0 else "Negativo")

    # Sensitivity
    st.markdown("---")
    st.subheader("Sensibilidad: ROI por Multiplicador de Uplift")
    mults = np.arange(0.5, 3.1, 0.25)
    rois = []
    for m in mults:
        rev = avg_uplift / max(uplift_mult, 0.01) * m * n_contactable
        r = (rev - cost_total) / max(cost_total, 1) * 100
        rois.append({"Multiplicador": m, "ROI %": round(r, 1), "Revenue Incr.": round(rev)})
    df_sens = pd.DataFrame(rois)
    fig = px.line(df_sens, x="Multiplicador", y="ROI %", markers=True,
                 title="ROI vs Multiplicador de Uplift")
    fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Break-even")
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

    # Segment breakdown
    if "cluster_name" in dff.columns and "propensity_score" in dff.columns:
        st.subheader("P(canje) y Uplift por Segmento")
        seg_stats = dff.groupby("cluster_name").agg(
            n=("cust_id", "count"),
            p_canje=("propensity_score", "mean"),
            uplift_avg=("uplift_x", "mean") if "uplift_x" in dff.columns else ("propensity_score", "mean"),
            tasa_real=("y", lambda x: (x > 0).mean()),
        ).round(3).reset_index()
        st.dataframe(seg_stats, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# VISTA 6: QUE PASO?
# ══════════════════════════════════════════════════════════════════════════
elif vista == "6. Que paso?":
    st.title("Que paso? — Predicho vs Real")
    st.markdown("Monitoreo de performance y alertas de drift.")

    dff = df_filtered.copy()

    # Calculate metrics per t0
    metrics = []
    for t0, grp in dff.groupby("t0"):
        tasa_real = (grp["y"] > 0).mean() * 100
        row = {"t0": t0, "tasa_canje_real": round(tasa_real, 2), "n": len(grp)}

        if "propensity_score" in grp.columns:
            tasa_pred = grp["propensity_score"].mean() * 100
            row["tasa_canje_pred"] = round(tasa_pred, 2)

            # Calibration: predicted vs actual in top decile
            top10 = grp.nlargest(max(1, len(grp) // 10), "propensity_score")
            lift = (top10["y"] > 0).mean() / max((grp["y"] > 0).mean(), 0.001)
            row["lift_top10"] = round(lift, 2)

            # Brier score
            from sklearn.metrics import brier_score_loss
            y_binary = (grp["y"] > 0).astype(int)
            brier = brier_score_loss(y_binary, grp["propensity_score"])
            row["brier"] = round(brier, 4)
        else:
            row["tasa_canje_pred"] = tasa_real
            row["lift_top10"] = 1.0
            row["brier"] = 0.15

        metrics.append(row)

    df_metrics = pd.DataFrame(metrics)

    # Alerts
    if len(df_metrics) > 0:
        latest_m = df_metrics.iloc[-1]
        alerts = []

        diff = abs(latest_m.get("tasa_canje_pred", 0) - latest_m.get("tasa_canje_real", 0))
        if diff > 5:
            alerts.append(("Tasa canje: pred vs real diverge >5pp", "Recalibrar modelo", "warning"))
        if latest_m.get("lift_top10", 5) < 2:
            alerts.append(("Lift top 10% < 2x", "Reentrenar", "error"))
        elif latest_m.get("lift_top10", 5) < 3:
            alerts.append(("Lift top 10% < 3x", "Monitorear", "warning"))
        if latest_m.get("brier", 0) > 0.18:
            alerts.append(("Brier score > 0.18", "Probabilidades descalibradas", "error"))

        if alerts:
            for msg, action, level in alerts:
                if level == "error":
                    st.error(f"**{msg}** — {action}")
                else:
                    st.warning(f"**{msg}** — {action}")
        else:
            st.success("Sin alertas activas. Modelo operando correctamente.")

        # KPIs
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tasa Canje Real", f"{latest_m['tasa_canje_real']:.1f}%")
        c2.metric("Tasa Canje Pred", f"{latest_m.get('tasa_canje_pred', 0):.1f}%")
        c3.metric("Lift Top 10%", f"{latest_m.get('lift_top10', 0):.1f}x")
        c4.metric("Brier Score", f"{latest_m.get('brier', 0):.4f}")

    st.markdown("---")

    # Charts
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_metrics["t0"], y=df_metrics["tasa_canje_pred"],
                                  name="Predicha", mode="lines+markers"))
        fig.add_trace(go.Scatter(x=df_metrics["t0"], y=df_metrics["tasa_canje_real"],
                                  name="Real", mode="lines+markers"))
        fig.update_layout(title="Tasa de Canje: Predicha vs Real", height=350, yaxis_title="%")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_metrics["t0"], y=df_metrics["lift_top10"],
                                  name="Lift", mode="lines+markers", fill="tozeroy"))
        fig.add_hline(y=5, line_dash="dash", line_color="green", annotation_text="Objetivo")
        fig.add_hline(y=3, line_dash="dash", line_color="orange", annotation_text="Minimo")
        fig.update_layout(title="Lift Top 10% (temporal)", height=350, yaxis_title="Lift (x)")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig = px.line(df_metrics, x="t0", y="brier", markers=True,
                     title="Brier Score (calibracion)")
        fig.add_hline(y=0.10, line_dash="dash", line_color="green", annotation_text="Ideal")
        fig.add_hline(y=0.18, line_dash="dash", line_color="red", annotation_text="Limite")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        # Feature drift (PSI approximation using std shift)
        if len(dff) > 0:
            train_t0s = sorted(dff["t0"].unique())[:18]
            test_t0s = sorted(dff["t0"].unique())[18:]
            if len(test_t0s) > 0:
                num_cols = dff.select_dtypes(include=[np.number]).columns[:10]
                train_means = dff[dff["t0"].isin(train_t0s)][num_cols].mean()
                test_means = dff[dff["t0"].isin(test_t0s)][num_cols].mean()
                train_stds = dff[dff["t0"].isin(train_t0s)][num_cols].std()
                drift = ((test_means - train_means) / train_stds.replace(0, 1)).abs()
                drift_df = drift.reset_index()
                drift_df.columns = ["Feature", "Drift (std units)"]
                drift_df = drift_df.sort_values("Drift (std units)", ascending=True)
                drift_df["Color"] = drift_df["Drift (std units)"].apply(
                    lambda x: "red" if x > 0.5 else ("orange" if x > 0.25 else "green")
                )

                fig = go.Figure(go.Bar(
                    x=drift_df["Drift (std units)"], y=drift_df["Feature"],
                    orientation="h",
                    marker_color=drift_df["Color"],
                ))
                fig.add_vline(x=0.25, line_dash="dash", line_color="orange")
                fig.add_vline(x=0.5, line_dash="dash", line_color="red")
                fig.update_layout(title="Feature Drift (train vs test)", height=350)
                st.plotly_chart(fig, use_container_width=True)

    # Decision Engine effectiveness
    if "prioridad" in dff.columns:
        st.markdown("---")
        st.subheader("Efectividad del Decision Engine")

        prio_stats = dff[dff["t0"] == latest_t0].groupby("prioridad").agg(
            n=("cust_id", "count"),
            tasa_canje_real=("y", lambda x: (x > 0).mean() * 100),
            p_canje_avg=("propensity_score", "mean") if "propensity_score" in dff.columns else ("y", "mean"),
            ev_avg=("expected_value", "mean") if "expected_value" in dff.columns else ("y", "mean"),
        ).round(2).reset_index()

        col_a, col_b = st.columns(2)
        with col_a:
            st.dataframe(prio_stats, use_container_width=True)
        with col_b:
            fig = px.bar(prio_stats, x="prioridad", y="tasa_canje_real",
                        title="Tasa Canje Real por Prioridad",
                        color="prioridad")
            fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    # Thresholds
    with st.expander("Umbrales de Reentrenamiento"):
        st.markdown("""
        | Metrica | Amarilla | Roja (reentrenar) |
        |---|---|---|
        | F1 drop | > 3 pts | > 7 pts |
        | PSI features | > 0.1 | > 0.2 en 3+ features |
        | Tasa canje diff pred vs real | > 10% | > 25% |
        | Lift top 10% | < 2.5x | < 2x |
        | Brier score | > 0.15 | > 0.18 |
        """)


# ══════════════════════════════════════════════════════════════════════════
# VISTA 7: EXPORTS
# ══════════════════════════════════════════════════════════════════════════
elif vista == "7. Exports":
    st.title("Exports — Listas Priorizadas para Marketing")
    st.markdown(f"Snapshot: **{latest_t0.strftime('%Y-%m')}** | Clientes: **{n_clients:,}**")

    dff = df_filtered.copy()
    dfo = df_out_filtered.copy()

    # Use latest t0 only
    dff_latest = dff[dff["t0"] == latest_t0].copy()
    dfo_latest = dfo[dfo["t0"] == latest_t0].copy() if "t0" in dfo.columns else dfo.copy()

    def to_csv_bytes(dataframe):
        return dataframe.to_csv(index=False).encode("utf-8")

    def to_excel_bytes(dataframe):
        from io import BytesIO
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            dataframe.to_excel(writer, index=False, sheet_name="data")
        return buf.getvalue()

    st.markdown("---")

    # ── 1. Decision Engine completo ────────────────────────────────────
    st.subheader("1. Decision Engine — Tabla Completa")
    st.caption(f"{len(dfo_latest):,} filas × {dfo_latest.shape[1]} columnas")

    with st.expander("Preview (primeras 20 filas)"):
        st.dataframe(dfo_latest.head(20), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.download_button(
            "Descargar CSV — Decision Engine",
            data=to_csv_bytes(dfo_latest),
            file_name=f"decision_engine_{latest_t0.strftime('%Y%m')}.csv",
            mime="text/csv",
        )
    with col_b:
        st.download_button(
            "Descargar Excel — Decision Engine",
            data=to_excel_bytes(dfo_latest),
            file_name=f"decision_engine_{latest_t0.strftime('%Y%m')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    st.markdown("---")

    # ── 2. Top 50 por Expected Value ───────────────────────────────────
    st.subheader("2. Top 50 Clientes por Expected Value")
    if "expected_value" in dfo_latest.columns:
        top_ev = dfo_latest[dfo_latest["prioridad"].isin(["Alta", "Media"])].nlargest(50, "expected_value")
        st.caption(f"{len(top_ev)} clientes | EV rango: ${top_ev['expected_value'].min():,.0f} — ${top_ev['expected_value'].max():,.0f}")

        with st.expander("Preview"):
            display_cols = [c for c in ["cust_id", "prioridad", "expected_value", "propensity_score",
                           "uplift_x", "accion", "retailer_recomendado", "canal", "timing"] if c in top_ev.columns]
            st.dataframe(top_ev[display_cols], use_container_width=True)

        st.download_button(
            "Descargar CSV — Top 50 EV",
            data=to_csv_bytes(top_ev),
            file_name=f"top50_ev_{latest_t0.strftime('%Y%m')}.csv",
            mime="text/csv",
        )

    st.markdown("---")

    # ── 3. Lista por Retailer ──────────────────────────────────────────
    st.subheader("3. Top 20 por Retailer")
    if "retailer_recomendado" in dfo_latest.columns and "expected_value" in dfo_latest.columns:
        retailers = sorted(dfo_latest["retailer_recomendado"].dropna().unique())
        selected_retailer = st.selectbox("Retailer", retailers)

        top_ret = (dfo_latest[dfo_latest["retailer_recomendado"] == selected_retailer]
                   .nlargest(20, "expected_value"))
        st.caption(f"{len(top_ret)} clientes para {selected_retailer}")

        with st.expander("Preview"):
            display_cols = [c for c in ["cust_id", "prioridad", "expected_value",
                           "accion", "canal", "timing", "cluster_name"] if c in top_ret.columns]
            st.dataframe(top_ret[display_cols], use_container_width=True)

        st.download_button(
            f"Descargar CSV — Top 20 {selected_retailer}",
            data=to_csv_bytes(top_ret),
            file_name=f"top20_{selected_retailer}_{latest_t0.strftime('%Y%m')}.csv",
            mime="text/csv",
        )

    st.markdown("---")

    # ── 4. Urgentes (fuga + puntos por vencer) ─────────────────────────
    st.subheader("4. Clientes Urgentes")
    if "timing" in dfo_latest.columns:
        urgentes = dfo_latest[dfo_latest["timing"].str.contains("Urgente", case=False, na=False)]
        st.caption(f"{len(urgentes)} clientes urgentes")

        if len(urgentes) > 0:
            timing_counts = urgentes["timing"].value_counts()
            for timing, n in timing_counts.items():
                st.write(f"- **{timing}**: {n} clientes")

            with st.expander("Preview"):
                display_cols = [c for c in ["cust_id", "prioridad", "expected_value", "timing",
                               "accion", "funnel_state_at_t0", "cluster_name"] if c in urgentes.columns]
                st.dataframe(urgentes[display_cols].head(30), use_container_width=True)

            st.download_button(
                "Descargar CSV — Urgentes",
                data=to_csv_bytes(urgentes),
                file_name=f"urgentes_{latest_t0.strftime('%Y%m')}.csv",
                mime="text/csv",
            )

    st.markdown("---")

    # ── 5. Reactivacion (dormidos + fuga) ──────────────────────────────
    st.subheader("5. Clientes para Reactivacion")
    reactivacion = pd.DataFrame()
    if "cluster_name" in dff_latest.columns and "funnel_state_at_t0" in dff_latest.columns:
        mask_dormidos = dff_latest["cluster_name"] == "Dormidos"
        mask_fuga = dff_latest["funnel_state_at_t0"] == "FUGA"
        ids_react = dff_latest[mask_dormidos | mask_fuga]["cust_id"].unique()

        if "cust_id" in dfo_latest.columns:
            reactivacion = dfo_latest[dfo_latest["cust_id"].isin(ids_react)]
            st.caption(f"{len(reactivacion)} clientes (Dormidos + Fuga)")

            if len(reactivacion) > 0:
                with st.expander("Preview"):
                    display_cols = [c for c in ["cust_id", "prioridad", "expected_value",
                                   "accion", "cluster_name", "funnel_state_at_t0", "timing"] if c in reactivacion.columns]
                    st.dataframe(reactivacion[display_cols].head(30), use_container_width=True)

                st.download_button(
                    "Descargar CSV — Reactivacion",
                    data=to_csv_bytes(reactivacion),
                    file_name=f"reactivacion_{latest_t0.strftime('%Y%m')}.csv",
                    mime="text/csv",
                )

    st.markdown("---")

    # ── 6. No Contactar ────────────────────────────────────────────────
    st.subheader("6. No Contactar (uplift negativo)")
    if "prioridad" in dfo_latest.columns:
        no_contact = dfo_latest[dfo_latest["prioridad"] == "No contactar"]
        st.caption(f"{len(no_contact)} clientes — contactarlos destruiria valor")

        with st.expander("Preview"):
            display_cols = [c for c in ["cust_id", "expected_value", "uplift_x",
                           "propensity_score", "cluster_name", "funnel_state_at_t0"] if c in no_contact.columns]
            st.dataframe(no_contact[display_cols].head(20), use_container_width=True)

        st.download_button(
            "Descargar CSV — No Contactar",
            data=to_csv_bytes(no_contact),
            file_name=f"no_contactar_{latest_t0.strftime('%Y%m')}.csv",
            mime="text/csv",
        )

    st.markdown("---")

    # ── 7. Resumen Ejecutivo ───────────────────────────────────────────
    st.subheader("7. Resumen Ejecutivo")
    if "prioridad" in dfo_latest.columns:
        resumen = dfo_latest.groupby("prioridad").agg(
            n=("cust_id", "count"),
            ev_avg=("expected_value", "mean") if "expected_value" in dfo_latest.columns else ("cust_id", "count"),
            p_canje_avg=("propensity_score", "mean") if "propensity_score" in dfo_latest.columns else ("cust_id", "count"),
        ).round(2).reset_index()
        resumen["pct"] = (resumen["n"] / resumen["n"].sum() * 100).round(1)
        resumen = resumen.sort_values("ev_avg", ascending=False)

        st.dataframe(resumen, use_container_width=True)

        # Full summary export (all lists in one Excel with multiple sheets)
        st.subheader("Export Completo (Excel multi-hoja)")
        from io import BytesIO
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            dfo_latest.to_excel(writer, index=False, sheet_name="Decision Engine")
            if "expected_value" in dfo_latest.columns:
                top_ev_all = dfo_latest[dfo_latest["prioridad"].isin(["Alta", "Media"])].nlargest(50, "expected_value")
                top_ev_all.to_excel(writer, index=False, sheet_name="Top 50 EV")
            if len(urgentes) > 0:
                urgentes.to_excel(writer, index=False, sheet_name="Urgentes")
            if len(reactivacion) > 0:
                reactivacion.to_excel(writer, index=False, sheet_name="Reactivacion")
            if len(no_contact) > 0:
                no_contact.to_excel(writer, index=False, sheet_name="No Contactar")
            resumen.to_excel(writer, index=False, sheet_name="Resumen")

        st.download_button(
            "Descargar Excel Completo (todas las listas)",
            data=buf.getvalue(),
            file_name=f"loyalty_analytics_export_{latest_t0.strftime('%Y%m')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
