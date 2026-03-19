import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
 
# ─── Config ───────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Demand Forecast", page_icon="📊", layout="wide")
 
st.markdown("""
<style>
    .main { background-color: #080d1a; }
    .block-container { padding-top: 2rem; }
    h1, h2, h3 { color: #c8d8f0; }
    .stDataFrame { border-radius: 8px; }
    .metric-card {
        background: #0d1525;
        border: 1px solid #1e2a3a;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 8px;
    }
    .stTabs [data-baseweb="tab"] { color: #8892aa; font-size: 13px; }
    .stTabs [aria-selected="true"] { color: #00ffc8 !important; border-bottom-color: #00ffc8 !important; }
</style>
""", unsafe_allow_html=True)
 
# ─── Constants ────────────────────────────────────────────────────────────────
COMPANIES = ["Inbalnor", "Skretting", "Biomar", "Cargill", "Haid"]
COLORS    = ["#60c8f0",  "#ef4444",   "#3b82f6", "#2dd4bf", "#16a34a"]
COLOR_MAP = dict(zip(COMPANIES, COLORS))
 
def generate_dates(extra=0):
    dates = []
    y, m = 2024, 1
    end_y, end_m = 2026, 2 + extra
    while True:
        dates.append(f"{y}-{m:02d}")
        if y == end_y + (end_m - 1) // 12 and m == ((end_m - 1) % 12) + 1:
            break
        m += 1
        if m > 12:
            m = 1
            y += 1
        if y * 12 + m > end_y * 12 + end_m:
            break
    return dates
 
# simpler approach
def gen_dates(extra=0):
    result = []
    y, m = 2024, 1
    total_months = (2026 - 2024) * 12 + 2 + extra  # Jan2024 -> Feb2026 + extra
    for _ in range(total_months):
        result.append(f"{y}-{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return result
 
# ─── Session state ────────────────────────────────────────────────────────────
if "extra_months" not in st.session_state:
    st.session_state.extra_months = 0
 
DATES = gen_dates(st.session_state.extra_months)
 
# ─── Header ───────────────────────────────────────────────────────────────────
col_title, col_add = st.columns([8, 1])
with col_title:
    st.markdown("## 📊 Demand Forecast — ML con Lags")
    st.caption("Random Forest · XGBoost · Validación Cruzada Walk-Forward · MAE · MAPE")
with col_add:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("➕ Mes", help="Agregar un mes más a la tabla"):
        st.session_state.extra_months += 1
        st.rerun()
 
st.divider()
 
# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab_input, tab_forecast, tab_metrics, tab_coverage = st.tabs(
    ["📥 Datos", "📈 Proyección", "🎯 Métricas CV", "📦 Cobertura"]
)
 
# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — INPUT
# ══════════════════════════════════════════════════════════════════════════════
with tab_input:
    col_m, col_h, col_btn = st.columns([2, 2, 2])
    with col_m:
        modelo_tipo = st.selectbox("Modelo ML", ["Random Forest", "XGBoost"])
    with col_h:
        horizonte = st.slider("Meses a proyectar", 1, 24, 12)
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        run = st.button("▶ Proyectar", type="primary", use_container_width=True)
 
    st.markdown("**Ingresa el consumo mensual por empresa** (pega desde Excel con Ctrl+V):")
 
    df_default = pd.DataFrame(
        np.zeros((len(COMPANIES), len(DATES)), dtype=int),
        index=COMPANIES,
        columns=DATES
    )
    df_default.index.name = "Empresa"
 
    df_editado = st.data_editor(
        df_default,
        use_container_width=True,
        num_rows="fixed",
        column_config={d: st.column_config.NumberColumn(d, min_value=0, step=1, format="%d") for d in DATES}
    )
 
# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING & MODELOS
# ══════════════════════════════════════════════════════════════════════════════
def build_features(series: pd.Series):
    """Dado un Series con índice de fecha, construye DataFrame de features."""
    df = pd.DataFrame({"consumo": series.values})
    df["t"]       = np.arange(len(df))
    df["mes"]     = [pd.to_datetime(d).month for d in series.index]
    df["mes_sin"] = np.sin(2 * np.pi * df["mes"] / 12)
    df["mes_cos"] = np.cos(2 * np.pi * df["mes"] / 12)
    df["lag1"]    = df["consumo"].shift(1)
    df["lag2"]    = df["consumo"].shift(2)
    df["lag3"]    = df["consumo"].shift(3)
    df = df.dropna()
    return df
 
FEATURES = ["t", "mes_sin", "mes_cos", "lag1", "lag2", "lag3"]
 
def get_model(tipo):
    if tipo == "Random Forest":
        return RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42, n_jobs=-1)
    else:
        return XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4,
                            random_state=42, verbosity=0)
 
def cross_validate_ts(df_feat, tipo, n_splits=4):
    """Walk-forward TimeSeriesSplit CV. Retorna MAE y MAPE promedio."""
    tscv  = TimeSeriesSplit(n_splits=n_splits)
    maes, mapes = [], []
    X = df_feat[FEATURES].values
    y = df_feat["consumo"].values
 
    for train_idx, test_idx in tscv.split(X):
        if len(train_idx) < 4:
            continue
        model = get_model(tipo)
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[test_idx])
        preds = np.maximum(0, preds)
        actuals = y[test_idx]
        mae = mean_absolute_error(actuals, preds)
        mask = actuals != 0
        mape = np.mean(np.abs((actuals[mask] - preds[mask]) / actuals[mask])) * 100 if mask.any() else np.nan
        maes.append(mae)
        if not np.isnan(mape):
            mapes.append(mape)
 
    return {
        "mae":      round(float(np.mean(maes)), 0)  if maes  else None,
        "mape":     round(float(np.mean(mapes)), 2) if mapes else None,
        "cv_folds": len(maes),
    }
 
def project(model, last_t, last_fecha, lag1, lag2, lag3, horizonte):
    """Proyección recursiva con actualización dinámica de lags."""
    future = []
    future_dates = pd.date_range(last_fecha, periods=horizonte + 1, freq="MS")[1:]
    for i, fecha in enumerate(future_dates):
        t_f     = last_t + i + 1
        mes     = fecha.month
        X_pred  = pd.DataFrame([{
            "t": t_f,
            "mes_sin": np.sin(2 * np.pi * mes / 12),
            "mes_cos": np.cos(2 * np.pi * mes / 12),
            "lag1": lag1, "lag2": lag2, "lag3": lag3,
        }])
        pred = float(model.predict(X_pred)[0])
        pred = max(0, pred)
        future.append({"Fecha": fecha, "consumo_proj": pred})
        lag3, lag2, lag1 = lag2, lag1, pred
    return pd.DataFrame(future)
 
# ══════════════════════════════════════════════════════════════════════════════
# RUN MODELS
# ══════════════════════════════════════════════════════════════════════════════
results_forecast = {}
results_metrics  = {}
 
if run or "forecast_done" in st.session_state:
 
    df_long = df_editado.reset_index().melt(
        id_vars="Empresa", var_name="Fecha", value_name="consumo"
    )
    df_long["Fecha"] = pd.to_datetime(df_long["Fecha"])
    df_long = df_long.sort_values(["Empresa", "Fecha"])
 
    if df_long["consumo"].sum() == 0:
        st.warning("⚠️ Ingresa datos antes de proyectar.")
        st.stop()
 
    df_long["mes"]     = df_long["Fecha"].dt.month
    df_long["t"]       = df_long.groupby("Empresa").cumcount()
    df_long["mes_sin"] = np.sin(2 * np.pi * df_long["mes"] / 12)
    df_long["mes_cos"] = np.cos(2 * np.pi * df_long["mes"] / 12)
    df_long["lag1"]    = df_long.groupby("Empresa")["consumo"].shift(1)
    df_long["lag2"]    = df_long.groupby("Empresa")["consumo"].shift(2)
    df_long["lag3"]    = df_long.groupby("Empresa")["consumo"].shift(3)
    df_clean = df_long.dropna()
 
    with st.spinner("Entrenando modelos y calculando métricas..."):
        modelos = {}
        for emp in COMPANIES:
            df_emp = df_clean[df_clean["Empresa"] == emp]
            if df_emp.empty or len(df_emp) < 5:
                continue
 
            X = df_emp[FEATURES]
            y = df_emp["consumo"]
 
            # Train full model
            model = get_model(modelo_tipo)
            model.fit(X, y)
            modelos[emp] = model
 
            # Cross-validation
            df_feat = df_emp[FEATURES + ["consumo"]].reset_index(drop=True)
            cv = cross_validate_ts(df_feat, modelo_tipo)
            results_metrics[emp] = cv
 
            # Project
            last_row   = df_emp.iloc[-1]
            lag1_val   = last_row["consumo"]
            lag2_val   = df_emp.iloc[-2]["consumo"] if len(df_emp) >= 2 else 0
            lag3_val   = df_emp.iloc[-3]["consumo"] if len(df_emp) >= 3 else 0
            proj_df    = project(model, last_row["t"], last_row["Fecha"],
                                 lag1_val, lag2_val, lag3_val, horizonte)
            proj_df["Empresa"] = emp
            results_forecast[emp] = proj_df
 
    st.session_state["forecast_done"]      = True
    st.session_state["results_forecast"]   = results_forecast
    st.session_state["results_metrics"]    = results_metrics
    st.session_state["modelo_tipo_usado"]  = modelo_tipo
    st.session_state["horizonte_usado"]    = horizonte
 
# Recuperar de session
if "results_forecast" in st.session_state:
    results_forecast  = st.session_state["results_forecast"]
    results_metrics   = st.session_state["results_metrics"]
    modelo_tipo_usado = st.session_state.get("modelo_tipo_usado", modelo_tipo)
    horizonte_usado   = st.session_state.get("horizonte_usado", horizonte)
 
    all_proj = pd.concat(results_forecast.values()) if results_forecast else pd.DataFrame()
 
# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — FORECAST
# ══════════════════════════════════════════════════════════════════════════════
with tab_forecast:
    if not results_forecast:
        st.info("Ingresa datos y presiona **▶ Proyectar** en la pestaña Datos.")
    else:
        st.markdown(f"#### Proyección — {modelo_tipo_usado} con Lags · {horizonte_usado} meses")
 
        # Pivot para tabla
        pivot = all_proj.pivot(index="Empresa", columns="Fecha", values="consumo_proj").fillna(0)
        pivot.columns = [c.strftime("%Y-%m") for c in pivot.columns]
        pivot = pivot.applymap(lambda x: round(x))
        st.dataframe(pivot.style.format("{:,.0f}"), use_container_width=True)
 
        # Gráfico
        chart_df = all_proj.copy()
        chart_df["Fecha"] = chart_df["Fecha"].dt.strftime("%Y-%m")
        chart_pivot = chart_df.pivot(index="Fecha", columns="Empresa", values="consumo_proj")
        st.line_chart(chart_pivot, use_container_width=True)
 
# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MÉTRICAS CV
# ══════════════════════════════════════════════════════════════════════════════
with tab_metrics:
    if not results_metrics:
        st.info("Ingresa datos y presiona **▶ Proyectar** para ver las métricas de validación cruzada.")
    else:
        st.markdown(f"#### Validación Cruzada Walk-Forward — {modelo_tipo_usado}")
        st.caption("Se entrena en los primeros N meses y se valida en los meses siguientes, repitiendo por folds.")
 
        rows = []
        for emp in COMPANIES:
            m = results_metrics.get(emp, {})
            if not m:
                continue
            mape = m.get("mape")
            mae  = m.get("mae")
            folds = m.get("cv_folds", 0)
            if mape is None:
                rating = "Sin datos"
            elif mape < 10:
                rating = "🟢 Excelente"
            elif mape < 20:
                rating = "🟡 Bueno"
            elif mape < 35:
                rating = "🟠 Regular"
            else:
                rating = "🔴 Pobre"
            rows.append({
                "Empresa":    emp,
                "MAE":        f"{mae:,.0f}" if mae is not None else "—",
                "MAPE":       f"{mape:.2f}%" if mape is not None else "—",
                "Folds CV":   folds,
                "Rating":     rating,
            })
 
        df_metrics = pd.DataFrame(rows).set_index("Empresa")
        st.dataframe(df_metrics, use_container_width=True)
 
        st.markdown("---")
        st.markdown("**Referencia MAPE:**")
        col1, col2, col3, col4 = st.columns(4)
        col1.success("**< 10%** — Excelente")
        col2.info("**10–20%** — Bueno")
        col3.warning("**20–35%** — Regular")
        col4.error("**> 35%** — Pobre")
 
        st.caption(
            "⚠️ Con menos de 12 meses de datos históricos por empresa, "
            "los folds de CV son pocos y las métricas menos representativas. "
            "Se recomiendan mínimo 18–24 meses para resultados confiables."
        )
 
# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — COBERTURA
# ══════════════════════════════════════════════════════════════════════════════
with tab_coverage:
    if not results_forecast:
        st.info("Ingresa datos y presiona **▶ Proyectar** para ver la cobertura.")
    else:
        st.markdown("#### Stock y Cobertura")
 
        stock_vals = {}
        cols = st.columns(len(COMPANIES))
        for i, emp in enumerate(COMPANIES):
            with cols[i]:
                stock_vals[emp] = st.number_input(f"Stock {emp}", value=100_000.0,
                                                  min_value=0.0, step=1000.0, key=f"stock_{emp}")
 
        def meses_reales(stock, consumos):
            rem = stock
            for i, c in enumerate(consumos):
                rem -= c
                if rem <= 0:
                    return i + 1
            return len(consumos)
 
        def meses_promedio(stock, consumos):
            prom = np.mean(consumos)
            return round(stock / prom, 2) if prom > 0 else float("inf")
 
        cov_rows = []
        for emp in COMPANIES:
            if emp not in results_forecast:
                continue
            consumos = results_forecast[emp]["consumo_proj"].values
            stock    = stock_vals[emp]
            mr       = meses_reales(stock, consumos)
            mp       = meses_promedio(stock, consumos)
            emoji    = "🔴" if mr <= 3 else "🟡" if mr <= 6 else "🟢"
            cov_rows.append({
                "Empresa":        emp,
                "Stock":          f"{stock:,.0f}",
                "Meses real":     f"{emoji} {mr}",
                "Meses promedio": mp,
            })
 
        df_cov = pd.DataFrame(cov_rows).set_index("Empresa")
        st.dataframe(df_cov, use_container_width=True)
