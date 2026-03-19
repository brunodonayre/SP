import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

st.title("📊 Proyección de Consumo + Cobertura por Empresa 🔥")

# =========================
# 1. INPUT DATOS
# =========================
st.subheader("Carga tus datos")

fechas = pd.date_range("2024-01-01", "2026-02-01", freq="MS")
fechas_str = [d.strftime("%Y-%m") for d in fechas]

empresas = ["Empresa1", "Empresa2", "Empresa3", "Empresa4", "Empresa5"]

df = pd.DataFrame(0, index=empresas, columns=fechas_str)
df.index.name = "Empresa"

df_editado = st.data_editor(df)

# =========================
# 2. TRANSFORMACIÓN
# =========================
df_reset = df_editado.reset_index()

df_long = df_reset.melt(id_vars="Empresa", var_name="Fecha", value_name="consumo")
df_long["Fecha"] = pd.to_datetime(df_long["Fecha"])

# Validación
if df_long["consumo"].sum() == 0:
    st.warning("⚠️ Ingresa datos antes de proyectar")
    st.stop()

df_long = df_long.sort_values(["Empresa", "Fecha"])

df_long["mes"] = df_long["Fecha"].dt.month
df_long["t"] = df_long.groupby("Empresa").cumcount()

# Estacionalidad
df_long["mes_sin"] = np.sin(2 * np.pi * df_long["mes"] / 12)
df_long["mes_cos"] = np.cos(2 * np.pi * df_long["mes"] / 12)

# =========================
# 3. MODELOS POR EMPRESA
# =========================
modelos = {}

for emp in df_long["Empresa"].unique():
    df_emp = df_long[df_long["Empresa"] == emp]
    
    X_emp = df_emp[["t", "mes_sin", "mes_cos"]]
    y_emp = df_emp["consumo"]
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_emp, y_emp)
    
    modelos[emp] = model

# =========================
# 4. PROYECCIÓN
# =========================
st.subheader("Proyección")

horizonte = st.slider("Meses a proyectar", 1, 12, 6)

future = []

for emp in df_long["Empresa"].unique():
    df_emp = df_long[df_long["Empresa"] == emp]
    last_t = df_emp["t"].max()
    last_fecha = df_emp["Fecha"].max()
    
    future_dates = pd.date_range(last_fecha, periods=horizonte+1, freq="MS")[1:]
    
    for i, fecha in enumerate(future_dates):
        t_future = last_t + i + 1
        
        mes = fecha.month
        mes_sin = np.sin(2 * np.pi * mes / 12)
        mes_cos = np.cos(2 * np.pi * mes / 12)
        
        future.append({
            "Fecha": fecha,
            "Empresa": emp,
            "t": t_future,
            "mes_sin": mes_sin,
            "mes_cos": mes_cos
        })

future_df = pd.DataFrame(future)

# Predicción por empresa
future_df["consumo_proj"] = 0.0

for emp in future_df["Empresa"].unique():
    mask = future_df["Empresa"] == emp
    X_future = future_df.loc[mask, ["t", "mes_sin", "mes_cos"]]
    future_df.loc[mask, "consumo_proj"] = modelos[emp].predict(X_future)

# =========================
# 5. RESULTADOS CONSUMO
# =========================
st.subheader("Resultados de Consumo")

pivot = future_df.pivot(index="Empresa", columns="Fecha", values="consumo_proj")

st.dataframe(pivot)

# Gráfico combinado (histórico + forecast)
df_plot_hist = df_long[["Fecha", "Empresa", "consumo"]].rename(columns={"consumo": "valor"})
df_plot_fut = future_df[["Fecha", "Empresa", "consumo_proj"]].rename(columns={"consumo_proj": "valor"})

df_plot = pd.concat([df_plot_hist, df_plot_fut])

st.line_chart(df_plot.pivot(index="Fecha", columns="Empresa", values="valor"))

# =========================
# 6. STOCK POR EMPRESA
# =========================
st.subheader("Stock actual por empresa")

stock_empresas = {}

for emp in empresas:
    stock_empresas[emp] = st.number_input(
        f"Stock actual - {emp}",
        min_value=0.0,
        value=1000.0,
        step=100.0,
        key=emp
    )

# =========================
# 7. FUNCIONES
# =========================
def calcular_meses_stock(stock_actual, consumos_proj):
    stock = stock_actual
    
    for i, consumo in enumerate(consumos_proj):
        stock -= consumo
        
        if stock <= 0:
            return i + 1
    
    return len(consumos_proj)

def evolucion_stock(stock_actual, consumos_proj):
    stock = stock_actual
    evolucion = []
    
    for consumo in consumos_proj:
        stock -= consumo
        evolucion.append(stock)
    
    return evolucion

# =========================
# 8. COBERTURA POR EMPRESA
# =========================
resultados = []

for emp in pivot.index:
    consumos = pivot.loc[emp].values
    stock = stock_empresas.get(emp, 0)
    
    meses = calcular_meses_stock(stock, consumos)
    
    resultados.append({
        "Empresa": emp,
        "Stock": stock,
        "Meses cobertura": meses
    })

df_resultados = pd.DataFrame(resultados)

st.subheader("Cobertura por empresa")
st.dataframe(df_resultados)

# =========================
# 9. STOCK TOTAL
# =========================
stock_total = sum(stock_empresas.values())
consumo_total = pivot.sum(axis=0).values

meses_total = calcular_meses_stock(stock_total, consumo_total)

st.subheader("Cobertura total")
st.metric("Meses de cobertura total", meses_total)

# =========================
# 10. EVOLUCIÓN STOCK TOTAL
# =========================
stock_evol = evolucion_stock(stock_total, consumo_total)

df_stock = pd.DataFrame({
    "Fecha": pivot.columns,
    "Stock Total": stock_evol
})

st.subheader("Evolución del Stock Total")
st.line_chart(df_stock.set_index("Fecha"))

# =========================
# 11. ALERTAS INTELIGENTES
# =========================
if min(stock_evol) < 0:
    st.error("⚠️ Quiebre de stock en el horizonte")
elif meses_total <= 3:
    st.error("⚠️ Riesgo alto: stock crítico")
elif meses_total <= 6:
    st.warning("⚠️ Atención: stock moderado")
else:
    st.success("✅ Stock saludable")
