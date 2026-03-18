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

df_long["mes"] = df_long["Fecha"].dt.month
df_long["t"] = np.arange(len(df_long))
df_long["empresa_id"] = df_long["Empresa"].astype("category").cat.codes

# =========================
# 3. MODELO
# =========================
X = df_long[["mes", "t", "empresa_id"]]
y = df_long["consumo"]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# =========================
# 4. PROYECCIÓN
# =========================
st.subheader("Proyección")

horizonte = st.slider("Meses a proyectar", 1, 12, 6)

future_dates = pd.date_range(df_long["Fecha"].max(), periods=horizonte+1, freq="MS")[1:]

future = []

for i, fecha in enumerate(future_dates):
    for emp in df_long["Empresa"].unique():
        future.append({
            "Fecha": fecha,
            "Empresa": emp,
            "mes": fecha.month,
            "t": len(df_long) + i,
            "empresa_id": df_long[df_long["Empresa"] == emp]["empresa_id"].iloc[0]
        })

future_df = pd.DataFrame(future)

X_future = future_df[["mes", "t", "empresa_id"]]
future_df["consumo_proj"] = model.predict(X_future)

# =========================
# 5. RESULTADOS CONSUMO
# =========================
st.subheader("Resultados de Consumo")

pivot = future_df.pivot(index="Empresa", columns="Fecha", values="consumo_proj")

st.dataframe(pivot)

st.line_chart(future_df.pivot(index="Fecha", columns="Empresa", values="consumo_proj"))

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

for emp in empresas:
    consumos = pivot.loc[emp].values
    stock = stock_empresas[emp]
    
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
# 9. STOCK TOTAL (OPCIONAL)
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
# 11. ALERTAS
# =========================
if meses_total <= 3:
    st.error("⚠️ Riesgo alto: stock crítico")
elif meses_total <= 6:
    st.warning("⚠️ Atención: stock moderado")
else:
    st.success("✅ Stock saludable")
