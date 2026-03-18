import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

st.title("📊 Proyección de Consumo + Cobertura de Stock 🔥")

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

# Features
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
# 6. STOCK
# =========================
st.subheader("Stock del Ingrediente (Total para las 5 empresas)")

stock_actual = st.number_input(
    "Ingrese el stock actual total",
    min_value=0.0,
    value=1000.0,
    step=100.0
)

# Consumo total mensual (suma de empresas)
consumos_proj = pivot.sum(axis=0).values

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
# 8. CÁLCULOS
# =========================
stock_evol = evolucion_stock(stock_actual, consumos_proj)

df_stock = pd.DataFrame({
    "Fecha": pivot.columns,
    "Stock": stock_evol
})

# =========================
# 9. VISUALIZACIÓN STOCK
# =========================
st.subheader("Evolución del Stock")

st.line_chart(df_stock.set_index("Fecha"))

meses = calcular_meses_stock(stock_actual, consumos_proj)

st.metric("Meses de cobertura", meses)

# =========================
# 10. ALERTAS
# =========================
if meses <= 3:
    st.error("⚠️ Riesgo alto: stock crítico")
elif meses <= 6:
    st.warning("⚠️ Atención: stock moderado")
else:
    st.success("✅ Stock saludable")
