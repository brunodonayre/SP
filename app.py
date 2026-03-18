import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

st.title("📊 Proyección de Consumo - Vitapro Style 🔥")

# =========================
# 1. INPUT (FORMATO FILAS=EMPRESA, COLUMNAS=FECHAS)
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

# convertir fecha
df_long["Fecha"] = pd.to_datetime(df_long["Fecha"])

# features
df_long["mes"] = df_long["Fecha"].dt.month
df_long["t"] = np.arange(len(df_long))

# encode empresa
df_long["empresa_id"] = df_long["Empresa"].astype("category").cat.codes

# =========================
# 3. MODELO
# =========================
X = df_long[["mes", "t", "empresa_id"]]
y = df_long["consumo"]

model = RandomForestRegressor()
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
# 5. RESULTADOS
# =========================
st.subheader("Resultados")

pivot = future_df.pivot(index="Empresa", columns="Fecha", values="consumo_proj")


# ====================
#  implementación de funcion calcular_meses_stock
# ==================

def calcular_meses_stock(stock_actual, consumos_proj):
    stock = stock_actual
    
    for i, consumo in enumerate(consumos_proj):
        stock -= consumo
        
        if stock <= 0:
            return i + 1  # meses que duró
    
    return len(consumos_proj)  # no se acabó

# ==================

st.dataframe(pivot)

st.line_chart(future_df.pivot(index="Fecha", columns="Empresa", values="consumo_proj"))
