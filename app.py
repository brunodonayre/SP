import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

st.title("📊 Forecast + Lags + ML 🔥")

# =========================
# 1. INPUT
# =========================
st.subheader("Carga tus datos")

fechas = pd.date_range("2024-01-01", "2026-02-01", freq="MS")
fechas_str = [d.strftime("%Y-%m") for d in fechas]

empresas = ["Biomar", "Cargill", "Haid", "Inbalnor", "Skretting"]

df = pd.DataFrame(0, index=empresas, columns=fechas_str)
df.index.name = "Empresa"

df_editado = st.data_editor(df)

# =========================
# 2. TRANSFORMACIÓN
# =========================
df_long = df_editado.reset_index().melt(
    id_vars="Empresa",
    var_name="Fecha",
    value_name="consumo"
)

df_long["Fecha"] = pd.to_datetime(df_long["Fecha"])
df_long = df_long.sort_values(["Empresa", "Fecha"])

if df_long["consumo"].sum() == 0:
    st.warning("Ingresa datos")
    st.stop()

df_long["mes"] = df_long["Fecha"].dt.month
df_long["t"] = df_long.groupby("Empresa").cumcount()

# estacionalidad
df_long["mes_sin"] = np.sin(2 * np.pi * df_long["mes"] / 12)
df_long["mes_cos"] = np.cos(2 * np.pi * df_long["mes"] / 12)

# =========================
# 🔥 LAGS
# =========================
df_long["lag1"] = df_long.groupby("Empresa")["consumo"].shift(1)
df_long["lag2"] = df_long.groupby("Empresa")["consumo"].shift(2)

df_long = df_long.dropna()

# =========================
# 3. SELECCIÓN MODELO
# =========================
modelo_tipo = st.selectbox(
    "Modelo",
    ["Random Forest", "XGBoost"]
)

# =========================
# 4. ENTRENAMIENTO
# =========================
modelos = {}

features = ["t", "mes_sin", "mes_cos", "lag1", "lag2"]

for emp in df_long["Empresa"].unique():
    df_emp = df_long[df_long["Empresa"] == emp]

    X = df_emp[features]
    y = df_emp["consumo"]

    if modelo_tipo == "Random Forest":
        model = RandomForestRegressor(n_estimators=200, random_state=42)
    else:
        model = XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            random_state=42
        )

    model.fit(X, y)
    modelos[emp] = model

# =========================
# 5. PROYECCIÓN CON LAGS
# =========================
st.subheader("Proyección")

horizonte = st.slider("Meses a proyectar", 1, 24, 12)

future = []

for emp in df_long["Empresa"].unique():
    df_emp = df_long[df_long["Empresa"] == emp]

    last_row = df_emp.iloc[-1]

    last_t = last_row["t"]
    last_fecha = last_row["Fecha"]

    lag1 = last_row["consumo"]
    lag2 = df_emp.iloc[-2]["consumo"]

    future_dates = pd.date_range(last_fecha, periods=horizonte+1, freq="MS")[1:]

    for i, fecha in enumerate(future_dates):
        t_future = last_t + i + 1

        mes = fecha.month
        mes_sin = np.sin(2 * np.pi * mes / 12)
        mes_cos = np.cos(2 * np.pi * mes / 12)

        X_pred = pd.DataFrame([{
            "t": t_future,
            "mes_sin": mes_sin,
            "mes_cos": mes_cos,
            "lag1": lag1,
            "lag2": lag2
        }])

        pred = modelos[emp].predict(X_pred)[0]
        pred = max(0, pred)

        future.append({
            "Empresa": emp,
            "Fecha": fecha,
            "consumo_proj": pred
        })

        # 🔥 actualizar lags dinámicamente
        lag2 = lag1
        lag1 = pred

future_df = pd.DataFrame(future)

# =========================
# 6. RESULTADOS
# =========================
pivot = future_df.pivot(
    index="Empresa",
    columns="Fecha",
    values="consumo_proj"
).fillna(0)

st.dataframe(pivot)

st.line_chart(
    future_df.pivot(index="Fecha", columns="Empresa", values="consumo_proj")
)

# =========================
# 7. STOCK
# =========================
st.subheader("Stock")

stock_empresas = {}

for emp in df_long["Empresa"].unique():
    stock_empresas[emp] = st.number_input(
        f"Stock - {emp}",
        value=100000.0,
        key=f"stock_{emp}"
    )

# =========================
# 8. COBERTURA
# =========================
def cobertura(stock, consumos):
    stock_actual = stock

    for i, c in enumerate(consumos):
        stock_actual -= c
        if stock_actual <= 0:
            return i + 1

    return len(consumos)

def cobertura_prom(stock, consumos):
    prom = np.mean(consumos)
    return stock / prom if prom > 0 else 0

resultados = []

for emp in pivot.index:
    consumos = pivot.loc[emp].values
    stock = stock_empresas.get(emp, 0)

    resultados.append({
        "Empresa": emp,
        "Stock": stock,
        "Meses real": cobertura(stock, consumos),
        "Meses promedio": round(cobertura_prom(stock, consumos), 2)
    })

st.subheader("Cobertura")
st.dataframe(pd.DataFrame(resultados))
