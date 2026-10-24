import datetime as dt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

# ==============================================================================
# 1. CONFIGURACIÓN DE PÁGINA Y ENCABEZADO
# ==============================================================================
st.set_page_config(page_title="FinanBot IA", page_icon="📈", layout="wide")

st.title("📈 FinanBot IA: Asistente Financiero & Cobertura Dinámica")
st.markdown(
    "Seleccioná tus activos de preferencia y ajustá el **monto** y **plazo de inversión** "
    "para obtener una recomendación adaptada dinámicamente a tu perfil de riesgo e historia de mercado."
)

# ==============================================================================
# 2. PARÁMETROS DEL INVERSOR (SIDEBAR)
# ==============================================================================
st.sidebar.header("⚙️ Parámetros del Inversor")

monto = st.sidebar.number_input(
    "Monto a invertir ($ ARS)", 
    min_value=10_000, 
    max_value=100_000_000,
    value=1_000_000, 
    step=50_000,
)
plazo_dias = st.sidebar.slider("Horizonte de inversión (días)", 30, 365, 90, 30)
perfil = st.sidebar.selectbox("Perfil de riesgo", ["Conservador", "Moderado", "Agresivo"])

# ==============================================================================
# 3. CATÁLOGO DE ACTIVOS Y SELECCIÓN
# ==============================================================================
CATALOGO = {
    "Bonos soberanos": {
        "AL29.BA": "Bonar 2029", "AL30.BA": "Bonar 2030", "AL35.BA": "Bonar 2035",
        "GD29.BA": "Global 2029", "GD30.BA": "Global 2030", "GD35.BA": "Global 2035",
    },
    "Acciones argentinas": {
        "GGAL.BA": "Galicia", "YPF.BA": "YPF", "PAMP.BA": "Pampa Energía",
        "BMA.BA": "Banco Macro", "CEPU.BA": "Central Puerto", "TXAR.BA": "Ternium Argentina",
    },
    "CEDEARs": {
        "AAPL.BA": "Apple", "MSFT.BA": "Microsoft", "NVDA.BA": "Nvidia",
        "MELI.BA": "Mercado Libre", "SPY": "S&P 500 (ETF)",
    },
    "Criptoassets": {
        "BTC-USD": "Bitcoin", "ETH-USD": "Ethereum",
    },
}
SELECCION_INICIAL = {"AL30.BA", "GD30.BA", "GGAL.BA", "YPF.BA", "PAMP.BA", "BMA.BA", "SPY", "BTC-USD"}

CATEGORIA = {t: cat for cat, d in CATALOGO.items() for t in d}
CATEGORIA.update({"BADLAR": "Tasas en pesos", "LECAP": "Tasas en pesos", "IPC_UVA": "Inflación"})

st.sidebar.divider()
st.sidebar.subheader("🧺 Activos a analizar")
seleccion = []
for cat, activos in CATALOGO.items():
    elegidos = st.sidebar.multiselect(
        cat,
        options=list(activos),
        default=[t for t in activos if t in SELECCION_INICIAL],
        format_func=lambda t, a=activos: f"{t.replace('.BA', '')} · {a[t]}",
    )
    seleccion += elegidos
st.sidebar.caption("BADLAR, LECAP e Inflación (IPC) se incluyen siempre como benchmark de tasa.")

if not seleccion:
    st.warning("Seleccioná al menos un activo en la barra lateral para comenzar.")
    st.stop()

TICKERS = tuple(sorted(seleccion))

# ==============================================================================
# 4. EXTRACCIÓN DE DATOS DE MERCADO
# ==============================================================================
TNA_BADLAR = 0.40
TEM_LECAP = 0.038
IPC_MENSUAL = 0.035

@st.cache_data(ttl=3600, show_spinner=False)
def cargar_datos(dias: int, tickers: tuple):
    fin = dt.date.today()
    inicio = fin - dt.timedelta(days=dias)

    series = {}
    for t in tickers:
        try:
            s = yf.Ticker(t).history(start=inicio, end=fin)["Close"]
            if not s.empty:
                s.index = s.index.tz_localize(None).normalize()
                series[t] = s
        except Exception:
            pass

    if not series:
        return None, None

    precios = pd.DataFrame(series).ffill().bfill()

    r_badlar = (1 + TNA_BADLAR) ** (1 / 365) - 1
    r_lecap = (1 + TEM_LECAP) ** (1 / 30) - 1
    r_ipc = (1 + IPC_MENSUAL) ** (1 / 30) - 1

    ret = precios.pct_change()
    ret["BADLAR"] = r_badlar
    ret["LECAP"] = r_lecap
    ret["IPC_UVA"] = r_ipc
    ret = ret.fillna(0)
    return precios, ret

with st.spinner("Descargando cotizaciones del mercado..."):
    precios, retornos = cargar_datos(plazo_dias, TICKERS)

if retornos is None:
    st.error("No se pudieron descargar cotizaciones. Verificá tu conexión o intentá nuevamente.")
    st.stop()

evolucion = (1 + retornos).cumprod() * 100

resumen = pd.DataFrame({
    "Rendimiento Acumulado (%)": (evolucion.iloc[-1] - 100).round(2),
    "Volatilidad Anualizada (%)": (retornos.std() * np.sqrt(252) * 100).round(2),
}).sort_values("Rendimiento Acumulado (%)", ascending=False)
resumen["Categoría"] = [CATEGORIA.get(t, "Otro") for t in resumen.index]

# ==============================================================================
# 5. ASIGNACIÓN DINÁMICA DE CARTERA (MONTO + PLAZO + PERFIL)
# ==============================================================================
def construir_cartera_dinamica(perfil_user, plazo, monto_user, columnas_disponibles):
    pesos_raw = {}
    
    # 1. Ponderación base por plazo de inversión
    if plazo <= 60:
        # Corto plazo: Privilegia liquidez y tasa fija
        pesos_raw["LECAP"] = 0.40
        pesos_raw["BADLAR"] = 0.30
        pesos_raw["IPC_UVA"] = 0.20
        pesos_raw["AL30.BA"] = 0.10
    elif plazo <= 180:
        # Mediano plazo: Equilibrio entre tasa y renta variable/soberanos
        pesos_raw["LECAP"] = 0.25
        pesos_raw["AL30.BA"] = 0.25
        pesos_raw["GGAL.BA"] = 0.20
        pesos_raw["YPF.BA"] = 0.15
        pesos_raw["SPY"] = 0.15
    else:
        # Largo plazo: Maximización de retorno en Renta Variable / Cripto
        pesos_raw["GGAL.BA"] = 0.25
        pesos_raw["YPF.BA"] = 0.25
        pesos_raw["AL30.BA"] = 0.20
        pesos_raw["SPY"] = 0.15
        pesos_raw["BTC-USD"] = 0.15

    # 2. Ajuste por perfil de riesgo
    if perfil_user == "Conservador":
        pesos_raw["LECAP"] = pesos_raw.get("LECAP", 0) + 0.25
        pesos_raw["BADLAR"] = pesos_raw.get("BADLAR", 0) + 0.20
        # Reducimos peso de renta variable
        for k in ["BTC-USD", "GGAL.BA", "YPF.BA"]:
            if k in pesos_raw:
                pesos_raw[k] *= 0.3
    elif perfil_user == "Agresivo":
        for k in ["BTC-USD", "GGAL.BA", "YPF.BA", "SPY"]:
            if k in pesos_raw:
                pesos_raw[k] *= 1.8
        for k in ["LECAP", "BADLAR"]:
            if k in pesos_raw:
                pesos_raw[k] *= 0.2

    # 3. Filtrar solo activos disponibles en los datos descargados
    pesos_filtrados = {k: v for k, v in pesos_raw.items() if k in columnas_disponibles and v > 0}

    # Si ningún activo clave está seleccionado, distribuimos proporcionalmente
    if not pesos_filtrados:
        pesos_filtrados = {k: 1.0 for k in columnas_disponibles if k not in ["IPC_UVA"]}

    # Normalización para que sumen 100%
    suma_pesos = sum(pesos_filtrados.values())
    return {k: v / suma_pesos for k, v in pesos_filtrados.items()}

pesos_optimos = construir_cartera_dinamica(perfil, plazo_dias, monto, retornos.columns)

# Cálculo de la rentabilidad de la cartera
ret_cartera = sum(retornos[k] * w for k, w in pesos_optimos.items())
evol_cartera = (1 + ret_cartera).cumprod() * 100
rend_cartera = evol_cartera.iloc[-1] - 100
monto_final = monto * (1 + rend_cartera / 100)
ganancia_pesos = monto_final - monto

# ==============================================================================
# 6. INTERFAZ Y RESULTADOS DÍNAMICOS
# ==============================================================================
st.subheader("📊 Diagnóstico y Rendimiento Estimado")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Monto a invertir", f"$ {monto:,.0f} ARS")
c2.metric("Horizonte de Análisis", f"{plazo_dias} días")
c3.metric("Activo Líder", resumen.index[0], f"{resumen.iloc[0, 0]:+.2f}%")
c4.metric(
    "Capital Estimado Cartera", 
    f"$ {monto_final:,.0f} ARS", 
    f"{ganancia_pesos:+,.0f} ARS ({rend_cartera:+.2f}%)"
)

fig = px.line(
    evolucion, x=evolucion.index, y=evolucion.columns,
    title=f"Evolución Comparativa del Mercado ({plazo_dias} días históricos)",
    labels={"value": "Capital (Base 100)", "variable": "Activo", "index": "Fecha"},
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("📋 Resumen de Desempeño por Activo")
st.dataframe(resumen.style.highlight_max(subset=["Rendimiento Acumulado (%)"], color="#c6f6d5"), use_container_width=True)

# ==============================================================================
# 7. DICTAMEN DE FINANBOT IA
# ==============================================================================
st.subheader("🤖 Recomendación Personalizada de FinanBot IA")

top3 = resumen.head(3).index.tolist()
mas_volatil = resumen["Volatilidad Anualizada (%)"].idxmax()
ipc_acum = resumen.loc["IPC_UVA", "Rendimiento Acumulado (%)"]
cartera_txt = " + ".join(f"{int(round(w * 100))}% {k}" for k, w in pesos_optimos.items())

if rend_cartera > ipc_acum:
    diagnostico_cobertura = f"La cartera logra superar la inflación proyectada del período ({ipc_acum}%), generando un rendimiento real positivo de **{rend_cartera - ipc_acum:.2f}%**."
else:
    diagnostico_cobertura = f"La cartera rinde **{rend_cartera:.2f}%**, quedando por debajo de la inflación acumulada del período ({ipc_acum}%). Se recomienda ajustar la ponderación hacia activos de mayor cobertura."

st.info(f"""
**Estrategia Dinámica para ${monto:,.0f} ARS a {plazo_dias} días:**
La combinación sugerida para tu perfil **{perfil}** a un plazo de **{plazo_dias} días** es:
👉 **{cartera_txt}**

**Diagnóstico de Mercado:**
En el plazo analizado, los activos más rentables fueron **{', '.join(top3)}**. {diagnostico_cobertura}

**Riesgo y Volatilidad:**
El activo con mayor fluctuación en este período fue **{mas_volatil}**. Se proyecta una ganancia estimada de **$ {ganancia_pesos:,.0f} ARS** sobre tu capital aportado.
""")

st.caption("⚠️ Herramienta educativa. Rendimientos pasados no garantizan rendimientos futuros. No constituye asesoramiento financiero formal.")
