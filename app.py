import datetime as dt

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

# ==============================================================================
# 1. CONFIGURACIÓN DE LA PÁGINA
# ==============================================================================
st.set_page_config(page_title="FinanBot IA", page_icon="📈", layout="wide")

st.title("📈 FinanBot IA: Asistente Financiero & Cobertura")
st.markdown(
    "Elegí los activos que quieras comparar (bonos, acciones, CEDEARs, oro, cripto) junto con "
    "LECAP, BADLAR e inflación, y obtené un diagnóstico según tu perfil de riesgo."
)

# ==============================================================================
# 2. SIDEBAR: PARÁMETROS DEL INVERSOR
# ==============================================================================
st.sidebar.header("⚙️ Parámetros del Inversor")

monto = st.sidebar.number_input(
    "Monto a invertir ($ ARS)", min_value=10_000, max_value=100_000_000,
    value=1_000_000, step=50_000,
)
plazo_dias = st.sidebar.slider("Plazo histórico a analizar (días)", 30, 365, 90, 30)
perfil = st.sidebar.selectbox("Perfil de riesgo", ["Conservador", "Moderado", "Agresivo"])

# ==============================================================================
# 3. DATOS Y CÁLCULOS
# ==============================================================================
# Catálogo de activos por categoría (ticker de Yahoo Finance -> nombre).
# Si algún ticker no devuelve datos en Yahoo, la app lo saltea sin romperse.
CATALOGO = {
    "Bonos soberanos": {
        "AL29.BA": "Bonar 2029", "AL30.BA": "Bonar 2030", "AL35.BA": "Bonar 2035",
        "GD29.BA": "Global 2029", "GD30.BA": "Global 2030", "GD35.BA": "Global 2035",
        "GD38.BA": "Global 2038", "GD41.BA": "Global 2041", "GD46.BA": "Global 2046",
    },
    "Acciones argentinas": {
        "GGAL.BA": "Galicia", "YPF.BA": "YPF", "PAMP.BA": "Pampa Energía",
        "BMA.BA": "Banco Macro", "SUPV.BA": "Supervielle", "BYMA.BA": "BYMA",
        "CEPU.BA": "Central Puerto", "TGSU2.BA": "Transp. Gas del Sur",
        "TXAR.BA": "Ternium Argentina", "ALUA.BA": "Aluar", "LOMA.BA": "Loma Negra",
        "CRES.BA": "Cresud", "EDN.BA": "Edenor", "TECO2.BA": "Telecom Argentina",
    },
    "CEDEARs": {
        "AAPL.BA": "Apple", "MSFT.BA": "Microsoft", "GOOGL.BA": "Alphabet",
        "AMZN.BA": "Amazon", "NVDA.BA": "Nvidia", "TSLA.BA": "Tesla",
        "MELI.BA": "Mercado Libre", "KO.BA": "Coca-Cola",
    },
    "Internacional, oro y cripto": {
        "SPY": "S&P 500 (ETF)", "QQQ": "Nasdaq 100 (ETF)", "GLD": "Oro (ETF)",
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
st.sidebar.caption("BADLAR, LECAP e inflación (UVA) se incluyen siempre como referencia.")

if not seleccion:
    st.warning("Elegí al menos un activo en la barra lateral para empezar.")
    st.stop()
TICKERS = tuple(sorted(seleccion))

# Tasas simuladas (podés cambiarlas o conectarlas a una API del BCRA)
TNA_BADLAR = 0.40
TEM_LECAP = 0.038
IPC_MENSUAL = 0.035

# Carteras sugeridas por perfil (los pesos suman 1)
CARTERAS = {
    "Conservador": {"LECAP": 0.35, "BADLAR": 0.35, "IPC_UVA": 0.30},
    "Moderado": {"LECAP": 0.50, "AL30.BA": 0.30, "GGAL.BA": 0.10, "YPF.BA": 0.10},
    "Agresivo": {"GGAL.BA": 0.25, "YPF.BA": 0.25, "AL30.BA": 0.15,
                 "GD30.BA": 0.15, "BTC-USD": 0.10, "SPY": 0.10},
}
EXPLICACIONES = {
    "Conservador": "Prioriza preservar el capital frente a la inflación, con baja exposición a la volatilidad bursátil.",
    "Moderado": "Combina la certeza de la tasa en pesos con la ganancia de capital potencial de bonos y acciones.",
    "Agresivo": "Busca maximizar el retorno aceptando la alta volatilidad de acciones y cripto.",
}


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
            pass  # si un ticker falla, seguimos con el resto

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


with st.spinner("Descargando cotizaciones..."):
    precios, retornos = cargar_datos(plazo_dias, TICKERS)

if retornos is None:
    st.error("No se pudieron descargar datos de Yahoo Finance. Probá de nuevo en unos minutos.")
    st.stop()

evolucion = (1 + retornos).cumprod() * 100

resumen = pd.DataFrame({
    "Rendimiento Acumulado (%)": (evolucion.iloc[-1] - 100).round(2),
    "Volatilidad Anualizada (%)": (retornos.std() * np.sqrt(252) * 100).round(2),
}).sort_values("Rendimiento Acumulado (%)", ascending=False)
resumen["Categoría"] = [CATEGORIA.get(t, "Otro") for t in resumen.index]

# Rendimiento de la cartera sugerida (rebalanceo diario, simple)
pesos = {k: v for k, v in CARTERAS[perfil].items() if k in retornos.columns}
if not pesos:  # ningún activo de la cartera típica está seleccionado
    pesos = {k: 1.0 for k in retornos.columns}
total = sum(pesos.values())
pesos = {k: v / total for k, v in pesos.items()}
ret_cartera = sum(retornos[k] * w for k, w in pesos.items())
evol_cartera = (1 + ret_cartera).cumprod() * 100
rend_cartera = evol_cartera.iloc[-1] - 100
monto_final = monto * (1 + rend_cartera / 100)

# ==============================================================================
# 4. INTERFAZ
# ==============================================================================
st.subheader("📊 Diagnóstico de Mercado")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Monto a invertir", f"$ {monto:,.0f}")
c2.metric("Perfil", perfil)
c3.metric("Activo líder", resumen.index[0], f"{resumen.iloc[0, 0]:+.2f}%")
c4.metric("Cartera sugerida (histórico)", f"$ {monto_final:,.0f}", f"{rend_cartera:+.2f}%")

fig = px.line(
    evolucion, x=evolucion.index, y=evolucion.columns,
    title="Evolución comparativa (Base 100 = inicio del período)",
    labels={"value": "Capital (base 100)", "variable": "Activo", "index": "Fecha"},
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("📋 Rendimiento y riesgo por activo")
st.dataframe(resumen.style.highlight_max(subset=["Rendimiento Acumulado (%)"], color="#c6f6d5"), use_container_width=True)

# ==============================================================================
# 5. DICTAMEN AUTOMÁTICO (según el perfil de riesgo)
# ==============================================================================
st.subheader("🤖 Recomendación de FinanBot")

top3 = resumen.head(3).index.tolist()
mas_volatil = resumen["Volatilidad Anualizada (%)"].idxmax()
ipc_acum = resumen.loc["IPC_UVA", "Rendimiento Acumulado (%)"]
cartera_txt = " + ".join(f"{int(round(w * 100))}% {k}" for k, w in pesos.items())

st.info(f"""
**Estrategia:** para **$ {monto:,.0f} ARS** y perfil **{perfil}**, la cartera sugerida es **{cartera_txt}**.

**Mercado:** en los últimos {plazo_dias} días, los mejores activos fueron **{', '.join(top3)}**. La inflación proyectada del período es **{ipc_acum}%**.

**Riesgo:** {EXPLICACIONES[perfil]} El activo más volátil fue **{mas_volatil}**, conviene seguirlo de cerca.
""")

st.caption("⚠️ Herramienta educativa. Rendimientos pasados no garantizan rendimientos futuros. No es asesoramiento financiero.")
