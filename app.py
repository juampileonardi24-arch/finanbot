import datetime as dt
import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf
import anthropic

# ==============================================================================
# 1. CONFIGURACIÓN DE LA PÁGINA
# ==============================================================================
st.set_page_config(page_title="FinanBot IA", page_icon="📈", layout="wide")

st.title("📈 FinanBot IA: Asistente Financiero & Cobertura")
st.markdown(
    "Analiza el rendimiento de una canasta de activos (bonos, acciones, LECAP, "
    "BADLAR, inflación, SPY y BTC) y arma un diagnóstico según tu perfil de riesgo."
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
TICKERS = ["AL30.BA", "GD30.BA", "GGAL.BA", "YPF.BA", "PAMP.BA", "BMA.BA", "SPY", "BTC-USD"]

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
def cargar_datos(dias: int):
    fin = dt.date.today()
    inicio = fin - dt.timedelta(days=dias)

    series = {}
    for t in TICKERS:
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
    precios, retornos = cargar_datos(plazo_dias)

if retornos is None:
    st.error("No se pudieron descargar datos de Yahoo Finance. Probá de nuevo en unos minutos.")
    st.stop()

evolucion = (1 + retornos).cumprod() * 100

resumen = pd.DataFrame({
    "Rendimiento Acumulado (%)": (evolucion.iloc[-1] - 100).round(2),
    "Volatilidad Anualizada (%)": (retornos.std() * np.sqrt(252) * 100).round(2),
}).sort_values("Rendimiento Acumulado (%)", ascending=False)

# Rendimiento de la cartera sugerida (rebalanceo diario, simple)
pesos = {k: v for k, v in CARTERAS[perfil].items() if k in retornos.columns}
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
st.dataframe(resumen.style.highlight_max(axis=0, color="#c6f6d5"), use_container_width=True)

# ==============================================================================
# 5. DICTAMEN (reglas; acá podrías enchufar una API de IA real)
# ==============================================================================
st.subheader("🤖 Recomendación de FinanBot IA")

top3 = resumen.head(3).index.tolist()
mas_volatil = resumen["Volatilidad Anualizada (%)"].idxmax()
ipc_acum = resumen.loc["IPC_UVA", "Rendimiento Acumulado (%)"]
cartera_txt = " + ".join(f"{int(round(w * 100))}% {k}" for k, w in pesos.items())

st.info(f"""
**Estrategia:** para **$ {monto:,.0f} ARS** y perfil **{perfil}**, la cartera sugerida es **{cartera_txt}**.

**Mercado:** en los últimos {plazo_dias} días, los mejores activos fueron **{', '.join(top3)}**. La inflación proyectada del período es **{ipc_acum}%**.

**Riesgo:** {EXPLICACIONES[perfil]} El activo más volátil fue **{mas_volatil}**, conviene seguirlo de cerca.
""")

# ==============================================================================
# 6. ASESOR CON IA REAL (Claude). Si no hay API key, la app sigue funcionando.
# ==============================================================================
st.subheader("💬 Preguntale a la IA")


def obtener_api_key():
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        return os.environ.get("ANTHROPIC_API_KEY")


def consultar_ia(pregunta: str, api_key: str) -> str:
    client = anthropic.Anthropic(api_key=api_key)
    contexto = f"""
Perfil del inversor: {perfil}
Monto a invertir: $ {monto:,.0f} ARS
Período histórico analizado: {plazo_dias} días
Cartera sugerida por la app: {cartera_txt}
Rendimiento histórico de esa cartera: {rend_cartera:+.2f}%

Tabla de rendimiento y volatilidad por activo:
{resumen.to_string()}
"""
    system = (
        "Sos FinanBot, un asistente financiero educativo para inversores argentinos. "
        "Respondé en español rioplatense, claro y concreto, basándote en los datos que te paso. "
        "Explicá riesgos, no prometas rendimientos futuros y aclará que no es asesoramiento financiero. "
        "Máximo 250 palabras."
    )
    msg = client.messages.create(
        model="claude-sonnet-5-5",
        max_tokens=1000,
        system=system,
        messages=[{"role": "user", "content": f"{contexto}\n\nConsulta: {pregunta}"}],
    )
    return msg.content[0].text


api_key = obtener_api_key()
if not api_key:
    st.caption(
        "Para activar el asesor con IA, cargá tu `ANTHROPIC_API_KEY` en los secrets de la app "
        "(ver instrucciones). Mientras tanto se muestra el dictamen automático de arriba."
    )
else:
    pregunta = st.text_input(
        "Tu consulta",
        value="Analizá la cartera sugerida y decime qué riesgos tiene.",
    )
    if st.button("Consultar a FinanBot"):
        with st.spinner("Pensando..."):
            try:
                st.markdown(consultar_ia(pregunta, api_key))
            except Exception as e:
                st.error(f"No se pudo consultar a la IA: {e}")

st.caption("⚠️ Herramienta educativa. Rendimientos pasados no garantizan rendimientos futuros. No es asesoramiento financiero.")
