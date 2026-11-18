import streamlit as st

# ==============================================================================
# NÚCLEO DE CÁLCULO (se usa igual en la web y en el notebook)
# ==============================================================================
import datetime as dt
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf

# Catálogo de activos por categoría (ticker de Yahoo Finance -> nombre).
# Los tickers con sufijo .BA cotizan en pesos; el resto cotiza en dólares y se
# convierte a pesos con el dólar CCL implícito.
CATALOGO = {
    "Bonos soberanos": {
        "AL29.BA": "Bonar 2029", "AL30.BA": "Bonar 2030", "AL35.BA": "Bonar 2035",
        "AE38.BA": "Bonar 2038", "AL41.BA": "Bonar 2041",
        "GD29.BA": "Global 2029", "GD30.BA": "Global 2030", "GD35.BA": "Global 2035",
        "GD38.BA": "Global 2038", "GD41.BA": "Global 2041", "GD46.BA": "Global 2046",
    },
    "Acciones argentinas": {
        "GGAL.BA": "Galicia", "YPF.BA": "YPF", "PAMP.BA": "Pampa Energía",
        "BMA.BA": "Banco Macro", "SUPV.BA": "Supervielle", "BYMA.BA": "BYMA",
        "CEPU.BA": "Central Puerto", "TGSU2.BA": "Transportadora Gas del Sur",
        "TGNO4.BA": "Transportadora Gas del Norte", "TXAR.BA": "Ternium Argentina",
        "ALUA.BA": "Aluar", "LOMA.BA": "Loma Negra", "CRES.BA": "Cresud",
        "EDN.BA": "Edenor", "TECO2.BA": "Telecom Argentina", "COME.BA": "Soc. Comercial del Plata",
        "MIRG.BA": "Mirgor", "VALO.BA": "Grupo Valores",
    },
    "CEDEARs (en pesos)": {
        "AAPL.BA": "Apple", "MSFT.BA": "Microsoft", "GOOGL.BA": "Alphabet (Google)",
        "AMZN.BA": "Amazon", "NVDA.BA": "Nvidia", "TSLA.BA": "Tesla",
        "META.BA": "Meta", "NFLX.BA": "Netflix", "AMD.BA": "AMD",
        "MELI.BA": "Mercado Libre", "KO.BA": "Coca-Cola", "DIS.BA": "Disney",
        "WMT.BA": "Walmart", "JPM.BA": "JPMorgan", "XOM.BA": "Exxon Mobil",
        "MCD.BA": "McDonald's",
    },
    "ETFs y oro (USD)": {
        "SPY": "S&P 500", "QQQ": "Nasdaq 100", "DIA": "Dow Jones", "IWM": "Small Caps EE.UU.",
        "EEM": "Mercados emergentes", "EWZ": "Brasil", "GLD": "Oro", "SLV": "Plata",
        "TLT": "Bonos del Tesoro 20+ años",
    },
    "Criptoactivos (USD)": {
        "BTC-USD": "Bitcoin", "ETH-USD": "Ethereum", "SOL-USD": "Solana",
        "BNB-USD": "BNB", "XRP-USD": "XRP", "ADA-USD": "Cardano",
    },
}
SELECCION_INICIAL = {"AL30.BA", "GD30.BA", "GGAL.BA", "YPF.BA", "PAMP.BA", "BMA.BA", "SPY", "BTC-USD"}

TASAS = ["BADLAR", "LECAP", "IPC_UVA"]  # instrumentos de referencia (siempre incluidos)
CATEGORIA = {t: cat for cat, d in CATALOGO.items() for t in d}
CATEGORIA.update({"BADLAR": "Tasas en pesos", "LECAP": "Tasas en pesos", "IPC_UVA": "Inflación"})


def es_usd(ticker):
    """Los tickers sin sufijo .BA cotizan en dólares (SPY, BTC-USD, ETH-USD)."""
    return not ticker.endswith(".BA")


def descargar_precios(dias, tickers):
    """Baja precios de cierre y pasa todo a PESOS.

    Los activos en USD se convierten con el dólar CCL implícito (YPF.BA / YPF,
    porque 1 ADR de YPF = 1 acción). Devuelve (precios_ars, fallidos, ccl_actual).
    """
    tickers = list(tickers)
    necesita_ccl = any(es_usd(t) for t in tickers)
    a_bajar = set(tickers) | ({"YPF.BA", "YPF"} if necesita_ccl else set())

    hoy = dt.date.today()
    inicio, fin = hoy - dt.timedelta(days=dias), hoy + dt.timedelta(days=1)

    def bajar(t):
        try:
            s = yf.Ticker(t).history(start=inicio, end=fin)["Close"]
            if s.empty:
                return t, None
            s.index = s.index.tz_localize(None).normalize()
            return t, s[~s.index.duplicated(keep="last")]
        except Exception:
            return t, None  # si un ticker falla seguimos con el resto

    # Descarga en paralelo: con muchos activos es mucho más rápido que de a uno
    with ThreadPoolExecutor(max_workers=8) as pool:
        resultados = list(pool.map(bajar, sorted(a_bajar)))
    series = {t: s for t, s in resultados if s is not None}

    fallidos = [t for t in tickers if t not in series]
    if not series:
        return None, fallidos, None

    # Calendario diario completo (7 días): así las tasas y los activos que cotizan
    # fines de semana (cripto) quedan en la misma escala de tiempo.
    precios = pd.DataFrame(series).sort_index().asfreq("D").ffill().bfill()

    ccl_actual = None
    if necesita_ccl:
        if "YPF.BA" in precios and "YPF" in precios:
            ccl = precios["YPF.BA"] / precios["YPF"]
            ccl_actual = float(ccl.iloc[-1])
            for t in tickers:
                if es_usd(t) and t in precios:
                    precios[t] = precios[t] * ccl
        else:  # sin CCL no se puede comparar en pesos: se descartan los activos en USD
            for t in tickers:
                if es_usd(t):
                    fallidos.append(t)
                    series.pop(t, None)

    columnas = [t for t in tickers if t in series and t in precios.columns]
    if not columnas:
        return None, sorted(set(fallidos)), ccl_actual
    return precios[columnas], sorted(set(fallidos)), ccl_actual


def calcular_retornos(precios, tna_badlar, tem_lecap, ipc_mensual):
    """Retornos diarios de los activos + las tres series de referencia en pesos."""
    ret = precios.pct_change()
    ret["BADLAR"] = tna_badlar / 365
    ret["LECAP"] = (1 + tem_lecap) ** (1 / 30) - 1
    ret["IPC_UVA"] = (1 + ipc_mensual) ** (1 / 30) - 1
    return ret.fillna(0)


def calcular_resumen(retornos):
    """Evolución base 100 y tabla de métricas por activo."""
    evolucion = (1 + retornos).cumprod() * 100
    rend = evolucion.iloc[-1] - 100
    vol = retornos.std() * np.sqrt(365) * 100
    caida_max = ((evolucion / evolucion.cummax()) - 1).min() * 100
    rend_anual = ((evolucion.iloc[-1] / 100) ** (365 / len(retornos)) - 1) * 100

    resumen = pd.DataFrame({
        "Rendimiento Acumulado (%)": rend.round(2),
        "Volatilidad Anualizada (%)": vol.round(2),
        "Caída Máxima (%)": caida_max.round(2),
        "Retorno/Riesgo": (rend_anual / vol.replace(0, np.nan)).round(2),
    }).sort_values("Rendimiento Acumulado (%)", ascending=False)
    resumen["Categoría"] = [CATEGORIA.get(t, "Otro") for t in resumen.index]
    return evolucion, resumen


def construir_cartera_dinamica(perfil, plazo, columnas):
    """Pesos sugeridos según plazo de inversión y perfil de riesgo."""
    pesos = {}

    # 1) Base según el plazo
    if plazo <= 60:       # corto plazo: liquidez y tasa
        pesos.update({"LECAP": 0.40, "BADLAR": 0.30, "IPC_UVA": 0.20, "AL30.BA": 0.10})
    elif plazo <= 180:    # mediano plazo: equilibrio
        pesos.update({"LECAP": 0.25, "AL30.BA": 0.25, "GGAL.BA": 0.20, "YPF.BA": 0.15, "SPY": 0.15})
    else:                 # largo plazo: renta variable y cripto
        pesos.update({"GGAL.BA": 0.25, "YPF.BA": 0.25, "AL30.BA": 0.20, "SPY": 0.15, "BTC-USD": 0.15})

    # 2) Ajuste según el perfil
    if perfil == "Conservador":
        pesos["LECAP"] = pesos.get("LECAP", 0) + 0.25
        pesos["BADLAR"] = pesos.get("BADLAR", 0) + 0.20
        for k in ["BTC-USD", "GGAL.BA", "YPF.BA"]:
            if k in pesos:
                pesos[k] *= 0.3
    elif perfil == "Agresivo":
        for k in ["BTC-USD", "GGAL.BA", "YPF.BA", "SPY"]:
            if k in pesos:
                pesos[k] *= 1.8
        for k in ["LECAP", "BADLAR"]:
            if k in pesos:
                pesos[k] *= 0.2

    # 3) Solo activos disponibles en los datos
    pesos = {k: v for k, v in pesos.items() if k in columnas and v > 0}
    if not pesos:  # ninguno de los activos típicos fue seleccionado: partes iguales
        pesos = {k: 1.0 for k in columnas if k != "IPC_UVA"}

    total = sum(pesos.values())
    return {k: v / total for k, v in pesos.items()}


def simular_cartera(retornos, pesos):
    """Backtest de la cartera (rebalanceo diario a los pesos objetivo)."""
    ret = sum(retornos[k] * w for k, w in pesos.items())
    evolucion = (1 + ret).cumprod() * 100
    return ret, evolucion


def tabla_distribucion(pesos, monto):
    """Cuántos pesos van a cada activo."""
    return pd.DataFrame({
        "Activo": list(pesos),
        "Peso (%)": [round(w * 100, 1) for w in pesos.values()],
        "Monto (ARS)": [f"$ {monto * w:,.0f}" for w in pesos.values()],
    })


def grafico_evolucion(evolucion, evol_cartera, plazo):
    fig = go.Figure()
    for col in evolucion.columns:
        fig.add_trace(go.Scatter(
            x=evolucion.index, y=evolucion[col], name=col, mode="lines", opacity=0.75,
            line=dict(width=1.5, dash="dash" if col == "IPC_UVA" else "solid"),
        ))
    fig.add_trace(go.Scatter(
        x=evol_cartera.index, y=evol_cartera, name="CARTERA SUGERIDA",
        mode="lines", line=dict(width=4, color="#e11d48"),
    ))
    fig.update_layout(
        title=f"Evolución comparativa en pesos ({plazo} días, base 100)",
        xaxis_title="Fecha", yaxis_title="Capital (base 100)",
        legend_title="Activo", hovermode="x unified",
    )
    return fig


def grafico_torta(df_dist):
    return px.pie(df_dist, names="Activo", values="Peso (%)", hole=0.45,
                  title="Distribución sugerida del capital")


def generar_dictamen(perfil, plazo, monto, pesos, resumen, rend_cartera):
    """Texto del diagnóstico (Markdown). Usa 'ARS' en vez de '$' para no activar fórmulas."""
    activos = resumen[~resumen.index.isin(TASAS)]
    top3 = ", ".join(activos.head(3).index)
    mas_volatil = activos["Volatilidad Anualizada (%)"].idxmax()
    ipc_acum = resumen.loc["IPC_UVA", "Rendimiento Acumulado (%)"]
    monto_final = monto * (1 + rend_cartera / 100)
    ganancia = monto_final - monto
    cartera_txt = " + ".join(f"{w * 100:.0f}% {k}" for k, w in pesos.items())

    if rend_cartera > ipc_acum:
        cobertura = (f"La cartera superó a la inflación del período ({ipc_acum:.2f}%) "
                     f"con un rendimiento real positivo de **{rend_cartera - ipc_acum:.2f} puntos**.")
    else:
        cobertura = (f"La cartera rindió **{rend_cartera:.2f}%** y quedó por debajo de la inflación "
                     f"del período ({ipc_acum:.2f}%). Conviene revisar la ponderación hacia activos de mayor cobertura.")

    return f"""**Cartera sugerida** para perfil **{perfil}** y horizonte de **{plazo} días**:

👉 **{cartera_txt}**

**Resultado histórico (backtest):** si hubieras invertido **ARS {monto:,.0f}** hace {plazo} días con esta combinación, hoy tendrías **ARS {monto_final:,.0f}** ({ganancia:+,.0f} ARS, {rend_cartera:+.2f}%).

**Cobertura:** {cobertura}

**Mercado:** los activos más rentables del período fueron **{top3}**, y el de mayor fluctuación fue **{mas_volatil}**.

*Son datos pasados: no constituyen una proyección ni garantizan resultados futuros.*"""



# ==============================================================================
# INTERFAZ WEB (Streamlit)
# ==============================================================================
st.set_page_config(page_title="FinanBot IA", page_icon="📈", layout="wide")

# La descarga se guarda en caché una hora para no volver a pedir datos a Yahoo
descargar_cacheado = st.cache_data(ttl=3600, show_spinner=False)(descargar_precios)

st.title("📈 FinanBot IA: Asistente Financiero & Cobertura Dinámica")
st.markdown(
    "Seleccioná tus activos, ajustá el **monto**, el **plazo** y las **tasas de referencia** "
    "para obtener una cartera sugerida según tu perfil de riesgo y el desempeño histórico del mercado."
)

# ------------------------------------------------------------------------------
# Barra lateral: parámetros
# ------------------------------------------------------------------------------
st.sidebar.header("⚙️ Parámetros del Inversor")
monto = st.sidebar.number_input(
    "Monto a invertir ($ ARS)", min_value=10_000, max_value=100_000_000, value=1_000_000, step=50_000,
)
plazo_dias = st.sidebar.slider("Horizonte de inversión (días)", 30, 365, 90, 30)
perfil = st.sidebar.selectbox("Perfil de riesgo", ["Conservador", "Moderado", "Agresivo"])

st.sidebar.divider()
st.sidebar.subheader("🏦 Tasas de referencia")
tna_badlar = st.sidebar.number_input("TNA BADLAR (%)", 0.0, 300.0, 40.0, 0.5) / 100
tem_lecap = st.sidebar.number_input("TEM LECAP (%)", 0.0, 30.0, 3.8, 0.1) / 100
ipc_mensual = st.sidebar.number_input("Inflación mensual (%)", 0.0, 30.0, 3.5, 0.1) / 100

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
st.sidebar.caption("BADLAR, LECAP e inflación se incluyen siempre como referencia. "
                   "Los activos en USD se pasan a pesos con el dólar CCL.")

if not seleccion:
    st.warning("Seleccioná al menos un activo en la barra lateral para comenzar.")
    st.stop()
TICKERS = tuple(sorted(seleccion))

# ------------------------------------------------------------------------------
# Datos y cálculos
# ------------------------------------------------------------------------------
with st.spinner("Descargando cotizaciones del mercado..."):
    precios, fallidos, ccl_actual = descargar_cacheado(plazo_dias, TICKERS)

if fallidos:
    st.warning("Sin datos (o sin poder convertir a pesos) para: **" + ", ".join(fallidos)
               + "**. Se excluyeron del análisis.")
if precios is None:
    st.error("No se pudieron descargar cotizaciones. Intentá nuevamente en unos minutos.")
    st.stop()

retornos = calcular_retornos(precios, tna_badlar, tem_lecap, ipc_mensual)
evolucion, resumen = calcular_resumen(retornos)
pesos = construir_cartera_dinamica(perfil, plazo_dias, list(retornos.columns))
ret_cartera, evol_cartera = simular_cartera(retornos, pesos)

rend_cartera = evol_cartera.iloc[-1] - 100
monto_final = monto * (1 + rend_cartera / 100)
ganancia = monto_final - monto
activos_puros = resumen[~resumen.index.isin(TASAS)]

# ------------------------------------------------------------------------------
# Resultados
# ------------------------------------------------------------------------------
st.subheader("📊 Diagnóstico y Rendimiento (backtest)")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Monto a invertir", f"$ {monto:,.0f} ARS")
c2.metric("Horizonte analizado", f"{plazo_dias} días")
c3.metric("Activo líder", activos_puros.index[0], f"{activos_puros.iloc[0, 0]:+.2f}%")
c4.metric(
    f"Si hubieras invertido hace {plazo_dias} días",
    f"$ {monto_final:,.0f} ARS",
    f"{ganancia:+,.0f} ARS ({rend_cartera:+.2f}%)",
)
if ccl_actual:
    st.caption(f"Los activos en USD (ETFs y cripto) se convirtieron a pesos con el dólar CCL implícito "
               f"(YPF.BA / YPF). Valor actual: $ {ccl_actual:,.2f}.")

st.plotly_chart(grafico_evolucion(evolucion, evol_cartera, plazo_dias), width="stretch")

st.subheader("💰 ¿Cómo repartir tu capital?")
df_dist = tabla_distribucion(pesos, monto)
col_a, col_b = st.columns([1, 1])
col_a.plotly_chart(grafico_torta(df_dist), width="stretch")
col_b.dataframe(df_dist, hide_index=True, width="stretch")

st.subheader("📋 Resumen de desempeño por activo")
st.dataframe(
    resumen.style.highlight_max(subset=["Rendimiento Acumulado (%)"], color="#c6f6d5").format(na_rep="—", precision=2),
    width="stretch",
)
st.download_button(
    "⬇️ Descargar tabla (CSV)",
    data=resumen.to_csv().encode("utf-8"),
    file_name="finanbot_resumen.csv",
    mime="text/csv",
)

# ------------------------------------------------------------------------------
# Dictamen
# ------------------------------------------------------------------------------
st.subheader("🤖 Recomendación de FinanBot")
st.info(generar_dictamen(perfil, plazo_dias, monto, pesos, resumen, rend_cartera))

with st.expander("📘 Metodología y supuestos"):
    st.markdown("""
- **Datos:** precios de cierre de Yahoo Finance (`yfinance`) en el plazo elegido.
- **Moneda:** todo se expresa en pesos. Los activos en USD se convierten con el dólar CCL implícito (YPF.BA / YPF).
- **Tasas:** BADLAR (TNA), LECAP (TEM) e inflación mensual son valores que cargás vos; se capitalizan día a día.
- **Calendario:** se usan los 7 días de la semana, repitiendo el último precio cuando el mercado está cerrado.
- **Cartera:** pesos según plazo y perfil, con rebalanceo diario (simplificación). No incluye comisiones ni impuestos.
- **Métricas:** volatilidad anualizada (raíz de 365), caída máxima desde el pico y retorno/riesgo (retorno anualizado dividido volatilidad).
- **Alcance:** es un *backtest* con datos pasados, no una proyección.
""")

st.caption("⚠️ Herramienta educativa. Rendimientos pasados no garantizan rendimientos futuros. No constituye asesoramiento financiero.")
