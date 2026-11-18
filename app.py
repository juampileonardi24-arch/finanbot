import streamlit as st

# ==============================================================================
# NÚCLEO DE CÁLCULO (se usa igual en la web y en el notebook)
# ==============================================================================
import datetime as dt
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
from scipy.optimize import minimize
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


# Pares (acción local, ADR en EE.UU., acciones locales por ADR) para estimar el dólar CCL
PARES_CCL = [("YPF.BA", "YPF", 1), ("GGAL.BA", "GGAL", 10), ("BMA.BA", "BMA", 10)]
DOLAR_OFICIAL = "ARS=X"  # respaldo: dólar oficial según Yahoo Finance


def calcular_dolar(precios):
    """Serie de pesos por dólar. Devuelve (serie, fuente) o (None, None).

    Primero usa el CCL implícito (precio local x ratio / precio del ADR). Con varios
    pares toma la mediana de cada día, así un dato raro no distorsiona el resultado.
    Si no hay ningún par disponible, usa el dólar oficial como último recurso.
    """
    pares, usados = [], []
    for local, adr, ratio in PARES_CCL:
        if local in precios and adr in precios:
            pares.append(precios[local] * ratio / precios[adr])
            usados.append(adr)
    if pares:
        return pd.concat(pares, axis=1).median(axis=1), "dólar CCL implícito (" + ", ".join(usados) + ")"
    if DOLAR_OFICIAL in precios:
        return precios[DOLAR_OFICIAL], "dólar oficial de Yahoo (no se pudo calcular el CCL)"
    return None, None


def descargar_precios(dias, tickers):
    """Baja precios de cierre y pasa todo a PESOS.

    Los activos en USD se convierten con el dólar CCL implícito (ver calcular_dolar).
    Devuelve (precios_ars, fallidos, dolar) donde dolar = {"valor", "fuente", "serie"}.
    """
    tickers = list(tickers)
    necesita_dolar = any(es_usd(t) for t in tickers)
    auxiliares = {DOLAR_OFICIAL}
    for local, adr, _ in PARES_CCL:
        auxiliares |= {local, adr}
    a_bajar = set(tickers) | auxiliares  # el dólar también se usa como alternativa de comparación

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
    if not [t for t in tickers if t in series]:
        return None, fallidos, None

    # Calendario diario completo (7 días): así las tasas y los activos que cotizan
    # fines de semana (cripto) quedan en la misma escala de tiempo.
    precios = pd.DataFrame(series).sort_index().asfreq("D").ffill().bfill()

    dolar = None
    serie_usd, fuente = calcular_dolar(precios)
    if serie_usd is not None:
        dolar = {"valor": float(serie_usd.iloc[-1]), "fuente": fuente, "serie": serie_usd}
        if necesita_dolar:
            for t in tickers:
                if es_usd(t) and t in precios.columns:
                    precios[t] = precios[t] * serie_usd
    elif necesita_dolar:  # sin dólar no se puede comparar en pesos: se descartan los activos en USD
        for t in tickers:
            if es_usd(t):
                fallidos.append(t)
                series.pop(t, None)

    columnas = [t for t in tickers if t in series and t in precios.columns]
    if not columnas:
        return None, sorted(set(fallidos)), dolar
    return precios[columnas], sorted(set(fallidos)), dolar


# ------------------------------------------------------------------------------
# Tasas de referencia desde la API oficial del BCRA (Estadísticas Monetarias v4.0)
# ------------------------------------------------------------------------------
URL_BCRA = "https://api.bcra.gob.ar/estadisticas/v4.0/monetarias"
ID_BADLAR = 7       # BADLAR de bancos privados, % nominal anual (serie diaria)
ID_INFLACION = 27   # Inflación mensual, % (serie mensual)


def _ultimo_valor_bcra(id_variable, dias_atras):
    """Último valor publicado de una variable del BCRA: (valor, fecha, certificado_ok)."""
    hoy = dt.date.today()
    params = {"desde": (hoy - dt.timedelta(days=dias_atras)).isoformat(),
              "hasta": hoy.isoformat(), "limit": 1000}
    url = f"{URL_BCRA}/{id_variable}"
    verificado = True
    try:
        r = requests.get(url, params=params, timeout=15)
    except requests.exceptions.SSLError:
        # A veces el servidor del BCRA no entrega la cadena de certificados completa.
        # Son datos públicos de solo lectura, así que se reintenta sin verificar y se avisa.
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        verificado = False
        r = requests.get(url, params=params, timeout=15, verify=False)
    r.raise_for_status()
    detalle = r.json()["results"][0]["detalle"]
    ultimo = max(detalle, key=lambda d: d["fecha"])
    return float(ultimo["valor"]), ultimo["fecha"], verificado


def obtener_tasas_bcra():
    """Lee BADLAR (TNA %) e inflación mensual (%) del BCRA.

    Devuelve {"badlar": (valor, fecha) | None, "ipc": (valor, fecha) | None, "verificado": bool}.
    Si algo falla devuelve None en ese dato y la app usa valores por defecto.
    """
    out = {"badlar": None, "ipc": None, "verificado": True}
    pedidos = (("badlar", ID_BADLAR, 20, (1, 300)), ("ipc", ID_INFLACION, 120, (-5, 50)))
    for clave, id_variable, dias, (minimo, maximo) in pedidos:
        try:
            valor, fecha, ok = _ultimo_valor_bcra(id_variable, dias)
            if minimo <= valor <= maximo:  # descarta datos absurdos
                out[clave] = (valor, fecha)
                out["verificado"] = out["verificado"] and ok
        except Exception:
            pass
    return out


def estimar_tem_lecap(tna_badlar_pct):
    """TEM de las LECAP estimada desde la BADLAR (en %). No hay fuente gratuita oficial:
    es una aproximación y conviene corregirla con la tasa de la última licitación."""
    return tna_badlar_pct * 30 / 365


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


def grafico_evolucion(evolucion, carteras, plazo, dolar_serie=None):
    """Líneas de cada activo + las carteras (gruesas) + el dólar como referencia."""
    fig = go.Figure()
    for col in evolucion.columns:
        fig.add_trace(go.Scatter(
            x=evolucion.index, y=evolucion[col], name=col, mode="lines", opacity=0.7,
            line=dict(width=1.5, dash="dash" if col == "IPC_UVA" else "solid"),
        ))
    if dolar_serie is not None:
        base = dolar_serie.reindex(evolucion.index).ffill().bfill()
        fig.add_trace(go.Scatter(x=base.index, y=base / base.iloc[0] * 100, name="Dólar (CCL)",
                                 mode="lines", line=dict(width=2.5, dash="dot", color="#6b7280")))
    estilos = [dict(width=4, color="#e11d48"), dict(width=3, color="#2563eb", dash="dashdot")]
    for (nombre, evol), estilo in zip(carteras.items(), estilos):
        fig.add_trace(go.Scatter(x=evol.index, y=evol, name=nombre, mode="lines", line=estilo))
    fig.update_layout(
        title=f"Evolución comparativa en pesos ({plazo} días, base 100)",
        xaxis_title="Fecha", yaxis_title="Capital (base 100)",
        legend_title="Activo", hovermode="x unified",
    )
    return fig


def grafico_torta(df_dist):
    return px.pie(df_dist, names="Activo", values="Peso (%)", hole=0.45,
                  title="Distribución sugerida del capital")


def generar_dictamen(perfil, plazo, monto, pesos, resumen, rend_cartera, metodo_txt=""):
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

    metodo = f"\n\n**Método:** {metodo_txt}" if metodo_txt else ""
    return f"""**Cartera sugerida** para perfil **{perfil}** y horizonte de **{plazo} días**:

👉 **{cartera_txt}**{metodo}

**Resultado histórico (backtest):** si hubieras invertido **ARS {monto:,.0f}** hace {plazo} días con esta combinación, hoy tendrías **ARS {monto_final:,.0f}** ({ganancia:+,.0f} ARS, {rend_cartera:+.2f}%).

**Cobertura:** {cobertura}

**Mercado:** los activos más rentables del período fueron **{top3}**, y el de mayor fluctuación fue **{mas_volatil}**.

*Son datos pasados: no constituyen una proyección ni garantizan resultados futuros.*"""



# ------------------------------------------------------------------------------
# Cartera optimizada (media-varianza) con los activos que elegiste
# ------------------------------------------------------------------------------
# objetivo: "min_vol" = mínima volatilidad | "sharpe" = mejor retorno por unidad de riesgo
# tope: máximo por activo | colchon: parte del capital que va a tasas (LECAP, BADLAR, inflación)
PERFILES_OPT = {
    "Conservador": {"objetivo": "min_vol", "tope": 0.25, "colchon": 0.60},
    "Moderado": {"objetivo": "sharpe", "tope": 0.30, "colchon": 0.30},
    "Agresivo": {"objetivo": "sharpe", "tope": 0.40, "colchon": 0.05},
}
COLCHON = {"LECAP": 0.5, "BADLAR": 0.3, "IPC_UVA": 0.2}  # cómo se reparte el colchón en tasas


def _resolver(mu, cov, objetivo, tope, rf):
    """Pesos long-only que minimizan la volatilidad o maximizan el ratio de Sharpe."""
    k = len(mu)
    if k == 1:
        return np.array([1.0])
    tope = max(tope, 1.0 / k + 1e-6)  # con pocos activos el tope tiene que dejar sumar 100%
    w0 = np.full(k, 1.0 / k)
    if objetivo == "min_vol":
        f = lambda w: 100 * (w @ cov @ w)
    else:
        f = lambda w: -(w @ mu - rf) / np.sqrt(max(w @ cov @ w, 1e-12))
    res = minimize(f, w0, method="SLSQP", bounds=[(0.0, tope)] * k,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                   options={"maxiter": 500, "ftol": 1e-10})
    w = res.x if np.all(np.isfinite(res.x)) else w0
    w = np.clip(w, 0, None)
    w[w < 0.01] = 0  # se descartan posiciones insignificantes (menos de 1%)
    if w.sum() == 0:
        w = w0
    return w / w.sum()


def optimizar_cartera(retornos, perfil, plazo, n_nube=3000):
    """Cartera de mínima volatilidad / máximo Sharpe con los activos con riesgo elegidos.

    Para que no sea tan sensible al pasado: los retornos esperados se acercan a su promedio
    (shrinkage), la covarianza se regulariza y ningún activo pasa del tope del perfil.
    Una parte del capital (colchón) va a tasas según perfil y plazo.
    Devuelve un diccionario con los pesos y los datos para graficar, o None si no hay activos.
    """
    cfg = PERFILES_OPT[perfil]
    riesgosos = [c for c in retornos.columns if c not in TASAS and retornos[c].std() > 0]
    if not riesgosos:
        return None

    colchon = cfg["colchon"] + (0.15 if plazo <= 60 else (-0.10 if plazo > 180 else 0.0))
    colchon = float(min(max(colchon, 0.0), 0.9))

    R = retornos[riesgosos]
    mu = R.mean().values * 365
    mu = 0.5 * mu + 0.5 * mu.mean()            # achica los extremos de la historia reciente
    cov = R.cov().values * 365
    cov = 0.9 * cov + 0.1 * np.diag(np.diag(cov))  # regulariza la covarianza
    rf = float((1 + retornos["LECAP"].iloc[-1]) ** 365 - 1)

    w = _resolver(mu, cov, cfg["objetivo"], cfg["tope"], rf)

    pesos = {t: float(wi) * (1 - colchon) for t, wi in zip(riesgosos, w) if wi > 0}
    for t, parte in COLCHON.items():
        if t in retornos.columns:
            pesos[t] = pesos.get(t, 0.0) + colchon * parte
    total = sum(pesos.values())
    pesos = {t: v / total for t, v in pesos.items()}

    def punto(pw):  # (volatilidad %, retorno esperado %) de una combinación de activos riesgosos
        return float(np.sqrt(pw @ cov @ pw) * 100), float(pw @ mu * 100)

    nube = None
    if n_nube > 0:
        W = np.random.default_rng(0).dirichlet(np.full(len(riesgosos), 0.3), size=n_nube)
        nube = pd.DataFrame({"Volatilidad (%)": np.sqrt(np.einsum("ij,jk,ik->i", W, cov, W)) * 100,
                             "Retorno esperado (%)": (W @ mu) * 100})
    return {
        "pesos": pesos, "colchon": colchon, "objetivo": cfg["objetivo"], "tope": cfg["tope"],
        "riesgosos": riesgosos, "nube": nube,
        "optimo": punto(w), "igual": punto(np.full(len(riesgosos), 1.0 / len(riesgosos))),
    }


def describir_metodo(opt):
    objetivo = "mínima volatilidad" if opt["objetivo"] == "min_vol" else "mejor relación retorno/riesgo"
    return (f"optimización sobre tus {len(opt['riesgosos'])} activos con riesgo (objetivo: {objetivo}; "
            f"máximo {opt['tope']:.0%} por activo) más un colchón de {opt['colchon']:.0%} en tasas "
            f"(LECAP, BADLAR e inflación).")


def validar_fuera_de_muestra(retornos, perfil, plazo, pesos_reglas, corte=0.7):
    """Optimiza con el primer 70% del período y mide qué pasó en el 30% restante."""
    n = len(retornos)
    punto = int(n * corte)
    if n < 60 or punto < 30:
        return None
    entreno, prueba = retornos.iloc[:punto], retornos.iloc[punto:]
    opt = optimizar_cartera(entreno, perfil, plazo, n_nube=0)
    if opt is None:
        return None

    def rendimiento(pesos):
        pesos = {k: v for k, v in pesos.items() if k in prueba.columns}
        total = sum(pesos.values())
        if total == 0:
            return float("nan")
        diario = sum(prueba[k] * v / total for k, v in pesos.items())
        return float(((1 + diario).prod() - 1) * 100)

    return {
        "dias_prueba": len(prueba),
        "optimizada": rendimiento(opt["pesos"]),
        "reglas": rendimiento(pesos_reglas),
        "inflacion": float(((1 + prueba["IPC_UVA"]).prod() - 1) * 100),
    }


def comparar_alternativas(monto, evolucion, carteras, dolar_serie):
    """Tabla: ¿le gana la cartera al dólar, al plazo fijo, a las LECAP y a la inflación?"""
    ipc = evolucion["IPC_UVA"].iloc[-1] - 100
    filas = []

    def fila(nombre, nominal):
        real = ((1 + nominal / 100) / (1 + ipc / 100) - 1) * 100
        filas.append({"Alternativa": nombre, "Rendimiento nominal (%)": round(nominal, 2),
                      "Rendimiento real (%)": round(real, 2),
                      "Capital final (ARS)": f"$ {monto * (1 + nominal / 100):,.0f}"})

    for nombre, evol in carteras.items():
        fila(nombre, evol.iloc[-1] - 100)
    if dolar_serie is not None:
        base = dolar_serie.reindex(evolucion.index).ffill().bfill()
        fila("Dólar (CCL)", (base.iloc[-1] / base.iloc[0] - 1) * 100)
    fila("Plazo fijo (BADLAR)", evolucion["BADLAR"].iloc[-1] - 100)
    fila("LECAP", evolucion["LECAP"].iloc[-1] - 100)
    fila("Pesos sin invertir", 0.0)
    return pd.DataFrame(filas)


def grafico_alternativas(df):
    df = df.copy()
    df["Resultado"] = np.where(df["Rendimiento real (%)"] >= 0, "Le gana a la inflación", "Pierde contra la inflación")
    return px.bar(df, x="Alternativa", y="Rendimiento real (%)", color="Resultado",
                  color_discrete_map={"Le gana a la inflación": "#16a34a", "Pierde contra la inflación": "#dc2626"},
                  title="Rendimiento real (descontando la inflación del período)")


def grafico_frontera(opt):
    """Nube de carteras posibles con los activos elegidos y dónde queda la optimizada."""
    fig = go.Figure()
    nube = opt["nube"]
    fig.add_trace(go.Scatter(x=nube["Volatilidad (%)"], y=nube["Retorno esperado (%)"], mode="markers",
                             name="Carteras posibles", marker=dict(size=4, opacity=0.35, color="#9ca3af")))
    fig.add_trace(go.Scatter(x=[opt["igual"][0]], y=[opt["igual"][1]], mode="markers",
                             name="Partes iguales", marker=dict(size=12, symbol="diamond", color="#2563eb")))
    fig.add_trace(go.Scatter(x=[opt["optimo"][0]], y=[opt["optimo"][1]], mode="markers",
                             name="Optimizada", marker=dict(size=16, symbol="star", color="#e11d48")))
    fig.update_layout(title="Riesgo vs retorno de las carteras posibles (parte con riesgo, anualizado)",
                      xaxis_title="Volatilidad anualizada (%)", yaxis_title="Retorno esperado ajustado (%)")
    return fig


# ==============================================================================
# INTERFAZ WEB (Streamlit)
# ==============================================================================
st.set_page_config(page_title="FinanBot IA", page_icon="📈", layout="wide")

# La descarga se guarda en caché una hora para no volver a pedir datos a Yahoo
descargar_cacheado = st.cache_data(ttl=3600, show_spinner=False)(descargar_precios)
tasas_cacheado = st.cache_data(ttl=3600, show_spinner=False)(obtener_tasas_bcra)

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
metodo = st.sidebar.radio(
    "Método de cartera",
    ["Optimizada con mis activos", "Por reglas (plazo y perfil)"],
    help="La optimizada reparte el capital entre TODOS los activos que elegiste. "
         "La de reglas usa una fórmula fija con unos pocos activos.",
)

st.sidebar.divider()
st.sidebar.subheader("🏦 Tasas de referencia")
usar_auto = st.sidebar.toggle("Cargar tasas automáticamente (BCRA)", value=True)
tasas_auto = tasas_cacheado()
badlar_auto, ipc_auto = tasas_auto["badlar"], tasas_auto["ipc"]

# Valores iniciales: los del BCRA si se pudieron leer; si no, valores de ejemplo
tna_ini = round(badlar_auto[0], 2) if badlar_auto else 40.0
ipc_ini = round(ipc_auto[0], 2) if ipc_auto else 3.5
lecap_ini = round(estimar_tem_lecap(tna_ini), 2) if badlar_auto else 3.8

# La clave cambia con el interruptor para que los campos tomen el valor correcto
tna_badlar = st.sidebar.number_input(
    "TNA BADLAR (%)", 0.0, 300.0, tna_ini, 0.5,
    key=f"tna_{usar_auto}", disabled=usar_auto and bool(badlar_auto)) / 100
tem_lecap = st.sidebar.number_input(
    "TEM LECAP (%) · estimada desde BADLAR" if badlar_auto else "TEM LECAP (%)",
    0.0, 30.0, lecap_ini, 0.1, key=f"lecap_{usar_auto}") / 100
ipc_mensual = st.sidebar.number_input(
    "Inflación mensual (%)", 0.0, 30.0, ipc_ini, 0.1,
    key=f"ipc_{usar_auto}", disabled=usar_auto and bool(ipc_auto)) / 100

if usar_auto:
    partes = []
    if badlar_auto:
        partes.append(f"BADLAR {badlar_auto[0]:.2f}% TNA ({badlar_auto[1]})")
    if ipc_auto:
        partes.append(f"inflación {ipc_auto[0]:.2f}% mensual ({ipc_auto[1]})")
    if partes:
        st.sidebar.caption("Fuente: BCRA · " + " · ".join(partes)
                           + ". La LECAP es una estimación: corregila si conocés la de la última licitación.")
        if not tasas_auto["verificado"]:
            st.sidebar.caption("⚠️ No se pudo verificar el certificado de seguridad del BCRA.")
    else:
        st.sidebar.warning("No se pudo leer el BCRA. Se usan valores de ejemplo; podés editarlos apagando el interruptor.")
else:
    st.sidebar.caption("Modo manual: ingresá las tasas que quieras probar.")

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
    precios, fallidos, dolar = descargar_cacheado(plazo_dias, TICKERS)

if fallidos:
    st.warning("Sin datos (o sin poder convertir a pesos) para: **" + ", ".join(fallidos)
               + "**. Se excluyeron del análisis.")
if precios is None:
    st.error("No se pudieron descargar cotizaciones. Intentá nuevamente en unos minutos.")
    st.stop()

retornos = calcular_retornos(precios, tna_badlar, tem_lecap, ipc_mensual)
evolucion, resumen = calcular_resumen(retornos)
pesos_reglas = construir_cartera_dinamica(perfil, plazo_dias, list(retornos.columns))
opt = optimizar_cartera(retornos, perfil, plazo_dias)

if metodo.startswith("Optimizada") and opt is None:
    st.info("No hay activos con riesgo para optimizar (solo tasas). Se usa la cartera por reglas.")
usa_opt = metodo.startswith("Optimizada") and opt is not None
pesos = opt["pesos"] if usa_opt else pesos_reglas

_, evol_reglas = simular_cartera(retornos, pesos_reglas)
carteras = {}
if opt is not None:
    carteras["CARTERA OPTIMIZADA"] = simular_cartera(retornos, opt["pesos"])[1]
carteras["CARTERA POR REGLAS"] = evol_reglas
evol_cartera = carteras["CARTERA OPTIMIZADA"] if usa_opt else evol_reglas

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
if dolar and any(es_usd(t) for t in TICKERS):
    st.caption(f"Los activos en USD (ETFs y cripto) se pasaron a pesos con el {dolar['fuente']}. "
               f"Valor actual: $ {dolar['valor']:,.2f} por dólar.")

st.plotly_chart(grafico_evolucion(evolucion, carteras, plazo_dias, dolar["serie"] if dolar else None), width="stretch")

st.subheader("💰 ¿Cómo repartir tu capital?")
df_dist = tabla_distribucion(pesos, monto)
col_a, col_b = st.columns([1, 1])
col_a.plotly_chart(grafico_torta(df_dist), width="stretch")
col_b.dataframe(df_dist, hide_index=True, width="stretch")

st.subheader("🎯 Cartera optimizada con tus activos")
if opt is None:
    st.info("Elegí al menos un activo con riesgo (acciones, bonos, CEDEARs, ETFs o cripto) para poder optimizar.")
else:
    st.markdown(
        f"El optimizador busca la combinación de **tus {len(opt['riesgosos'])} activos** que, para el perfil "
        f"**{perfil}**, logra la {'**menor volatilidad**' if opt['objetivo'] == 'min_vol' else '**mejor relación retorno/riesgo**'}, "
        f"sin pasar de {opt['tope']:.0%} en un solo activo, y suma un **colchón de {opt['colchon']:.0%}** en tasas."
    )
    st.plotly_chart(grafico_frontera(opt), width="stretch")
    val = validar_fuera_de_muestra(retornos, perfil, plazo_dias, pesos_reglas)
    if val:
        st.markdown(f"**Prueba de validación:** se optimizó con el primer 70% del período y se midió qué pasó en los "
                    f"últimos {val['dias_prueba']} días, que el optimizador no vio.")
        v1, v2, v3 = st.columns(3)
        v1.metric("Cartera optimizada", f"{val['optimizada']:+.2f}%")
        v2.metric("Cartera por reglas", f"{val['reglas']:+.2f}%")
        v3.metric("Inflación del tramo", f"{val['inflacion']:+.2f}%")
        st.caption("Es una prueba corta y ruidosa: sirve para detectar si la optimización se pasó de ajustada al pasado, "
                   "no para predecir resultados.")
    else:
        st.caption("Con plazos menores a 60 días no hay datos suficientes para la prueba de validación.")

st.subheader("⚖️ ¿Le gana a las alternativas?")
df_alt = comparar_alternativas(monto, evolucion, carteras, dolar["serie"] if dolar else None)
alt_a, alt_b = st.columns([1, 1])
alt_a.dataframe(df_alt, hide_index=True, width="stretch")
alt_b.plotly_chart(grafico_alternativas(df_alt), width="stretch")
st.caption("Rendimiento real = rendimiento nominal descontando la inflación del período. "
           "Si falló el dato del dólar, esa fila no aparece.")

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
metodo_txt = describir_metodo(opt) if usa_opt else "cartera por reglas según el plazo y el perfil de riesgo."
st.info(generar_dictamen(perfil, plazo_dias, monto, pesos, resumen, rend_cartera, metodo_txt))

with st.expander("📘 Metodología y supuestos"):
    st.markdown("""
- **Datos:** precios de cierre de Yahoo Finance (`yfinance`) en el plazo elegido.
- **Moneda:** todo se expresa en pesos. Los activos en USD se convierten con el dólar CCL implícito (mediana de YPF, GGAL y BMA contra sus ADR); si no se puede calcular, se usa el dólar oficial de Yahoo.
- **Tasas:** BADLAR (TNA) e inflación mensual se leen de la API oficial del BCRA; la TEM de LECAP se estima desde BADLAR (no hay fuente gratuita oficial). Podés pasar a modo manual y se capitalizan día a día.
- **Calendario:** se usan los 7 días de la semana, repitiendo el último precio cuando el mercado está cerrado.
- **Cartera por reglas:** pesos fijos según plazo y perfil.
- **Cartera optimizada:** minimiza la volatilidad (conservador) o maximiza el ratio de Sharpe (moderado y agresivo) con tus activos, sin ventas en corto y con un tope por activo. Los retornos esperados se acercan a su promedio y la covarianza se regulariza para que no dependa tanto de unos pocos días. Un colchón en tasas depende del perfil y del plazo.
- **Supuestos comunes:** rebalanceo diario (simplificación), sin comisiones ni impuestos.
- **Métricas:** volatilidad anualizada (raíz de 365), caída máxima desde el pico y retorno/riesgo (retorno anualizado dividido volatilidad).
- **Alcance:** es un *backtest* con datos pasados, no una proyección.
""")

st.caption("⚠️ Herramienta educativa. Rendimientos pasados no garantizan rendimientos futuros. No constituye asesoramiento financiero.")
