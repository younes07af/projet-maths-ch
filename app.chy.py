import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuration de la page
st.set_page_config(page_title="Analyse Financière Pro", layout="wide")

# --- 6.1 ACQUISITION DES DONNÉES ---
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    return data if not data.empty else None

# --- 6.2 TRAITEMENT MATHÉMATIQUE ---
def compute_returns(df, method="Arithmétique"):
    if method == "Arithmétique":
        return df['Close'].pct_change().dropna()
    else:
        return np.log(df['Close'] / df['Close'].shift(1)).dropna()

# --- 6.3 PROBABILITÉS ET STATISTIQUES ---
def get_stats(returns):
    # Volatilité annualisée (σ_annuel = σ_quotidien * √252)
    vol_ann = returns.std() * np.sqrt(252)
    # Test de normalité Jarque-Bera
    jb_stat, p_value = stats.jarque_bera(returns)
    
    return {
        "Moyenne": returns.mean(),
        "Médiane": returns.median(),
        "Volatilité Ann.": vol_ann,
        "Skewness": stats.skew(returns),
        "Kurtosis": stats.kurtosis(returns),
        "P-Value (JB)": p_value
    }

# --- 6.4 INDICATEURS TECHNIQUES ---
def add_indicators(df):
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    # Bandes de Bollinger
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['Std20'] = df['Close'].rolling(window=20).std()
    df['BB_Up'] = df['SMA20'] + (df['Std20'] * 2)
    df['BB_Low'] = df['SMA20'] - (df['Std20'] * 2)
    return df

# --- 6.5 BACKTESTING ---
def backtest_strategy(df, short=20, long=50):
    df['SMA_S'] = df['Close'].rolling(window=short).mean()
    df['SMA_L'] = df['Close'].rolling(window=long).mean()
    # Signaux
    df['Position'] = np.where(df['SMA_S'] > df['SMA_L'], 1, 0)
    df['Strat_Ret'] = df['Position'].shift(1) * df['Close'].pct_change()
    # Capital (Base 1000)
    df['Equity'] = 1000 * (1 + df['Strat_Ret']).fillna(0).cumprod()
    return df

# --- INTERFACE (DASHBOARD) ---
st.title("📈 Plateforme d'Analyse Financière Quant")
st.sidebar.header("Paramètres")

ticker = st.sidebar.text_input("Actif (ex: AAPL, BTC-USD)", "AAPL")
start = st.sidebar.date_input("Date début", pd.to_datetime("2022-01-01"))
end = st.sidebar.date_input("Date fin", pd.to_datetime("2024-01-01"))

if st.sidebar.button("Lancer l'analyse"):
    data = load_data(ticker, start, end)
    
    if data is not None:
        # Section Statistique
        st.header("1. Analyse Statistique")
        rets = compute_returns(data)
        s = get_stats(rets)
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Volatilité Ann.", f"{s['Volatilité Ann.']:.2%}")
        c2.metric("Skewness", f"{s['Skewness']:.2f}")
        c3.metric("Kurtosis", f"{s['Kurtosis']:.2f}")
        c4.metric("P-Value JB", f"{s['P-Value (JB)']:.4f}")
        
        # Formules LaTeX (Exigence du projet)
        st.latex(r"\sigma_{annuel} = \sigma_{quotidien} \times \sqrt{252}")

        # Graphiques Statistiques
        fig_st, ax_st = plt.subplots(1, 2, figsize=(10, 4))
        sns.histplot(rets, kde=True, ax=ax_st[0])
        ax_st[0].set_title("Distribution des Rendements")
        stats.probplot(rets, dist="norm", plot=ax_st[1])
        st.pyplot(fig_st)

        # Section Indicateurs
        st.header("2. Indicateurs Techniques")
        data = add_indicators(data)
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(x=data.index, y=data['Close'], name="Prix"))
        fig_price.add_trace(go.Scatter(x=data.index, y=data['BB_Up'], name="Bollinger Haut", line=dict(dash='dash')))
        fig_price.add_trace(go.Scatter(x=data.index, y=data['BB_Low'], name="Bollinger Bas", line=dict(dash='dash')))
        st.plotly_chart(fig_price)

        # Section Backtesting
        st.header("3. Backtesting (SMA Crossover)")
        bt = backtest_strategy(data)
        st.line_chart(bt['Equity'])
        
        rendement_total = (bt['Equity'].iloc[-1] - 1000) / 1000
        st.success(f"Rendement total de la stratégie : {rendement_total:.2%}")
