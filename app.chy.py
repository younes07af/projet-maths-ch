import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats



def load_data(ticker, start, end):
    """Télécharge les données OHLC depuis Yahoo Finance"""
    data = yf.download(ticker, start=start, end=end)
    data = data.dropna()
    return data

def rendement_arithmetique(prices):
    return prices.pct_change().dropna()


def rendement_log(prices):
    return np.log(prices / prices.shift(1)).dropna()



def stats_descriptives(r):
    return {
        "Moyenne": r.mean(),
        "Médiane": r.median(),
        "Volatilité": r.std(),
        "Skewness": stats.skew(r),
        "Kurtosis": stats.kurtosis(r)
    }



def SMA(prices, n):
    return prices.rolling(n).mean()


def EMA(prices, n):
    return prices.ewm(span=n, adjust=False).mean()



def backtest_sma(data, short=20, long=50, capital=1000):
    df = data.copy()
    df['SMA_short'] = SMA(df['Close'], short)
    df['SMA_long'] = SMA(df['Close'], long)

    # Signaux
    df['Position'] = np.where(df['SMA_short'] > df['SMA_long'], 1, 0)
    df['Rendement'] = rendement_arithmetique(df['Close'])
    df['Rendement_strat'] = df['Position'].shift(1) * df['Rendement']

    df['Capital'] = capital * (1 + df['Rendement_strat']).cumprod()
    return df



st.title("📈 Plateforme simple d'analyse financière")


ticker = st.text_input("Actif financier", "AAPL")
start = st.date_input("Date début", pd.to_datetime("2022-01-01"))
end = st.date_input("Date fin", pd.to_datetime("2024-01-01"))

if st.button("Charger les données"):
    data = load_data(ticker, start, end)

    st.subheader("Données OHLC")
    st.dataframe(data.tail())

    
    r = rendement_arithmetique(data['Close'])

    st.subheader("Statistiques des rendements")
    stats_r = stats_descriptives(r)
    st.write(stats_r)

    
    sma20 = SMA(data['Close'], 20)
    sma50 = SMA(data['Close'], 50)

    fig, ax = plt.subplots()
    ax.plot(data['Close'], label='Prix')
    ax.plot(sma20, label='SMA 20')
    ax.plot(sma50, label='SMA 50')
    ax.legend()
    st.pyplot(fig)

   
    st.subheader("Backtesting SMA")
    bt = backtest_sma(data)

    fig2, ax2 = plt.subplots()
    ax2.plot(bt['Capital'], label='Capital')
    ax2.legend()
    st.pyplot(fig2)

    rendement_total = (bt['Capital'].iloc[-1] - 1000) / 1000
    st.write("Rendement total:", rendement_total)