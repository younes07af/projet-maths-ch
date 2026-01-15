import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Finance Analytics Pro", layout="wide")

# --- FONCTIONS DE CALCUL ---

def load_data(ticker, start, end):
    """Télécharge les données OHLC depuis Yahoo Finance"""
    try:
        data = yf.download(ticker, start=start, end=end)
        if data.empty:
            return None
        return data
    except Exception as e:
        st.error(f"Erreur lors du téléchargement : {e}")
        return None

def compute_returns(prices, method="Arithmétique"):
    """Calcule les rendements selon la méthode choisie"""
    if method == "Arithmétique":
        return prices.pct_change().dropna()
    else:
        return np.log(prices / prices.shift(1)).dropna()

def get_full_stats(r):
    """Calcule les statistiques descriptives complètes selon les exigences"""
    vol_daily = r.std()
    vol_ann = vol_daily * np.sqrt(252)
    
    # Tests de normalité
    jb_stat, jb_p = stats.jarque_bera(r)
    
    stats_dict = {
        "Moyenne": r.mean(),
        "Médiane": r.median(),
        "Volatilité Quotidienne": vol_daily,
        "Volatilité Annualisée": vol_ann,
        "Skewness": stats.skew(r),
        "Kurtosis": stats.kurtosis(r),
        "Max": r.max(),
        "Min": r.min(),
        "VaR (95%)": np.percentile(r, 5),
        "P-Value (Jarque-Bera)": jb_p
    }
    return stats_dict

# --- INDICATEURS TECHNIQUES ---

def add_indicators(df, sma_n=20, lma_n=50, rsi_n=14):
    df = df.copy()
    # SMA
    df['SMA_S'] = df['Close'].rolling(window=sma_n).mean()
    df['SMA_L'] = df['Close'].rolling(window=lma_n).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_n).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_n).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bandes de Bollinger
    df['BB_Mid'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Up'] = df['BB_Mid'] + (df['BB_Std'] * 2)
    df['BB_Low'] = df['BB_Mid'] - (df['BB_Std'] * 2)
    
    return df

# --- BACKTESTING ---

def run_backtest(data, short_n=20, long_n=50, initial_capital=1000):
    df = data.copy()
    df['SMA_S'] = df['Close'].rolling(window=short_n).mean()
    df['SMA_L'] = df['Close'].rolling(window=long_n).mean()
    
    # Signaux : 1 si SMA courte > SMA longue, sinon 0
    df['Position'] = np.where(df['SMA_S'] > df['SMA_L'], 1, 0)
    df['Signal'] = df['Position'].shift(1)
    
    # Rendements
    df['Asset_Ret'] = df['Close'].pct_change()
    df['Strat_Ret'] = df['Signal'] * df['Asset_Ret']
    
    # Capital
    df['Equity_Curve'] = initial_capital * (1 + df['Strat_Ret']).fillna(0).cumprod()
    
    # Métriques
    total_ret = (df['Equity_Curve'].iloc[-1] / initial_capital) - 1
    sharpe = (df['Strat_Ret'].mean() / df['Strat_Ret'].std()) * np.sqrt(252) if df['Strat_Ret'].std() != 0 else 0
    
    # Max Drawdown
    rolling_max = df['Equity_Curve'].cummax()
    drawdown = (df['Equity_Curve'] / rolling_max) - 1
    max_drawdown = drawdown.min()
    
    return df, total_ret, sharpe, max_drawdown

# --- INTERFACE STREAMLIT ---

st.title("🏛️ Dashboard d'Analyse Financière & Quantitative")
st.markdown("""---""")

# Sidebar de configuration
st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Symbole (ex: AAPL, BTC-USD)", "AAPL")
start_date = st.sidebar.date_input("Date de début", pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("Date de fin", pd.to_datetime("2024-01-01"))
ret_method = st.sidebar.selectbox("Méthode de rendement", ["Arithmétique", "Logarithmique"])

if st.sidebar.button("Exécuter l'Analyse"):
    data = load_data(ticker, start_date, end_date)
    
    if data is not None:
        # 1. PRÉSENTATION DES DONNÉES
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader(f"Prix de Clôture - {ticker}")
            st.line_chart(data['Close'])
        with col2:
            st.subheader("Dernières Données OHLC")
            st.write(data.tail())

        # 2. ANALYSE MATHÉMATIQUE ET STATISTIQUE
        st.markdown("---")
        st.header("📊 Analyse Statistique des Rendements")
        
        returns = compute_returns(data['Close'], method=ret_method)
        stats_metrics = get_full_stats(returns)
        
        # Affichage des métriques en colonnes
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rendement Moyen", f"{stats_metrics['Moyenne']:.4%}")
        m2.metric("Volatilité Ann.", f"{stats_metrics['Volatilité Annualisée']:.2%}")
        m3.metric("Skewness", f"{stats_metrics['Skewness']:.2f}")
        m4.metric("Kurtosis", f"{stats_metrics['Kurtosis']:.2f}")
        
        # Justification Mathématique
        with st.expander("Voir les formules mathématiques utilisées"):
            st.latex(r"R_{arith} = \frac{P_t - P_{t-1}}{P_{t-1}} \quad | \quad r_{log} = \ln\left(\frac{P_t}{P_{t-1}}\right)")
            st.latex(r"\sigma_{annuel} = \sigma_{quotidien} \times \sqrt{252}")
            st.info("Un test de Jarque-Bera est effectué pour vérifier la normalité. " + 
                    f"P-Value actuelle : {stats_metrics['P-Value (Jarque-Bera)']:.4f}")

        # Visualisations statistiques
        fig_stats, (ax_hist, ax_qq) = plt.subplots(1, 2, figsize=(12, 4))
        sns.histplot(returns, kde=True, ax=ax_hist, color='skyblue')
        ax_hist.set_title("Distribution des Rendements")
        
        stats.probplot(returns, dist="norm", plot=ax_qq)
        ax_qq.set_title("QQ-Plot (Test de Normalité)")
        st.pyplot(fig_stats)

        # 3. INDICATEURS TECHNIQUES
        st.markdown("---")
        st.header("📈 Indicateurs Techniques")
        data_ind = add_indicators(data)
        
        fig_ind = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                vertical_spacing=0.1, subplot_titles=(f'Prix & Bandes de Bollinger', 'RSI'),
                                row_width=[0.3, 0.7])
        
        # Prix et BB
        fig_ind.add_trace(go.Scatter(x=data_ind.index, y=data_ind['Close'], name='Prix'), row=1, col=1)
        fig_ind.add_trace(go.Scatter(x=data_ind.index, y=data_ind['BB_Up'], name='BB High', line=dict(dash='dash', color='gray')), row=1, col=1)
        fig_ind.add_trace(go.Scatter(x=data_ind.index, y=data_ind['BB_Low'], name='BB Low', line=dict(dash='dash', color='gray')), row=1, col=1)
        
        # RSI
        fig_ind.add_trace(go.Scatter(x=data_ind.index, y=data_ind['RSI'], name='RSI', line=dict(color='purple')), row=2, col=1)
        fig_ind.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig_ind.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        fig_ind.update_layout(height=600, template="plotly_white")
        st.plotly_chart(fig_ind, use_container_width=True)

        # 4. BACKTESTING
        st.markdown("---")
        st.header("🧪 Backtesting : Stratégie Croisement SMA")
        
        s_sma = st.slider("Fenêtre courte", 5, 50, 20)
        l_sma = st.slider("Fenêtre longue", 51, 200, 100)
        
        bt_data, t_ret, sharpe, mdd = run_backtest(data, s_sma, l_sma)
        
        res1, res2, res3 = st.columns(3)
        res1.metric("Rendement Stratégie", f"{t_ret:.2%}")
        res2.metric("Ratio de Sharpe", f"{sharpe:.2f}")
        res3.metric("Max Drawdown", f"{mdd:.2%}")
        
        st.subheader("Évolution du Capital (Initial : 1000€)")
        st.line_chart(bt_data['Equity_Curve'])
        
        st.success("Analyse terminée avec succès.")
    else:
        st.error("Impossible de récupérer les données pour ce symbole.")
else:
    st.info("Entrez un ticker et cliquez sur 'Exécuter l'Analyse' dans la barre latérale.")
