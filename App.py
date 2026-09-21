"""
LSTM Collateral Valuation — Interactive Dashboard
Portfolio companion app for: "LSTM-Based Time-Series Forecasting for Equity Collateral Value Prediction in Kenya" by Nickson Wafula Masai, KCA University

Run locally with:   streamlit run app.py
Deploy for free at: https://share.streamlit.io
"""

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="NSE Collateral Valuation — LSTM Dashboard",
    page_icon="\U0001F4C8",
    layout="wide",
)

# ----------------------------------------------------------------------------
# DATA — transcribed directly from the dissertation's results tables
# (Table 4.6, 4.7, 4.8, 4.9, 4.10)
# ----------------------------------------------------------------------------

PER_COUNTER = pd.DataFrame([
    ("ABSA", 100, 1.4625, 1.1547, 6.06), ("BAMB", 17, 5.0422, 3.3925, 6.00),
    ("BAT", 30, 9.6020, 6.9881, 1.83), ("BOC", 100, 3.8162, 2.5586, 2.54),
    ("BRIT", 11, 0.7735, 0.6278, 10.82), ("CARB", 20, 1.6775, 1.4534, 7.16),
    ("CGEN", 61, 1.2672, 0.9583, 3.59), ("CIC", 35, 0.1783, 0.1305, 4.41),
    ("COOP", 74, 0.6136, 0.3937, 2.19), ("CRWN", 17, 2.0995, 1.5449, 3.86),
    ("CTUM", 26, 0.6137, 0.4150, 3.61), ("DTK", 17, 3.1031, 2.2142, 3.26),
    ("EABL", 12, 10.4343, 7.2370, 4.19), ("EGAD", 40, 0.7595, 0.5256, 3.72),
    ("EQTY", 55, 1.3426, 0.8979, 1.72), ("EVRD", 15, 0.0796, 0.0539, 4.50),
    ("HFCK", 17, 0.6183, 0.4411, 7.14), ("JUB", 26, 12.9996, 10.5160, 5.67),
    ("KAPC", 46, 17.9560, 12.6050, 4.81), ("KCB", 29, 1.6620, 1.2097, 3.50),
    ("KEGN", 21, 0.4348, 0.3245, 7.63), ("KNRE", 22, 0.1929, 0.1046, 5.71),
    ("KPLC", 30, 0.6826, 0.4551, 7.13), ("KPLC-P4", 23, 0.0713, 0.0486, 1.17),
    ("KPLC-P7", 17, 0.0002, 0.0002, 0.00), ("KUKZ", 100, 10.1707, 5.4730, 1.36),
    ("LBTY", 14, 0.6241, 0.4782, 6.88), ("LIMT", 82, 7.9372, 5.5170, 1.57),
    ("LKL", 11, 1.1680, 1.1338, 45.52), ("NCBA", 57, 4.3058, 3.0111, 4.90),
    ("OCH", 44, 0.2472, 0.1619, 3.70), ("PORT", 19, 4.9990, 2.8775, 9.87),
    ("SASN", 62, 0.5367, 0.4309, 2.52), ("SBIC", 22, 8.6301, 6.9820, 4.42),
    ("SCAN", 11, 1.0510, 0.9988, 42.03), ("SCBK", 29, 11.3692, 9.0736, 3.42),
    ("SCOM", 37, 0.8262, 0.5943, 2.98), ("SGL", 31, 3.3415, 3.3183, 55.36),
    ("SLAM", 12, 0.6149, 0.4628, 7.11), ("SMER", 37, 1.0549, 0.5779, 7.18),
    ("TCL", 31, 0.2105, 0.1905, 37.35), ("TOTL", 65, 1.3964, 0.9088, 3.28),
    ("TPSE", 13, 0.9193, 0.6599, 4.30), ("UCHM", 14, 0.1518, 0.0686, 18.61),
    ("UNGA", 60, 0.9539, 0.7297, 4.00), ("WTK", 90, 6.6050, 3.8877, 1.71),
], columns=["Ticker", "Epochs", "RMSE", "MAE", "MAPE"])

LTV_DATA = pd.DataFrame([
    ("ABSA", 12.96, 1.4625, 11.28), ("BAMB", 108.23, 5.0422, 4.66),
    ("BAT", 566.67, 9.6020, 1.69), ("BOC", 91.52, 3.8162, 4.17),
    ("BRIT", 10.56, 0.7735, 7.32), ("CARB", 24.40, 1.6775, 6.88),
    ("CGEN", 30.24, 1.2672, 4.19), ("CIC", 4.16, 0.1783, 4.29),
    ("COOP", 15.20, 0.6136, 4.04), ("CRWN", 58.55, 2.0995, 3.59),
    ("CTUM", 27.42, 0.6137, 2.24), ("DTK", 126.07, 3.1031, 2.46),
    ("EABL", 217.06, 10.4343, 4.81), ("EGAD", 19.03, 0.7595, 3.99),
    ("EQTY", 41.95, 1.3426, 3.20), ("EVRD", 1.80, 0.0796, 4.42),
    ("HFCK", 12.74, 0.6183, 4.85), ("JUB", 339.52, 12.9996, 3.83),
    ("KAPC", 129.23, 17.9560, 13.89), ("KCB", 41.27, 1.6620, 4.03),
    ("KEGN", 6.77, 0.4348, 6.42), ("KNRE", 10.03, 0.1929, 1.92),
    ("KPLC", 7.24, 0.6826, 9.43), ("KPLC-P4", 5.15, 0.0713, 1.38),
    ("KPLC-P7", 5.85, 0.0002, 0.00), ("KUKZ", 317.45, 10.1707, 3.20),
    ("LBTY", 11.21, 0.6241, 5.57), ("LIMT", 521.80, 7.9372, 1.52),
    ("LKL", 5.90, 1.1680, 19.80), ("NCBA", 40.23, 4.3058, 10.70),
    ("OCH", 3.36, 0.2472, 7.36), ("PORT", 29.78, 4.9990, 16.79),
    ("SASN", 18.83, 0.5367, 2.85), ("SBIC", 100.57, 8.6301, 8.58),
    ("SCAN", 19.13, 1.0510, 5.49), ("SCBK", 215.43, 11.3692, 5.28),
    ("SCOM", 22.10, 0.8262, 3.74), ("SGL", 21.81, 3.3415, 15.32),
    ("SLAM", 33.85, 0.6149, 1.82), ("SMER", 3.85, 1.0549, 27.40),
    ("TCL", 7.43, 0.2105, 2.83), ("TOTL", 23.29, 1.3964, 6.00),
    ("TPSE", 23.16, 0.9193, 3.97), ("UCHM", 4.18, 0.1518, 3.63),
    ("UNGA", 28.75, 0.9539, 3.32), ("WTK", 196.73, 6.6050, 3.36),
], columns=["Ticker", "MeanPrice", "RMSE", "RFR"])

BENCHMARK = pd.DataFrame([
    ("LSTM", 1.1115, 0.9034, 4.35),
    ("ARIMA(5,1,0)", 0.4458, 0.2936, 1.81),
    ("Random Forest", 1.5309, 1.0691, 3.78),
    ("Naive Persistence", 0.4554, 0.2817, 1.74),
    ("SMA (30-day)", 1.3502, 0.7676, 5.06),
], columns=["Model", "RMSE", "MAE", "MAPE"])

CAP_TIER = pd.DataFrame([
    ("Large-Cap / High-Vol", 16, 3.82, 3.64, 9.87),
    ("Mid-Cap / Med-Vol", 15, 4.54, 3.72, 7.18),
    ("Small-Cap / Low-Vol", 15, 16.74, 7.14, 55.36),
], columns=["Tier", "Count", "Mean_MAPE", "Median_MAPE", "Max_MAPE"])

MULTIVARIATE = pd.DataFrame([
    ("Univariate (primary model)", 4.35, "\u2014"),
    ("Multivariate, original architecture", 13.90, "4 / 46"),
    ("Multivariate, engineered features", 10.03, "4 / 46"),
    ("Multivariate, increased capacity", 11.66, "4 / 46"),
], columns=["Configuration", "Median MAPE (%)", "Counters beating univariate"])

# ----------------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------------

st.sidebar.title("\U0001F4C8 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Model Performance", "Benchmark Comparison", "Multivariate Experiment", "LTV Framework"],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**About this project**\n\n"
    "An LSTM time-series model forecasting NSE share prices for 46 counters, "
    "used to build a counter-specific loan-to-value framework for share-backed "
    "lending in Kenya.\n\n"
    "[Full write-up](#) · [GitHub](#) · [Notebooks](#)"
)

# ----------------------------------------------------------------------------
# PAGE: OVERVIEW
# ----------------------------------------------------------------------------

if page == "Overview":
    st.title("LSTM-Based Collateral Valuation for NSE Shares")
    st.markdown(
        "Kenyan lenders are reluctant to accept NSE-listed shares as loan "
        "collateral because share prices are volatile and hard to trust. "
        "This project forecasts share prices with an LSTM model trained "
        "independently on 46 NSE counters, benchmarks it honestly against "
        "four traditional methods, and converts each counter's measured "
        "forecast uncertainty into a usable loan-to-value recommendation."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Counters modelled", "46")
    col2.metric("Study window", "2013\u20132025")
    col3.metric("Median MAPE (LSTM)", "4.35%")
    col4.metric("Daily observations", "148,442")

    st.markdown("### What this dashboard covers")
    st.markdown(
        "- **Model Performance** \u2014 per-counter RMSE / MAE / MAPE across all 46 counters\n"
        "- **Benchmark Comparison** \u2014 LSTM vs. ARIMA, Random Forest, SMA, and naive persistence\n"
        "- **Multivariate Experiment** \u2014 what happened when macro/regulatory variables were added\n"
        "- **LTV Framework** \u2014 an interactive calculator for the counter-specific lending model"
    )

# ----------------------------------------------------------------------------
# PAGE: MODEL PERFORMANCE
# ----------------------------------------------------------------------------

elif page == "Model Performance":
    st.title("Per-Counter Model Performance")
    st.caption("Test-set results for the univariate LSTM, trained independently per counter.")

    sort_by = st.selectbox("Sort by", ["MAPE", "RMSE", "MAE", "Epochs"], index=0)
    ascending = st.checkbox("Ascending", value=False)
    st.dataframe(
        PER_COUNTER.sort_values(sort_by, ascending=ascending).reset_index(drop=True),
        use_container_width=True,
        height=460,
    )

    fig = px.bar(
        PER_COUNTER.sort_values("MAPE", ascending=False),
        x="Ticker", y="MAPE",
        title="Test-set MAPE by counter",
        color="MAPE", color_continuous_scale="RdYlGn_r",
    )
    fig.update_layout(xaxis_tickangle=-90, height=450)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Performance by market-cap / liquidity tier")
    st.dataframe(CAP_TIER, use_container_width=True)
    fig2 = px.bar(CAP_TIER, x="Tier", y="Median_MAPE", color="Tier",
                  title="Median MAPE by liquidity tier")
    st.plotly_chart(fig2, use_container_width=True)
    st.caption(
        "Small-cap, thinly-traded counters show inflated MAPE due to low absolute "
        "prices \u2014 their RMSE/MAE (shown in the table above) are not proportionally worse."
    )

# ----------------------------------------------------------------------------
# PAGE: BENCHMARK COMPARISON
# ----------------------------------------------------------------------------

elif page == "Benchmark Comparison":
    st.title("LSTM vs. Traditional Forecasting Models")
    st.dataframe(BENCHMARK, use_container_width=True)

    fig = px.bar(
        BENCHMARK, x="Model", y="MAPE", color="Model",
        title="Median test-set MAPE across the 46-counter panel",
        text_auto=".2f",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.warning(
        "**Naive persistence beat the LSTM on 45 of 46 counters** on raw "
        "one-step-ahead price accuracy. This isn't a failure of the model \u2014 "
        "daily NSE closing prices behave close to a random walk, consistent "
        "with classical market-efficiency theory. Median LSTM directional "
        "accuracy was 39.7%, below the 50% expected from a coin flip."
    )

# ----------------------------------------------------------------------------
# PAGE: MULTIVARIATE EXPERIMENT
# ----------------------------------------------------------------------------

elif page == "Multivariate Experiment":
    st.title("Did Adding More Variables Help?")
    st.caption(
        "Volume, 30-day rolling volatility, Central Bank Rate, inflation, and "
        "regulatory/market-shock dummies were added to the closing-price-only model."
    )
    st.dataframe(MULTIVARIATE, use_container_width=True)

    fig = px.bar(
        MULTIVARIATE, x="Configuration", y="Median MAPE (%)",
        title="Median MAPE: univariate vs. three multivariate configurations",
        text_auto=".2f",
    )
    fig.update_layout(xaxis_tickangle=-15)
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "All three multivariate configurations underperformed the univariate "
        "model, consistently \u2014 only 4 of 46 counters improved in every "
        "configuration. This negative result is reported in full rather than "
        "dropped, since it was part of the original study design."
    )

# ----------------------------------------------------------------------------
# PAGE: LTV FRAMEWORK (interactive)
# ----------------------------------------------------------------------------

elif page == "LTV Framework":
    st.title("Interactive Loan-to-Value Calculator")
    st.markdown(
        "**Recommended LTV = max( Floor LTV, Base LTV \u00d7 (1 \u2212 k \u00d7 RFR) )**\n\n"
        "Adjust the parameters below and watch the recommended LTV recalculate "
        "for all 46 counters using their actual Relative Forecast Risk (RFR)."
    )

    c1, c2, c3 = st.columns(3)
    base_ltv = c1.slider("Base LTV (%)", 10, 90, 50)
    k = c2.slider("Sensitivity (k)", 0.5, 5.0, 2.0, step=0.1)
    floor_ltv = c3.slider("Floor LTV (%)", 0, 50, 20)

    df = LTV_DATA.copy()
    df["RecommendedLTV"] = (base_ltv * (1 - k * df["RFR"] / 100)).clip(lower=floor_ltv)
    df = df.sort_values("RFR")

    st.dataframe(
        df.rename(columns={
            "MeanPrice": "Mean Price (KES)", "RFR": "RFR (%)",
            "RecommendedLTV": "Recommended LTV (%)",
        }).round(2),
        use_container_width=True,
        height=420,
    )

    fig = px.bar(
        df, x="Ticker", y="RecommendedLTV",
        title=f"Recommended LTV by counter (Base={base_ltv}%, k={k}, Floor={floor_ltv}%)",
        color="RFR", color_continuous_scale="RdYlGn_r",
    )
    fig.update_layout(xaxis_tickangle=-90, height=450)
    st.plotly_chart(fig, use_container_width=True)

    lookup = st.selectbox("Look up a specific counter", sorted(df["Ticker"].unique()))
    row = df[df["Ticker"] == lookup].iloc[0]
    lc1, lc2, lc3 = st.columns(3)
    lc1.metric("Mean Price (KES)", f"{row['MeanPrice']:.2f}")
    lc2.metric("Relative Forecast Risk", f"{row['RFR']:.2f}%")
    lc3.metric("Recommended LTV", f"{row['RecommendedLTV']:.1f}%")