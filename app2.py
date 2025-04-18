import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import streamlit.components.v1 as components
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ---------------- Setup ----------------
st.set_page_config(page_title="Bandwidth Forecast App", layout="wide")

# Sidebar navigation
page = st.sidebar.selectbox(
    "Select a Page",
    ["Introduction", "KPI Dashboard", "Forecast Dashboard"]
)

# === Load & Prepare Data ===
@st.cache_data

def load_data():
    df = pd.read_csv("Cleaned_Full_Data.csv")
    df['Start Time'] = pd.to_datetime(df['Start Time'])
    df.set_index('Start Time', inplace=True)
    return df[['Bandwidth Value']]

df = load_data()

# === Clean Raw Uploaded Data ===
def preprocess_uploaded_data(raw_df):
    raw_df = raw_df[raw_df['Sensor Number'] == 1]  # Use sensor 1
    raw_df['Start Time'] = pd.to_datetime(
        raw_df['Date Time'].str.extract(r'(^\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2})')[0],
        format='%m/%d/%Y %H:%M:%S'
    )
    raw_df = raw_df[['Start Time', 'Value(RAW)']].rename(columns={'Value(RAW)': 'Bandwidth Value'})
    raw_df.set_index('Start Time', inplace=True)
    return raw_df


# Forecasting function using SARIMA
def forecast_sarima(input_df, forecast_horizon=168):
    df_hourly = input_df['Bandwidth Value'].resample('H').mean().dropna()
    model = SARIMAX(df_hourly, order=(2, 1, 2), seasonal_order=(1, 1, 1, 24),
                    enforce_stationarity=False, enforce_invertibility=False)
    sarima_fit = model.fit(disp=False)

    forecast = sarima_fit.get_forecast(steps=forecast_horizon)
    forecast_index = pd.date_range(df_hourly.index[-1], periods=forecast_horizon+1, freq='H')[1:]
    forecast_mean = forecast.predicted_mean
    forecast_ci = forecast.conf_int()

    # Evaluation on historical
    y_true = df_hourly[-len(sarima_fit.fittedvalues):]
    y_pred = sarima_fit.fittedvalues
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    return forecast_index, forecast_mean, forecast_ci, rmse, mae, mape

# ---------------- Page 1: Introduction ----------------
if page == "Introduction":
    st.image("logo.jpg", width=150)
    st.title("Bandwidth Forecasting & Insights Dashboard")

    st.markdown("""
    Welcome to the **Bandwidth Forecasting & Insights App** — an advanced tool designed to support
    **data-driven decision-making** at Ogero by delivering accurate, interpretable, and actionable
    forecasts of bandwidth usage across the national network.
    """)

    st.markdown("---")

    st.header(" Strategic Objectives of the Tool")
    st.markdown("""
    This platform provides stakeholders with predictive visibility into bandwidth consumption trends by leveraging the power of **SARIMA time series modeling**. The tool shows:

    -  **7-day forecasts** to anticipate short-term bandwidth demand  
    -  **Precision-driven planning** for infrastructure upgrades and network load balancing  
    -  **Proactive operations** through better alignment of maintenance schedules and resource deployment  
    -  **Performance diagnostics** via key metrics such as RMSE, MAE, MAPE, and AIC  
    """)

    st.markdown("---")

    st.header(" Why Forecasting Matters for Ogero")
    st.markdown("""
    As Lebanon’s national telecom operator, Ogero must continuously adapt to rising bandwidth demands, unexpected usage spikes, and evolving consumption patterns.

    By embedding predictive intelligence into daily workflows, this dashboard helps:
    - Prevent bottlenecks and service interruptions  
    - Optimize infrastructure investment timing  
    - Improve network responsiveness and SLA compliance  
    - Enhance the end-user experience
    """)

    st.image("network.jpg", caption="Sensor-based time series data enables precision forecasting", use_column_width=True)

# ---------------- Page 2: KPI Dashboard ----------------
elif page == "KPI Dashboard":
    st.title(" Bandwidth KPI Dashboard")
    st.markdown("Explore bandwidth sensor data and system KPIs below:")

    st.markdown("### Tableau Dashboard:")
    components.iframe(
        "https://public.tableau.com/views/OgeroDashboard/Dashboard1?:embed=y&:display_count=yes&:showVizHome=no",
        height=1400,
        width=1200
    )

# ---------------- Page 3: Forecast Dashboard ----------------
elif page == "Forecast Dashboard":
    st.title(" Optimized SARIMA Forecast Tool")
    st.markdown("This page presents the performance benchmarking of sensors followed by the 7-day SARIMA bandwidth forecast.")

    # ---------- Sensor Benchmark Chart ----------
    st.subheader(" Sensor Performance Benchmarking")
    st.image("sensor_benchmark.jpg", use_column_width=True, caption="Sensor Performance Categorized by Benchmark Scores")

    st.markdown("### Key Insights for Ogero:")
    sensor_insights = [
        "🔴 **Sensor 1 and Sensor 2 underperform**, indicating possible calibration issues or data transmission delays. These sensors should be flagged for maintenance or replacement.",
        "⚙️ **Sensor 3 and Sensor 6 show average reliability** — they are operational but might benefit from periodic diagnostics to ensure long-term stability.",
        "🔵 **Sensor 4 and Sensor 5 consistently outperform**, making them ideal candidates for replicating setup standards or locations across similar environments.",
        "📡 **Benchmarking helps Ogero prioritize troubleshooting efforts**, improve sensor coverage in weak zones, and maintain network data integrity for accurate forecasting."
    ]
    for i in sensor_insights:
        st.markdown(i)

    st.markdown("---")
    st.markdown("Upload new hourly bandwidth data to generate a 7-day forecast.")

    uploaded_file = st.file_uploader("Upload raw or cleaned CSV file", type='csv')

    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
            if 'Date Time' in raw_df.columns:
                st.info("Detected uncleaned data. Preprocessing in progress...")
                new_df = preprocess_uploaded_data(raw_df)
            else:
                new_df = raw_df.copy()
                new_df['Start Time'] = pd.to_datetime(new_df['Start Time'])
                new_df.set_index('Start Time', inplace=True)

            # Forecast
            forecast_index, forecast_mean, forecast_ci, rmse, mae, mape = forecast_sarima(new_df)

            # Plot
            fig, ax = plt.subplots(figsize=(12, 5))
            new_df['Bandwidth Value'].resample('H').mean().plot(ax=ax, label='Observed', color='blue')
            forecast_mean.plot(ax=ax, label='Forecast', color='green', linestyle='--')
            ax.fill_between(forecast_index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], alpha=0.2, color='green')
            ax.set_title("7-Day Forecast (Hourly)")
            ax.set_xlabel("Time")
            ax.set_ylabel("Bandwidth Value")
            ax.legend()
            st.pyplot(fig)

            # Metrics
            st.markdown("### Forecast Evaluation Metrics:")
            st.success(f"RMSE: {rmse:.2f}")
            st.info(f"MAE: {mae:.2f}")
            st.info(f"MAPE: {mape:.2f}%")

            st.download_button(
                label="📥 Download Forecast CSV",
                data=forecast_mean.reset_index().rename(columns={0: 'Forecast'}).to_csv(index=False),
                file_name="sarima_forecast_output.csv"
            )

        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
    else:
        st.warning("Please upload a CSV to proceed.")

    st.markdown("---")
    st.header("💡 Business Insights from SARIMA Forecast")

    insights = [
    " **Proactive Infrastructure Scaling**: With a low RMSE and MAPE, Ogero’s teams can confidently rely on SARIMA forecasts to anticipate usage spikes and proactively scale up local infrastructure—especially in areas where weekday evening demand peaks are common. This means that by December 30th 2024 Ogero coould have predicted bandwidth usage and needs till the 8th of January 2025",

    " **Maintenance Window Optimization**: The model’s high precision (low MAE) helps identify consistent low-usage periods across the week. This enables Ogero teams to schedule field operations like fiber line testing or DSL cabinet repairs with minimal customer disruption.",

    " **Targeted Bandwidth Distribution**: SARIMA’s reliable forecasting allows network planners to redistribute bandwidth capacity efficiently—e.g., by boosting provisioning in areas forecasted to hit saturation thresholds, particularly between 6–10 PM as seen in recent usage trends.",

    " **Cost Efficiency & service level agreement (SLA) Management**: The low AIC score validates that the model achieves high accuracy without overfitting, supporting Ogero in making lean infrastructure investments while confidently meeting SLA commitments on throughput and latency. Ogero can confidently defer or minimize unnecessary infrastructure upgrades (e.g., adding new lines, increasing backhaul capacity) while ensuring compliance with SLA commitments.",

    " **Data-Driven Decision Support**: By integrating this forecast into its network operations center (NOC) dashboards which is a real-time monitoring interface used by telecoms, ISPs, and IT teams, Ogero’s decision-makers gain a forward-looking view of demand fluctuations, transforming reactive bandwidth adjustments into proactive policy and rollout strategies. By embedding this SARIMA forecast into Ogero’s NOC dashboard, weekly bandwidth monitoring transforms from reactive reporting into proactive action."
    ]



    for insight in insights:
        st.markdown(insight)