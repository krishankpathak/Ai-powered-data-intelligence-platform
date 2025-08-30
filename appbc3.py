"""
AI-Powered Data Intelligence Platform (Full Enhanced Version)
-------------------------------------------------------------
Features:
- Upload or sample datasets
- Data Overview: stats, correlations, missing data, unique counts
- Visual Explorer: histogram, scatter, boxplot, heatmap, 3D scatter
- AutoML: classification & regression
- Forecasting (Prophet)
- Anomaly Detection (Isolation Forest)
- SQL Queries (pandasql)
- AI Search & Report (OpenAI GPT)
"""

from __future__ import annotations
import io, warnings, requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import openai
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype

# Prophet for forecasting
try:
    from prophet import Prophet
    _HAS_PROPHET = True
except Exception:
    _HAS_PROPHET = False

# PandasSQL for queries
try:
    from pandasql import sqldf
    _HAS_PANDASQL = True
except Exception:
    _HAS_PANDASQL = False

warnings.filterwarnings("ignore")
st.set_page_config(page_title="AI Data Intelligence Platform", layout="wide")

# ===================== CUSTOM STYLING ===================== #
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f172a, #1e293b);
        color: white !important;
    }
    h1, h2, h3, h4 {
        color: #38bdf8 !important;
    }
    [data-testid="stMetricValue"] {
        color: #facc15 !important;
        font-weight: bold;
    }
    .stDataFrame {
        background: white;
        border-radius: 12px;
    }
    div.stButton > button {
        background: #0ea5e9;
        color: white;
        border-radius: 12px;
        padding: 0.6em 1.2em;
        font-weight: bold;
    }
    div.stButton > button:hover {
        background: #0369a1;
        color: #facc15;
    }
    </style>
""", unsafe_allow_html=True)

# ===================== UTILS ===================== #
def mem_usage(df: pd.DataFrame) -> str:
    bytes_ = df.memory_usage(deep=True).sum()
    for unit in ["B","KB","MB","GB"]:
        if bytes_ < 1024.0:
            return f"{bytes_:,.2f} {unit}"
        bytes_ /= 1024.0
    return f"{bytes_:,.2f} TB"

@st.cache_data(show_spinner=False)
def load_csv(file: bytes, **kwargs) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file), **kwargs)

# ===================== FILE UPLOAD ===================== #
st.title("🤖 AI Data Intelligence Platform")
st.caption("Next-gen analytics, AutoML, anomaly detection, forecasting, SQL queries & AI reporting.")

uploaded = st.file_uploader("📂 Upload CSV file", type=["csv"])
sample_datasets = {
    "Titanic": "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
    "California Housing": "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv",
    "Iris": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
    "Flights": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/flights.csv"
}
if st.toggle("Use sample dataset"):
    name = st.selectbox("Pick a sample", list(sample_datasets))
    url = sample_datasets[name]
    r = requests.get(url, timeout=30)
    uploaded = io.BytesIO(r.content)
    uploaded.name = name + ".csv"

if not uploaded:
    st.info("👆 Upload a CSV or choose a sample dataset to begin.")
    st.stop()

# Load data
raw_bytes = uploaded.read() if hasattr(uploaded, "read") else uploaded.getvalue()
df = load_csv(raw_bytes)
st.success(f"Loaded **{uploaded.name}** · Shape {df.shape[0]} x {df.shape[1]} · {mem_usage(df)}")

# ===================== MAIN TABS ===================== #
tab_overview, tab_viz, tab_automl, tab_forecast, tab_anomaly, tab_sql, tab_ai = st.tabs(
    ["📊 Overview", "📈 Visual Explorer", "🤖 AutoML", "📅 Forecast", "⚠️ Anomalies", "📝 SQL", "🔍 AI Search"]
)

# ===================== TAB: OVERVIEW ===================== #
with tab_overview:
    st.header("📊 Data Overview")

    col1, col2, col3 = st.columns([2,2,1])
    with col1:
        st.subheader("Preview")
        st.dataframe(df.head(50), use_container_width=True)
    with col2:
        st.subheader("Missing Values")
        missing_data = df.isna().sum().reset_index()
        missing_data.columns = ["Column", "Missing"]
        fig = px.bar(missing_data, x="Column", y="Missing", title="Missing Data")
        st.plotly_chart(fig, use_container_width=True)
    with col3:
        st.subheader("Quick Stats")
        st.metric("Rows", f"{len(df):,}")
        st.metric("Cols", f"{df.shape[1]:,}")
        st.metric("Duplicates", int(df.duplicated().sum()))

    st.subheader("📌 Descriptive Statistics")
    st.dataframe(df.describe(include="all").T, use_container_width=True)

    st.subheader("📌 Correlation Heatmap")
    num_df = df.select_dtypes(include=np.number)
    if not num_df.empty:
        corr = num_df.corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Heatmap", color_continuous_scale="viridis")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📌 Unique Value Counts")
    unique_counts = df.nunique().reset_index()
    unique_counts.columns = ["Column", "Unique Values"]
    st.dataframe(unique_counts, use_container_width=True)

# ===================== TAB: VISUAL EXPLORER ===================== #
with tab_viz:
    st.header("📈 Visual Explorer (Futuristic)")
    chart_type = st.selectbox("Select Chart Type", ["Histogram", "Boxplot", "Scatter", "Heatmap", "3D Scatter"])

    if chart_type == "Histogram":
        colx = st.selectbox("Select column", df.columns)
        if is_numeric_dtype(df[colx]):
            fig = px.histogram(df, x=colx, nbins=30, title=f"Distribution of {colx}", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Boxplot":
        colx = st.selectbox("Select column", df.columns)
        fig = px.box(df, y=colx, title=f"Boxplot of {colx}", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Scatter":
        x_col = st.selectbox("X-axis", df.columns)
        y_col = st.selectbox("Y-axis", df.columns)
        fig = px.scatter(df, x=x_col, y=y_col, color=df.columns[0], title="Scatter Plot", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Heatmap":
        num_df = df.select_dtypes(include=np.number)
        if not num_df.empty:
            fig = px.imshow(num_df.corr(), text_auto=True, aspect="auto", title="Heatmap", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "3D Scatter":
        num_cols = df.select_dtypes(include=np.number).columns
        if len(num_cols) >= 3:
            x_col = st.selectbox("X-axis", num_cols, index=0)
            y_col = st.selectbox("Y-axis", num_cols, index=1)
            z_col = st.selectbox("Z-axis", num_cols, index=2)
            fig = px.scatter_3d(df, x=x_col, y=y_col, z=z_col, color=num_cols[0], title="3D Scatter Plot", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

# ===================== TAB: AUTOML ===================== #
with tab_automl:
    st.header("🤖 AutoML")
    target = st.selectbox("Pick target column", df.columns)
    if target:
        X = df.dropna(subset=[target])
        y = X.pop(target)
        cat_cols = [c for c in X.columns if X[c].dtype=="object"]
        num_cols = [c for c in X.columns if is_numeric_dtype(X[c])]
        ct = ColumnTransformer([
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ], remainder="drop")
        if y.dtype=="object":
            model = Pipeline([("prep", ct), ("clf", RandomForestClassifier())])
            Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=0.2,random_state=42)
            model.fit(Xtr,ytr)
            yp = model.predict(Xte)
            st.metric("Accuracy", f"{accuracy_score(yte, yp):.2%}")
        else:
            model = Pipeline([("prep", ct), ("reg", RandomForestRegressor())])
            Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=0.2,random_state=42)
            model.fit(Xtr,ytr)
            yp = model.predict(Xte)
            st.metric("R²", f"{r2_score(yte, yp):.3f}")

# ===================== TAB: FORECAST ===================== #
with tab_forecast:
    st.header("📅 Forecasting")
    if not _HAS_PROPHET:
        st.warning("Install Prophet: `pip install prophet`")
    else:
        time_col = st.selectbox("Select datetime column", [c for c in df.columns if is_datetime64_any_dtype(df[c])])
        val_col = st.selectbox("Select value column", [c for c in df.columns if is_numeric_dtype(df[c])])
        if time_col and val_col:
            data = df[[time_col,val_col]].rename(columns={time_col:"ds",val_col:"y"}).dropna()
            m = Prophet()
            m.fit(data)
            fut = m.make_future_dataframe(periods=30)
            fc = m.predict(fut)
            fig = px.line(fc, x="ds", y="yhat", title="Forecast", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

# ===================== TAB: ANOMALY DETECTION ===================== #
with tab_anomaly:
    st.header("⚠️ Anomaly Detection")
    num_cols = [c for c in df.columns if is_numeric_dtype(df[c])]
    sel = st.multiselect("Select numeric columns", num_cols, default=num_cols[:2])
    if sel:
        iso = IsolationForest(contamination=0.05)
        preds = iso.fit_predict(df[sel].dropna())
        df["Anomaly"] = preds
        st.dataframe(df[df["Anomaly"]==-1])

# ===================== TAB: SQL ===================== #
with tab_sql:
    st.header("📝 SQL Querying")
    if not _HAS_PANDASQL:
        st.warning("Install pandasql: `pip install pandasql`")
    else:
        q = st.text_area("Write SQL (use table name `df`)", "SELECT * FROM df LIMIT 5;")
        if st.button("Run Query"):
            res = sqldf(q, {"df":df})
            st.dataframe(res)

# ===================== TAB: AI SEARCH ===================== #
with tab_ai:
    st.header("🔍 AI Search & Report")
    st.caption("Ask natural language questions about your dataset.")
    query = st.text_input("Enter your question:")
    if query:
        with st.spinner("Thinking..."):
            client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role":"system","content":"You are a data analyst generating insights from a dataset."},
                    {"role":"user","content":f"Dataset: {df.head(50).to_dict()} \n\nQuestion: {query}"}
                ]
            )
            st.success(response.choices[0].message.content)
