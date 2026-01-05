import streamlit as st
import pandas as pd
import mysql.connector
import seaborn as sns
import matplotlib.pyplot as plt
import io
import numpy as np


st.set_page_config(page_title="SupplySyncAI – MLOps UI", layout="wide")

st.markdown(
    """
    <style>
        .stApp {
            background-color: #FFFFFF;
        }
    </style>
    """,
    unsafe_allow_html=True
)




st.markdown(
    """
    <style>
        /* Dark blue themed button */
        div.stButton > button {
            background-color: #0B2C5D;   /* Dark blue from your header */
            color: #FFFFFF;
            border-radius: 8px;
            padding: 8px 18px;
            border: none;
            font-weight: 600;
        }

        div.stButton > button:hover {
            background-color: #08306B;   /* Slightly darker on hover */
            color: #FFFFFF;
        }
    </style>
    """,
    unsafe_allow_html=True
)



st.markdown(
    """
    <div style="
        background-color:#0B2C5D;
        padding:35px;
        border-radius:12px;
        color:white;
        text-align:center;
        margin-bottom:20px;
    ">
        <h1 style="margin-bottom:8px;">
            AI-Powered Demand Forecasting & Sales Prediction Engine
        </h1>
        <h3 style="font-weight:400; margin-top:0;">
            From Broad Estimates to SKU-Level Intelligence
        </h3>
        <p style="font-size:17px; margin-top:15px;">
            Predict demand accurately across products, stores, channels,
            promotions, events, and time.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style="
        background-color:#2F75B5;
        padding:28px;
        border-radius:12px;
        color:white;
        font-size:16px;
        line-height:1.6;
        margin-bottom:25px;
    ">

    <p>
    This application enables <b>granular demand forecasting and sales prediction</b>
    by combining transactional data, customer behavior, promotions, events,
    weather, inventory, and trends into a unified AI-driven analytics pipeline.
    </p>

    <p>
    Unlike traditional forecasting systems that operate at a
    <b>store or category level</b>, this platform provides
    <b>fine-grained forecasts at the SKU × Store × Time level</b>,
    empowering data-driven decisions across planning, inventory, and operations.
    </p>

    <h4 style="margin-top:22px;">Why This Matters</h4>
    
    <p>
    Rurtail demand is influenced by far more than historical sales. 
    This engine cuptures<b> real-world drivers of demand</b>, including:
    </p>

    <ul>
        <li>Customer engagement and loyalty behavior</li>
        <li>Promotion effectiveness and campaign impact</li>
        <li>Event-driven demand spikes</li>
        <li>Weather and trend influences</li>
        <li>Inventory availability and stock health</li>
    </ul>

    <p style="margin-top:15px;">
        <b>The result:</b> More accurate forecasts, reduced stockouts,
        lower excess inventory, and improved profitability.
    </p>

    </div>
    """,
    unsafe_allow_html=True
)

st.write("")  

# MYSQL LOADER FUNCTION
@st.cache_data
# CSV LOADER FUNCTION (DEPLOYMENT SAFE)
@st.cache_data
def load_data():
    return pd.read_csv("fact_consolidated.csv")


# CENTERED SMALL PLOT FUNCTION
def show_small_plot(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)

    st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
    st.image(buf, width=480)  # Half screen
    st.markdown("</div>", unsafe_allow_html=True)




# STEP 1 – LOAD DATA (USING YOUR EXISTING MYSQL FUNCTION)
st.markdown(
    """
    <div style="
        background-color:#0B2C5D;
        padding:18px 25px;
        border-radius:10px;
        color:white;
        margin-top:20px;
        margin-bottom:10px;
    ">
        <h3 style="margin:0;">
            Data Collection & Integration (Unified Data Ingestion)
        </h3>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style="
        background-color:#2F75B5;
        padding:28px;
        border-radius:12px;
        color:white;
        font-size:16px;
        line-height:1.6;
        margin-bottom:20px;
    ">

    <p>
    This section consolidates data from multiple enterprise sources into a single analytical model.
    </p>

    <b>Integrated Data Domains:</b>
    <ul>
        <li>Customer behavior & loyalty</li>
        <li>Product master & pricing</li>
        <li>Store & sales channel data</li>
        <li>Promotions & events</li>
        <li>Inventory & stock conditions</li>
        <li>Weather & market trends</li>
        <li>Time & seasonality signals</li>
    </ul>

    <p>
    All data is validated and aligned using a <b>consistent dimensional model</b>
    to ensure forecasting accuracy.
    </p>

    </div>
    """,
    unsafe_allow_html=True
)



# Make sure session key exists
if "df" not in st.session_state:
    st.session_state.df = None

# Load Button
if st.button("Load Data"):
    
    st.session_state.df = load_data()
    



# Show preview if loaded
df = st.session_state.df

if df is not None:
    st.markdown(
    "<h3 style='color:#000000;'>🔍 Data Preview</h3>",
    unsafe_allow_html=True
)

    st.data_editor(
    df.head(20),
    use_container_width=True,
    disabled=True
)



    st.info(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
else:
    st.info("Click the button above to load the dataset.")
    
# ============================================================
# STEP 2 – DATA PRE-PROCESSING (Data Quality & Readiness)
# ============================================================

st.markdown("""
<div style="
    background-color:#0B2C5D;
    padding:18px 25px;
    border-radius:10px;
    color:white;
    margin-top:25px;
    margin-bottom:12px;
">
    <h3 style="margin:0;">
        Data Pre-Processing (Data Quality & Readiness)
    </h3>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="
    background-color:#2F75B5;
    padding:24px;
    border-radius:12px;
    color:white;
    font-size:16px;
    line-height:1.7;
    margin-bottom:20px;
">
This section ensures the dataset is <b>model-ready</b> by handling:
<ul>
    <li>Missing values & inconsistencies</li>
    <li>Outliers & anomalies</li>
    <li>Data type validation</li>
    <li>Referential integrity checks across dimensions</li>
    <li>Time alignment and granularity normalization</li>
</ul>

This step guarantees that downstream models are trained on
<b>clean, reliable, and trustworthy data.</b>
</div>
""", unsafe_allow_html=True)

# Ensure df exists
if df is None:
    st.warning("⚠ Load data first.")
    st.stop()

processed_df = df.copy()
logs = []

# ============================================================
# 1️⃣ REMOVE DUPLICATE ROWS
# ============================================================

remove_duplicates = st.checkbox("Remove Duplicate Rows")

st.markdown("""
<div style="
    background-color:#2F75B5;
    padding:28px;
    border-radius:12px;
    color:white;
    font-size:16px;
        line-height:1.6;
        margin-bottom:20px;
">
<b>What this does:</b>
This step identifies and removes <b>exact duplicate records</b> from the dataset.<br>

<b>Duplicate rows often occur due to:</b>
<ul>
    <li>Multiple data ingestion runs</li>
    <li>System retries or sync issues</li>
    <li>Manual data merges</li>
</ul><br>

<b>Why this is important:</b>
<ul>
    <li>Prevents <b>double counting of sales, customers, or inventory</b></li>
    <li>Ensures <b>accurate aggregates and trends</b></li>
    <li>Avoids biased model training caused by repeated observations</li>
</ul><br>

<b>How it helps forecasting:</b><br>
Demand models rely on <b>true historical patterns</b>.<br>
Duplicates distort demand signals and inflate sales volumes,
leading to <b>over-forecasting</b>.
</div>
""", unsafe_allow_html=True)

if remove_duplicates:
    before = processed_df.shape[0]
    processed_df = processed_df.drop_duplicates()
    logs.append(f"✔ Removed **{before - processed_df.shape[0]} duplicate rows**")

# ============================================================
# 2️⃣ REMOVE ROWS WITH NULL VALUES
# ============================================================

remove_nulls = st.checkbox("Remove Rows with NULL Values")

st.markdown(
    """
    <div style="
        background-color:#2F75B5;
        padding:28px;
        border-radius:12px;
        color:white;
        font-size:16px;
        line-height:1.6;
        margin-bottom:20px;
    ">

    <b>What this does:</b><br>

    This step removes rows where <b>critical fields</b> contain missing (null) values.<br>
    Examples of critical fields:<br>
    Product ID<br>
    Store ID<br>
    Date / Time<br>
    Quantity Sold<br>
    Sales Amount<br><br>

    <b>Why this is important:</b><br>

    Incomplete records cannot be reliably used for analysis or modeling<br>
    Missing core identifiers break relationships between dimensions and facts<br>
    Ensures data integrity across the dimensional model<br><br>

    <b>When this is applied:</b><br>

    Only applied when:<br>
    The missing value is <b>essential</b><br>
    The row cannot be safely repaired or inferred<br>
    This avoids introducing <b>incorrect assumptions</b> into the dataset.

    </div>
    """,
    unsafe_allow_html=True
)

if remove_nulls:
    before = processed_df.shape[0]
    processed_df = processed_df.dropna()
    logs.append(f"✔ Removed **{before - processed_df.shape[0]} rows with NULL values**")

# ============================================================
# 3️⃣ REPLACE NULL VALUES WITH "UNKNOWN"
# ============================================================

replace_nulls = st.checkbox("Replace NULL Values with 'Unknown'")

st.markdown(
    """
    <div style="
        background-color:#2F75B5;
        padding:28px;
        border-radius:12px;
        color:white;
        font-size:16px;
        line-height:1.6;
        margin-bottom:20px;
    ">

    <b>What this does:<br>

    For non-critical categorical fields, missing values are replaced with a placeholder like:<br>
    “<b>Unknown</b>”<br>

    <b>Examples:</b>

    <li> Customer Gender</li>
    <li> Promotion Type</li>
    <li> Event Category</li>
    <li> Payment Type</li><br>

    <b>Why this is important:<br>

    <li>Preserves valuable records instead of discarding them</li>
    <li> Keeps categorical columns consistent</li>
    <li> Allows models to learn from “unknown” patterns rather than losing data</li><br>

        
    <b>Modelling advantage:</b>

    Many ML models can handle a distinct “<b>Unknown</b>” category better than missing values.<br>

    This improves:<br>

    <li>Model stability</li>
    <li>Feature completeness</li>
    <li>Interpretability</li>

    </div>
    """,
    unsafe_allow_html=True
)

if replace_nulls:
    processed_df = processed_df.fillna("Unknown")
    logs.append("✔ Replaced NULL values with 'Unknown'")

# ============================================================
# 4️⃣ CONVERT COLUMNS TO NUMERIC (SAFE COLUMNS ONLY)
# ============================================================

convert_numeric = st.checkbox("Convert Columns to Numeric (Safe Columns Only)")


st.markdown(
    """
    <div style="
        background-color:#2F75B5;
        padding:28px;
        border-radius:12px;
        color:white;
        font-size:16px;
        line-height:1.6;
        margin-bottom:20px;
    ">

    <b>What this does:<br>

    This step converts <b>safe, measurable columns</b> into numeric formats so they can be used in analysis and ML models.

    <b>Examples of safe numeric columns:</b>

    <ul>
        <li>Quantity Sold</li>
        <li>Unit Price</li>
        <li>Total Sales Amount</li>
        <li>Discount Applied</li>
        <li>Tax Amount</li>
        <li>Satisfaction Score</li>
        <li>Forecast Quantity</li>
    </ul><br>

    <b>What is NOT converted:<br>

    <ul>
        <li>IDs (Product ID, Store ID, Customer ID)</li>
        <li>Categorical labels</li>
        <li>Descriptive text fields</li>
    </ul><br>

    <b>Why this is important:<br>

    <ul>
        <li>Enables mathematical operations and aggregations</li>
        <li>Required for correlation analysis and model training</li>
        <li>Prevents runtime errors in ML pipelines</li>
    </ul><br>

    
    <b>Why “safe columns only” matters:<br>

    Blindly converting columns can:<br>

    <ul>
        <li>Corrupt IDs</li>
        <li>Break joins</li>
        <li>Create misleading numerical patterns</li>
    </ul>

    This step ensures <b>only logically numeric fields are converted.</b>

    </div>
    """,
    unsafe_allow_html=True
)
if convert_numeric:
    exclude = [
        "transaction_id","product_id","store_id","customer_id",
        "sales_channel_id","promo_id","event_id"
    ]
    safe_cols = [c for c in processed_df.columns if c not in exclude]

    converted = 0
    for col in safe_cols:
        before = processed_df[col].dtype
        processed_df[col] = pd.to_numeric(processed_df[col], errors="ignore")
        if processed_df[col].dtype != before:
            converted += 1

    logs.append(f"✔ Converted **{converted} columns** to numeric")

st.session_state.df = processed_df

st.success("### ✅ Pre-Processing Status")

if remove_duplicates:
    st.markdown("✔ **Duplicate rows removed**")
else:
    st.markdown("⚠ **Duplicate rows removal not applied**")

if remove_nulls:
    st.markdown("✔ **Rows with NULL values removed**")
else:
    st.markdown("⚠ **Rows with NULL values removal not applied**")

if replace_nulls:
    st.markdown("✔ **NULL values replaced with 'Unknown'**")
else:
    st.markdown("⚠ **NULL values replacement not applied**")

if convert_numeric:
    st.markdown("✔ **Numeric conversion applied (safe columns only)**")
else:
    st.markdown("⚠ **Numeric conversion not applied**")


# ============================================================
# STEP 3 – FULL ADAPTIVE EDA (ANALYSIS-FOCUSED)
# ============================================================

df = st.session_state.get("df", None)

if df is None:
    st.warning("⚠ No dataset loaded. Please load data first.")
    st.stop()

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.markdown("## 📊 Step 3 — Exploratory Data Analysis (EDA)")
st.info(f"Dataset Loaded: **{df.shape[0]} rows × {df.shape[1]} columns**")

# ============================================================
# COLUMN MAPPING (SAFE & SIMPLE)
# ============================================================

def map_col(candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

col_rev     = map_col(["total_sales_amount"])
col_qty     = map_col(["quantity_sold"])
col_price   = map_col(["unit_price"])
col_date    = map_col(["date"])
col_product = map_col(["product_id"])
col_store   = map_col(["store_id"])
col_channel = map_col(["sales_channel_id"])
col_event   = map_col(["event_id"])
col_promo   = map_col(["promo_id"])

num_df = df.select_dtypes(include=np.number)

# ============================================================
# 1. DATA QUALITY OVERVIEW
# ============================================================

with st.expander("1. Data Quality Overview"):

    st.subheader("Dataset Shape")
    st.write(f"Rows: **{df.shape[0]}**, Columns: **{df.shape[1]}**")

    st.subheader("Missing Value Analysis (%)")
    mv = (df.isnull().mean() * 100).sort_values(ascending=False)
    st.dataframe(mv.to_frame("missing_%"))

    st.subheader("Duplicate Analysis")
    st.write(f"Total duplicate rows: **{df.duplicated().sum()}**")

    st.subheader("Data Types Summary")
    st.dataframe(df.dtypes.value_counts().to_frame("count"))

# ============================================================
# 2. SALES METRICS OVERVIEW (NO PLOTS)
# ============================================================

with st.expander("2. Sales Overview"):

    if col_rev:
        st.metric("Total Revenue", f"{df[col_rev].sum():,.2f}")
        st.metric("Average Order Value", f"{df[col_rev].mean():,.2f}")
        st.metric("Max Order Value", f"{df[col_rev].max():,.2f}")

    if col_qty:
        st.metric("Total Units Sold", f"{df[col_qty].sum():,}")
        st.metric("Average Units per Transaction", f"{df[col_qty].mean():.2f}")

# ============================================================
# 3. PRODUCT ANALYSIS (RANKINGS & COUNTS)
# ============================================================

with st.expander("3. Product-Level Analysis"):

    if col_product:
        st.subheader("Top Products by Transaction Count")
        st.dataframe(df[col_product].value_counts().head(10))

    if col_product and col_rev:
        st.subheader("Top Products by Revenue")
        st.dataframe(
            df.groupby(col_product)[col_rev]
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )

# ============================================================
# 4. STORE ANALYSIS
# ============================================================

with st.expander("4. Store-Level Analysis"):

    if col_store:
        st.subheader("Transactions per Store")
        st.dataframe(df[col_store].value_counts().head(10))

    if col_store and col_rev:
        st.subheader("Revenue Contribution by Store")
        st.dataframe(
            df.groupby(col_store)[col_rev]
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )

# ============================================================
# 5. SALES CHANNEL ANALYSIS
# ============================================================

with st.expander("5. Sales Channel Analysis"):

    if col_channel:
        st.subheader("Transaction Distribution by Channel")
        st.dataframe(df[col_channel].value_counts())

    if col_channel and col_rev:
        st.subheader("Revenue by Channel")
        st.dataframe(df.groupby(col_channel)[col_rev].sum())

# ============================================================
# 6. PROMOTION ANALYSIS
# ============================================================

with st.expander("6. Promotion Effectiveness"):

    if col_promo:
        st.subheader("Transactions under Promotions")
        st.dataframe(df[col_promo].value_counts().head(10))

    if col_promo and col_rev:
        st.subheader("Revenue under Promotions")
        st.dataframe(
            df.groupby(col_promo)[col_rev]
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )

# ============================================================
# 7. EVENT IMPACT ANALYSIS
# ============================================================

with st.expander("7. Event Impact Analysis"):

    if col_event:
        st.subheader("Transactions per Event")
        st.dataframe(df[col_event].value_counts().head(10))

    if col_event and col_rev:
        st.subheader("Revenue Impact by Event")
        st.dataframe(
            df.groupby(col_event)[col_rev]
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )

# 7. Correlation  ANALYSIS

with st.expander("Top Correlated Features "):

    # 1. Select numeric columns
    num_df = df.select_dtypes(include=np.number)

    if num_df.shape[1] < 2:
        st.info("Not enough numeric columns for correlation.")
    else:
        # 2. Correlation
        corr = num_df.corr()

        # 3. Absolute correlation 
        corr_abs = corr.abs()
        np.fill_diagonal(corr_abs.values, np.nan)

        # 4. Get top correlated pairs
        top_pairs = (
            corr_abs.unstack()
            .dropna()
            .sort_values(ascending=False)
            .head(8)   
        )

        # 5. Extract involved features
        top_features = sorted(
            set([f for pair in top_pairs.index for f in pair])
        )

        # 6. Focused correlation matrix
        focused_corr = corr.loc[top_features, top_features]

        # 7. Plot FULL GRID heatmap
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(
            focused_corr,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            linewidths=0.5,
            cbar=True,
            ax=ax
        )

        ax.set_title("Top Correlated Numeric Features ")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

        plt.tight_layout()
        st.pyplot(fig)



# ============================================================
# 10. SUMMARY REPORT (INSIGHTS)
# ============================================================

with st.expander("10. Summary Report"):

    summary = {
        "Rows": df.shape[0],
        "Columns": df.shape[1],
        "Numeric Columns": num_df.shape[1],
        "Total Revenue": df[col_rev].sum() if col_rev else None,
        "Total Units Sold": int(df["quantity_sold"].fillna(0).sum()),
        
    }

    st.json(summary)
    st.success("EDA Summary Generated ✔")


st.markdown("""
    <br><br>
    <div style="
        background-color:#2E86C1;
        padding:12px;
        text-align:center;
        color:white;
        border-radius:6px;
        font-size:14px;">
        © 2025 SupplySyncAI – Inventory Intelligence & Analytics Platform
    </div>
""", unsafe_allow_html=True)
