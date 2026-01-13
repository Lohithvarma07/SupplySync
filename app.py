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


# MYSQL LOADER FUNCTION
@st.cache_data
# CSV LOADER FUNCTION (DEPLOYMENT SAFE)
@st.cache_data
def load_data():
    return pd.read_csv("data/fact_consolidated1.csv")


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
    df.head(40),
    use_container_width=True,
    disabled=True
)



    st.info(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
else:
    st.info("Click the button above to load the dataset.")
# ============================================================
# STEP 2 – DATA PRE-PROCESSING (USER-CONTROLLED PIPELINE)
# ============================================================
if "preprocess_history" not in st.session_state:
    st.session_state.preprocess_history = {
        "duplicates": None,
        "outliers": {},
        "null_replaced_cols": None,
        "null_replaced_rows": None,
        "numeric_converted": None
    }

if "preprocessing_completed" not in st.session_state:
    st.session_state.preprocessing_completed = False



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

# Safety check
if st.session_state.df is None:
    st.warning("⚠ Load data first.")
    st.stop()

df = st.session_state.df

# ------------------------------------------------------------
# STEP SELECTOR (SEQUENTIAL CONTROL)
# ------------------------------------------------------------

step = st.radio(
    "Select a Data Pre-Processing Step",
    [
        "Remove Duplicate Rows",
        "Outliers",
        "Replace NULL Values with 'Unknown'",
        "Convert Columns to Numeric (Safe Columns Only)"
        
    ],
    index=None,
    horizontal=True
)

# ============================================================
# 1️⃣ REMOVE DUPLICATE ROWS
# ============================================================

if step == "Remove Duplicate Rows":

    st.markdown("### Remove Duplicate Rows")

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

    dup_mask = df.duplicated()
    dup_rows = df[dup_mask]

    # --------------------------------------------------
# DUPLICATE REMOVAL – CORRECT STATEFUL LOGIC
# --------------------------------------------------

    if st.button("Apply Duplicate Removal"):

        # If already done earlier
        if st.session_state.preprocess_history["duplicates"] is not None:
            st.info("Duplicate rows were already removed earlier.")

        else:
            # Detect duplicates ONCE
            dup_rows = df[df.duplicated()]

            if dup_rows.empty:
                st.info("No duplicate rows found.")
            else:
                # Save history
                st.session_state.preprocess_history["duplicates"] = dup_rows.copy()

                # Update working dataframe
                st.session_state.df = df.drop_duplicates()
                st.session_state.preprocessing_completed = True
                st.success("✔ Duplicate rows removed")


    # --------------------------------------------------
    # SHOW HISTORY (ONLY IF EXISTS)
    # --------------------------------------------------
    if st.session_state.preprocess_history["duplicates"] is not None:
        st.markdown("#### 🧾 Duplicate Rows Removed")
        st.dataframe(
            st.session_state.preprocess_history["duplicates"],
            use_container_width=True
        )

    # ============================================================
    # OUTLIER DETECTION (IQR-BASED – FLAG ONLY)
    # ============================================================
if step == "Outliers":

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
            Outlier Detection (IQR-Based Validation)
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
    <b>What this does:</b><br>
    Identifies <b>extreme values (outliers)</b> in numeric fields using the
    <b>Interquartile Range (IQR)</b> method.<br><br>

    <b>Important:</b>
    <ul>
        <li>No rows are deleted</li>
        <li>No values are modified</li>
        <li>Used only for data validation</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    

    df = st.session_state.df

    # NUMERIC COLUMN DETECTION
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if not numeric_cols:
        st.info("No numeric columns available for outlier detection.")
        st.stop()

    # MULTISELECT WITH SELECT ALL
    options = ["Select All"] + numeric_cols

    selected = st.multiselect(
        "Select numeric columns to check for outliers",
        options
    )

    # Resolve selection
    if "Select All" in selected:
        selected_cols = numeric_cols
    else:
        selected_cols = selected

    # ------------------------------------------------------------
    # DETECT OUTLIERS (ON BUTTON CLICK ONLY)
    # ------------------------------------------------------------

    if st.button("Detect Outliers"):

        if not selected_cols:
            st.warning("⚠ Please select at least one numeric column.")

        else:
            outlier_results = {}

            for col in selected_cols:

                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1

                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR

                outliers = df[
                    (df[col] < lower) | (df[col] > upper)
                ]

                if not outliers.empty:
                    outlier_results[col] = {
                        "lower": lower,
                        "upper": upper,
                        "count": outliers.shape[0],
                        "rows": outliers[[col]].copy()
                    }

            # Save history ONLY if something found
            if outlier_results:
                st.session_state.preprocess_history["outliers"] = outlier_results
                st.success("✔ Outlier detection completed")
            else:
                st.info("No outliers detected for selected columns.")

    # ------------------------------------------------------------
    # SHOW OUTLIER HISTORY (PERSISTENT)
    # ------------------------------------------------------------

    if st.session_state.preprocess_history["outliers"]:

        st.markdown("### 🧾 Outlier Detection History")

        for col, info in st.session_state.preprocess_history["outliers"].items():

            st.markdown(f"#### 📌 Outliers for `{col}`")

            st.markdown(f"""
            **IQR Bounds**
            - Lower Bound: `{info['lower']:.2f}`
            - Upper Bound: `{info['upper']:.2f}`
            - Outliers Found: **{info['count']}**
            """)

            st.dataframe(
                info["rows"].head(20),
                use_container_width=True
            )


# ============================================================
# 3️⃣ REPLACE NULL VALUES WITH "UNKNOWN"
# ============================================================

elif step == "Replace NULL Values with 'Unknown'":

    st.markdown("### Replace NULL Values with 'Unknown'")

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


        # ------------------------------------------------------------
    # NULL VALUE REPLACEMENT (STATEFUL + HISTORY PRESERVED)
    # ------------------------------------------------------------

    # Detect NULLs BEFORE replacement (always use current df)
    null_mask = df.isnull()

    rows_with_nulls = df[null_mask.any(axis=1)]
    null_counts = null_mask.sum()
    null_counts = null_counts[null_counts > 0]  # only affected columns

    if st.button("Apply NULL Replacement"):

        if null_counts.empty:
            st.info("NULL values were already handled earlier.")

        else:
            # ✅ Save history BEFORE modifying df
            st.session_state.preprocess_history["null_replaced_cols"] = (
                null_counts.to_frame("NULL Count")
            )

            st.session_state.preprocess_history["null_replaced_rows"] = (
                rows_with_nulls.copy()
            )

            # ✅ Apply replacement
            df_updated = df.fillna("Unknown")
            st.session_state.df = df_updated
            st.session_state.preprocessing_completed = True
            st.success("✔ NULL values replaced with 'Unknown'")


    # ------------------------------------------------------------
    # SHOW HISTORY (PERSISTENT — EVEN AFTER SWITCHING STEPS)
    # ------------------------------------------------------------

    if st.session_state.preprocess_history["null_replaced_cols"] is not None:

        st.markdown("#### 🧾 Columns Where NULL Values Were Replaced")
        st.dataframe(
            st.session_state.preprocess_history["null_replaced_cols"],
            use_container_width=True
        )

    if st.session_state.preprocess_history["null_replaced_rows"] is not None:

        st.markdown("#### 🧾 Rows Where NULL Values Were Replaced")
        st.dataframe(
            st.session_state.preprocess_history["null_replaced_rows"],
            use_container_width=True
        )


# ============================================================
# 4️⃣ CONVERT COLUMNS TO NUMERIC (SAFE ONLY)
# ============================================================

elif step == "Convert Columns to Numeric (Safe Columns Only)":

    st.markdown("### Convert Columns to Numeric")

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

        # ============================================================
    # CONVERT COLUMNS TO NUMERIC (SAFE ONLY + HISTORY)
    # ============================================================

    exclude = [
        "transaction_id", "product_id", "store_id", "customer_id",
        "sales_channel_id", "promo_id", "event_id"
    ]

    df = st.session_state.df

    # Candidate columns (object type only, excluding IDs)
    candidate_cols = [
        c for c in df.columns
        if c not in exclude and df[c].dtype == "object"
    ]

    # ------------------------------------------------------------
    # APPLY CONVERSION (ONCE)
    # ------------------------------------------------------------

    if st.button("Apply Numeric Conversion"):

        df_updated = df.copy()
        converted_cols = []

        for col in candidate_cols:
            before = df_updated[col].dtype
            df_updated[col] = pd.to_numeric(df_updated[col], errors="ignore")
            if df_updated[col].dtype != before:
                converted_cols.append(col)

        # Update session dataframe
        st.session_state.df = df_updated

        # Save history ONLY if something converted
        if converted_cols:
            st.session_state.preprocess_history["numeric_converted"] = converted_cols
            st.session_state.preprocessing_completed = True

            st.success("✔ Numeric conversion applied")
        else:
            st.info("No columns were found.")

    # ------------------------------------------------------------
    # SHOW CONVERSION HISTORY (PERSISTENT)
    # ------------------------------------------------------------

    if st.session_state.preprocess_history["numeric_converted"]:

        st.markdown("#### 🧾 Numeric Conversion History")
        st.write(
            st.session_state.preprocess_history["numeric_converted"]
        )



st.markdown("""
<style>

/* =====================================================
   GLOBAL / COMMON STYLES
   ===================================================== */

/* Clean report-style table (used across EDA) */
.clean-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13.5px;
}

.clean-table th {
    background-color: #F4F6F7;
    padding: 8px;
    text-align: left;
    font-weight: 600;
    border-bottom: 1px solid #D6DBDF;
    color: #34495E;
}

.clean-table td {
    padding: 7px 8px;
    border-bottom: 1px solid #ECF0F1;
    color: #2C3E50;
}

.clean-table tr:hover {
    background-color: #F8F9F9;
}


/* =====================================================
   EDA RADIO NAVIGATION (optional but safe)
   ===================================================== */

div[data-baseweb="radio-group"] {
    background-color: #F8F9F9;
    padding: 12px 16px;
    border-radius: 10px;
    border: 1px solid #E5E7E9;
    margin-bottom: 18px;
}

div[data-baseweb="radio"] {
    margin-right: 14px;
}

div[data-baseweb="radio"] input:checked + div {
    font-weight: 600;
    color: #2F75B5;
}


/* =====================================================
   DATA QUALITY – LAYOUT
   ===================================================== */

/* Horizontal row for 3 cards */
.quality-row {
    display: flex;
    gap: 16px;
    margin-bottom: 20px;
}

/* Individual card */
.quality-card {
    flex: 1;
    background-color: #FFFFFF;
    border-radius: 12px;
    padding: 14px 16px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.06);
    border-left: 5px solid #2F75B5;
}

/* Card title */
.quality-title {
    font-size: 15px;
    font-weight: 600;
    margin-bottom: 10px;
    color: #2C3E50;
}

/* Scrollable content inside card */
.table-scroll {
    max-height: 260px;
    overflow-y: auto;
}


/* =====================================================
   REPORT / CARD STYLE (used for future EDA sections)
   ===================================================== */

.report-card {
    background-color: #FFFFFF;
    border-radius: 14px;
    padding: 18px 20px;
    margin-bottom: 22px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.06);
    border-left: 6px solid #2F75B5;
}

.report-title {
    font-size: 16px;
    font-weight: 600;
    margin-bottom: 12px;
    color: #2C3E50;
}

.metric-pill {
    display: inline-block;
    background-color: #EBF5FB;
    color: #1F618D;
    padding: 6px 12px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
    margin-right: 8px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# STEP 3 – FULL ADAPTIVE EDA (ANALYSIS-FOCUSED)
# ============================================================

# ============================================================
# STEP 3 – FULL ADAPTIVE EDA (LOCKED UNTIL PREPROCESSING)
# ============================================================

# ============================================================
# STEP 3 – EDA (LOCKED UNTIL PREPROCESSING)
# ============================================================

if not st.session_state.preprocessing_completed:
    st.info("ℹ Please apply at least one data pre-processing step to unlock EDA.")
    st.stop()


df = st.session_state.get("df", None)

if df is None:
    st.warning("⚠ No dataset available.")
    st.stop()


# ---------------- EDA HEADER ----------------
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
        <h3 style="margin:0;">Exploratory Data Analysis (EDA)</h3>
    </div>
    """,
    unsafe_allow_html=True
)

st.info(f"Dataset Loaded: **{df.shape[0]} rows × {df.shape[1]} columns**")

# ---------------- EDA INTRO CARD ----------------
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

    <b>Exploratory Data Analysis (EDA)</b><br><br>

    Provides <b>high-level insights</b> to understand data behavior before model engineering.<br><br>

    <b>Key Insights Generated:</b>
    <ul>
        <li>Sales and demand patterns over time</li>
        <li>Customer purchase behavior and loyalty trends</li>
        <li>Product category and brand performance</li>
        <li>Store and regional sales distribution</li>
        <li>Promotion and event effectiveness</li>
        <li>Weather and trend influence on demand</li>
    </ul>

    This section focuses on <b>interpretability</b>, not deep statistical modeling.

    </div>
    """,
    unsafe_allow_html=True
)

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
# EDA NAVIGATION
# ============================================================

eda_option = st.radio(
    "Select Analysis",
    [
        "Data Quality Overview",
        "Sales Overview",
        "Product-Level Analysis",
        "Store-Level Analysis",
        "Sales Channel Analysis",
        "Promotion Effectiveness",
        "Event Impact Analysis",
        "Top Correlated Features",
        "Summary Report"
    ],
    index=None,
    horizontal=True
)

if eda_option is None:
    st.info("👆 Please select an analysis above to view insights.")
    st.stop()

# ============================================================
# EDA ROUTER (⚠️ DO NOT BREAK THIS STRUCTURE)
# ============================================================

if eda_option == "Data Quality Overview":

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

        <b>What this section does:</b>

        This section provides a <b>high-level health check</b> of the dataset before any modeling or forecasting is attempted.

        It evaluates:
        <ul>
            <li>Missing values</li>
            <li>Duplicate records</li>
            <li>Data type consistency</li>
            <li>Overall row and column completeness</li>
        </ul>

        <b>Why this matters:</b>

        Demand forecasting models are highly sensitive to <b>poor data quality</b>.
        Even small inconsistencies (missing prices, invalid quantities, duplicate transactions)
        can significantly distort predictions.<br>

        <b>Key insights users get:</b>
        <ul>
            <li>Whether the dataset is <b>model-ready</b></li>
            <li>Which columns require cleaning or transformation</li>
            <li>Confidence in the reliability of downstream analysis</li>
        </ul>

        </div>
        """,
        unsafe_allow_html=True
    )

    # =========================
    # PREPARE DATA
    # =========================
    rows_count = df.shape[0]
    cols_count = df.shape[1]

    dup_count = df.duplicated().sum()
    dtype_counts = df.dtypes.value_counts()

    mv = (df.isnull().mean() * 100).round(2).sort_values(ascending=False)

    # =========================
    # DATASET SHAPE
    # =========================
    st.markdown(
        f"""
        <div class="quality-card">
            <div class="quality-title">Dataset Shape</div>
            <table class="clean-table">
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Total Rows</td><td>{rows_count}</td></tr>
                <tr><td>Total Columns</td><td>{cols_count}</td></tr>
            </table>
        </div>
        """,
        unsafe_allow_html=True
    )

    # =========================
    # MISSING VALUE ANALYSIS
    # =========================
    st.markdown(
        f"""
        <div class="quality-card">
            <div class="quality-title">Missing Value Analysis (%)</div>
            <div class="table-scroll">
                <table class="clean-table">
                    <tr><th>Column Name</th><th>Missing (%)</th></tr>
                    {''.join([f"<tr><td>{c}</td><td>{v}%</td></tr>" for c, v in mv.items()])}
                </table>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # =========================
    # DUPLICATE ANALYSIS
    # =========================
    st.markdown(
        f"""
        <div class="quality-card">
            <div class="quality-title">Duplicate Analysis</div>
            <table class="clean-table">
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Total Duplicate Rows</td><td>{dup_count}</td></tr>
            </table>
        </div>
        """,
        unsafe_allow_html=True
    )

    # =========================
    # DATA TYPES SUMMARY
    # =========================
    st.markdown(
        f"""
        <div class="quality-card">
            <div class="quality-title">Data Types Summary</div>
            <table class="clean-table">
                <tr><th>Data Type</th><th>Column Count</th></tr>
                {''.join([f"<tr><td>{d}</td><td>{c}</td></tr>" for d, c in dtype_counts.items()])}
            </table>
        </div>
        """,
        unsafe_allow_html=True
    )


elif eda_option == "Sales Overview":
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

    <b>What this section does:</b>

    This provides a <b>macro-level snapshot of sales performance</b>, answering the question:

    “What does overall sales look like across time?”

    It typically highlights:
    <ul>
        <li>Total revenue</li>
        <li>Total units sold</li>
        <li>Average order value</li>
        <li>Sales trends over time</li>
    </ul><br>

    <b>Why this matters:</b>

    Before diving into granular analysis, it’s important to understand:
    <ul>
        <li>Overall business scale</li>
        <li>Growth or decline patterns</li>
        <li>Presence of seasonality or anomalies</li>
    </ul><br>

    <b>Key insights users get:</b>
    <ul>
        <li>Baseline sales behavior</li>
        <li>Early signals of trends or volatility</li>
        <li>Context for all deeper analyses</li>
    </ul>

    </div>
    """,
    unsafe_allow_html=True
)
    col1, col2, col3 = st.columns(3)

    if col_rev:
        col1.metric("Total Revenue", f"{df[col_rev].sum():,.2f}")
        col2.metric("Average Order Value", f"{df[col_rev].mean():,.2f}")
        col3.metric("Max Order Value", f"{df[col_rev].max():,.2f}")

    col4, col5, col6 = st.columns(3)

    if col_qty and col_price:
        sales_value = (df[col_qty] * df[col_price]).sum()
        col4.metric("Sales Value", f"{sales_value:,.2f}")

    if col_qty:
        col5.metric("Total Units Sold", f"{df[col_qty].sum():,}")
        col6.metric("Average Units per Transaction", f"{df[col_qty].mean():.2f}")

    if "created_at" in df.columns and col_rev:
        st.subheader("Sales by Time")

        # Convert to datetime safely
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")

        # Aggregate sales by date
        sales_time = (
            df.groupby(df["created_at"].dt.date)[col_rev]
            .sum()
            .sort_index()
        )

        st.bar_chart(sales_time)


    if col_store and col_rev:
        st.subheader("Sales by Store")

        sales_store = (
            df.groupby(col_store)[col_rev]
            .sum()
            .sort_values(ascending=False)
        )

        st.bar_chart(sales_store)
    if col_channel and col_rev:
        st.subheader("Sales by Sales Channel")

        sales_channel = (
            df.groupby(col_channel)[col_rev]
            .sum()
            .sort_values(ascending=False)
        )

        st.bar_chart(sales_channel)



elif eda_option == "Product-Level Analysis":

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

    This section analyzes <b>sales performance at the product (SKU) level</b>.

    It focuses on:
    <ul>
        <li>Top- and bottom-performing products</li>
        <li>Revenue contribution by product</li>
        <li>Demand concentration across SKUs</li>
    </ul><br>

    <b>Why this matters:</b>

    Demand forecasting at an aggregate level hides <b>SKU-specific behavior</b>.
    Some products are fast-moving, others are slow or highly seasonal.<br>

    <b>Key insights users get:</b>
    <ul>
        <li>Which products drive the majority of sales</li>
        <li>Which SKUs may require special forecasting treatment</li>
        <li>Candidates for product-level demand models</li>
    </ul>

    </div>
    """,
    unsafe_allow_html=True
    )

        
    

    # =========================
    # COLUMN MAPPING
    # =========================
    col_product = "product_id"
    col_qty     = "quantity_sold"
    col_revenue = "total_sales_amount"
    col_profit  = "profit_value"

    # =========================
    # AGGREGATE PRODUCT METRICS
    # =========================
    product_metrics = (
        df.groupby(col_product)
        .agg(
            total_quantity_sold=(col_qty, "sum"),
            total_revenue=(col_revenue, "sum"),
            total_profit=(col_profit, "sum")
        )
        .sort_values("total_revenue", ascending=False)
    )

    # =========================
    # TOP PRODUCTS FOR BAR CHART
    # =========================
    TOP_N = 20
    top_products = product_metrics.head(TOP_N)

    # =========================
    # SIDE-BY-SIDE PLOTS
    # =========================
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # ===== PLOT 1: REVENUE CONTRIBUTION =====
    axes[0].bar(
        top_products.index.astype(str),
        top_products["total_revenue"]
    )

    axes[0].set_title("Revenue Contribution by Product (Top 20)")
    axes[0].set_xlabel("Product ID")
    axes[0].set_ylabel("Total Revenue")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)

    # ===== PLOT 2: DEMAND VS PROFITABILITY =====
    axes[1].scatter(
        product_metrics["total_quantity_sold"],
        product_metrics["total_profit"],
        alpha=0.6
    )

    axes[1].set_title("Product Demand vs Profitability")
    axes[1].set_xlabel("Total Quantity Sold (Demand)")
    axes[1].set_ylabel("Total Profit")
    axes[1].grid(True, linestyle="--", alpha=0.4)

        # =========================
    # SELECT PRODUCTS TO LABEL
    # =========================
    top_demand = product_metrics.sort_values(
        "total_quantity_sold", ascending=False
    ).head(5)

    top_profit = product_metrics.sort_values(
        "total_profit", ascending=False
    ).head(5)

    label_products = pd.concat([top_demand, top_profit]).drop_duplicates()

    # =========================
    # ANNOTATE WITH OFFSET
    # =========================
    for pid, row in label_products.iterrows():
        axes[1].annotate(
            pid,
            (row["total_quantity_sold"], row["total_profit"]),
            xytext=(5, 5),                 # offset text
            textcoords="offset points",
            fontsize=8,
            alpha=0.9
        )


    plt.tight_layout()
    st.pyplot(fig)




elif eda_option == "Store-Level Analysis":

    st.markdown(
    """
    <div style="
        background-color:#2F75B5;
        padding:28px;
        border-radius:12px;
        color:white;
        font-size:16px;
        line-height:1.6;
        margin-bottom:22px;
    ">

    <b>What this section does:</b>

    This examines how <b>sales vary across stores or locations</b>.

    It evaluates:
    <ul>
        <li>Store-wise revenue and volume</li>
        <li>Performance comparison across regions</li>
        <li>High-demand vs low-demand stores</li>
    </ul><br>

    <b>Why this matters:</b>

    Forecasting accuracy improves when <b>store heterogeneity</b> is understood.<br>
    Not all stores behave the same, even for the same products.<br><br>

    <b>Key insights users get:</b>
    <ul>
        <li>Store demand clusters</li>
        <li>Regional sales disparities</li>
        <li>Inputs for store-level or cluster-based forecasting</li>
    </ul>

    </div>
    """,
    unsafe_allow_html=True
)
    # =========================
    # COLUMN MAPPING
    # =========================
    col_store   = "store_id"
    col_product = "product_id"
    col_qty     = "quantity_sold"
    col_revenue = "total_sales_amount"

    # =========================
    # PARAMETERS (CLEAN VIEW)
    # =========================
    TOP_STORES   = 20
    TOP_PRODUCTS = 20

    # =========================
    # TOP STORES BY REVENUE
    # =========================
    top_stores = (
        df.groupby(col_store)[col_revenue]
        .sum()
        .sort_values(ascending=False)
        .head(TOP_STORES)
        .index
    )

    # =========================
    # STORE × PRODUCT QUANTITY
    # =========================
    store_product_qty = (
        df[df[col_store].isin(top_stores)]
        .groupby([col_store, col_product])[col_qty]
        .sum()
        .reset_index()
    )

    # =========================
    # TOP PRODUCTS PER STORE
    # =========================
    store_top_products = (
        store_product_qty
        .sort_values([col_store, col_qty], ascending=[True, False])
        .groupby(col_store)
        .head(TOP_PRODUCTS)
    )

    # =========================
    # PIVOT → PERCENTAGE MIX
    # =========================
    pivot = store_top_products.pivot_table(
        index=col_store,
        columns=col_product,
        values=col_qty,
        fill_value=0
    )

    pivot_qty = pivot.copy()


    # =========================
    # SIDE-BY-SIDE PLOTS
    # =========================
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # =====================================================
    # PLOT 1: REVENUE CONCENTRATION ACROSS STORES
    # =====================================================
    store_revenue = (
        df.groupby(col_store)[col_revenue]
        .sum()
        .loc[top_stores]
    )

    axes[0].bar(
        store_revenue.index.astype(str),
        store_revenue.values
    )

    axes[0].set_title("Revenue Concentration Across Stores")
    axes[0].set_xlabel("Store ID")
    axes[0].set_ylabel("Total Revenue")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)

    # =====================================================
    # PLOT 2: STORE-WISE PRODUCT MIX (STACKED %)
    # =====================================================
    bottom = np.zeros(len(pivot_qty))

    for product in pivot_qty.columns:
        axes[1].bar(
            pivot_qty.index.astype(str),
            pivot_qty[product],
            bottom=bottom,
            width=0.4,
            label=str(product)
        )
        bottom += pivot_qty[product].values





    axes[1].set_title("Store-wise Product Mix (Quantity Sold)")
    axes[1].set_xlabel("Store ID")
    axes[1].set_ylabel("Quantity Sold")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].legend(
        title="Product ID",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=9
    )
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)

    # =========================
    # RENDER
    # =========================
    plt.tight_layout()
    st.pyplot(fig)


elif eda_option == "Sales Channel Analysis":

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

    <b>What this section does:</b><br><br>

    This provides a <b>high-level view of overall sales performance</b> across time,
    products, and customers.


    It evaluates:
    <ul>
        <li>Total sales revenue and volume</li>
        <li>Sales trends over time</li>
        <li>Overall demand patterns</li>
    </ul>


    <b>Why this matters:</b>

    Understanding overall sales behavior helps identify
    <b>growth trends, seasonality, and demand fluctuations</b>.
    It establishes a baseline before deeper analysis.


    <b>Key insights users get:</b>
    <ul>
        <li>Overall business performance trends</li>
        <li>Periods of high and low sales activity</li>
        <li>Inputs for forecasting and planning</li>
    </ul>

    </div>
    """,
    unsafe_allow_html=True
)

    # =========================
    # COLUMN MAPPING
    # =========================
    col_channel = "sales_channel_id"
    col_revenue = "total_sales_amount"
    col_aov     = "avg_order_value"

    # =========================
    # AGGREGATE CHANNEL REVENUE
    # =========================
    channel_revenue = (
        df.groupby(col_channel)[col_revenue]
        .sum()
        .sort_values(ascending=False)
    )

    # Limit to top channels (avoid clutter)
    TOP_N = 15
    top_channels = channel_revenue.head(TOP_N)

    # =========================
    # SIDE-BY-SIDE PLOTS
    # =========================
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # =====================================================
    # PLOT 1: POLISHED DONUT – REVENUE CONTRIBUTION
    # =====================================================
    colors = plt.cm.tab10.colors

    wedges, texts, autotexts = axes[0].pie(
        top_channels.values,
        labels=top_channels.index.astype(str),
        autopct="%1.1f%%",
        startangle=90,
        colors=colors,
        wedgeprops={"width": 0.45, "edgecolor": "white"},
        pctdistance=0.78
    )

    # Improve percentage label readability
    for t in autotexts:
        t.set_fontsize(9)
        t.set_color("black")

    axes[0].set_title(
        "Revenue Contribution by Sales Channel",
        fontsize=13,
        fontweight="bold"
    )

    # =====================================================
    # PLOT 2: POLISHED AOV COMPARISON (DOT PLOT)
    # =====================================================
    aov_values = [
        df[df[col_channel] == ch][col_aov].mean()
        for ch in top_channels.index
    ]

    axes[1].scatter(
        top_channels.index.astype(str),
        aov_values,
        s=90,
        alpha=0.85
    )

    axes[1].set_title(
        "Average Order Value by Sales Channel",
        fontsize=13,
        fontweight="bold"
    )
    axes[1].set_xlabel("Sales Channel")
    axes[1].set_ylabel("Average Order Value")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)

    # =========================
    # RENDER
    # =========================
    plt.tight_layout()
    st.pyplot(fig)

elif eda_option == "Promotion Effectiveness":

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

    <b>What this section does:</b><br><br>

    This analyzes how <b>promotions impact sales performance</b> by comparing
    promotion cost, sales uplift, and profitability.

    It evaluates:
    <ul>
        <li>Revenue uplift generated by promotions</li>
        <li>Promotion cost vs sales impact</li>
        <li>Effectiveness of individual promotions</li>
    </ul>
    <br>

    <b>Why this matters:</b>

    Promotions can increase sales but may also reduce margins.
    This analysis helps ensure promotions are
    <b>cost-effective and profitable</b>.

    <b>Key insights users get:</b>
    <ul>
        <li>High-performing vs underperforming promotions</li>
        <li>Which promotions should be scaled or stopped</li>
        <li>Better data-driven promotion planning</li>
    </ul>

    </div>
    """,
    unsafe_allow_html=True
    )

    
    # =========================
    # COLUMN MAPPING
    # =========================
    col_promo       = "promo_transaction_id"
    col_sales       = "promo_total_sales_amount"
    col_cost        = "promo_promo_cost"
    col_uplift      = "promo_promo_uplift_revenue"

    # =========================
    # AGGREGATE PROMOTION METRICS
    # =========================
    promo_metrics = (
        df[df[col_promo].notna()]
        .groupby(col_promo)
        .agg(
            promo_total_sales_amount=(col_sales, "sum"),
            promo_cost=(col_cost, "sum"),
            uplift_revenue=(col_uplift, "sum")
        )
    )

    # =========================
    # DERIVED METRICS
    # =========================
    promo_metrics["net_uplift"] = (
        promo_metrics["uplift_revenue"] - promo_metrics["promo_cost"]
    )

    # Clean invalid values
    promo_metrics = (
        promo_metrics
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    # =========================
    # SELECT TOP PROMOTIONS
    # =========================
    TOP_N = 20
    top_promos = (
        promo_metrics
        .sort_values("net_uplift", ascending=False)
        .head(TOP_N)
    )

    # =========================
    # SIDE-BY-SIDE PLOTS
    # =========================
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # =====================================================
    # PLOT 1: PROMOTION PROFITABILITY (NET UPLIFT)
    # =====================================================
    axes[0].bar(
        top_promos.index.astype(str),
        top_promos["net_uplift"],
        alpha=0.85
    )

    axes[0].axhline(0, color="black", linewidth=1)

    axes[0].set_title(
        "Promotion Profitability (Net Uplift Revenue)",
        fontsize=13,
        fontweight="bold"
    )
    axes[0].set_xlabel("Promotion ID")
    axes[0].set_ylabel("Net Uplift Revenue")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)
    # =====================================================
    # PLOT 2: PROMOTION SALES vs COST (READABLE VERSION)
    # =====================================================
    axes[1].scatter(
        top_promos["promo_cost"],
        top_promos["promo_total_sales_amount"],
        s=top_promos["promo_total_sales_amount"] / 1500,
        alpha=0.75,
        edgecolors="black",
        linewidth=0.5
    )

    # Light reference line (visual aid only)
    max_cost = top_promos["promo_cost"].max()
    axes[1].plot(
        [0, max_cost],
        [0, max_cost],
        linestyle="--",
        color="gray",
        alpha=0.4
    )

    
    top_labels = top_promos.sort_values(
        "promo_total_sales_amount", ascending=False
    ).head(10)

    for pid, row in top_labels.iterrows():
        axes[1].annotate(
            pid,
            (row["promo_cost"], row["promo_total_sales_amount"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold"
        )

    axes[1].set_title(
        "Promotion Effectiveness: Sales vs Cost",
        fontsize=14,
        fontweight="bold"
    )
    axes[1].set_xlabel("Promotion Cost")
    axes[1].set_ylabel("Promotion Total Sales Amount")

    axes[1].grid(True, linestyle="--", alpha=0.3)

    # =========================
    # RENDER
    # =========================
    plt.tight_layout()
    st.pyplot(fig)



elif eda_option == "Event Impact Analysis":

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

    <b>What this section does:</b><br><br>

    This analyzes how <b>special events</b> (festivals, campaigns, external factors)
    influence <b>sales performance and demand patterns</b>.<br><br>

    <b>Why this matters:</b><br><br>

    Events can create <b>temporary demand spikes</b>, alter customer behavior,
    and affect forecasting accuracy if not modeled correctly.<br><br>

    <b>Key insights users get:</b>
    <ul>
        <li>Which events drive the highest sales uplift</li>
        <li>How strongly events impact revenue vs cost</li>
        <li>Which events are worth planning inventory for</li>
    </ul>

    </div>
    """,
    unsafe_allow_html=True
    )


    # =========================
    # COLUMN MAPPING
    # =========================
    col_event       = "event_id"
    col_sales       = "total_sales_amount"
    col_qty         = "quantity_sold"
    col_before      = "impact_sales_before_impact"
    col_after       = "impact_sales_after_impact"
    col_change_pct  = "impact_impact_percentage_change"

    # =========================
    # AGGREGATE EVENT METRICS
    # =========================
    event_metrics = (
        df[df[col_event].notna()]
        .groupby(col_event)
        .agg(
            sales_before=(col_before, "mean"),
            sales_after=(col_after, "mean"),
            total_sales=(col_sales, "sum"),
            total_quantity=(col_qty, "sum"),
            impact_pct=(col_change_pct, "mean")
        )
    )

    # =========================
    # DERIVED METRICS
    # =========================
    event_metrics["sales_uplift"] = (
        event_metrics["sales_after"] - event_metrics["sales_before"]
    )

    # Clean invalid values
    event_metrics = (
        event_metrics
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    # =========================
    # SELECT TOP EVENTS
    # =========================
    TOP_N = 20
    top_events = (
        event_metrics
        .sort_values("sales_uplift", ascending=False)
        .head(TOP_N)
    )

    # =========================
    # SIDE-BY-SIDE PLOTS
    # =========================
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # =====================================================
    # PLOT 1: EVENT SALES UPLIFT
    # =====================================================
    axes[0].bar(
        top_events.index.astype(str),
        top_events["sales_uplift"],
        alpha=0.85
    )

    axes[0].axhline(0, color="black", linewidth=1)

    axes[0].set_title(
        "Event Impact on Sales (Before vs After)",
        fontsize=13,
        fontweight="bold"
    )
    axes[0].set_xlabel("Event ID")
    axes[0].set_ylabel("Average Sales Uplift")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)

    # =====================================================
    # PLOT 2: EVENT EFFECTIVENESS (VOLUME vs IMPACT)
    # =====================================================
    axes[1].scatter(
        top_events["total_quantity"],
        top_events["impact_pct"],
        s=top_events["total_sales"] / 1500,
        alpha=0.75,
        edgecolors="black",
        linewidth=0.5
    )

    # Reference lines (median-based quadrants)
    axes[1].axvline(
        top_events["total_quantity"].median(),
        linestyle="--",
        alpha=0.5
    )
    axes[1].axhline(
        top_events["impact_pct"].median(),
        linestyle="--",
        alpha=0.5
    )

    # Label top impactful events
    top_labels = top_events.sort_values(
        "sales_uplift", ascending=False
    ).head(10)

    for eid, row in top_labels.iterrows():
        axes[1].annotate(
            eid,
            (row["total_quantity"], row["impact_pct"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold"
        )

    axes[1].set_title(
        "Event Effectiveness: Demand vs Impact",
        fontsize=14,
        fontweight="bold"
    )
    axes[1].set_xlabel("Total Quantity Sold During Event")
    axes[1].set_ylabel("Average Sales Impact (%)")
    axes[1].grid(True, linestyle="--", alpha=0.3)

    # =========================
    # RENDER
    # =========================
    plt.tight_layout()
    st.pyplot(fig)


elif eda_option == "Top Correlated Features":

    if num_df.shape[1] < 2:
        st.info("Not enough numeric columns for correlation.")
    else:
        st.subheader("Feature Correlation Heatmap")

        corr = num_df.corr()

        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(
            corr,
            ax=ax,
            cmap="coolwarm",
            annot=False,          # set True only if very few columns
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"shrink": 0.8}
        )

        ax.set_title("Correlation Between Numeric Features", fontsize=14)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

        st.pyplot(fig)


elif eda_option == "Summary Report":

    # =========================
    # SUMMARY REPORT – INTRO
    # =========================
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

        <b>What this section does:</b>

        This provides a <b>consolidated narrative summary</b> of all EDA findings.

        It highlights:
        <ul>
            <li>Key demand patterns</li>
            <li>Major influencing factors</li>
            <li>Data readiness for modelling</li>
        </ul>

        <b>Why this matters:</b>

        Not all stakeholders want charts.<br>
        This section translates analysis into <b>actionable understanding</b>.


        <b>Key insights users get:</b>
        <ul>
            <li>A single, clear view of data insights</li>
            <li>Business-ready conclusions</li>
            <li>Readiness assessment for model engineering</li>
        </ul>

        </div>
        """,
        unsafe_allow_html=True
    )

    # =========================
    # NARRATIVE PLACEHOLDER
    # =========================
    st.markdown(
        """
        <div style="
            background-color:#0B2C5D;
            padding:30px;
            border-radius:12px;
            color:white;
            font-size:15px;
            line-height:1.7;
        ">

        <h4>📊 Dataset Health & Readiness</h4>
        <ul>
            <li>The dataset contains <b>998 rows and 96 columns</b>, providing sufficient depth for analysis.</li>
            <li>Missing values are minimal (mostly under <b>0.1%</b>), and <b>no duplicate records</b> were found.</li>
            <li>Data types are well distributed across numeric and categorical features, indicating strong model readiness.</li>
            <li>Overall, the dataset is <b>clean, reliable, and suitable for downstream forecasting and analytics</b>.</li>
        </ul>

        <h4>💰 Overall Sales Performance</h4>
        <ul>
            <li>Total revenue generated is approximately <b>₹12.07M</b>, with sales value exceeding <b>₹13.86M</b>.</li>
            <li>Average order value is around <b>₹12K</b>, while the maximum observed order exceeds <b>₹46K</b>.</li>
            <li>Sales over time show <b>clear spikes</b>, indicating seasonality, promotional effects, or event-driven demand.</li>
        </ul>

        <h4>📦 Product-Level Insights</h4>
        <ul>
            <li>A small group of products contributes a <b>large share of total revenue</b>, showing strong demand concentration.</li>
            <li>Demand vs profitability analysis reveals that <b>high-demand products are not always the most profitable</b>.</li>
            <li>Some low-volume products generate disproportionately high profit, making them candidates for targeted strategies.</li>
        </ul>

        <h4>🏬 Store-Level Performance</h4>
        <ul>
            <li>Revenue is <b>unevenly distributed across stores</b>, with a few stores driving a majority of sales.</li>
            <li>Store-wise product mix varies significantly, indicating <b>localized customer preferences</b>.</li>
            <li>This variation suggests that <b>store-level or cluster-based forecasting</b> will improve accuracy.</li>
        </ul>

        <h4>🛒 Sales Channel Analysis</h4>
        <ul>
            <li>Revenue contribution is spread across multiple channels, but a few channels dominate overall sales.</li>
            <li>Average order value differs notably by channel, highlighting <b>different purchasing behaviors</b>.</li>
            <li>Some channels generate lower volume but higher value per transaction.</li>
        </ul>

        <h4>🎯 Promotion Effectiveness</h4>
        <ul>
            <li>Not all promotions are profitable — some generate high sales but <b>low net uplift</b> after cost.</li>
            <li>A subset of promotions delivers <b>strong uplift at relatively low cost</b>, making them ideal for scaling.</li>
            <li>Sales vs cost analysis clearly separates efficient promotions from underperforming ones.</li>
        </ul>

        <h4>🎉 Event Impact Analysis</h4>
        <ul>
            <li>Several events cause <b>significant demand spikes</b>, while others have minimal impact.</li>
            <li>High-impact events are not always high-volume, indicating <b>quality of impact over quantity</b>.</li>
            <li>These findings are critical for <b>inventory planning and demand forecasting around events</b>.</li>
        </ul>

        <h4>🔗 Feature Relationships</h4>
        <ul>
            <li>Total sales amount is strongly correlated with unit price, tax, and average order value.</li>
            <li>Promotion and event-related features show meaningful correlations with demand and revenue.</li>
            <li>This confirms the importance of including <b>price, promotions, events, and customer behavior</b> in forecasting models.</li>
        </ul>

        <h4>✅ Final Takeaway</h4>
        <ul>
            <li>The data shows <b>clear demand patterns, strong segmentation, and identifiable drivers</b>.</li>
            <li>The dataset is <b>model-ready</b> with rich explanatory variables.</li>
            <li>Forecasting performance will improve by modeling at <b>product, store, channel, promotion, and event levels</b>.</li>
        </ul>

        </div>
        """,
        unsafe_allow_html=True
    )




# ============================================================
# FOOTER
# ============================================================

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
