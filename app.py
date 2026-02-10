import streamlit as st
import pandas as pd
import mysql.connector
import seaborn as sns
import matplotlib.pyplot as plt
import io
import numpy as np
from utils.html_table import render_html_table
import streamlit as st
import altair as alt



st.set_page_config(page_title="SupplySyncAI – MLOps UI", layout="wide")

st.markdown("""
<style>

/* App background */
.stApp {
    background-color: #FFFFFF;
    margin: 0;
    padding: 0;
}

/* 🔥 Remove ALL app-level top spacing */
.block-container {
    padding-top: 0rem !important;
    margin-top: 0rem !important;
}

/* 🔥 Remove internal main section padding */
section.main > div:first-child {
    padding-top: 0rem !important;
    margin-top: 0rem !important;
}

/* ✅ KEEP header visible (do NOT hide it) */
header {
    position: relative;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

/* =========================================
   RADIO CONTAINER – FULL WIDTH
   ========================================= */
div.stRadio {
    width: 100%;
    padding: 0 !important;
    margin: 0 !important;
}

/* =========================================
   Teal WRAP BOX – FULL PAGE WIDTH
   ========================================= */
div.stRadio > div {
    background-color:  #0D9488;
    padding: 16px 400px;
    border-radius: 8px;
    width: 100%;              
    box-sizing: border-box;
}

/* =========================================
   RADIO GROUP ALIGNMENT
   ========================================= */
div[data-baseweb="radio-group"] {
    display: flex;
    justify-content: center;  /* center options inside */
}

/* =========================================
   RADIO OPTION TEXT
   ========================================= */
/* RADIO LABEL TEXT – FORCE WHITE */
div[data-baseweb="radio"] label,
div[data-baseweb="radio"] label span {
    font-size: 18px !important;
    font-weight: 800 !important;
    color: #FFFFFF !important;
}


/* =========================================
   SPACE BETWEEN OPTIONS
   ========================================= */
div[data-baseweb="radio"] {
    margin-right: 28px;
}

          

</style>
""", unsafe_allow_html=True)


st.markdown(""" 
 <style> /* Expander outer card */ 
    div[data-testid="stExpander"]
        { background-color: #2F75B5;
        border-radius: 20px; 
        border: 1px solid #9EDAD0; 
        overflow: hidden; /* 🔑 fixes unfinished edges */ }
    /* Hide expander header completely */
    div[data-testid="stExpander"]:nth-of-type(1)
             summary { display: none; }
    /* Inner content padding fix */
     div[data-testid="stExpander"]:nth-of-type(1) > 
            div { padding: 22px 18px; } 
            </style> """, unsafe_allow_html=True)







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

st.markdown("""
<style>

/* =========================================
   SUMMARY GRID (CENTERED, SMALL, EQUAL BOXES)
   ========================================= */
.summary-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 14px;
    margin: 6px 0 10px 0;
    justify-content: center;
    
}

/* =========================================
   SUMMARY CARD (TABLE CONTAINER)
   ========================================= */
.summary-card {
    border: 2px solid #6B7280;
    border-radius: 2px;
    background-color: #E5E7EB;
    overflow: hidden;
    text-align: center;
}

/* =========================================
   HEADER ROW (NO WRAP, SAME HEIGHT)
   ========================================= */
.summary-title {
    background-color: #9CA3AF;
    color: #000000;
    font-size: 14px;
    font-weight: 700;
    padding: 8px 6px;
    border-bottom: 1px solid #6B7280;

    white-space: nowrap;       /* 🔥 stop wrapping */
    overflow: hidden;
    text-overflow: ellipsis;
}

/* =========================================
   VALUE CELL (COMPACT)
   ========================================= */
.summary-value {
    font-size: 22px;
    font-weight: 600;
    color: #000000;
    padding: 1px 0;
}

</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>
/* Outer gray wrap */
.gray-analytics-wrap {
    background-color: #E6E6E6;
    padding: 16px 400px;
    border-radius: 8px;
    width: 100%;              
    box-sizing: border-box;
}

/* Inner blue analytics bar */
.analytics-container {
    background-color:#1F6FB2;
    padding:18px;
    border-radius:14px;
}
</style>
""", unsafe_allow_html=True)



st.markdown(
    """
    <div style="
        background-color:#0B2C5D;
        padding:35px;
        border-radius:12px;
        color:white;
        text-align:center;
        margin:0 0 20px 0;
    ">
        <h1 style="margin:0 0 8px 0;">
            AI-Powered Demand Forecasting & Sales Prediction Engine
        </h1>
        <h3 style="font-weight:400; margin:0;">
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
    return pd.read_csv("data/fact_consolidated.csv")


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
    "<h3 style='color:#000000;'>Data Preview</h3>",
    unsafe_allow_html=True
)

    render_html_table(
        df.head(20),
        max_height=260
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
st.markdown(
    "<div style='font-size:20px; font-weight:600; margin-bottom:8px;'>"
    "Select a Data Pre-Processing Step"
    "</div>",
    unsafe_allow_html=True
)
st.write("")


step = st.radio(
    "",
    [
        "Remove Duplicate Rows",
        "Remove Outliers",
        "Replace Missing Values"
    ],
    index=None,
    horizontal=True,
    label_visibility="collapsed"

)


# ============================================================
# 1️⃣ REMOVE DUPLICATE ROWS
# ============================================================

if step == "Remove Duplicate Rows":

    st.markdown("### Remove Duplicate Rows")
    st.write("")

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

    # --------------------------------------------------
    # DUPLICATE REMOVAL – FIXED BEFORE / AFTER LOGIC
    # --------------------------------------------------

    # Init session keys (SAFE)
    if "dup_before_df" not in st.session_state:
        st.session_state.dup_before_df = None
    if "dup_after_df" not in st.session_state:
        st.session_state.dup_after_df = None
    if "dup_removed_df" not in st.session_state:
        st.session_state.dup_removed_df = None


    if st.button("Apply Duplicate Row Removal"):
        st.write("")
        st.write("")
        # Prevent re-run
        if st.session_state.dup_removed_df is not None:
            st.info("Duplicate rows were already removed earlier.")

        else:
            # 🔒 SNAPSHOT BEFORE (CRITICAL)
            before_df = st.session_state.df.copy()

            # Detect duplicates from BEFORE snapshot
            dup_mask = before_df.duplicated()
            dup_rows = before_df[dup_mask]

            if dup_rows.empty:
                st.info("No duplicate rows found.")
            else:
                # Cleaned version
                after_df = before_df.drop_duplicates().reset_index(drop=True)


                # ✅ STORE ALL THREE STATES (IMMUTABLE)
                st.session_state.dup_before_df = before_df
                st.session_state.dup_removed_df = dup_rows
                st.session_state.dup_after_df = after_df

                # ✅ UPDATE WORKING DF ONLY ONCE
                st.session_state.df = after_df
                st.session_state.preprocessing_completed = True

                st.success("✔ Duplicate rows removed")


    # --------------------------------------------------
    # OUTPUT SECTION – ALWAYS USE SNAPSHOTS
    # --------------------------------------------------

    if st.session_state.dup_removed_df is not None:

        before_df = st.session_state.dup_before_df   # 🔒 frozen
        after_df = st.session_state.dup_after_df     # 🔒 frozen
        removed_df = st.session_state.dup_removed_df     
        st.markdown("####  Duplicate Removal Summary")
        st.write("")
        st.markdown("""
        <div class="summary-grid">
            <div class="summary-card">
                <div class="summary-title">Rows Before</div>
                <div class="summary-value">{}</div>
            </div>
            <div class="summary-card">
                <div class="summary-title">Rows After</div>
                <div class="summary-value">{}</div>
            </div>
            <div class="summary-card">
                <div class="summary-title">Duplicates Removed</div>
                <div class="summary-value">{}</div>
            </div>
        </div>
        """.format(
            before_df.shape[0],
            after_df.shape[0],
            removed_df.shape[0]
        ), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        # ===== BEFORE =====
        st.markdown(
            f"#### Before Duplicate Removal ({before_df.shape[0]} Rows)"
        )
        st.write("")
        render_html_table(
            before_df,
            title=None,
            max_height=300
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # ===== AFTER =====
        st.markdown(
            f"####  After Duplicate Removal ({after_df.shape[0]} Rows)"
        )
        st.write("")
        render_html_table(
            after_df,
            title=None,
            max_height=300
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # ===== REMOVED =====
        st.markdown(
            f"#### Duplicates Removed ({removed_df.shape[0]} Rows)"
        )
        st.write("")
        render_html_table(
            removed_df,
            title=None,
            max_height=300  # smaller is fine here
        )



    # ============================================================
    # OUTLIER DETECTION (IQR-BASED – FLAG ONLY)
    # ============================================================
if step == "Remove Outliers":

    st.markdown("### Remove Outliers")
    st.write("")

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
    This step identifies and handles <b>statistical outliers</b> in numeric fields using a
    <b>robust IQR-based method</b>.

    Outlier handling is performed <b>internally</b> and follows a <b>two-level strategy</b>:
    <ul>
        <li><b>Mild anomalies</b> are <b>capped</b> to safe bounds (no row deletion)</li>
        <li><b>Extreme anomalies</b> in <b>critical columns</b> are <b>removed</b></li>
    </ul>

    <br>

    <b>Why this is important:</b>
    <ul>
        <li>Prevents extreme values from <b>skewing averages and distributions</b></li>
        <li>Reduces noise without discarding valuable data</li>
        <li>Ensures numeric stability for downstream models</li>
        <li>Avoids over-cleaning by deleting only <b>truly abnormal records</b></li>
    </ul>
    <br>

    <b>How it helps forecasting:</b>
    <li>
    Demand forecasting models are highly sensitive to extreme numeric values.
    By controlling these extremes, the model learns from realistic historical behavior
    rather than rare or erroneous spikes.
    </li>

    <li>
    This improves forecasting by preserving <b>true demand signals</b>, reducing noise,
    preventing overreaction to anomalies, and ensuring forecasts remain
    <b>stable, generalizable, and business-relevant</b> across time, products, and stores.
    </li>


    </div>
    """, unsafe_allow_html=True)

    

    df = st.session_state.df

    # --------------------------------------------------
    # NUMERIC COLUMN DETECTION
    # --------------------------------------------------
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if not numeric_cols:
        st.info("No numeric columns available for outlier detection.")
        st.stop()

    # --------------------------------------------------
    # BASE COLUMNS (MOST TRUSTWORTHY FOR DELETION)
    # --------------------------------------------------
    DELETE_COLS = ["quantity_sold", "unit_price"]

    # --------------------------------------------------
    # INIT SESSION KEYS
    # --------------------------------------------------
    if "out_before_df" not in st.session_state:
        st.session_state.out_before_df = None
    if "out_after_df" not in st.session_state:
        st.session_state.out_after_df = None
    if "out_removed_df" not in st.session_state:
        st.session_state.out_removed_df = None

    # --------------------------------------------------
    # APPLY AGGRESSIVE OUTLIER HANDLING
    # --------------------------------------------------
    if st.button("Apply Outlier Removal"):

        if st.session_state.out_removed_df is not None:
            st.info("Outliers were already handled earlier.")

        else:
            before_df = df.copy()
            after_df = before_df.copy()

            # Count how many columns flag each row
            outlier_count = pd.Series(0, index=before_df.index)

            for col in numeric_cols:
                Q1 = before_df[col].quantile(0.25)
                Q3 = before_df[col].quantile(0.75)
                IQR = Q3 - Q1

                mild_lower = Q1 - 1.5 * IQR
                mild_upper = Q3 + 1.5 * IQR

                # More aggressive extreme bounds
                extreme_lower = Q1 - 2.0 * IQR
                extreme_upper = Q3 + 2.0 * IQR

                # Count mild outliers
                is_mild = (
                    (before_df[col] < mild_lower) |
                    (before_df[col] > mild_upper)
                )

                outlier_count += is_mild.astype(int)

                # Hard delete if base column is extreme
                if col in DELETE_COLS:
                    outlier_count += (
                        (before_df[col] < extreme_lower) |
                        (before_df[col] > extreme_upper)
                    ).astype(int) * 2  # heavier weight

                # Cap all numeric columns
                after_df[col] = after_df[col].clip(mild_lower, mild_upper)

            # 🔥 DELETE RULE (AGGRESSIVE BUT LOGICAL)
            # Remove rows flagged in 3+ signals
            extreme_mask = outlier_count >= 4

            removed_df = before_df[extreme_mask]
            after_df = after_df[~extreme_mask].reset_index(drop=True)

            # Save snapshots
            st.session_state.out_before_df = before_df
            st.session_state.out_removed_df = removed_df
            st.session_state.out_after_df = after_df

            st.session_state.df = after_df
            st.session_state.preprocessing_completed = True

            st.success("Outliers handled successfully")

    # --------------------------------------------------
    # OUTPUT SECTION (UNCHANGED)
    # --------------------------------------------------
    if st.session_state.out_removed_df is not None:

        before_df = st.session_state.out_before_df
        after_df = st.session_state.out_after_df
        removed_df = st.session_state.out_removed_df

        st.markdown("####  Outlier Removal Summary")
        st.write("")
        st.markdown("""
        <div class="summary-grid">
            <div class="summary-card">
                <div class="summary-title">Rows Before</div>
                <div class="summary-value">{}</div>
            </div>
            <div class="summary-card">
                <div class="summary-title">Rows After</div>
                <div class="summary-value">{}</div>
            </div>
            <div class="summary-card">
                <div class="summary-title">Outliers Removed</div>
                <div class="summary-value">{}</div>
            </div>
        </div>
        """.format(
            before_df.shape[0],
            after_df.shape[0],
            removed_df.shape[0]
        ), unsafe_allow_html=True)
        st.write("")
            # ===== BEFORE =====
        st.markdown(f"#### Before Outlier Handling ({before_df.shape[0]} Rows)")
        st.write("")
        render_html_table(before_df, max_height=300)
        st.write("")

        # ===== AFTER =====
        st.markdown(f"#### After Outlier Handling ({after_df.shape[0]} Rows)")
        st.write("")
        render_html_table(after_df, max_height=300)
        

        st.markdown("<br>", unsafe_allow_html=True)

        # ===== REMOVED =====
        st.markdown(f"####  Outliers Removed ({removed_df.shape[0]} Rows)")
        st.write("")
        render_html_table(removed_df, max_height=300)




# ============================================================
# 3️⃣ REPLACE NULL VALUES WITH "UNKNOWN"
# ============================================================

elif step == "Replace Missing Values":

    st.markdown("### Replace Missing Values")
    st.write("")

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


    # ============================================================
    # NULL VALUE REPLACEMENT (STATEFUL + AFFECTED ROWS ONLY)
    # ============================================================

    df = st.session_state.df

    # ------------------------------------------------------------
    # INIT SESSION KEYS
    # ------------------------------------------------------------
    if "null_before_rows" not in st.session_state:
        st.session_state.null_before_rows = None
    if "null_after_rows" not in st.session_state:
        st.session_state.null_after_rows = None
    if "null_replaced_cols" not in st.session_state:
        st.session_state.null_replaced_cols = None


    # ------------------------------------------------------------
    # DETECT NULLS (CURRENT DF)
    # ------------------------------------------------------------
    null_mask = df.isnull()
    affected_rows_before = df[null_mask.any(axis=1)]
    null_counts = null_mask.sum()
    null_counts = null_counts[null_counts > 0]


    # ------------------------------------------------------------
    # APPLY NULL REPLACEMENT
    # ------------------------------------------------------------
    if st.button("Apply NULL Replacement"):

        if null_counts.empty:
            st.info("NULL values were already handled earlier.")

        else:
            # 🔒 SNAPSHOT ONLY AFFECTED ROWS (BEFORE)
            st.session_state.null_before_rows = affected_rows_before.copy()

            # SAVE COLUMN IMPACT
            st.session_state.null_replaced_cols = (
                null_counts.to_frame("NULL Count")
            )

            # APPLY REPLACEMENT
            df_updated = df.fillna("Unknown")
            st.session_state.df = df_updated
            st.session_state.preprocessing_completed = True

            # 🔒 SNAPSHOT SAME ROWS AFTER REPLACEMENT
            st.session_state.null_after_rows = df_updated.loc[
                affected_rows_before.index
            ].copy()

            st.success(" NULL values replaced with 'Unknown'")


    # ------------------------------------------------------------
    # OUTPUT SECTION – AFFECTED ROWS ONLY
    # ------------------------------------------------------------
    if (
    st.session_state.null_before_rows is not None and
    st.session_state.null_after_rows is not None and
    st.session_state.null_replaced_cols is not None
):


        before_rows = st.session_state.null_before_rows
        after_rows = st.session_state.null_after_rows
        replaced_cols = st.session_state.null_replaced_cols
        # ===================== COLUMNS =====================
        st.markdown("####  Columns Where NULL Values Were Replaced")
        st.write("")

        if not replaced_cols.empty:
            value_col = replaced_cols.columns[0]

            html_cards = "".join(
                f"""
                <div class="summary-card">
                    <div class="summary-title">{str(idx).replace('_', ' ').title()}</div>
                    <div class="summary-value">{row[value_col]}</div>
                </div>
                """
                for idx, row in replaced_cols.iterrows()
            )

            st.markdown(
                f"""
                <div class="summary-grid">
                
                    {html_cards}
                </div>
                """,
                unsafe_allow_html=True   # 🔥 THIS IS CRITICAL
            )
        else:
            st.info("No NULL values were replaced.")

        st.write("")
        # ===================== BEFORE =====================
        st.markdown(
            f"#### Rows Before Missing Values Replacement ({before_rows.shape[0]} Rows)"
        )
        st.write("")
        render_html_table(before_rows)
        
        # ===================== AFTER =====================
        st.markdown(
            f"####  Rows After Missing Values Replacement ({after_rows.shape[0]} Rows)"
        )
        st.write("")
        render_html_table(after_rows)



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
   DATA QUALITY – LAYOUT (FINAL, CLEAN)
   ===================================================== */

/* Horizontal row for 3 cards */
.quality-row {
    display: flex;
    gap: 16px;
    margin-bottom: 48px;   /* clear gap between rows */
}

/* Individual card */
.quality-card {
    flex: 1;
    background-color: white;
    border-radius: 12px;
    padding: 16px 18px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.06);
    border-left: 5px solid #2F75B5;
    margin-bottom: 48px;   /* ~5 line gap between sections */
}

/* Section title with light blue band (AS PER IMAGE) */
.quality-title {
    font-size: 15px;
    font-weight: 600;
    color: #ffffff;
    background-color:#123A72;
    padding: 10px 14px;
    border-radius: 6px;
    margin-bottom: 18px;
}

/* Scrollable content inside card */
.table-scroll {
    max-height: 260px;
    overflow-y: auto;
}

/* ===============================
   TABLE APPEARANCE (NO RENAMES)
   =============================== */

.quality-card table {
    width: 100%;
    border-collapse: collapse;
    background-color: #FFFFFF;
    font-size: 14px;
}

/* Table header */
.quality-card th {
    background-color: #E5ECF4;   /* slightly darker */
    color: #1F2937;
    font-weight: 600;
    text-align: left;
    padding: 10px 12px;
    border-bottom: 1px solid #D6DEE8;
}

/* Table cells */
.quality-card td {
    padding: 9px 12px;
    color: #111827;
    border-bottom: 1px solid #EEF2F7;
}

/* Zebra rows (LIKE IMAGE) */
.quality-card tr:nth-child(even) td {
    background-color: #FFFFFF;
}

.quality-card tr:nth-child(odd) td {
    background-color: #F3F6FA;
}

/* Subtle hover */
.quality-card tr:hover td {
    background-color: #E9F1FF;
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


# Global transparent theme
def transparent_theme():
    return {
        "config": {
            "background": "transparent",
            "view": {
                "fill": "transparent",
                "stroke": "transparent"
            },
            "axis": {
                "labelColor": "rgba(255,255,255,0.8)",
                "titleColor": "rgba(255,255,255,0.9)",
                "gridColor": "rgba(255,255,255,0.25)",
                "domainColor": "rgba(255,255,255,0.4)"
            },
            "text": {"color": "white"}
        }
    }

alt.themes.register("transparent_theme", transparent_theme)
alt.themes.enable("transparent_theme")






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
st.write("")
st.info(f"Dataset Loaded: **{df.shape[0]} rows × {df.shape[1]} columns**")
st.write("")
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
# =========================
# EDA NAVIGATION – ACTIVE BUTTON HIGHLIGHT (SAFE)
# =========================

# =========================
# EDA NAVIGATION (INSTANT COLOR CHANGE)
# =========================

st.markdown("###  List of Analytics")
st.markdown(
    "<div style='margin-top:6px'></div>",
    unsafe_allow_html=True
)



if "eda_option" not in st.session_state:
    st.session_state.eda_option = None


def nav_button(label, value):
    
    """Instant active highlight + no size change"""
    if st.session_state.eda_option == value:
        st.markdown(
            f"""
            <div style="
                background-color:#4F97EE
;
                color:white;
                padding:14px;
                border-radius:10px;
                font-weight:600;
                text-align:center;
                margin-bottom:12px;
            ">
                {label}
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        if st.button(label, use_container_width=True):
            st.session_state.eda_option = value
            st.rerun() 

with st.expander(" ", expanded=True):
    row1 = st.columns(5)
    row2 = st.columns(4)

    with row1[0]:
        nav_button("Data Quality Overview", "Data Quality Overview")
    with row1[1]:
        nav_button("Sales Overview", "Sales Overview")
    with row1[2]:
        nav_button("Promotion Effectiveness", "Promotion Effectiveness")
    with row1[3]:
        nav_button("Product-Level Analysis", "Product-Level Analysis")
    with row1[4]:
        nav_button("Customer-Level Analysis", "Customer-Level Analysis")

    with row2[0]:
        nav_button("Event Impact Analysis", "Event Impact Analysis")
    with row2[1]:
        nav_button("Store-Level Analysis", "Store-Level Analysis")
    with row2[2]:
        nav_button("Sales Channel Analysis", "Sales Channel Analysis")
    with row2[3]:
        nav_button("Summary Report", "Summary Report")


eda_option = st.session_state.eda_option
st.markdown(
    "<div style='margin-top:6px'></div>",
    unsafe_allow_html=True
)

if eda_option is None:
    st.info("Select an analysis to view insights.")
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
    st.markdown("###  Sales Overview")

        # ---------- ROW 1 ----------
    st.markdown(
        """
        <div class="summary-grid">
            <div class="summary-card">
                <div class="summary-title">Total Revenue</div>
                <div class="summary-value">{}</div>
            </div>
            <div class="summary-card">
                <div class="summary-title">Average Order Value</div>
                <div class="summary-value">{}</div>
            </div>
            <div class="summary-card">
                <div class="summary-title">Maximum Order Value</div>
                <div class="summary-value">{}</div>
            </div>
        </div>
        """.format(
            f"${df[col_rev].sum():,.2f}" if col_rev else "NA",
            f"${df[col_rev].mean():,.2f}" if col_rev else "NA",
            f"${df[col_rev].max():,.2f}" if col_rev else "NA",
        ),
        unsafe_allow_html=True
        )

        # ---------- ROW 2 ----------
    st.markdown(
        """
        <div class="summary-grid">
            <div class="summary-card">
                <div class="summary-title">Total Sales</div>
                <div class="summary-value">{}</div>
            </div>
            <div class="summary-card">
                <div class="summary-title">Total Units Sold</div>
                <div class="summary-value">{}</div>
            </div>
            <div class="summary-card">
                <div class="summary-title">Average Units / Transaction</div>
                <div class="summary-value">{}</div>
            </div>
        </div>
        """.format(
            f"${(df[col_qty] * df[col_price]).sum():,.2f}" if col_qty and col_price else "NA",
            f"{df[col_qty].sum():,}" if col_qty else "NA",
            f"{df[col_qty].mean():.2f}" if col_qty else "NA",
        ),
        unsafe_allow_html=True
        )

    st.write("")
    st.write("")


    if "created_at" in df.columns and col_rev:
   
            st.markdown(
        """
        <div style="
            background-color:#2F75B5;
            padding:18px 25px;
            border-radius:10px;
            font-size:20px;
            color:white;
            margin-top:20px;
            margin-bottom:10px;
            text-align:center;
        ">
            <b>Sales By Year</b>
        </div>
        """,
        unsafe_allow_html=True
    )
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df = df.dropna(subset=["created_at"])

    df["Year"] = df["created_at"].dt.year
    df["Quarter"] = df["created_at"].dt.to_period("Q").astype(str)
    df["Month"] = df["created_at"].dt.to_period("M").astype(str)

    sales_by_year = (
        df.groupby("Year")[col_rev]
        .sum()
        .sort_index()
    )
    
    chart = (
            alt.Chart(sales_by_year.reset_index())
            .mark_bar(color="#001F5C",cornerRadiusEnd=6)
            .encode(
                x=alt.X("Year:O", title="Year"),
                y=alt.Y(f"{col_rev}:Q", title="Revenue",scale=alt.Scale(padding=10)),
                tooltip=["Year", col_rev]
            )
            .properties(
                height=380,
                background="#00D05E",
                padding={"top": 10, "left": 10, "right": 10, "bottom": 10}
            )
            .configure_view(
                fill="#00D05E",
                strokeOpacity=0
            )
            .configure_axis(
                labelColor="#000000",
                titleColor="#000000",
                gridColor="rgba(0,0,0,0.2)",
                domainColor="rgba(0,0,0,0.3)"
            )
        )

    st.altair_chart(chart, use_container_width=True)
   
    

    st.markdown(
        """
        <div style="
            background-color:#2F75B5;
            padding:18px 25px;
            border-radius:10px;
            font-size:20px;
            color:white;
            margin-top:20px;
            margin-bottom:10px;
            text-align:center;
        ">
            <b>Sales By Quaters</b>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Aggregate revenue by quarter
    sales_by_quarter = (
        df.groupby("Quarter")[col_rev]
        .sum()
        .sort_index()
    )

    # Altair chart with SAME layout/template as yearly chart
    chart_quarter = (
        alt.Chart(sales_by_quarter.reset_index())
        .mark_bar(color="#001F5C", cornerRadiusEnd=6)
        .encode(
            x=alt.X("Quarter:O", title="Quarter"),
            y=alt.Y(f"{col_rev}:Q", title="Revenue", scale=alt.Scale(padding=10)),
            tooltip=["Quarter", col_rev]
        )
        .properties(
            height=380,
            background="#00D05E",
            padding={"top": 10, "left": 10, "right": 10, "bottom": 10}
        )
        .configure_view(
            fill="#00D05E",
            strokeOpacity=0
        )
        .configure_axis(
            labelColor="#000000",
            titleColor="#000000",
            gridColor="rgba(0,0,0,0.2)",
            domainColor="rgba(0,0,0,0.3)"
        )
    )

    st.altair_chart(chart_quarter, use_container_width=True)


    st.markdown(
        """
        <div style="
            background-color:#2F75B5;
            padding:18px 25px;
            border-radius:10px;
            font-size:20px;
            color:white;
            margin-top:20px;
            margin-bottom:10px;
            text-align:center;
        ">
            <b>Sales By Month</b>
        </div>
        """,
        unsafe_allow_html=True
    )
    # Aggregate revenue by month
    sales_by_month = (
        df.groupby("Month")[col_rev]
        .sum()
        .sort_index()
    )

    # Altair chart with SAME layout/template
    chart_month = (
        alt.Chart(sales_by_month.reset_index())
        .mark_bar(color="#001F5C", cornerRadiusEnd=6)
        .encode(
            x=alt.X("Month:O", title="Month"),
            y=alt.Y(f"{col_rev}:Q", title="Revenue", scale=alt.Scale(padding=10)),
            tooltip=["Month", col_rev]
        )
        .properties(
            height=380,
            background="#00D05E",
            padding={"top": 10, "left": 10, "right": 10, "bottom": 10}
        )
        .configure_view(
            fill="#00D05E",
            strokeOpacity=0
        )
        .configure_axis(
            labelColor="#000000",
            titleColor="#000000",
            gridColor="rgba(0,0,0,0.2)",
            domainColor="rgba(0,0,0,0.3)"
        )
    )

    st.altair_chart(chart_month, use_container_width=True)





    if col_store and col_rev:
            st.markdown(
        """
        <div style="
            background-color:#2F75B5;
            padding:18px 25px;
            border-radius:10px;
            font-size:20px;
            color:white;
            margin-top:20px;
            margin-bottom:10px;
            text-align:center;
        ">
            <b>Sales By Store</b>
        </div>
        """,
        unsafe_allow_html=True
    )
    # Aggregate revenue by store
    sales_store = (
        df.groupby(col_store)[col_rev]
        .sum()
        .sort_values(ascending=False)
    )

    # Altair chart with SAME layout/template
    chart_store = (
        alt.Chart(sales_store.reset_index())
        .mark_bar(color="#001F5C", cornerRadiusEnd=6)
        .encode(
            x=alt.X(f"{col_store}:O", title="Store"),
            y=alt.Y(f"{col_rev}:Q", title="Revenue", scale=alt.Scale(padding=10)),
            tooltip=[col_store, col_rev]
        )
        .properties(
            height=380,
            background="#00D05E",
            padding={"top": 10, "left": 10, "right": 10, "bottom": 10}
        )
        .configure_view(
            fill="#00D05E",
            strokeOpacity=0
        )
        .configure_axis(
            labelColor="#000000",
            titleColor="#000000",
            gridColor="rgba(0,0,0,0.2)",
            domainColor="rgba(0,0,0,0.3)"
        )
    )

    st.altair_chart(chart_store, use_container_width=True)

    if col_channel and col_rev:
            st.markdown(
        """
        <div style="
            background-color:#2F75B5;
            padding:18px 25px;
            border-radius:10px;
            font-size:20px;
            color:white;
            margin-top:20px;
            margin-bottom:10px;
            text-align:center;
        ">
            <b>Sales By Sales Channels</b>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Aggregate revenue by channel
    sales_channel = (
        df.groupby(col_channel)[col_rev]
        .sum()
        .sort_values(ascending=False)
    )

    # Altair chart with SAME layout/template
    chart_channel = (
        alt.Chart(sales_channel.reset_index())
        .mark_bar(color="#001F5C", cornerRadiusEnd=6)
        .encode(
            x=alt.X(f"{col_channel}:O", title="Channel"),
            y=alt.Y(f"{col_rev}:Q", title="Revenue", scale=alt.Scale(padding=10)),
            tooltip=[col_channel, col_rev]
        )
        .properties(
            height=380,
            background="#00D05E",
            padding={"top": 10, "left": 10, "right": 10, "bottom": 10}
        )
        .configure_view(
            fill="#00D05E",
            strokeOpacity=0
        )
        .configure_axis(
            labelColor="#000000",
            titleColor="#000000",
            gridColor="rgba(0,0,0,0.2)",
            domainColor="rgba(0,0,0,0.3)"
        )
    )

    st.altair_chart(chart_channel, use_container_width=True)


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
    <b>What this section does:</b>
    <li>This section analyzes <b>sales performance at the product (SKU) level</li>

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
    # =========================================================
    # ENSURE PRODUCT METRICS & TOP PRODUCTS ARE DEFINED
    # =========================================================
    col_product = "product_id"
    col_qty     = "quantity_sold"
    col_revenue = "total_sales_amount"
    col_profit  = "profit_value"

    product_metrics = (
        df.groupby(col_product)
        .agg(
            total_quantity_sold=(col_qty, "sum"),
            total_revenue=(col_revenue, "sum"),
            total_profit=(col_profit, "sum")
        )
        .sort_values("total_revenue", ascending=False)
    )

    TOP_N = 20
    top_products = product_metrics.head(TOP_N)

    # Products to label in scatter (same logic you already had)
    top_demand = product_metrics.sort_values(
        "total_quantity_sold", ascending=False
    ).head(5)

    top_profit = product_metrics.sort_values(
        "total_profit", ascending=False
    ).head(5)

    label_products = pd.concat([top_demand, top_profit]).drop_duplicates()


    # =========================================================
    # BLUE TITLE BOX (SAME STYLE FOR ALL CHARTS)
    # =========================================================
    def blue_title(title):
        st.markdown(
            f"""
            <div style="
                background-color:#2F75B5;
                padding:14px;
                border-radius:8px;
                font-size:16px;
                color:white;
                margin-bottom:8px;
                text-align:center;
                font-weight:600;
            ">
                {title}
            </div>
            """,
            unsafe_allow_html=True
        )
    # ================= THEME COLORS (DEFINE ONCE) =================
    GREEN_BG = "#00D05E"
    GRID_GREEN = "#3B3B3B"

    BAR_BLUE = "#001F5C"


    # =========================================================
    # ROW 1 — EXISTING TWO PLOTS (LOGIC UNTOUCHED)
    # =========================================================
    col1, col2 = st.columns(2)

    # ---------- PLOT 1: Revenue Contribution ----------
    with col1:
        blue_title("Revenue Contribution by Product ")

        fig1, ax1 = plt.subplots(figsize=(7, 4))

        # 🔑 GREEN THEME
        fig1.patch.set_facecolor(GREEN_BG)
        ax1.set_facecolor(GREEN_BG)
        fig1.subplots_adjust(
    left=0.08,
    right=0.98,
    top=0.92,
    bottom=0.28   # enough for rotated labels
)

        ax1.bar(
            top_products.index.astype(str),
            top_products["total_revenue"],
            color=BAR_BLUE
        )

        ax1.set_xlabel("Product ID")
        ax1.set_ylabel("Total Revenue")
        ax1.tick_params(axis="x", rotation=45)
        ax1.grid(axis="y", linestyle="-", color=GRID_GREEN, alpha=0.5)

        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        
        st.pyplot(fig1)
        plt.close(fig1)


    # ---------- PLOT 2: Demand vs Profitability ----------
    with col2:
        blue_title("Product Demand vs Profitability")

        fig2, ax2 = plt.subplots(figsize=(7, 4))
        # 🔑 GREEN THEME
        fig2.patch.set_facecolor(GREEN_BG)
        ax2.set_facecolor(GREEN_BG)
        fig2.subplots_adjust(
    left=0.08,
    right=0.98,
    top=0.92,
    bottom=0.13   # enough for rotated labels
)
        ax2.scatter(
            product_metrics["total_quantity_sold"],
            product_metrics["total_profit"],
            alpha=0.6,
            color=BAR_BLUE
        )

        ax2.set_xlabel("Total Quantity Sold (Demand)")
        ax2.set_ylabel("Total Profit")
        ax2.grid(True, linestyle="-", color=GRID_GREEN, alpha=0.5)

        for pid, row in label_products.iterrows():
            ax2.annotate(
                pid,
                (row["total_quantity_sold"], row["total_profit"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                alpha=0.9
            )

        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        st.pyplot(fig2)
        plt.close(fig2)


    # =========================================================
    # ROW 2 — TWO NEW 2D ANALYSES (SAME DESIGN)
    # =========================================================
    col3, col4 = st.columns(2)

    # ---------- PLOT 3: Revenue vs Discount ----------
    with col3:
        blue_title("Revenue vs Discount by Product ")

        product_metrics_viz = (
            df.groupby("product_id")
            .agg(
                total_revenue=("total_sales_amount", "sum"),
                total_discount=("discount_applied", "sum")
            )
            .sort_values("total_revenue", ascending=False)
            .head(20)
        )

        fig3, ax3 = plt.subplots(figsize=(7, 4))

        # 🔑 GREEN THEME
        fig3.patch.set_facecolor(GREEN_BG)
        ax3.set_facecolor(GREEN_BG)
        fig3.subplots_adjust(
    left=0.08,
    right=0.98,
    top=0.92,
    bottom=0.28   # enough for rotated labels
)

        x = np.arange(len(product_metrics_viz))
        width = 0.35

        ax3.bar(
            x - width/2,
            product_metrics_viz["total_revenue"],
            width,
            label="Revenue",
            color=BAR_BLUE
        )

        ax3.bar(
            x + width/2,
            product_metrics_viz["total_discount"],
            width,
            label="Discount Given",
            color="#F59E0B"
        )

        ax3.set_xlabel("Product ID")
        ax3.set_xticks(x)
        ax3.set_xticklabels(
            product_metrics_viz.index.astype(str),
            rotation=45,
            ha="right"
        )

        ax3.set_ylabel("Amount")
        ax3.legend()
        ax3.grid(axis="y", linestyle="-", color=GRID_GREEN, alpha=0.5)

        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)

        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)


    # ---------- PLOT 4: Stock Sold vs Stock Damaged ----------
    with col4:
        blue_title("Stock Sold vs Stock Damaged ")

        stock_metrics = (
            df.groupby("product_id")
            .agg(
                stock_sold=("stock_sold_qty", "sum"),
                stock_damaged=("stock_damaged_qty", "sum")
            )
            .sort_values("stock_sold", ascending=False)
            .head(20)
        )

        fig4, ax4 = plt.subplots(figsize=(7, 4))

        # 🔑 GREEN THEME
        fig4.patch.set_facecolor(GREEN_BG)
        ax4.set_facecolor(GREEN_BG)
        fig4.subplots_adjust(
    left=0.08,
    right=0.98,
    top=0.92,
    bottom=0.17  # enough for rotated labels
)

        x = np.arange(len(stock_metrics))
        width = 0.35

        ax4.bar(
            x - width/2,
            stock_metrics["stock_sold"],
            width,
            label="Stock Sold",
            color=BAR_BLUE
        )

        ax4.bar(
            x + width/2,
            stock_metrics["stock_damaged"],
            width,
            label="Stock Damaged",
            color="#EF4444"
        )

        ax4.set_xlabel("Product ID")
        ax4.set_xticks(x)
        ax4.set_xticklabels(
            stock_metrics.index.astype(str),
            rotation=45,
            ha="right"
        )

        ax4.set_ylabel("Quantity")
        ax4.legend()
        ax4.grid(axis="y", linestyle="-", color=GRID_GREEN, alpha=0.5)

        ax4.spines["top"].set_visible(False)
        ax4.spines["right"].set_visible(False)

        st.pyplot(fig4)
        plt.close(fig4)


elif eda_option == "Customer-Level Analysis":

    # =========================================================
    # INTRO / CONTEXT CARD
    # =========================================================
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

    This section analyzes <b>customer behavior and value patterns</b>.

    It focuses on:
    <ul>
        <li>Customer spending and purchase frequency</li>
        <li>Loyalty engagement and retention signals</li>
        <li>High-value vs low-value customer segments</li>
    </ul><br>

    <b>Why this matters:</b>
    <li>Customer demand is not uniform. Some customers are frequent, loyal, and high-value,
    while others are occasional or price-sensitive.</li><br>

    <b>Key insights users get:</b>
    <ul>
        <li>Identification of high-value customers</li>
        <li>Understanding loyalty effectiveness</li>
        <li>Signals for churn risk and retention planning</li>
    </ul>

    </div>
    """,
    unsafe_allow_html=True
    )

    # =========================================================
    # CUSTOMER METRICS AGGREGATION
    # =========================================================
    col_customer = "customer_id"

    customer_metrics = (
        df.groupby(col_customer)
        .agg(
            total_revenue=("total_sales_amount", "sum"),
            total_purchases=("customer_total_purchases", "max"),
            total_visits=("customer_total_visits", "max"),
            avg_purchase_value=("customer_avg_purchase_value", "mean"),
            loyalty_points_earned=("customer_loyalty_points_earned", "sum"),
            satisfaction_score=("customer_satisfaction_score", "mean"),
            days_since_last_purchase=("customer_days_since_last_purchase", "mean")
        )
    )

    TOP_N = 20
    top_customers = customer_metrics.sort_values(
        "total_revenue", ascending=False
    ).head(TOP_N)

    # =========================================================
    # BLUE TITLE BOX HELPER
    # =========================================================
    def blue_title(title):
        st.markdown(
            f"""
            <div style="
                background-color:#2F75B5;
                padding:14px;
                border-radius:8px;
                font-size:16px;
                color:white;
                margin-bottom:8px;
                text-align:center;
                font-weight:600;
            ">
                {title}
            </div>
            """,
            unsafe_allow_html=True
        )
    # ================= THEME COLORS (DEFINE ONCE) =================
    GREEN_BG = "#00D05E"
    GRID_GREEN = "#3B3B3B"
    BAR_BLUE = "#001F5C"

    # =========================================================
    # ROW 1 — CUSTOMER VALUE & FREQUENCY
    # =========================================================
    col1, col2 = st.columns(2)

    # ---------- PLOT 1: Revenue Contribution by Customer ----------
    with col1:
        blue_title("Revenue Contribution by Customer ")

        fig1, ax1 = plt.subplots(figsize=(7, 4))

        # 🔑 GREEN THEME
        fig1.patch.set_facecolor(GREEN_BG)
        ax1.set_facecolor(GREEN_BG)
        fig1.subplots_adjust(
            left=0.08,
            right=0.98,
            top=0.92,
            bottom=0.28
        )

        x = np.arange(len(top_customers))

        ax1.bar(
            x,
            top_customers["total_revenue"],
            color=BAR_BLUE
        )

        ax1.set_xlabel("Customer ID")
        ax1.set_ylabel("Total Revenue")

        ax1.set_xticks(x)
        ax1.set_xticklabels(
            top_customers.index.astype(str),
            rotation=45,
            ha="right"
        )

        ax1.grid(axis="y", linestyle="-", color=GRID_GREEN, alpha=0.5)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)

        st.pyplot(fig1)
        plt.close(fig1)


    # ---------- PLOT 2: Avg Order Value vs Discount Dependency ----------
    with col2:
        blue_title("Customer Order Value vs Discount Dependency")

        customer_discount_metrics = (
            df.groupby("customer_id")
            .agg(
                avg_order_value=("avg_order_value", "mean"),
                avg_discount=("discount_applied", "mean")
            )
            .dropna()
        )

        top_customers = (
            customer_discount_metrics
            .sort_values("avg_order_value", ascending=False)
            .head(20)
        )

        x = np.arange(len(top_customers))
        width = 0.35

        fig2, ax2 = plt.subplots(figsize=(7, 4))

        # 🔑 GREEN THEME
        fig2.patch.set_facecolor(GREEN_BG)
        ax2.set_facecolor(GREEN_BG)
        fig2.subplots_adjust(
            left=0.08,
            right=0.98,
            top=0.92,
            bottom=0.30
        )

        ax2.bar(
            x - width / 2,
            top_customers["avg_order_value"],
            width,
            label="Avg Order Value",
            color=BAR_BLUE
        )

        ax2.bar(
            x + width / 2,
            top_customers["avg_discount"],
            width,
            label="Avg Discount Applied",
            color="#F59E0B"
        )

        ax2.set_xticks(x)
        ax2.set_xticklabels(
            top_customers.index.astype(str),
            rotation=45,
            ha="right"
        )

        ax2.set_ylabel("Amount")
        ax2.set_xlabel("Customer ID")
        ax2.legend()
        ax2.grid(axis="y", linestyle="-", color=GRID_GREEN, alpha=0.5)

        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        st.pyplot(fig2)
        plt.close(fig2)


    # =========================================================
    # ROW 2 — LOYALTY & RETENTION SIGNALS
    # =========================================================
    col3, col4 = st.columns(2)

    # ---------- PLOT 3: Revenue vs Loyalty Contribution (%) ----------
    with col3:
        blue_title("Revenue vs Loyalty Contribution (%)")

        top_loyal_customers = (
            customer_metrics
            .sort_values("total_revenue", ascending=False)
            .head(20)
            .copy()
        )

        top_loyal_customers["revenue_pct"] = (
            top_loyal_customers["total_revenue"]
            / top_loyal_customers["total_revenue"].sum()
        ) * 100

        top_loyal_customers["loyalty_pct"] = (
            top_loyal_customers["loyalty_points_earned"]
            / top_loyal_customers["loyalty_points_earned"].sum()
        ) * 100

        fig3, ax3 = plt.subplots(figsize=(7, 4))

        # 🔑 GREEN THEME
        fig3.patch.set_facecolor(GREEN_BG)
        ax3.set_facecolor(GREEN_BG)
        fig3.subplots_adjust(
            left=0.08,
            right=0.98,
            top=0.92,
            bottom=0.28
        )

        x = np.arange(len(top_loyal_customers))
        width = 0.35

        ax3.bar(
            x - width / 2,
            top_loyal_customers["revenue_pct"],
            width,
            label="Revenue Contribution (%)",
            color=BAR_BLUE
        )

        ax3.bar(
            x + width / 2,
            top_loyal_customers["loyalty_pct"],
            width,
            label="Loyalty Contribution (%)",
            color="#F59E0B"
        )

        ax3.set_xticks(x)
        ax3.set_xticklabels(
            top_loyal_customers.index.astype(str),
            rotation=45,
            ha="right"
        )

        ax3.set_xlabel("Customer ID")
        ax3.set_ylabel("Percentage Contribution (%)")
        ax3.legend()
        ax3.grid(axis="y", linestyle="-", color=GRID_GREEN, alpha=0.5)

        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)

        st.pyplot(fig3)
        plt.close(fig3)


    # ---------- PLOT 4: Customer Satisfaction vs Recency ----------
    with col4:
        blue_title("Customer Satisfaction vs Recency")

        customer_metrics["recency_bucket"] = pd.cut(
            customer_metrics["days_since_last_purchase"],
            bins=[0, 30, 90, 180, 365],
            labels=["0–30 Days", "31–90 Days", "91–180 Days", "181–365 Days"]
        )

        recency_summary = (
            customer_metrics
            .groupby("recency_bucket", observed=True)
            .agg(
                avg_satisfaction=("satisfaction_score", "mean"),
                customer_count=("satisfaction_score", "count")
            )
        )

        fig4, ax4 = plt.subplots(figsize=(7, 4))

        # 🔑 GREEN THEME
        fig4.patch.set_facecolor(GREEN_BG)
        ax4.set_facecolor(GREEN_BG)
        fig4.subplots_adjust(
            left=0.10,
            right=0.98,
            top=0.92,
            bottom=0.22
        )

        bars = ax4.bar(
            recency_summary.index.astype(str),
            recency_summary["avg_satisfaction"],
            color=BAR_BLUE
        )

        ax4.set_xlabel("Days Since Last Purchase")
        ax4.set_ylabel("Average Customer Satisfaction")
        ax4.set_ylim(0, 5)
        ax4.grid(axis="y", linestyle="-", color=GRID_GREEN, alpha=0.5)

        ax4.spines["top"].set_visible(False)
        ax4.spines["right"].set_visible(False)

        # ---- Add customer count labels ----
        for bar, count in zip(bars, recency_summary["customer_count"]):
            ax4.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                f"{count} customers",
                ha="center",
                fontsize=9
            )

        st.pyplot(fig4)
        plt.close(fig4)

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

    # =========================================================
    # BLUE TITLE BOX
    # =========================================================
    def blue_title(title):
        st.markdown(
            f"""
            <div style="
                background-color:#2F75B5;
                padding:14px;
                border-radius:8px;
                font-size:16px;
                color:white;
                margin-bottom:8px;
                text-align:center;
                font-weight:600;
            ">
                {title}
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
    col_returns = "returns_quantity_returned"

    # =========================
    # PARAMETERS
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

    store_top_products = (
        store_product_qty
        .sort_values([col_store, col_qty], ascending=[True, False])
        .groupby(col_store)
        .head(TOP_PRODUCTS)
    )

    pivot_qty = store_top_products.pivot_table(
        index=col_store,
        columns=col_product,
        values=col_qty,
        fill_value=0
    )
    # ================= THEME COLORS (DEFINE ONCE) =================
    GREEN_BG = "#00D05E"
    GRID_GREEN = "#3B3B3B"

    BAR_BLUE = "#001F5C"

    # =========================================================
    # ROW 1 — EXISTING PLOTS (THEMED ONLY)
    # =========================================================
    col1, col2 = st.columns(2)

    # ---------- PLOT 1: Revenue Concentration Across Stores ----------
    with col1:
        blue_title("Revenue Concentration Across Stores")

        store_revenue = (
            df.groupby(col_store)[col_revenue]
            .sum()
            .loc[top_stores]
        )

        fig1, ax1 = plt.subplots(figsize=(7, 4))

        # 🔑 GREEN THEME
        fig1.patch.set_facecolor(GREEN_BG)
        ax1.set_facecolor(GREEN_BG)
        fig1.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.16)

        ax1.bar(
            store_revenue.index.astype(str),
            store_revenue.values,
            color=BAR_BLUE
        )

        ax1.set_xlabel("Store ID")
        ax1.set_ylabel("Total Revenue")
        ax1.tick_params(axis="x", rotation=45)
        ax1.grid(axis="y", linestyle="-", color=GRID_GREEN, alpha=0.5)

        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)

        st.pyplot(fig1)
        plt.close(fig1)


    # ---------- PLOT 2: Store-wise Product Mix ----------
    with col2:
        blue_title("Store-wise Product Mix (Quantity Sold)")

        fig2, ax2 = plt.subplots(figsize=(7, 4))

        # 🔑 GREEN THEME
        fig2.patch.set_facecolor(GREEN_BG)
        ax2.set_facecolor(GREEN_BG)
        fig2.subplots_adjust(left=0.08, right=0.78, top=0.92, bottom=0.25)

        bottom = np.zeros(len(pivot_qty))

        for product in pivot_qty.columns:
            ax2.bar(
                pivot_qty.index.astype(str),
                pivot_qty[product],
                bottom=bottom,
                width=0.6,
                label=str(product)
            )
            bottom += pivot_qty[product].values

        ax2.set_xlabel("Store ID")
        ax2.set_ylabel("Quantity Sold")
        ax2.tick_params(axis="x", rotation=45)

        for label in ax2.get_xticklabels():
            label.set_ha("right")

        ax2.grid(axis="y", linestyle="-", color=GRID_GREEN, alpha=0.5)

        ax2.legend(
            title="Product ID",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            fontsize=8
        )

        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        st.pyplot(fig2)
        plt.close(fig2)


    # =========================================================
    # ROW 2 — NEW 2D BAR PLOTS (THEMED)
    # =========================================================
    col3, col4 = st.columns(2)

    # ---------- PLOT 3: Store Sales vs Returned Quantity ----------
    with col3:
        blue_title("Store Sales vs Returned Quantity")

        store_returns = (
            df.groupby(col_store)
            .agg(
                total_sales=(col_qty, "sum"),
                total_returns=(col_returns, "sum")
            )
            .loc[top_stores]
        )

        x = np.arange(len(store_returns))
        width = 0.35

        fig3, ax3 = plt.subplots(figsize=(7, 4))

        # 🔑 GREEN THEME
        fig3.patch.set_facecolor(GREEN_BG)
        ax3.set_facecolor(GREEN_BG)
        fig3.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.28)

        ax3.bar(
            x - width / 2,
            store_returns["total_sales"],
            width,
            label="Units Sold",
            color=BAR_BLUE
        )

        ax3.bar(
            x + width / 2,
            store_returns["total_returns"],
            width,
            label="Returned Units",
            color="#EF4444"
        )

        ax3.set_xticks(x)
        ax3.set_xticklabels(store_returns.index.astype(str), rotation=45, ha="right")
        ax3.set_ylabel("Quantity")
        ax3.set_xlabel("Store ID")
        ax3.legend()
        ax3.grid(axis="y", linestyle="-", color=GRID_GREEN, alpha=0.5)

        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)

        st.pyplot(fig3)
        plt.close(fig3)


    # ---------- PLOT 4: Units Sold vs Revenue ----------
    with col4:
        blue_title("Units Sold vs Revenue")

        store_efficiency = (
            df.groupby(col_store)
            .agg(
                total_units_sold=(col_qty, "sum"),
                total_revenue=(col_revenue, "sum")
            )
            .loc[top_stores]
        )

        x = np.arange(len(store_efficiency))
        width = 0.35

        fig4, ax1 = plt.subplots(figsize=(7, 4))

        # 🔑 GREEN THEME
        fig4.patch.set_facecolor(GREEN_BG)
        ax1.set_facecolor(GREEN_BG)
        fig4.subplots_adjust(left=0.10, right=0.90, top=0.92, bottom=0.26)

        # Units Sold — LEFT AXIS
        ax1.bar(
            x - width / 2,
            store_efficiency["total_units_sold"],
            width,
            label="Units Sold",
            color=BAR_BLUE
        )
        ax1.set_ylabel("Units Sold")

        # Revenue — RIGHT AXIS
        ax2 = ax1.twinx()
        ax2.bar(
            x + width / 2,
            store_efficiency["total_revenue"],
            width,
            label="Revenue",
            color="#F59E0B"
        )
        ax2.set_ylabel("Revenue")

        ax1.set_xticks(x)
        ax1.set_xticklabels(
            store_efficiency.index.astype(str),
            rotation=45,
            ha="right"
        )

        # Combined legend
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="upper right")

        ax1.set_xlabel("Store ID")
        ax1.grid(axis="y", linestyle="-", color=GRID_GREEN, alpha=0.5)

        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax2.spines["top"].set_visible(False)

        st.pyplot(fig4)
        plt.close(fig4)



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
    
    # =========================================================
    # BLUE TITLE BOX
    # =========================================================
    def blue_title(title):
        st.markdown(
            f"""
            <div style="
                background-color:#2F75B5;
                padding:14px;
                border-radius:8px;
                font-size:16px;
                color:white;
                margin-bottom:8px;
                text-align:center;
                font-weight:600;
            ">
                {title}
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
    col_qty     = "quantity_sold"

    # =========================
    # PARAMETERS
    # =========================
    TOP_CHANNELS = 15

    # =========================
    # TOP CHANNELS BY REVENUE
    # =========================
    top_channels = (
        df.groupby(col_channel)[col_revenue]
        .sum()
        .sort_values(ascending=False)
        .head(TOP_CHANNELS)
        .index
    )
    # ================= THEME COLORS (DEFINE ONCE) =================
    GREEN_BG = "#00D05E"
    GRID_GREEN = "#3B3B3B"

    BAR_BLUE = "#001F5C"

    # =========================================================
    # ROW 1 — EXISTING PLOTS (THEMED ONLY)
    # =========================================================
    col1, col2 = st.columns(2)

    # ---------- PLOT 1: Revenue Contribution (DONUT) ----------
    with col1:
        blue_title("Revenue Contribution by Sales Channel")

        channel_revenue = (
            df.groupby(col_channel)[col_revenue]
            .sum()
            .loc[top_channels]
        )

        fig1, ax1 = plt.subplots(figsize=(2, 2))

        # 🔑 THEME (IMPORTANT FOR PIE)
        fig1.patch.set_facecolor(GREEN_BG)
        ax1.set_facecolor(GREEN_BG)

        wedges, texts, autotexts = ax1.pie(
            channel_revenue.values,
            labels=channel_revenue.index.astype(str),
            autopct="%1.1f%%",
            startangle=90,
            colors=plt.cm.tab10.colors,
            wedgeprops={
                "width": 0.55,
                "edgecolor": "white"
            },
            pctdistance=0.75
        )

        for t in autotexts:
            t.set_fontsize(4)
            t.set_color("black")

        for t in texts:
            t.set_fontsize(4)

        ax1.set_aspect("equal")

        st.pyplot(fig1)
        plt.close(fig1)


    # ---------- PLOT 2: Average Order Value ----------
    with col2:
        blue_title("Average Order Value by Sales Channel")

        aov_values = [
            df[df[col_channel] == ch][col_aov].mean()
            for ch in top_channels
        ]

        fig2, ax2 = plt.subplots(figsize=(7, 5))

        # 🔑 GREEN THEME
        fig2.patch.set_facecolor(GREEN_BG)
        ax2.set_facecolor(GREEN_BG)
        fig2.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.03)

        ax2.scatter(
            top_channels.astype(str),
            aov_values,
            s=90,
            alpha=0.85,
            color=BAR_BLUE
        )

        for x, y in zip(top_channels.astype(str), aov_values):
            ax2.text(
                x,
                y + (max(aov_values) * 0.02),
                f"{int(y)}",
                ha="center",
                va="bottom",
                fontsize=9,
                color="black"
            )

        ax2.set_xlabel("Sales Channel")
        ax2.set_ylabel("Average Order Value")

        ax2.tick_params(axis="x", rotation=45)
        for label in ax2.get_xticklabels():
            label.set_ha("right")

        ax2.grid(axis="y", linestyle="-", color=GRID_GREEN, alpha=0.5)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        st.pyplot(fig2)
        plt.close(fig2)


    # =========================================================
    # ROW 2 — NEW 2D BAR PLOTS (THEMED)
    # =========================================================
    col3, col4 = st.columns(2)

    # ---------- PLOT 3: Units Sold vs Revenue ----------
    with col3:
        blue_title("Units Sold vs Revenue by Sales Channel")

        channel_volume = (
            df.groupby(col_channel)
            .agg(
                total_units_sold=(col_qty, "sum"),
                total_revenue=(col_revenue, "sum")
            )
            .loc[top_channels]
        )

        x = np.arange(len(channel_volume))
        width = 0.35

        fig3, ax1 = plt.subplots(figsize=(7, 4))

        # 🔑 GREEN THEME
        fig3.patch.set_facecolor(GREEN_BG)
        ax1.set_facecolor(GREEN_BG)
        fig3.subplots_adjust(left=0.10, right=0.90, top=0.92, bottom=0.28)

        ax1.bar(
            x - width / 2,
            channel_volume["total_units_sold"],
            width,
            label="Units Sold",
            color=BAR_BLUE
        )
        ax1.set_ylabel("Units Sold")

        ax2 = ax1.twinx()
        ax2.bar(
            x + width / 2,
            channel_volume["total_revenue"],
            width,
            label="Revenue",
            color="#F59E0B"
        )
        ax2.set_ylabel("Revenue")

        ax1.set_xticks(x)
        ax1.set_xticklabels(
            channel_volume.index.astype(str),
            rotation=45,
            ha="right"
        )
        ax1.set_xlabel("Sales Channel")

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="upper right")

        ax1.grid(axis="y", linestyle="-", color=GRID_GREEN, alpha=0.5)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax2.spines["top"].set_visible(False)

        st.pyplot(fig3)
        plt.close(fig3)


    # ---------- PLOT 4: Revenue vs Profit ----------
    with col4:
        blue_title("Sales Channel Revenue vs Profit")

        channel_finance = (
            df.groupby(col_channel)
            .agg(
                total_revenue=(col_revenue, "sum"),
                total_profit=("profit_value", "sum")
            )
            .sort_values("total_revenue", ascending=False)
            .head(15)
        )

        x = np.arange(len(channel_finance))
        width = 0.35

        fig4, ax4 = plt.subplots(figsize=(8, 4))

        # 🔑 GREEN THEME
        fig4.patch.set_facecolor(GREEN_BG)
        ax4.set_facecolor(GREEN_BG)
        fig4.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.14)

        ax4.bar(
            x - width / 2,
            channel_finance["total_revenue"],
            width,
            label="Total Revenue",
            color=BAR_BLUE
        )

        ax4.bar(
            x + width / 2,
            channel_finance["total_profit"],
            width,
            label="Total Profit",
            color="#0A6849"
        )

        ax4.set_xticks(x)
        ax4.set_xticklabels(
            channel_finance.index.astype(str),
            rotation=45,
            ha="right"
        )

        ax4.set_xlabel("Sales Channel")
        ax4.set_ylabel("Amount")
        ax4.legend()
        ax4.grid(axis="y", linestyle="-", color=GRID_GREEN, alpha=0.5)

        ax4.spines["top"].set_visible(False)
        ax4.spines["right"].set_visible(False)

        st.pyplot(fig4)
        plt.close(fig4)

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

        # =========================================================
    # BLUE TITLE BOX (REUSE SAME FUNCTION)
    # =========================================================
    def blue_title(title):
        st.markdown(
            f"""
            <div style="
                background-color:#2F75B5;
                padding:14px;
                border-radius:8px;
                font-size:16px;
                color:white;
                margin-bottom:8px;
                text-align:center;
                font-weight:600;
            ">
                {title}
            </div>
            """,
            unsafe_allow_html=True
        )

    # =========================
    # COLUMN MAPPING
    # =========================
    col_promo   = "promo_transaction_id"
    col_sales   = "promo_total_sales_amount"
    col_cost    = "promo_promo_cost"
    col_uplift  = "promo_promo_uplift_revenue"
    col_total_sales = "total_sales_amount"

    # =========================
    # AGGREGATE PROMOTION METRICS (UNCHANGED LOGIC)
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

    promo_metrics["net_uplift"] = (
        promo_metrics["uplift_revenue"] - promo_metrics["promo_cost"]
    )

    promo_metrics = (
        promo_metrics
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    TOP_N = 20
    top_promos = (
        promo_metrics
        .sort_values("net_uplift", ascending=False)
        .head(TOP_N)
    )
    # ================= THEME COLORS (DEFINE ONCE) =================
    GREEN_BG = "#00D05E"
    GRID_GREEN = "#3B3B3B"

    BAR_BLUE = "#001F5C"

    # =========================================================
    # ROW 1 — EXISTING PLOTS (SAME LOGIC, THEMED)
    # =========================================================
    col1, col2 = st.columns(2)

    # ---------- PLOT 1: PROMOTION PROFITABILITY ----------
    with col1:
        blue_title("Promotion Profitability (Net Uplift Revenue)")

        fig, ax = plt.subplots(figsize=(7, 4))

        # 🔑 THEME
        fig.patch.set_facecolor(GREEN_BG)
        ax.set_facecolor(GREEN_BG)
        fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.28)

        ax.bar(
            top_promos.index.astype(str),
            top_promos["net_uplift"],
            alpha=0.85,
            color=BAR_BLUE
        )

        ax.axhline(0, color="black", linewidth=1)
        ax.set_xlabel("Promotion ID")
        ax.set_ylabel("Net Uplift Revenue")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", linestyle="-", color=GRID_GREEN, alpha=0.5)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        st.pyplot(fig)
        plt.close(fig)


    # ---------- PLOT 2: SALES vs COST ----------
    with col2:
        blue_title("Promotion Effectiveness: Sales vs Cost")

        fig, ax = plt.subplots(figsize=(7, 4))

        # 🔑 THEME
        fig.patch.set_facecolor(GREEN_BG)
        ax.set_facecolor(GREEN_BG)
        fig.subplots_adjust(left=0.10, right=0.98, top=0.92, bottom=0.13)

        ax.scatter(
            top_promos["promo_cost"],
            top_promos["promo_total_sales_amount"],
            s=top_promos["promo_total_sales_amount"] / 1500,
            alpha=0.75,
            color=BAR_BLUE,
            edgecolors="black",
            linewidth=0.5
        )

        max_cost = top_promos["promo_cost"].max()
        ax.plot(
            [0, max_cost],
            [0, max_cost],
            linestyle="--",
            color=GRID_GREEN,
            alpha=0.6
        )

        top_labels = top_promos.sort_values(
            "promo_total_sales_amount", ascending=False
        ).head(10)

        for pid, row in top_labels.iterrows():
            ax.annotate(
                pid,
                (row["promo_cost"], row["promo_total_sales_amount"]),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=9
            )

        ax.set_xlabel("Promotion Cost")
        ax.set_ylabel("Promotion Total Sales Amount")
        ax.grid(True, linestyle="-", color=GRID_GREEN, alpha=0.5)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        st.pyplot(fig)
        plt.close(fig)


    # =========================================================
    # ROW 2 — NEW 2D BAR PLOTS (THEMED)
    # =========================================================
    col3, col4 = st.columns(2)

    # ---------- PLOT 3: QUANTITY SOLD vs RETURNS ----------
    with col3:
        blue_title("Promotion Effect on Quantity Sold vs Returns (Quality Check)")

        col_promo = "promo_transaction_id"
        col_qty_sold = "promo_total_quantity_sold"
        col_qty_returned = "returns_quantity_returned"

        promo_qty = (
            df[df[col_promo].notna()]
            .groupby(col_promo)
            .agg(
                total_quantity_sold=(col_qty_sold, "sum"),
                total_quantity_returned=(col_qty_returned, "sum")
            )
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        TOP_N = 15
        top_promo_qty = (
            promo_qty
            .sort_values("total_quantity_sold", ascending=False)
            .head(TOP_N)
        )

        x = np.arange(len(top_promo_qty))
        width = 0.35

        fig, ax = plt.subplots(figsize=(8, 4))

        # 🔑 THEME
        fig.patch.set_facecolor(GREEN_BG)
        ax.set_facecolor(GREEN_BG)
        fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.18)

        ax.bar(
            x - width/2,
            top_promo_qty["total_quantity_sold"],
            width,
            label="Quantity Sold",
            color=BAR_BLUE
        )

        ax.bar(
            x + width/2,
            top_promo_qty["total_quantity_returned"],
            width,
            label="Quantity Returned",
            color="#EF4444"
        )

        ax.set_xticks(x)
        ax.set_xticklabels(
            top_promo_qty.index.astype(str),
            rotation=45,
            ha="right"
        )

        ax.set_xlabel("Promotion ID")
        ax.set_ylabel("Quantity")
        ax.legend()
        ax.grid(axis="y", linestyle="-", color=GRID_GREEN, alpha=0.5)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        st.pyplot(fig)
        plt.close(fig)


    # ---------- PLOT 4: PROMO COST vs UPLIFT REVENUE ----------
    with col4:
        blue_title("Promotion Cost vs Revenue Uplift")

        promo_compare = (
            promo_metrics
            .sort_values("uplift_revenue", ascending=False)
            .head(15)
        )

        x = np.arange(len(promo_compare))
        width = 0.35

        fig, ax = plt.subplots(figsize=(7, 4))

        # 🔑 THEME
        fig.patch.set_facecolor(GREEN_BG)
        ax.set_facecolor(GREEN_BG)
        fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.28)

        ax.bar(
            x - width/2,
            promo_compare["promo_cost"],
            width,
            label="Promotion Cost",
            color=BAR_BLUE
        )

        ax.bar(
            x + width/2,
            promo_compare["uplift_revenue"],
            width,
            label="Uplift Revenue",
            color="#10B981"
        )

        ax.set_xticks(x)
        ax.set_xticklabels(
            promo_compare.index.astype(str),
            rotation=45,
            ha="right"
        )

        ax.set_xlabel("Promotion ID")
        ax.set_ylabel("Amount")
        ax.legend()
        ax.grid(axis="y", linestyle="-", color=GRID_GREEN, alpha=0.5)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        st.pyplot(fig)
        plt.close(fig)



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

    # =========================================================
    # BLUE TITLE BOX
    # =========================================================
    def blue_title(title):
        st.markdown(
            f"""
            <div style="
                background-color:#2F75B5;
                padding:14px;
                border-radius:8px;
                font-size:16px;
                color:white;
                margin-bottom:8px;
                text-align:center;
                font-weight:600;
            ">
                {title}
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

    # Optional influence columns (only used in Plot 4)
    col_weather = "impact_weather_influence_score"
    col_trend   = "impact_trend_influence_score"

    # =========================
    # AGGREGATE EVENT METRICS (UNCHANGED LOGIC)
    # =========================
    event_metrics = (
        df[df[col_event].notna()]
        .groupby(col_event)
        .agg(
            sales_before=(col_before, "mean"),
            sales_after=(col_after, "mean"),
            total_sales=(col_sales, "sum"),
            total_quantity=(col_qty, "sum"),
            impact_pct=(col_change_pct, "mean"),
            weather_score=(col_weather, "mean"),
            trend_score=(col_trend, "mean")
        )
    )

    # =========================
    # DERIVED METRICS
    # =========================
    event_metrics["sales_uplift"] = (
        event_metrics["sales_after"] - event_metrics["sales_before"]
    )

    event_metrics = (
        event_metrics
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    # =========================
    # SELECT TOP EVENTS
    # =========================
    TOP_N = 15
    top_events = (
        event_metrics
        .sort_values("sales_uplift", ascending=False)
        .head(TOP_N)
    )
    # ================= THEME COLORS (DEFINE ONCE) =================
    GREEN_BG = "#00D05E"
    GRID_GREEN = "#3B3B3B"

    BAR_BLUE = "#001F5C"

    # =========================================================
    # ROW 1 — EXISTING PLOTS (THEMED ONLY)
    # =========================================================
    col1, col2 = st.columns(2)

    # ---------- PLOT 1: EVENT SALES UPLIFT ----------
    with col1:
        blue_title("Event Sales Uplift ")

        fig, ax = plt.subplots(figsize=(7, 4))

        # 🔑 THEME
        fig.patch.set_facecolor(GREEN_BG)
        ax.set_facecolor(GREEN_BG)
        fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.28)

        ax.bar(
            top_events.index.astype(str),
            top_events["sales_uplift"],
            alpha=0.85,
            color=BAR_BLUE
        )

        ax.axhline(0, color="black", linewidth=1)
        ax.set_xlabel("Event ID")
        ax.set_ylabel("Average Sales Uplift")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", linestyle="-", color=GRID_GREEN, alpha=0.5)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        st.pyplot(fig)
        plt.close(fig)


    # ---------- PLOT 2: EVENT EFFECTIVENESS ----------
    with col2:
        blue_title("Event Effectiveness: Demand vs Impact")

        fig, ax = plt.subplots(figsize=(7, 4))

        # 🔑 THEME
        fig.patch.set_facecolor(GREEN_BG)
        ax.set_facecolor(GREEN_BG)
        fig.subplots_adjust(left=0.10, right=0.98, top=0.92, bottom=0.17)

        ax.scatter(
            top_events["total_quantity"],
            top_events["impact_pct"],
            s=top_events["total_sales"] / 1500,
            alpha=0.75,
            color=BAR_BLUE,
            edgecolors="black",
            linewidth=0.5
        )

        ax.axvline(
            top_events["total_quantity"].median(),
            linestyle="--",
            color=GRID_GREEN,
            alpha=0.6
        )
        ax.axhline(
            top_events["impact_pct"].median(),
            linestyle="--",
            color=GRID_GREEN,
            alpha=0.6
        )

        top_labels = top_events.sort_values(
            "sales_uplift", ascending=False
        ).head(7)

        for eid, row in top_labels.iterrows():
            ax.annotate(
                eid,
                (row["total_quantity"], row["impact_pct"]),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=9
            )

        ax.set_xlabel("Total Quantity Sold During Event")
        ax.set_ylabel("Average Sales Impact (%)")
        ax.grid(True, linestyle="-", color=GRID_GREEN, alpha=0.5)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        st.pyplot(fig)
        plt.close(fig)


    # =========================================================
    # ROW 2 — NEW 2D BAR PLOTS (THEMED)
    # =========================================================
    col3, col4 = st.columns(2)

    # ---------- PLOT 3: EVENT-WISE SALES BEFORE vs AFTER ----------
    with col3:
        blue_title("Event-wise Sales Before vs After Impact")

        x = np.arange(len(top_events))
        width = 0.35

        fig, ax = plt.subplots(figsize=(8, 4))

        # 🔑 THEME
        fig.patch.set_facecolor(GREEN_BG)
        ax.set_facecolor(GREEN_BG)
        fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.25)

        ax.bar(
            x - width/2,
            top_events["sales_before"],
            width,
            label="Sales Before Event",
            color=BAR_BLUE
        )

        ax.bar(
            x + width/2,
            top_events["sales_after"],
            width,
            label="Sales After Event",
            color="#649283"
        )

        ax.set_xticks(x)
        ax.set_xticklabels(
            top_events.index.astype(str),
            rotation=45,
            ha="right"
        )

        ax.set_xlabel("Event ID")
        ax.set_ylabel("Average Sales Amount")
        ax.legend()
        ax.grid(axis="y", linestyle="-", color=GRID_GREEN, alpha=0.5)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        st.pyplot(fig)
        plt.close(fig)


    # ---------- PLOT 4: EVENT INFLUENCE BREAKDOWN ----------
    with col4:
        blue_title("Event Influence Breakdown (Weather vs Trend)")

        x = np.arange(len(top_events))
        width = 0.35

        fig, ax = plt.subplots(figsize=(8, 4))

        # 🔑 THEME
        fig.patch.set_facecolor(GREEN_BG)
        ax.set_facecolor(GREEN_BG)
        fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.28)

        ax.bar(
            x - width/2,
            top_events["weather_score"],
            width,
            label="Weather Influence",
            color=BAR_BLUE
        )

        ax.bar(
            x + width/2,
            top_events["trend_score"],
            width,
            label="Trend Influence",
            color="#F59E0B"
        )

        ax.set_xticks(x)
        ax.set_xticklabels(
            top_events.index.astype(str),
            rotation=45,
            ha="right"
        )

        ax.set_xlabel("Event ID")
        ax.set_ylabel("Influence Score")
        ax.legend()
        ax.grid(axis="y", linestyle="-", color=GRID_GREEN, alpha=0.5)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        st.pyplot(fig)
        plt.close(fig)



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
    # FINAL EDA SUMMARY NARRATIVE (FULLY GROUNDED IN OUTPUTS)
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

        <h4>Data Health & Readiness</h4>
        <ul>
            <li>The dataset consists of <b>942 rows and 96 columns</b>, offering rich coverage across products, customers, stores, channels, promotions, and events.</li>
            <li><b>No duplicate records</b> were detected, ensuring transactional integrity.</li>
            <li>Core identifiers (transaction, product, store, customer, sales channel, promotion) have <b>0% missing values</b>.</li>
            <li>Data types are well balanced (categorical, numeric, datetime), confirming the dataset is <b>model-ready</b>.</li>
        </ul>

        <h4>Overall Sales Performance</h4>
        <ul>
            <li>Sales over time exhibit <b>sharp spikes</b>, with several days exceeding <b>₹400K–₹600K</b>, indicating event-driven and promotional demand.</li>
            <li>Revenue distribution is highly uneven, validating the need for deeper segmentation.</li>
            <li>Store-wise and channel-wise sales confirm that a subset of entities drives the majority of revenue.</li>
        </ul>

        <h4> Product-Level Insights</h4>
        <ul>
            <li>Revenue contribution is strongly concentrated — products such as <b>P_000034, P_000029, and P_000019</b> dominate total revenue.</li>
            <li>Demand vs profitability analysis shows <b>no linear relationship</b> between volume and profit.</li>
            <li>Products like <b>P_000050 and P_000058</b> achieve high profitability at moderate demand.</li>
            <li>Discount-heavy products do not consistently yield higher revenue.</li>
            <li>Stock damaged quantities for several top sellers reveal <b>operational loss exposure</b>.</li>
        </ul>

        <h4> Customer-Level Behavior</h4>
        <ul>
            <li>A small group of customers contributes a <b>disproportionate share of revenue</b>, led by <b>C_000034 and C_000029</b>.</li>
            <li>Loyalty contribution varies widely and does not scale proportionally with revenue.</li>
            <li>High-value customers generally show <b>lower discount dependency</b>.</li>
            <li>Customer satisfaction declines as inactivity increases, with the <b>181–365 day</b> segment showing the lowest satisfaction.</li>
        </ul>

        <h4> Store-Level Performance</h4>
        <ul>
            <li>Revenue is concentrated in a few stores, notably <b>S_000034 and S_000029</b>.</li>
            <li>Store-wise product mix varies significantly, confirming <b>localized demand patterns</b>.</li>
            <li>Some stores exhibit <b>high return volumes</b> relative to sales, indicating fulfillment or quality issues.</li>
            <li>High unit sales do not always translate into proportional revenue, highlighting efficiency gaps.</li>
        </ul>

        <h4>Sales Channel Analysis</h4>
        <ul>
            <li>Revenue contribution is dominated by channels such as <b>CH_000034 (10.9%)</b> and <b>CH_000029 (10.0%)</b>.</li>
            <li>Average order value varies significantly across channels, ranging roughly from <b>₹1.8K to ₹4.3K</b>.</li>
            <li>Some channels generate high volume but lower profitability.</li>
            <li>This confirms the need for <b>channel-specific pricing, promotion, and inventory strategies</b>.</li>
        </ul>

        <h4> Promotion Effectiveness</h4>
        <ul>
            <li>Promotion-level analysis shows that <b>not all high-cost promotions are profitable</b>.</li>
            <li>Promotions such as <b>T_000044 and T_000024</b> generate the highest net uplift revenue.</li>
            <li>Quantity sold vs returned analysis reveals promotions that drive volume but also increase returns.</li>
            <li>Sales vs cost scatter clearly separates <b>efficient promotions from underperformers</b>.</li>
        </ul>

        <h4>Event Impact Analysis</h4>
        <ul>
            <li>Events consistently show <b>higher sales after impact</b> compared to before.</li>
            <li>Events like <b>E_000028 and E_000039</b> produce the highest average sales uplift.</li>
            <li>Demand vs impact analysis shows that <b>high demand does not always mean high impact</b>.</li>
            <li>Influence breakdown reveals that some events are <b>trend-driven</b>, while others are <b>weather-sensitive</b>.</li>
        </ul>

        <h4> Cross-Dimensional Insights</h4>
        <ul>
            <li>Revenue and demand are concentrated across products, customers, stores, and channels.</li>
            <li>Discounts, returns, and damaged stock act as <b>hidden profitability leakages</b>.</li>
            <li>Events and promotions introduce strong non-linear effects on demand.</li>
        </ul>

        <h4> Final Takeaway</h4>
        <ul>
            <li>The dataset is <b>clean, consistent, and enterprise-grade</b>.</li>
            <li>Clear demand drivers and inefficiencies are observable across multiple dimensions.</li>
            <li>Forecasting accuracy will significantly improve by modeling at <b>SKU × Store × Channel × Event × Promotion</b> levels.</li>
            <li>The EDA strongly supports downstream use cases in <b>demand forecasting, inventory optimization, and promotion intelligence</b>.</li>
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
