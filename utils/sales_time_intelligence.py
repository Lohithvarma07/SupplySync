import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_plotly_events import plotly_events
import calendar


def render_sales_time_intelligence(
    df: pd.DataFrame,
    date_col: str = "created_at",
    value_col: str = "total_sales_amount"
):
    # ---------- DATA PREP ----------
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[date_col, value_col])

    if df.empty:
        st.warning("No valid data available")
        return

    # ---------- STATE ----------
    st.session_state.setdefault("level", "year")
    st.session_state.setdefault("year", None)
    st.session_state.setdefault("quarter", None)
    st.session_state.setdefault("month", None)

    # ---------- HEADER ----------
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
            <b>Sales By Time</b>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ================= YEAR =================
    if st.session_state.level == "year":
        data = (
            df.groupby(df[date_col].dt.year)[value_col]
            .sum()
            .reset_index(name="sales")
        )

        fig = px.bar(data, x=date_col, y="sales", labels={date_col: "Year"})
        fig.update_layout(clickmode="event+select")

        selected = plotly_events(fig, click_event=True)

        if selected:
            st.session_state.year = int(selected[0]["x"])
            st.session_state.level = "quarter"
            st.rerun()

    # ================= QUARTER =================
    elif st.session_state.level == "quarter":
        df_y = df[df[date_col].dt.year == st.session_state.year]
        df_y["quarter"] = df_y[date_col].dt.to_period("Q").astype(str)

        data = df_y.groupby("quarter")[value_col].sum().reset_index(name="sales")

        fig = px.bar(data, x="quarter", y="sales")
        fig.update_layout(
            title=f"Year {st.session_state.year}",
            clickmode="event+select"
        )

        selected = plotly_events(fig, click_event=True)

        if selected:
            st.session_state.quarter = selected[0]["x"]
            st.session_state.level = "month"
            st.rerun()

    # ================= MONTH =================
    elif st.session_state.level == "month":
        df_q = df.copy()
        df_q["quarter"] = df_q[date_col].dt.to_period("Q").astype(str)
        df_q = df_q[df_q["quarter"] == st.session_state.quarter]

        df_q["month"] = df_q[date_col].dt.to_period("M").astype(str)

        data = df_q.groupby("month")[value_col].sum().reset_index(name="sales")

        fig = px.bar(data, x="month", y="sales")
        fig.update_layout(
            title=f"{st.session_state.quarter}",
            clickmode="event+select"
        )

        selected = plotly_events(fig, click_event=True)

        if selected:
            st.session_state.month = selected[0]["x"]
            st.session_state.level = "day"
            st.rerun()

    # ================= DAY =================
    elif st.session_state.level == "day":
        df_m = df.copy()
        df_m["month"] = df_m[date_col].dt.to_period("M").astype(str)
        df_m = df_m[df_m["month"] == st.session_state.month]

        df_m["day"] = df_m[date_col].dt.day
        sales = df_m.groupby("day")[value_col].sum()

        year, month = map(int, st.session_state.month.split("-"))
        days = range(1, calendar.monthrange(year, month)[1] + 1)
        sales = sales.reindex(days, fill_value=0)

        fig = px.bar(
            x=list(days),
            y=sales.values,
            labels={"x": "Day", "y": "Sales"},
            title=f"{st.session_state.month}"
        )

        st.plotly_chart(fig, use_container_width=True)
