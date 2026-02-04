import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


def render_html_table(
    df: pd.DataFrame,
    title: str | None = None,
    max_height: int = 300   # 🔥 cap for large tables
):
    if df is None or df.empty:
        st.info("No data to display.")
        return

    if title:
        st.markdown(f"### 📋 {title}")

    # ---------- AUTO HEIGHT CALCULATION ----------
    ROW_HEIGHT = 32
    HEADER_HEIGHT = 44
    PADDING = 20

    rows = len(df)
    calculated_height = HEADER_HEIGHT + (rows * ROW_HEIGHT) + PADDING
    final_height = min(calculated_height, max_height)

    table_html = df.to_html(
        index=False,
        classes="display compact",
        table_id="stickyTable",
        escape=False
    )

    html = f"""
    <html>
    <head>
        <link rel="stylesheet"
              href="https://cdn.datatables.net/1.13.6/css/jquery.dataTables.min.css">

        <style>
            .table-container {{
                max-height: {final_height}px;
                overflow-y: auto;
                overflow-x: auto;
                border: none;
                margin: 0;
                padding: 0;
            }}

            table.dataTable {{
                width: 100% !important;
                font-size: 13.5px;
                border-collapse: collapse;
                border: none !important;
            }}

            table.dataTable thead th {{
                position: sticky;
                top: 0;
                background-color: #F3F4F6;
                font-weight: 600;
                text-align: center !important;
                color: #1F2937;
                padding: 6px 8px;
                border-bottom: 1px solid #D1D5DB;
                white-space: nowrap;
            }}

            table.dataTable td {{
                padding: 3px 5px;
                border-bottom: 1px solid #E5E7EB;
                white-space: nowrap;
                text-align: center !important;
            }}

            table.dataTable th,
            table.dataTable td {{
                border-left: none !important;
                border-right: none !important;
            }}

            table.dataTable tbody tr:nth-child(even) {{
                background-color: #EEF2F7;
            }}

            table.dataTable tbody tr:hover {{
                background-color: #E5EDFF;
            }}

            .dataTables_filter,
            .dataTables_length,
            .dataTables_info {{
                display: none;
            }}

            .dataTables_wrapper {{
                margin: 0 !important;
                padding: 0 !important;
            }}
        </style>
    </head>

    <body>
        <div class="table-container">
            {table_html}
        </div>

        <script src="https://code.jquery.com/jquery-3.7.0.min.js"></script>
        <script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>

        <script>
            $(document).ready(function() {{
                $('#stickyTable').DataTable({{
                    paging: false,
                    ordering: true,
                    searching: false,
                    info: false,
                    scrollX: true
                }});
            }});
        </script>
    </body>
    </html>
    """

    # 🔥 KEY FIXES HERE
    components.html(
        html,
        height=final_height,
        scrolling=False   # 🔥 prevent iframe scrollbar
    )
