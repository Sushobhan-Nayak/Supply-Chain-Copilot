import streamlit as st
import pandas as pd
import graphviz
import altair as alt
import json

from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
from streamlit_flow.state import StreamlitFlowState
from streamlit_flow.layouts import LayeredLayout

from cortex_agent import (
    ask_agent_stream,
    AgentResult,
    run_sql,
    create_thread,
    list_threads,
    get_thread_messages,
    set_thread_name,
    delete_thread,
    _extract_from_response_payload,
)


st.set_page_config(
    page_title="Supply Chain Copilot",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# STYLING
# ============================================================================
st.markdown(
    """
<style>
    #MainMenu { visibility: hidden !important; }
    footer { visibility: hidden !important; }

    .stApp, body {
        background-color: #ffffff !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    }

    [data-testid="stSidebarResizeHandle"] {
        display: none !important;
    }

    [data-testid="stAppViewContainer"] {
        gap: 0 !important;
    }

    [data-testid="stSidebar"] {
        margin-right: 0 !important;
    }

    [data-testid="stMain"] {
        padding-left: 0 !important;
        margin-left: 0 !important;
    }

    [data-testid="stMainBlockContainer"],
    .block-container {
        max-width: 100% !important;
        padding-top: 1.25rem !important;
        padding-bottom: 2rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }

    .nav-group {
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #98a1ac;
        margin: 18px 0 4px 2px;
    }

    [data-testid="stSidebar"] {
        background: #fafbfc !important;
    }

    [data-testid="stSidebar"] [role="radiogroup"] {
        gap: 2px !important;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label {
        display: flex !important;
        align-items: center !important;
        padding: 8px 12px !important;
        border-radius: 8px !important;
        margin: 0 !important;
        width: 100% !important;
        cursor: pointer;
    }

    [data-testid="stSidebar"] [role="radiogroup"] input {
        display: none !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div:first-of-type {
        display: none !important;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: #eef2f7 !important;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
        background: #e8f0fc !important;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p {
        color: #2568c4 !important;
        font-weight: 600 !important;
    }

    [data-testid="stSidebar"] [role="radiogroup"] p {
        font-size: 14px !important;
        color: #3c4149 !important;
    }

    [data-testid="stSidebar"] .brand-container {
        margin-bottom: 16px;
    }

    .st-key-new_chat button {
        background: #ffffff !important;
        color: #2568c4 !important;
        font-weight: 600 !important;
        border: 1.5px solid #2568c4 !important;
        border-radius: 8px !important;
        font-size: 13px !important;
    }

    .st-key-new_chat button:hover {
        background: #eef4fd !important;
        border-color: #1f57a8 !important;
    }

    .hist-empty {
        font-size: 12px;
        color: #98a1ac;
        padding: 8px 4px;
        font-style: italic;
    }

    [data-testid="stSidebar"] .st-key-hist_active button {
        background: #e8f0fc !important;
        color: #2568c4 !important;
    }

    .brand-container {
        display: flex;
        align-items: center;
        gap: 11px;
        font-weight: 700;
        font-size: 19px;
        color: #1c1e21 !important;
    }

    .brand-logo {
        width: 34px;
        height: 34px;
        border-radius: 9px;
        background: #2568c4 !important;
        color: #fff !important;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 14px;
        font-weight: 700;
        flex-shrink: 0;
    }

    /* Segmented tabs */
    [data-testid="stPills"] button {
        border-radius: 20px !important;
        border: 0.5px solid #e3e5e9 !important;
        background: #ffffff !important;
        color: #565c66 !important;
        font-size: 13px !important;
        font-weight: 500 !important;
    }

    [data-testid="stPills"] button[aria-selected="true"],
    [data-testid="stPills"] button[kind="pillsActive"] {
        background: #e8f0fc !important;
        color: #2568c4 !important;
        font-weight: 600 !important;
        border-color: transparent !important;
    }

    /* KPI cards */
    .kpi-inner {
        background: #f8f9fb;
        border: 1px solid #eceef1;
        border-radius: 10px;
        border-left: 3px solid #ccc;
        padding: 12px 16px;
    }

    .kpi-inner .label {
        font-size: 12px;
        color: #8a909b !important;
    }

    .kpi-inner .value {
        font-size: 24px;
        font-weight: 600;
        color: #1c1e21 !important;
        margin-top: 2px;
    }

    .band-rule {
        height: 1px;
        background: #e3e5e9;
        margin: 20px 0 18px;
    }

    /* Dashboard */
    .dash-heading {
        font-size: 28px;
        font-weight: 700;
        color: #1c1e21 !important;
        margin: 4px 0 20px;
        letter-spacing: -0.01em;
    }

    [class*="st-key-chartcard_"] {
        border: 1px solid #e3e5e9 !important;
        border-radius: 12px !important;
        background: #ffffff !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
        padding: 0 0 14px !important;
        overflow: hidden !important;
        margin-bottom: 6px !important;
    }

    .chart-card-title {
        background: #f4f6f9;
        border-bottom: 1px solid #e3e5e9;
        margin: 0 0 14px;
        padding: 12px 16px;
        font-size: 15px;
        font-weight: 700;
        color: #1c1e21 !important;
        text-align: center;
    }

    [class*="st-key-chartcard_"] [data-testid="stElementContainer"] {
        padding: 0 14px !important;
    }

    [class*="st-key-chartcard_"] .chart-card-title {
        padding: 12px 16px !important;
    }

    /* Main 3-column layout */
    .st-key-main_row > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(2),
    .st-key-main_row > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(3) {
        border-left: 1px solid #e3e5e9 !important;
        padding-left: 24px !important;
    }

    /* Left rail */
    .st-key-main_row > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child:nth-last-child(2) {
        background-color: #f7f8fa !important;
        border-right: 1px solid #e3e5e9 !important;
        border-radius: 12px !important;
        padding: 16px !important;
    }

    .st-key-main_row > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child:nth-last-child(2)
        + [data-testid="stColumn"] {
        border-left: none !important;
        padding-left: 16px !important;
    }

    .col-label {
        font-size: 11px;
        color: #8a909b !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        padding-left: 9px;
        margin: 20px 0 8px;
        font-weight: 600;
    }

    /* Buttons */
    [data-testid="stButton"] button {
        border-radius: 8px !important;
        border: 0.5px solid transparent !important;
        background: transparent !important;
        color: #565c66 !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        text-align: left !important;
        justify-content: flex-start !important;
        padding: 7px 9px !important;
        min-height: 0 !important;
    }

    [data-testid="stButton"] button:hover {
        background: #f4f5f7 !important;
        color: #1c1e21 !important;
    }

    [data-testid="stSidebar"] [class*="st-key-hist_"] button,
    [data-testid="stSidebar"] [class*="st-key-hist_"] button * {
        justify-content: flex-start !important;
        text-align: left !important;
        font-size: 13px !important;
    }

    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {
        flex-wrap: nowrap !important;
        gap: 2px !important;
        align-items: center !important;
    }

    [class*="st-key-del_"] button {
        background: transparent !important;
        border: none !important;
        padding: 4px 6px !important;
        min-height: 0 !important;
        opacity: 0.4 !important;
        font-size: 13px !important;
    }

    [class*="st-key-del_"] button:hover {
        opacity: 1 !important;
        background: #f4d9d9 !important;
        color: #c0392b !important;
    }

    /* Chat */
    .user-row {
        display: flex;
        justify-content: flex-end;
        margin: 8px 0 16px;
    }

    .user-bubble {
        background: #2568c4 !important;
        color: #fff !important;
        padding: 11px 15px;
        border-radius: 16px 16px 4px 16px;
        font-size: 14px;
        line-height: 1.5;
        max-width: 85%;
    }

    .assistant-answer-value {
        font-size: 32px;
        font-weight: 600;
        color: #1c1e21 !important;
        letter-spacing: -0.01em;
    }

    .assistant-delta {
        font-size: 12px;
        color: #177a52 !important;
        background: #e5f5ee !important;
        padding: 2px 9px;
        border-radius: 20px;
        font-weight: 500;
        margin-left: 8px;
    }

    .assistant-sub {
        font-size: 13px;
        color: #565c66 !important;
        margin-top: 5px;
    }

    .assistant-prose {
        font-size: 14px;
        color: #1c1e21 !important;
        line-height: 1.6;
        margin-top: 12px;
    }

    .empty-greeting {
        text-align: center;
        color: #8a909b;
        font-size: 14px;
        margin: 22vh 0 30px;
    }

    .empty-greeting .big {
        display: block;
        font-size: 24px;
        font-weight: 600;
        color: #1c1e21;
        margin-bottom: 6px;
    }

    /* Trace box */
    .trace-box {
        border: 0.5px solid #e3e5e9 !important;
        border-radius: 12px;
        background: #ffffff !important;
        overflow: hidden;
        margin: 4px 0 14px;
    }

    .trace-head {
        display: flex;
        align-items: center;
        gap: 7px;
        padding: 9px 13px;
        background: #f4f5f7 !important;
        border-bottom: 0.5px solid #e3e5e9 !important;
        font-size: 12px;
    }

    .trace-head .title {
        font-weight: 600;
        color: #1c1e21 !important;
    }

    .trace-head .meta {
        color: #8a909b !important;
    }

    .trace-body {
        padding: 10px 14px;
        display: flex;
        flex-direction: column;
        gap: 8px;
    }

    .trace-step {
        font-size: 13px;
        color: #565c66 !important;
    }

    .trace-step code {
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace !important;
        color: #1c1e21 !important;
        background: #f4f5f7 !important;
        padding: 1px 5px;
        border-radius: 4px;
        font-size: 12px;
    }

    .trace-check {
        color: #177a52 !important;
        font-weight: bold;
        margin-right: 6px;
    }

    .answer-card {
        border: 0.5px solid #e3e5e9 !important;
        border-radius: 14px;
        padding: 16px 18px;
        background: #f4f5f7 !important;
        margin-bottom: 12px;
    }

    /* Grounding */
    .g-head {
        font-size: 13px;
        font-weight: 600;
        color: #177a52 !important;
        margin-bottom: 12px;
        padding-bottom: 10px;
        border-bottom: 0.5px solid #e3e5e9 !important;
    }

    .g-label {
        font-size: 11px;
        color: #8a909b !important;
        margin-top: 12px;
    }

    .g-value {
        font-size: 13px;
        font-weight: 600;
        color: #1c1e21 !important;
    }

    .g-text {
        font-size: 12px;
        color: #565c66 !important;
        line-height: 1.5;
    }

    .g-sql {
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace !important;
        font-size: 11px;
        background: #f4f5f7 !important;
        border: 0.5px solid #e3e5e9 !important;
        border-radius: 8px;
        padding: 10px 12px;
        color: #565c66 !important;
        line-height: 1.7;
        margin-top: 5px;
    }

    .g-divider {
        height: 0.5px;
        background: #e3e5e9 !important;
        margin: 14px 0;
    }

    /* Ontology detail card */
    .st-key-onto_stage {
        position: relative !important;
    }

    .st-key-onto_detail_card {
        position: absolute !important;
        top: 14px;
        right: 14px;
        z-index: 20 !important;
        width: 320px !important;
        max-height: 560px !important;
        overflow-y: auto !important;
        background: #ffffff !important;
        border: 1px solid #e3e5e9 !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 16px rgba(0,0,0,0.10) !important;
        padding: 14px 16px !important;
    }

    .g-row {
        display: flex;
        justify-content: space-between;
        font-size: 12px;
        padding: 5px 0;
    }

    .g-row .name {
        color: #565c66 !important;
    }

    .g-row .val {
        font-weight: 600;
        color: #1c1e21 !important;
    }

    /* Ontology */
    .onto-group-title {
        font-size: 11px;
        color: #8a909b !important;
        margin: 14px 0 8px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .onto-item {
        display: flex;
        justify-content: space-between;
        font-size: 13px;
        color: #565c66 !important;
        padding: 4px 0;
    }

    .onto-item .count {
        color: #8a909b !important;
        font-size: 12px;
    }

    .metric-card {
        border: 0.5px solid #e3e5e9 !important;
        border-radius: 10px;
        padding: 10px 12px;
        margin-bottom: 10px;
        background: #ffffff !important;
    }

    .metric-card .m-name {
        font-size: 13px;
        font-weight: 600;
        color: #1c1e21 !important;
    }

    .metric-card .m-def {
        font-size: 12px;
        color: #565c66 !important;
        margin-top: 2px;
        line-height: 1.4;
    }

    .metric-card .m-view {
        font-size: 11px;
        color: #8a909b !important;
        margin-top: 5px;
        font-family: ui-monospace, Menlo, Consolas, monospace !important;
    }

    /* Scrollable chat */
    .st-key-msg_scroll {
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
        padding: 0 8px 90px !important;
        max-width: 820px !important;
        margin: 0 auto !important;
    }

    /* Composer */
    [data-testid="stChatInput"] {
        border-radius: 14px !important;
        border: 1px solid #e3e5e9 !important;
    }

    [data-testid="stChatInput"] textarea {
        font-size: 14px !important;
    }

    [data-testid="stBottom"] {
        background: transparent !important;
    }

    [data-testid="stBottom"] > div {
        max-width: 100% !important;
        padding: 0 !important;
    }

    [data-testid="stBottomBlockContainer"] {
        max-width: 100% !important;
        padding: 0 2rem !important;
        margin: 0 !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        width: 300px !important;
        min-width: 300px !important;
    }

    [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarCollapse"],
    [data-testid="collapsedControl"] {
        display: none !important;
    }

    /* Fixed composer dock */
    .st-key-composer_dock {
        position: fixed !important;
        bottom: 0 !important;
        left: 300px !important;
        right: 0 !important;
        width: auto !important;
        box-sizing: border-box !important;
        z-index: 100 !important;
        background: linear-gradient(
            to top,
            #ffffff 62%,
            rgba(255,255,255,0)
        ) !important;
        padding: 10px 2rem 16px !important;
    }

    /* Composer bar */
    .st-key-composer_bar {
        border: 1px solid #e3e5e9 !important;
        border-radius: 20px !important;
        background: #ffffff !important;
        padding: 0 10px !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05) !important;
        margin: 0 auto !important;
        max-width: 760px !important;
    }

    .st-key-composer_bar [data-testid="stHorizontalBlock"] {
        gap: 6px !important;
        align-items: center !important;
    }

    .st-key-composer_bar [data-testid="stTextInput"] > div > div {
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
        padding: 0 !important;
    }

    .st-key-composer_bar [data-testid="stTextInput"] input {
        font-size: 14px !important;
        padding: 2px 8px !important;
        min-height: 0 !important;
        height: 30px !important;
    }

    .st-key-composer_bar [data-testid="stButton"] button {
        border-radius: 18px !important;
        height: 32px !important;
        min-height: 32px !important;
        margin: 0 !important;
        padding: 0 14px !important;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================================
# SESSION STATE
# ============================================================================
if "active_view" not in st.session_state:
    st.session_state.active_view = "Chat"

if "active_persona" not in st.session_state:
    st.session_state.active_persona = "Planning"

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = None

if "parent_message_id" not in st.session_state:
    st.session_state.parent_message_id = 0

if "active_entity" not in st.session_state:
    st.session_state.active_entity = None


# ============================================================================
# GOVERNED KPI SNAPSHOT
# ============================================================================
KPI_SNAPSHOT_QUERY = (
    "SELECT METRIC_ID, METRIC_NAME, METRIC_VALUE, UNIT "
    "FROM SUPPLY_CHAIN_ONTOLOGY.CORE.V_DASHBOARD_KPI_SNAPSHOT"
)

KPI_COLOR = {
    "outbound_otd": "#378ADD",
    "inbound_otd": "#E0B84B",
    "fill_rate": "#1D9E75",
    "dii": "#D85A30",
    "landed_cost": "#7F77DD",
}


# ============================================================================
# PERSONA CONFIGURATION
# ============================================================================
PERSONA_CONFIG = {
    "Planning": {
        "kpi_order": [
            "outbound_otd",
            "fill_rate",
            "dii",
            "landed_cost",
            "inbound_otd",
        ],
        "accent": "#2568c4",
        "accent_dark": "#1f57a8",
        "accent_soft": "#e8f0fc",
        "accent_hover": "#f0f5fd",
    },
    "Procurement": {
        "kpi_order": [
            "inbound_otd",
            "landed_cost",
            "dii",
            "fill_rate",
            "outbound_otd",
        ],
        "accent": "#1d9e75",
        "accent_dark": "#177f5e",
        "accent_soft": "#e3f5ee",
        "accent_hover": "#eefaf5",
    },
    "Logistics": {
        "kpi_order": [
            "outbound_otd",
            "inbound_otd",
            "fill_rate",
            "dii",
            "landed_cost",
        ],
        "accent": "#d0463a",
        "accent_dark": "#ad392f",
        "accent_soft": "#fbe9e7",
        "accent_hover": "#fdf3f1",
    },
}


def _fmt_kpi(value, unit: str) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "—"

    if unit == "PERCENT":
        return f"{v * 100:.1f}%"

    if unit == "DAYS":
        return f"{v:.1f}"

    if unit == "USD":
        return f"${v:,.2f}"

    return f"{v:,.2f}"


@st.cache_data(ttl=300, show_spinner=False)
def load_kpis() -> tuple[dict[str, dict], list[str]]:
    """Read the governed KPI snapshot."""
    errors = []
    metrics: dict[str, dict] = {}

    try:
        cols, rows = run_sql(KPI_SNAPSHOT_QUERY)
        idx = {c: i for i, c in enumerate(cols)}

        for r in rows:
            mid = r[idx["METRIC_ID"]]

            metrics[mid] = {
                "name": r[idx["METRIC_NAME"]],
                "value": _fmt_kpi(
                    r[idx["METRIC_VALUE"]],
                    r[idx["UNIT"]],
                ),
                "color": KPI_COLOR.get(mid, "#8a909b"),
            }

    except Exception as e:
        errors.append(
            f"KPI snapshot: {type(e).__name__}: {e}"
        )

    return metrics, errors


# ============================================================================
# DASHBOARD SQL
# ============================================================================
_O = "SUPPLY_CHAIN_ONTOLOGY.CORE.VW_ON_TIME_DELIVERY_OUTBOUND"
_I = "SUPPLY_CHAIN_ONTOLOGY.CORE.VW_ON_TIME_DELIVERY_INBOUND"
_F = "SUPPLY_CHAIN_ONTOLOGY.CORE.VW_FILL_RATE"
_D = "SUPPLY_CHAIN_ONTOLOGY.CORE.VW_DAYS_OF_INVENTORY"
_L = "SUPPLY_CHAIN_ONTOLOGY.CORE.VW_LANDED_COST"
_SUP = "SUPPLY_CHAIN_ONTOLOGY.RAW.DIM_SUPPLIER"

_LANDED_NUM = (
    "(UNIT_PRICE_USD*QUANTITY_SHIPPED+"
    "FREIGHT_COST_USD+INSURANCE_COST_USD+CUSTOMS_COST_USD)"
)

_LANDED_NUM_L = (
    "(l.UNIT_PRICE_USD*l.QUANTITY_SHIPPED+"
    "l.FREIGHT_COST_USD+l.INSURANCE_COST_USD+l.CUSTOMS_COST_USD)"
)


PERSONA_DASHBOARD_QUERIES = {
    "Planning": [
        (
            "Outbound OTD trend",
            f"""
            SELECT
                DELIVERY_MONTH AS MONTH,
                ROUND(AVG(IS_ON_TIME)*100,2) AS OTD_PCT
            FROM {_O}
            GROUP BY DELIVERY_MONTH
            ORDER BY DELIVERY_MONTH
            """,
            "line",
            "MONTH",
            "OTD_PCT",
            "#2568c4",
        ),
        (
            "Fill rate trend",
            f"""
            SELECT
                ORDER_MONTH AS MONTH,
                ROUND(
                    SUM(QUANTITY_SHIPPED) /
                    NULLIF(SUM(QUANTITY_ORDERED),0) * 100,
                    2
                ) AS FILL_PCT
            FROM {_F}
            GROUP BY ORDER_MONTH
            ORDER BY ORDER_MONTH
            """,
            "line",
            "MONTH",
            "FILL_PCT",
            "#1D9E75",
        ),
        (
            "Days of inventory by plant",
            f"""
            SELECT
                PLANT_ID,
                ROUND(
                    SUM(ON_HAND_QTY) /
                    NULLIF(SUM(AVG_DAILY_USAGE),0),
                    1
                ) AS DII
            FROM {_D}
            GROUP BY PLANT_ID
            ORDER BY DII DESC
            """,
            "bar",
            "PLANT_ID",
            "DII",
            "#D85A30",
        ),
        (
            "Days of inventory by part category",
            f"""
            SELECT
                PART_CATEGORY_ID,
                ROUND(
                    SUM(ON_HAND_QTY) /
                    NULLIF(SUM(AVG_DAILY_USAGE),0),
                    1
                ) AS DII
            FROM {_D}
            GROUP BY PART_CATEGORY_ID
            ORDER BY DII DESC
            """,
            "bar",
            "PART_CATEGORY_ID",
            "DII",
            "#7F77DD",
        ),
        (
            "Inventory on hand by plant",
            f"""
            SELECT
                PLANT_ID,
                SUM(ON_HAND_QTY) AS ON_HAND
            FROM {_D}
            GROUP BY PLANT_ID
            ORDER BY ON_HAND DESC
            """,
            "bar",
            "PLANT_ID",
            "ON_HAND",
            "#378ADD",
        ),
    ],

    "Procurement": [
        (
            "Inbound OTD by supplier (worst 12)",
            f"""
            SELECT
                s.SUPPLIER_NAME AS SUPPLIER,
                ROUND(AVG(i.IS_ON_TIME)*100,2) AS OTD_PCT
            FROM {_I} i
            LEFT JOIN {_SUP} s
                ON i.SUPPLIER_ID=s.SUPPLIER_ID
            GROUP BY s.SUPPLIER_NAME
            ORDER BY OTD_PCT ASC
            LIMIT 12
            """,
            "bar",
            "SUPPLIER",
            "OTD_PCT",
            "#E0B84B",
        ),
        (
            "Landed cost by supplier (top 12)",
            f"""
            SELECT
                s.SUPPLIER_NAME AS SUPPLIER,
                ROUND(
                    SUM({_LANDED_NUM_L}) /
                    NULLIF(SUM(l.QUANTITY_SHIPPED),0),
                    2
                ) AS LANDED
            FROM {_L} l
            LEFT JOIN {_SUP} s
                ON l.SUPPLIER_ID=s.SUPPLIER_ID
            GROUP BY s.SUPPLIER_NAME
            ORDER BY LANDED DESC
            LIMIT 12
            """,
            "bar",
            "SUPPLIER",
            "LANDED",
            "#7F77DD",
        ),
        (
            "Landed cost by part (top 12)",
            f"""
            SELECT
                PART_ID,
                ROUND(
                    SUM({_LANDED_NUM}) /
                    NULLIF(SUM(QUANTITY_SHIPPED),0),
                    2
                ) AS LANDED
            FROM {_L}
            GROUP BY PART_ID
            ORDER BY LANDED DESC
            LIMIT 12
            """,
            "bar",
            "PART_ID",
            "LANDED",
            "#D85A30",
        ),
        (
            "Landed cost breakdown",
            f"""
            SELECT
                SUM(UNIT_PRICE_USD*QUANTITY_SHIPPED) AS PRODUCT,
                SUM(FREIGHT_COST_USD) AS FREIGHT,
                SUM(INSURANCE_COST_USD) AS INSURANCE,
                SUM(CUSTOMS_COST_USD) AS CUSTOMS
            FROM {_L}
            """,
            "components",
            "",
            "",
            "#2568c4",
        ),
        (
            "Supplier reliability vs inbound OTD",
            f"""
            SELECT
                s.SUPPLIER_RELIABILITY_SCORE AS RELIABILITY,
                ROUND(AVG(i.IS_ON_TIME)*100,2) AS OTD_PCT
            FROM {_I} i
            JOIN {_SUP} s
                ON i.SUPPLIER_ID=s.SUPPLIER_ID
            GROUP BY s.SUPPLIER_RELIABILITY_SCORE
            HAVING COUNT(*)>0
            ORDER BY RELIABILITY
            """,
            "scatter",
            "RELIABILITY",
            "OTD_PCT",
            "#1D9E75",
        ),
    ],

    "Logistics": [
        (
            "Outbound OTD trend",
            f"""
            SELECT
                DELIVERY_MONTH AS MONTH,
                ROUND(AVG(IS_ON_TIME)*100,2) AS OTD_PCT
            FROM {_O}
            GROUP BY DELIVERY_MONTH
            ORDER BY DELIVERY_MONTH
            """,
            "line",
            "MONTH",
            "OTD_PCT",
            "#2568c4",
        ),
        (
            "OTD by plant",
            f"""
            SELECT
                PLANT_ID,
                ROUND(AVG(IS_ON_TIME)*100,2) AS OTD_PCT
            FROM {_O}
            GROUP BY PLANT_ID
            ORDER BY OTD_PCT ASC
            """,
            "bar",
            "PLANT_ID",
            "OTD_PCT",
            "#378ADD",
        ),
        (
            "OTD by carrier (worst 12)",
            f"""
            SELECT
                CARRIER,
                ROUND(AVG(IS_ON_TIME)*100,2) AS OTD_PCT
            FROM {_O}
            GROUP BY CARRIER
            ORDER BY OTD_PCT ASC
            LIMIT 12
            """,
            "bar",
            "CARRIER",
            "OTD_PCT",
            "#E0B84B",
        ),
        (
            "OTD by mode",
            f"""
            SELECT
                MODE,
                ROUND(AVG(IS_ON_TIME)*100,2) AS OTD_PCT
            FROM {_O}
            GROUP BY MODE
            ORDER BY OTD_PCT ASC
            """,
            "bar",
            "MODE",
            "OTD_PCT",
            "#1D9E75",
        ),
        (
            "Late shipments by carrier (top 12)",
            f"""
            SELECT
                CARRIER,
                SUM(
                    CASE
                        WHEN IS_ON_TIME=0 THEN 1
                        ELSE 0
                    END
                ) AS LATE
            FROM {_O}
            GROUP BY CARRIER
            ORDER BY LATE DESC
            LIMIT 12
            """,
            "bar",
            "CARRIER",
            "LATE",
            "#D85A30",
        ),
        (
            "Delivery delay distribution",
            f"""
            SELECT
                CASE
                    WHEN DELIVERY_DATE<=PROMISED_DELIVERY_DATE
                        THEN 'On time'
                    WHEN DATEDIFF(
                        'day',
                        PROMISED_DELIVERY_DATE,
                        DELIVERY_DATE
                    )=1
                        THEN '1 day late'
                    WHEN DATEDIFF(
                        'day',
                        PROMISED_DELIVERY_DATE,
                        DELIVERY_DATE
                    ) BETWEEN 2 AND 3
                        THEN '2-3 days late'
                    WHEN DATEDIFF(
                        'day',
                        PROMISED_DELIVERY_DATE,
                        DELIVERY_DATE
                    ) BETWEEN 4 AND 7
                        THEN '4-7 days late'
                    ELSE '8+ days late'
                END AS BUCKET,
                COUNT(*) AS SHIPMENTS
            FROM {_O}
            GROUP BY 1
            """,
            "bar_v",
            "BUCKET",
            "SHIPMENTS",
            "#2568c4",
        ),
    ],
}


_NON_NUMERIC = {
    "MONTH",
    "PLANT_ID",
    "SUPPLIER",
    "PART_ID",
    "PART_CATEGORY_ID",
    "CARRIER",
    "MODE",
    "BUCKET",
}


@st.cache_data(ttl=300, show_spinner=False)
def load_df(sql: str) -> pd.DataFrame:
    try:
        cols, rows = run_sql(sql)
        df = pd.DataFrame(rows, columns=cols)

        for c in df.columns:
            if c not in _NON_NUMERIC:
                df[c] = pd.to_numeric(
                    df[c],
                    errors="coerce",
                )

        return df

    except Exception:
        return pd.DataFrame()


def line_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str,
    y_title: str,
):
    lo = max(0, float(df[y].min()) - 2)
    hi = float(df[y].max()) + 2

    return (
        alt.Chart(df)
        .mark_line(
            point=True,
            color=color,
            strokeWidth=2.5,
        )
        .encode(
            x=alt.X(
                f"{x}:N",
                title=None,
                axis=alt.Axis(labelAngle=-45),
            ),
            y=alt.Y(
                f"{y}:Q",
                scale=alt.Scale(domain=[lo, hi]),
                title=y_title,
            ),
            tooltip=[x, y],
        )
        .properties(height=240)
    )


def bar_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str,
    y_title: str,
    horizontal: bool = True,
):
    if horizontal:
        enc = dict(
            y=alt.Y(
                f"{x}:N",
                sort="-x",
                title=None,
            ),
            x=alt.X(
                f"{y}:Q",
                title=y_title,
            ),
        )
    else:
        enc = dict(
            x=alt.X(
                f"{x}:N",
                sort="-y",
                title=None,
            ),
            y=alt.Y(
                f"{y}:Q",
                title=y_title,
            ),
        )

    return (
        alt.Chart(df)
        .mark_bar(
            color=color,
            cornerRadius=3,
        )
        .encode(
            tooltip=[x, y],
            **enc,
        )
        .properties(height=260)
    )


def scatter_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str,
    y_title: str,
):
    return (
        alt.Chart(df)
        .mark_circle(
            size=90,
            color=color,
            opacity=0.7,
        )
        .encode(
            x=alt.X(
                f"{x}:Q",
                title=x.replace("_", " ").title(),
            ),
            y=alt.Y(
                f"{y}:Q",
                title=y_title,
            ),
            tooltip=[x, y],
        )
        .properties(height=260)
    )


def render_persona_chart(spec, idx: int = 0) -> None:
    title, sql, kind, x, y, color = spec

    with st.container(
        key=f"chartcard_{idx}",
        border=True,
    ):
        st.markdown(
            f'<div class="chart-card-title">{title}</div>',
            unsafe_allow_html=True,
        )

        df = load_df(sql)

        if df.empty:
            st.caption("No data.")
            return

        if kind == "line":
            st.altair_chart(
                line_chart(df, x, y, color, y),
                width="stretch",
            )

        elif kind == "bar":
            st.altair_chart(
                bar_chart(
                    df,
                    x,
                    y,
                    color,
                    y,
                    horizontal=True,
                ),
                width="stretch",
            )

        elif kind == "bar_v":
            st.altair_chart(
                bar_chart(
                    df,
                    x,
                    y,
                    color,
                    y,
                    horizontal=False,
                ),
                width="stretch",
            )

        elif kind == "scatter":
            st.altair_chart(
                scatter_chart(
                    df,
                    x,
                    y,
                    color,
                    y,
                ),
                width="stretch",
            )

        elif kind == "components":
            comp = pd.DataFrame(
                {
                    "Component": [
                        "Product",
                        "Freight",
                        "Insurance",
                        "Customs",
                    ],
                    "USD": [
                        df.iloc[0]["PRODUCT"],
                        df.iloc[0]["FREIGHT"],
                        df.iloc[0]["INSURANCE"],
                        df.iloc[0]["CUSTOMS"],
                    ],
                }
            )

            st.altair_chart(
                bar_chart(
                    comp,
                    "Component",
                    "USD",
                    color,
                    "Total USD",
                ),
                width="stretch",
            )


# ============================================================================
# CHAT HELPERS
# ============================================================================
def _rows_as_dicts(
    cols: list[str],
    rows: list[list],
) -> list[dict]:
    return [
        dict(zip(cols, r))
        for r in rows
    ]


def _payload_text(payload) -> str:
    """Extract plain text from a thread message payload."""
    try:
        obj = (
            json.loads(payload)
            if isinstance(payload, str)
            else payload
        )
    except (json.JSONDecodeError, TypeError):
        return str(payload)

    if isinstance(obj, dict):
        texts = [
            c.get("text", "")
            for c in obj.get("content", [])
            if c.get("type") == "text"
        ]

        joined = " ".join(
            t for t in texts if t
        ).strip()

        return joined or ""

    return str(payload)


@st.cache_data(ttl=600, show_spinner=False)
def thread_title(thread_id: int) -> str:
    """Title for an unnamed thread."""
    try:
        for m in get_thread_messages(
            thread_id,
            page_size=20,
        ):
            if m.get("role") == "user":
                text = _payload_text(
                    m.get("message_payload")
                ).strip()

                if text:
                    return (
                        text[:48]
                        + ("…" if len(text) > 48 else "")
                    )

    except Exception:
        pass

    return "Untitled conversation"


def restore_thread(thread_id: int) -> None:
    """Load a previous thread into the chat."""
    msgs = get_thread_messages(thread_id)

    chat = []
    last_assistant_id = 0

    for m in msgs:
        role = m.get("role")
        payload = m.get("message_payload")

        if role == "user":
            text = _payload_text(payload)

            if text:
                chat.append(
                    {
                        "role": "user",
                        "content": text,
                    }
                )

        elif role == "assistant":
            res = AgentResult()

            try:
                obj = (
                    json.loads(payload)
                    if isinstance(payload, str)
                    else payload
                )

                _extract_from_response_payload(
                    obj,
                    res,
                )

            except (json.JSONDecodeError, TypeError):
                res.answer_text = _payload_text(
                    payload
                )

            chat.append(
                {
                    "role": "assistant",
                    "result": res,
                    "question": None,
                }
            )

            if m.get("message_id"):
                last_assistant_id = m["message_id"]

    st.session_state.chat_messages = chat
    st.session_state.thread_id = thread_id
    st.session_state.parent_message_id = last_assistant_id


def submit_question(prompt: str) -> None:
    """Queue a question for streaming."""
    st.session_state.chat_messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    st.session_state.pending_question = prompt


def finalize_answer(
    prompt: str,
    result: AgentResult,
) -> None:
    """Finalize the streamed answer."""
    if result.assistant_message_id is not None:
        st.session_state.parent_message_id = (
            result.assistant_message_id
        )

    if (
        st.session_state.get("just_created_thread")
        and st.session_state.thread_id
    ):
        try:
            set_thread_name(
                st.session_state.thread_id,
                prompt,
            )
        except Exception:
            pass

        st.session_state.just_created_thread = False

    st.session_state.chat_messages.append(
        {
            "role": "assistant",
            "result": result,
            "question": prompt,
        }
    )


# ============================================================================
# ONTOLOGY LOADERS
# ============================================================================
@st.cache_data(ttl=600, show_spinner=False)
def load_ontology_entities() -> list[dict]:
    cols, rows = run_sql(
        """
        SELECT
            entity_id,
            display_name,
            entity_type,
            description,
            source_object,
            business_domain
        FROM V_ONTOLOGY_ENTITIES
        WHERE is_active = 'true'
        ORDER BY entity_type, display_name
        """
    )

    return _rows_as_dicts(cols, rows)


@st.cache_data(ttl=600, show_spinner=False)
def load_ontology_graph() -> list[dict]:
    cols, rows = run_sql(
        """
        SELECT
            source_id,
            source_name,
            edge_label,
            target_id,
            target_name,
            relationship_type,
            cardinality
        FROM V_ONTOLOGY_GRAPH
        """
    )

    return _rows_as_dicts(cols, rows)


@st.cache_data(ttl=600, show_spinner=False)
def load_ontology_attributes(
    entity_id: str,
) -> list[dict]:
    safe = entity_id.replace("'", "''")

    cols, rows = run_sql(
        """
        SELECT
            attribute_name,
            display_name,
            data_type,
            is_key,
            is_foreign_key
        FROM V_ONTOLOGY_ATTRIBUTES
        WHERE entity_id = '"""
        + safe
        + """'
        AND is_active = 'true'
        """
    )

    return _rows_as_dicts(cols, rows)


@st.cache_data(ttl=600, show_spinner=False)
def load_entity_metrics(
    entity_id: str,
) -> list[dict]:
    safe = entity_id.replace("'", "''")

    cols, rows = run_sql(
        """
        SELECT
            metric_name,
            definition,
            canonical_object,
            role
        FROM V_ONTOLOGY_ENTITY_METRICS
        WHERE entity_id = '"""
        + safe
        + """'
        """
    )

    return _rows_as_dicts(cols, rows)


@st.cache_data(ttl=600, show_spinner=False)
def load_metric_catalog() -> list[dict]:
    cols, rows = run_sql(
        """
        SELECT
            display_name,
            description,
            definition,
            grain,
            canonical_object,
            business_domain
        FROM V_ONTOLOGY_METRICS
        WHERE is_active = 'true'
        ORDER BY display_name
        """
    )

    return _rows_as_dicts(cols, rows)


# ============================================================================
# CHAT AUTOSCROLL
# ============================================================================
def _autoscroll_chat(nonce: int) -> None:
    """Keep the fixed-height message pane scrolled to the bottom."""
    st.html(
        f"""
        <script>
        (function() {{
            const nonce = {nonce};

            function scroll() {{
                const el = document.querySelector(
                    '.st-key-msg_scroll'
                );

                if (!el) return;

                el.scrollTop = el.scrollHeight;

                el.querySelectorAll('*').forEach(c => {{
                    if (c.scrollHeight > c.clientHeight) {{
                        c.scrollTop = c.scrollHeight;
                    }}
                }});
            }}

            scroll();
            setTimeout(scroll, 100);
            setTimeout(scroll, 400);
        }})();
        </script>
        """,
        unsafe_allow_javascript=True,
    )


def render_user(text: str) -> None:
    st.markdown(
        f'<div class="user-row">'
        f'<div class="user-bubble">{text}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_assistant(res: AgentResult) -> None:
    if res.error:
        st.error(res.error)
        return

    if res.tools_used:
        tools = "".join(
            f'<div class="trace-step">'
            f'<span class="trace-check">✓</span>'
            f'Used tool → <code>{t}</code>'
            f'</div>'
            for t in res.tools_used
        )

        st.markdown(
            '<div class="trace-box">'
            '<div class="trace-head">'
            '✨ <span class="title">Agent trace</span>'
            f'<span class="meta">· {len(res.tools_used)} tool(s)</span>'
            '</div>'
            f'<div class="trace-body">{tools}</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    if res.answer_text:
        st.markdown(res.answer_text)

    if getattr(res, "chart_spec", None):
        st.vega_lite_chart(
            res.chart_spec,
            width="stretch",
        )

    if res.result_rows and res.column_names:
        df = pd.DataFrame(
            res.result_rows,
            columns=res.column_names,
        )

        st.dataframe(
            df,
            width="stretch",
            hide_index=True,
        )


# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown(
        '<div class="brand-container">'
        '<div class="brand-logo">SC</div>'
        'Supply chain copilot'
        '</div>',
        unsafe_allow_html=True,
    )

    if st.button(
        "+ New chat",
        key="new_chat",
        width="stretch",
    ):
        st.session_state.chat_messages = []
        st.session_state.thread_id = None
        st.session_state.parent_message_id = 0
        st.session_state.active_view = "Chat"
        st.rerun()

    st.markdown(
        '<div class="nav-group">Navigation</div>',
        unsafe_allow_html=True,
    )

    view_icons = {
        "Dashboard": ":material/dashboard: Dashboard",
        "Chat": ":material/chat: Chat",
        "Ontology explorer": ":material/account_tree: Ontology explorer",
    }

    views = [
        "Dashboard",
        "Chat",
        "Ontology explorer",
    ]

    cur = (
        st.session_state.active_view
        if st.session_state.active_view in views
        else "Chat"
    )

    view = st.radio(
        "View",
        views,
        index=views.index(cur),
        format_func=lambda v: view_icons[v],
        label_visibility="collapsed",
        key="nav_radio",
    )

    if view != st.session_state.active_view:
        st.session_state.active_view = view
        st.rerun()

    if st.session_state.active_view != "Ontology explorer":
        st.markdown(
            '<div class="nav-group">Viewing as</div>',
            unsafe_allow_html=True,
        )

        persona_icons = {
            "Planning": ":material/bar_chart: Planning",
            "Procurement": ":material/shopping_cart: Procurement",
            "Logistics": ":material/local_shipping: Logistics",
        }

        personas = [
            "Planning",
            "Procurement",
            "Logistics",
        ]

        persona = st.radio(
            "Viewing as",
            personas,
            index=personas.index(
                st.session_state.active_persona
            ),
            format_func=lambda p: persona_icons[p],
            label_visibility="collapsed",
            key="persona_radio",
        )

        if (
            persona
            and persona != st.session_state.active_persona
        ):
            st.session_state.active_persona = persona
            st.rerun()

    st.markdown(
        '<div class="nav-group">Conversations</div>',
        unsafe_allow_html=True,
    )

    try:
        threads = list_threads()
    except Exception:
        threads = []

    if not threads:
        st.markdown(
            '<div class="hist-empty">'
            'No past conversations yet'
            '</div>',
            unsafe_allow_html=True,
        )

    for t in threads:
        tid = t.get("thread_id")
        name = (
            t.get("thread_name")
            or thread_title(tid)
        )

        is_active = (
            tid == st.session_state.thread_id
        )

        btn_key = (
            "hist_active"
            if is_active
            else f"hist_{tid}"
        )

        c_open, c_del = st.columns(
            [5, 1],
            gap="small",
            vertical_alignment="center",
        )

        with c_open:
            if st.button(
                name,
                key=btn_key,
                width="stretch",
            ):
                with st.spinner(
                    "Loading conversation…"
                ):
                    restore_thread(tid)

                st.session_state.active_view = "Chat"
                st.rerun()

        with c_del:
            if st.button(
                "🗑",
                key=f"del_{tid}",
                help="Delete conversation",
            ):
                try:
                    delete_thread(tid)
                except Exception:
                    pass

                if is_active:
                    st.session_state.chat_messages = []
                    st.session_state.thread_id = None
                    st.session_state.parent_message_id = 0

                st.rerun()


# ============================================================================
# PERSONA ACCENT THEMING
# ============================================================================
_pc = PERSONA_CONFIG.get(
    st.session_state.active_persona,
    PERSONA_CONFIG["Planning"],
)

_A = _pc["accent"]
_AD = _pc["accent_dark"]
_AS = _pc["accent_soft"]
_AH = _pc["accent_hover"]

st.markdown(
    f"""
    <style>
    .brand-logo {{
        background: {_A} !important;
    }}

    [data-testid="stSidebar"]
    [role="radiogroup"]
    label:has(input:checked) {{
        background: {_AS} !important;
    }}

    [data-testid="stSidebar"]
    [role="radiogroup"]
    label:has(input:checked) p {{
        color: {_A} !important;
    }}

    .st-key-new_chat button {{
        color: {_A} !important;
        border-color: {_A} !important;
    }}

    .st-key-new_chat button:hover {{
        background: {_AH} !important;
        color: {_AD} !important;
        border-color: {_AD} !important;
    }}

    [data-testid="stSidebar"]
    .st-key-hist_active button {{
        background: {_AS} !important;
        color: {_A} !important;
    }}

    .user-bubble {{
        background: {_A} !important;
    }}

    .st-key-composer_bar {{
        border: 1.5px solid {_A} !important;
        box-shadow: 0 2px 10px {_AS} !important;
    }}

    a,
    a:visited {{
        color: {_A} !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# DASHBOARD VIEW
# ============================================================================
if st.session_state.active_view == "Dashboard":
    persona = st.session_state.active_persona

    specs = PERSONA_DASHBOARD_QUERIES.get(
        persona,
        PERSONA_DASHBOARD_QUERIES["Planning"],
    )

    with st.spinner("Loading dashboard…"):
        kpis, kpi_errors = load_kpis()

    order = PERSONA_CONFIG.get(
        persona,
        PERSONA_CONFIG["Planning"],
    )["kpi_order"]

    ordered_ids = (
        [m for m in order if m in kpis]
        + [m for m in kpis if m not in order]
    )

    kpi_cols = st.columns(
        len(ordered_ids) or 1
    )

    for col, mid in zip(
        kpi_cols,
        ordered_ids,
    ):
        m = kpis[mid]

        with col:
            st.markdown(
                f'<div class="kpi-inner" '
                f'style="border-left-color:{m["color"]};">'
                f'<div class="label">{m["name"]}</div>'
                f'<div class="value">{m["value"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    if kpi_errors:
        with st.expander(
            "⚠ KPI load errors"
        ):
            for err in kpi_errors:
                st.write(err)

    st.markdown(
        '<div class="band-rule"></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="dash-heading">'
        f'📊 {persona} Reports &amp; Analytics'
        f'</div>',
        unsafe_allow_html=True,
    )

    for i in range(
        0,
        len(specs),
        2,
    ):
        left, right = st.columns(
            2,
            gap="medium",
        )

        with left:
            render_persona_chart(
                specs[i],
                idx=i,
            )

        if i + 1 < len(specs):
            with right:
                render_persona_chart(
                    specs[i + 1],
                    idx=i + 1,
                )


# ============================================================================
# CHAT VIEW
# ============================================================================
elif st.session_state.active_view == "Chat":
    main = st.container(
        key="main_row"
    )

    with main:
        col_chat = st.container()

        with col_chat:
            if not st.session_state.chat_messages:
                st.markdown(
                    '<div class="empty-greeting">'
                    '<span class="big">'
                    'How can I help with your supply chain?'
                    '</span>'
                    'Ask about suppliers, plants, orders, inventory…'
                    '</div>',
                    unsafe_allow_html=True,
                )

            else:
                with st.container(
                    height=640,
                    key="msg_scroll",
                ):
                    for msg in st.session_state.chat_messages:
                        if msg["role"] == "user":
                            render_user(
                                msg["content"]
                            )
                        else:
                            render_assistant(
                                msg["result"]
                            )

                    # Stream queued question.
                    pending = st.session_state.get(
                        "pending_question"
                    )

                    if pending:
                        st.session_state.pending_question = None

                        if st.session_state.thread_id is None:
                            try:
                                st.session_state.thread_id = (
                                    create_thread()
                                )
                                st.session_state.parent_message_id = 0
                                st.session_state.just_created_thread = True

                            except Exception:
                                st.session_state.thread_id = None

                        persona = (
                            st.session_state.active_persona
                        )

                        contextual = (
                            f"The user is viewing the supply chain "
                            f"as the {persona} persona. "
                            f"Frame the answer for that role.\n\n"
                            f"User question:\n{pending}"
                        )

                        gen = ask_agent_stream(
                            contextual,
                            thread_id=st.session_state.thread_id,
                            parent_message_id=st.session_state.parent_message_id,
                        )

                        st.write_stream(gen)

                        result = getattr(
                            ask_agent_stream,
                            "result",
                            AgentResult(),
                        )

                        finalize_answer(
                            pending,
                            result,
                        )

                        st.rerun()

                st.markdown(
                    '<div id="chat-scroll-anchor"></div>',
                    unsafe_allow_html=True,
                )

                _autoscroll_chat(
                    len(st.session_state.chat_messages)
                )


# ============================================================================
# ONTOLOGY VIEW
# ============================================================================
else:
    try:
        entities = load_ontology_entities()
        graph = load_ontology_graph()

    except Exception as e:
        entities, graph = [], []
        st.error(
            f"Could not load ontology metadata: {e}"
        )

    ENTITY_TYPE_COLOR = {
        "MASTER": "#378ADD",
        "REFERENCE": "#7F77DD",
        "FACT": "#D85A30",
        "LOGISTICS": "#1D9E75",
        "TRANSACTION": "#E0883B",
        "ASSET": "#C0508A",
        "BRIDGE": "#8a909b",
    }

    st.markdown(
        '<div class="dash-heading">'
        '🕸 Ontology explorer'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="g-text">'
        'Click any entity in the map to see its attributes, '
        'relationships, and metrics.'
        '</div>',
        unsafe_allow_html=True,
    )

    if graph:
        name_by_id = {
            e["ENTITY_ID"]: e["DISPLAY_NAME"]
            for e in entities
        }

        type_by_id = {
            e["ENTITY_ID"]: e["ENTITY_TYPE"]
            for e in entities
        }

        node_ids = set()

        for edge in graph:
            node_ids.add(
                edge["SOURCE_ID"]
            )
            node_ids.add(
                edge["TARGET_ID"]
            )

        flow_nodes = []

        for nid in node_ids:
            col = ENTITY_TYPE_COLOR.get(
                type_by_id.get(nid, ""),
                "#8A909B",
            )

            flow_nodes.append(
                StreamlitFlowNode(
                    id=nid,
                    pos=(0, 0),
                    data={
                        "content": name_by_id.get(
                            nid,
                            nid,
                        )
                    },
                    node_type="default",
                    source_position="right",
                    target_position="left",
                    selectable=True,
                    style={
                        "background": "#ffffff",
                        "border": f"2px solid {col}",
                        "borderRadius": "8px",
                        "fontSize": "12px",
                        "color": "#1c1e21",
                        "padding": "6px 10px",
                    },
                )
            )

        flow_edges = [
            StreamlitFlowEdge(
                id=(
                    f"{edge['SOURCE_ID']}-"
                    f"{edge['TARGET_ID']}-"
                    f"{i}"
                ),
                source=edge["SOURCE_ID"],
                target=edge["TARGET_ID"],
                label=edge.get(
                    "EDGE_LABEL",
                    "",
                ),
                animated=False,
                edge_type="smoothstep",
            )
            for i, edge in enumerate(graph)
        ]

        if "onto_flow_state" not in st.session_state:
            st.session_state.onto_flow_state = (
                StreamlitFlowState(
                    flow_nodes,
                    flow_edges,
                )
            )

        stage = st.container(
            key="onto_stage"
        )

        with stage:
            new_state = streamlit_flow(
                "onto_flow",
                st.session_state.onto_flow_state,
                layout=LayeredLayout(
                    direction="right",
                    node_node_spacing=40,
                    node_layer_spacing=110,
                ),
                fit_view=True,
                height=600,
                get_node_on_click=True,
                show_minimap=True,
                show_controls=True,
                hide_watermark=True,
            )

            eid = st.session_state.active_entity

            ent = (
                next(
                    (
                        e
                        for e in entities
                        if e["ENTITY_ID"] == eid
                    ),
                    None,
                )
                if eid
                else None
            )

            with st.container(
                key="onto_detail_card"
            ):
                if not ent:
                    st.markdown(
                        '<div class="onto-group-title">'
                        'Entity detail'
                        '</div>'
                        '<div class="g-text">'
                        'Click a node to see its attributes, '
                        'relationships, and metrics.'
                        '</div>',
                        unsafe_allow_html=True,
                    )

                else:
                    parts = [
                        f'<div class="onto-group-title">'
                        f'{ent["ENTITY_TYPE"].title()}'
                        f'</div>',

                        f'<div class="g-value">'
                        f'{ent["DISPLAY_NAME"]}'
                        f'</div>',

                        f'<div class="g-text">'
                        f'{ent.get("DESCRIPTION", "")}'
                        f'</div>',

                        '<div class="g-label">Source</div>',

                        f'<div class="g-text">'
                        f'{ent.get("SOURCE_OBJECT", "")}'
                        f'</div>',
                    ]

                    attrs = load_ontology_attributes(
                        eid
                    )

                    if attrs:
                        parts.append(
                            '<div class="g-label">'
                            'Attributes'
                            '</div>'
                        )

                        parts += [
                            f'<div class="onto-item">'
                            f'<span>'
                            f'{a["DISPLAY_NAME"] or a["ATTRIBUTE_NAME"]}'
                            f'{" 🔑" if str(a.get("IS_KEY")).lower() == "true" else ""}'
                            f'</span>'
                            f'<span class="count">'
                            f'{a["DATA_TYPE"]}'
                            f'</span>'
                            f'</div>'
                            for a in attrs
                        ]

                    rels = [
                        g
                        for g in graph
                        if g["SOURCE_ID"] == eid
                        or g["TARGET_ID"] == eid
                    ]

                    if rels:
                        parts.append(
                            '<div class="g-label">'
                            'Relationships'
                            '</div>'
                        )

                        parts += [
                            f'<div class="onto-item">'
                            f'<span>'
                            f'{r["SOURCE_NAME"]} '
                            f'{r["EDGE_LABEL"]} '
                            f'{r["TARGET_NAME"]}'
                            f'</span>'
                            f'<span class="count">'
                            f'{r.get("CARDINALITY", "")}'
                            f'</span>'
                            f'</div>'
                            for r in rels
                        ]

                    ems = load_entity_metrics(
                        eid
                    )

                    if ems:
                        parts.append(
                            '<div class="g-label">'
                            'Metrics'
                            '</div>'
                        )

                        parts += [
                            f'<div class="onto-item">'
                            f'<span>'
                            f'{m["METRIC_NAME"]}'
                            f'</span>'
                            f'<span class="count">'
                            f'{m.get("ROLE", "")}'
                            f'</span>'
                            f'</div>'
                            for m in ems
                        ]

                    st.markdown(
                        "".join(parts),
                        unsafe_allow_html=True,
                    )

        if (
            new_state
            and new_state.selected_id
            and new_state.selected_id
            != st.session_state.active_entity
        ):
            st.session_state.active_entity = (
                new_state.selected_id
            )
            st.rerun()

    else:
        st.info(
            "No relationship data available."
        )

    st.markdown(
        '<div class="band-rule"></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="onto-group-title">'
        'Governed metric catalog'
        '</div>',
        unsafe_allow_html=True,
    )

    _metric_colors = [
        "#378ADD",
        "#E0B84B",
        "#1D9E75",
        "#D85A30",
        "#7F77DD",
    ]

    try:
        catalog = load_metric_catalog()
    except Exception:
        catalog = []

    cat_cols = st.columns(
        len(catalog) or 1
    )

    for col, m, color in zip(
        cat_cols,
        catalog,
        _metric_colors * 3,
    ):
        with col:
            st.markdown(
                f'<div class="metric-card" '
                f'style="border-left:3px solid {color};">'
                f'<div class="m-name">'
                f'{m["DISPLAY_NAME"]}'
                f'</div>'
                f'<div class="m-def">'
                f'{m.get("DEFINITION", "")}'
                f'</div>'
                f'<div class="m-view">'
                f'{m.get("CANONICAL_OBJECT", "")}'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ============================================================================
# TEXT-ONLY CHAT COMPOSER
# ============================================================================
#
# Voice/audio/transcription has intentionally been removed.
#
# The composer now follows:
#
#   User types question
#          ↓
#   Enter OR Send
#          ↓
#   submit_question()
#          ↓
#   Cortex Agent streaming
#
# No st.audio_input()
# No transcribe_audio()
# No audio file state
# No transcription dependency
# ============================================================================

if st.session_state.active_view == "Chat":

    if "composer_input" not in st.session_state:
        st.session_state.composer_input = ""

    if "pending_clear" not in st.session_state:
        st.session_state.pending_clear = False

    # Clear the input before the widget is created.
    if st.session_state.pending_clear:
        st.session_state.composer_input = ""
        st.session_state.pending_clear = False

    composer_dock = st.container(
        key="composer_dock"
    )

    with composer_dock, st.container(
        key="composer_bar"
    ):
        text_col, send_col = st.columns(
            [8.5, 1.2],
            vertical_alignment="center",
        )

        with text_col:
            st.text_input(
                "Ask",
                placeholder=(
                    "Ask about suppliers, plants, "
                    "orders, inventory…"
                ),
                label_visibility="collapsed",
                key="composer_input",
                on_change=lambda: st.session_state.update(
                    enter_pressed=True
                ),
            )

        with send_col:
            send = st.button(
                "Send",
                type="primary",
                width="stretch",
            )

    submitted = (
        send
        or st.session_state.pop(
            "enter_pressed",
            False,
        )
    )

    if (
        submitted
        and st.session_state.composer_input.strip()
    ):
        q = st.session_state.composer_input.strip()

        # Clear the input on the next Streamlit run.
        st.session_state.pending_clear = True

        submit_question(q)

        st.rerun()