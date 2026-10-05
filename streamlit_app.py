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
    ask_agent, ask_agent_stream, AgentResult, run_sql,
    create_thread, list_threads, get_thread_messages, set_thread_name, delete_thread,
    transcribe_audio, _extract_from_response_payload,
)

st.set_page_config(
    page_title="Supply Chain Copilot",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded",
)


@alt.theme.register("sc_dark", enable=True)
def _sc_dark_theme() -> alt.theme.ThemeConfig:
    try:
        persona = st.session_state.get("active_persona", "Planning")
        surf = PERSONA_CONFIG[persona]
        bg, grid = surf["card"], surf["border"]
    except (AttributeError, KeyError):
        bg, grid = "#132f4c", "#2a4a6b"
    text, muted = "#e3f0ff", "#8ba3c0"
    return alt.theme.ThemeConfig({
        "config": {
            "background": bg,
            "view": {"stroke": "transparent"},
            "title": {"color": text, "subtitleColor": muted},
            "axis": {
                "domainColor": grid, "tickColor": grid, "gridColor": grid,
                "labelColor": muted, "titleColor": text,
            },
            "legend": {"labelColor": muted, "titleColor": text},
        }
    })

# STYLING NOTES (read before editing):
# - Hybrid layout: a dashboard shell (top bar + KPI strip + 3 columns) wrapping
#   a Claude-style chat in the center column. History = left column, grounding =
#   right column (both in-card, NOT st.sidebar).
# - DARK THEME (Snowflake-inspired deep navy). Palette anchors — keep in sync with
#   .streamlit/config.toml:
#     bg #0a1929 | card #132f4c | elevated #1a3a5c | border #2a4a6b
#     text #e3f0ff | muted #8ba3c0 | subtle #5f7a99
#     persona blue #4d9fff | green #3dd598 | red #ff6b5a
# - !important is required throughout: Streamlit-in-Snowflake injects a theme
#   with higher specificity than plain rules; without it our colors lose.
# - Streamlit internals are targeted by [data-testid=...] and .st-key-{key};
#   these can change across Streamlit versions, so this block is the fragile part.
#   .st-key-{key} = the reliable per-widget selector (from a widget's key= arg).
st.markdown("""
<style>
    #MainMenu {
        visibility: hidden !important;
    }

    footer {
        visibility: hidden !important;
    }

    /* Hide Streamlit's top header / toolbar */
    header[data-testid="stHeader"],
    [data-testid="stToolbar"] {
        display: none !important;
    }

    /* Remove the space reserved for the hidden header */
    [data-testid="stAppViewContainer"] {
        padding-top: 0 !important;
        gap: 0 !important;
        background: #0a1929 !important;
    }

    .stAppViewContainer > .main {
        padding-top: 0 !important;
    }

    /* Global app theme */
    .stApp,
    body {
        background-color: #0a1929 !important;
        color: #e3f0ff !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    }

    [data-testid="stSidebarResizeHandle"] {
        display: none !important;
    }

    [data-testid="stSidebar"] {
        margin-right: 0 !important;
        background: #061320 !important;
        border-right: 1px solid #2a4a6b !important;
    }

    [data-testid="stMain"] {
        padding-left: 0 !important;
        margin-left: 0 !important;
        background: #0a1929 !important;
    }

    [data-testid="stMainBlockContainer"],
    .block-container {
        max-width: 100% !important;
        padding-top: 1.25rem !important;
        padding-bottom: 2rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }

    /* Navigation */
    .nav-group {
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #5f7a99;
        margin: 18px 0 4px 2px;
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
        background: #1a3a5c !important;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
        background: #1a3a5c !important;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p {
        color: #4d9fff !important;
        font-weight: 600 !important;
    }

    [data-testid="stSidebar"] [role="radiogroup"] p {
        font-size: 14px !important;
        color: #b8ccea !important;
    }

    [data-testid="stSidebar"] .brand-container {
        margin-bottom: 16px;
    }

    /* New chat button */
    .st-key-new_chat button {
        background: #132f4c !important;
        color: #4d9fff !important;
        font-weight: 600 !important;
        border: 1.5px solid #4d9fff !important;
        border-radius: 8px !important;
        font-size: 13px !important;
    }

    .st-key-new_chat button:hover {
        background: #1a3a5c !important;
        border-color: #79b8ff !important;
    }

    .hist-empty {
        font-size: 12px;
        color: #5f7a99;
        padding: 8px 4px;
        font-style: italic;
    }

    [data-testid="stSidebar"] .st-key-hist_active button {
        background: #1a3a5c !important;
        color: #4d9fff !important;
    }

    /* Brand */
    .brand-container {
        display: flex;
        align-items: center;
        gap: 11px;
        font-weight: 700;
        font-size: 19px;
        color: #e3f0ff !important;
    }

    .brand-logo {
        width: 34px;
        height: 34px;
        border-radius: 9px;
        background: #4d9fff !important;
        color: #0a1929 !important;
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
        border: 0.5px solid #2a4a6b !important;
        background: #132f4c !important;
        color: #8ba3c0 !important;
        font-size: 13px !important;
        font-weight: 500 !important;
    }

    [data-testid="stPills"] button[aria-selected="true"],
    [data-testid="stPills"] button[kind="pillsActive"] {
        background: #1a3a5c !important;
        color: #4d9fff !important;
        font-weight: 600 !important;
        border-color: transparent !important;
    }

    /* KPI cards */
    .kpi-inner {
        background: #132f4c;
        border: 1px solid #2a4a6b;
        border-radius: 10px;
        border-left: 3px solid #4d9fff;
        padding: 12px 16px;
    }

    .kpi-inner .label {
        font-size: 12px;
        color: #8ba3c0 !important;
    }

    .kpi-inner .value {
        font-size: 24px;
        font-weight: 600;
        color: #e3f0ff !important;
        margin-top: 2px;
    }

    /* Dashboard divider */
    .band-rule {
        height: 1px;
        background: #2a4a6b;
        margin: 20px 0 18px;
    }

    /* Dashboard heading */
    .dash-heading {
        font-size: 28px;
        font-weight: 700;
        color: #e3f0ff !important;
        margin: 4px 0 20px;
        letter-spacing: -0.01em;
    }

    /* Dashboard chart cards */
    [class*="st-key-chartcard_"] {
        border: 1px solid #2a4a6b !important;
        border-radius: 12px !important;
        background: #132f4c !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.3) !important;
        padding: 0 0 14px !important;
        overflow: hidden !important;
        margin-bottom: 6px !important;
    }

    .chart-card-title {
        background: #1a3a5c;
        border-bottom: 1px solid #2a4a6b;
        margin: 0 0 14px;
        padding: 12px 16px;
        font-size: 15px;
        font-weight: 700;
        color: #e3f0ff !important;
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
        border-left: 1px solid #2a4a6b !important;
        padding-left: 24px !important;
    }

    /* Left rail panel */
    .st-key-main_row > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child:nth-last-child(2) {
        background-color: #132f4c !important;
        border-right: 1px solid #2a4a6b !important;
        border-radius: 12px !important;
        padding: 16px !important;
    }

    .st-key-main_row > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child:nth-last-child(2) + [data-testid="stColumn"] {
        border-left: none !important;
        padding-left: 16px !important;
    }

    /* History */
    .col-label {
        font-size: 11px;
        color: #8ba3c0 !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        padding-left: 9px;
        margin: 20px 0 8px;
        font-weight: 600;
    }

    /* General buttons */
    [data-testid="stButton"] button {
        border-radius: 8px !important;
        border: 0.5px solid transparent !important;
        background: transparent !important;
        color: #b8ccea !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        text-align: left !important;
        justify-content: flex-start !important;
        padding: 7px 9px !important;
        min-height: 0 !important;
    }

    [data-testid="stButton"] button:hover {
        background: #1a3a5c !important;
        color: #e3f0ff !important;
    }

    /* Sidebar history rows */
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

    /* Delete buttons */
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
        background: #3d1a16 !important;
        color: #ff6b5a !important;
    }

    /* Chat */
    .user-row {
        display: flex;
        justify-content: flex-end;
        margin: 8px 0 16px;
    }

    .user-bubble {
        background: #4d9fff !important;
        color: #0a1929 !important;
        padding: 11px 15px;
        border-radius: 16px 16px 4px 16px;
        font-size: 14px;
        line-height: 1.5;
        max-width: 85%;
    }

    .assistant-answer-value {
        font-size: 32px;
        font-weight: 600;
        color: #e3f0ff !important;
        letter-spacing: -0.01em;
    }

    .assistant-delta {
        font-size: 12px;
        color: #3dd598 !important;
        background: #153a2a !important;
        padding: 2px 9px;
        border-radius: 20px;
        font-weight: 500;
        margin-left: 8px;
    }

    .assistant-sub {
        font-size: 13px;
        color: #8ba3c0 !important;
        margin-top: 5px;
    }

    .assistant-prose {
        font-size: 14px;
        color: #e3f0ff !important;
        line-height: 1.6;
        margin-top: 12px;
    }

    .empty-greeting {
        text-align: center;
        color: #8ba3c0;
        font-size: 14px;
        margin: 22vh 0 30px;
    }

    .empty-greeting .big {
        display: block;
        font-size: 24px;
        font-weight: 600;
        color: #e3f0ff;
        margin-bottom: 6px;
    }

    /* Trace box */
    .trace-box {
        border: 0.5px solid #2a4a6b !important;
        border-radius: 12px;
        background: #132f4c !important;
        overflow: hidden;
        margin: 4px 0 14px;
    }

    .trace-head {
        display: flex;
        align-items: center;
        gap: 7px;
        padding: 9px 13px;
        background: #1a3a5c !important;
        border-bottom: 0.5px solid #2a4a6b !important;
        font-size: 12px;
    }

    .trace-head .title {
        font-weight: 600;
        color: #e3f0ff !important;
    }

    .trace-head .meta {
        color: #8ba3c0 !important;
    }

    .trace-body {
        padding: 10px 14px;
        display: flex;
        flex-direction: column;
        gap: 8px;
    }

    .trace-step {
        font-size: 13px;
        color: #b8ccea !important;
    }

    .trace-step code {
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace !important;
        color: #e3f0ff !important;
        background: #1a3a5c !important;
        padding: 1px 5px;
        border-radius: 4px;
        font-size: 12px;
    }

    .trace-check {
        color: #3dd598 !important;
        font-weight: bold;
        margin-right: 6px;
    }

    /* Answer card */
    .answer-card {
        border: 0.5px solid #2a4a6b !important;
        border-radius: 14px;
        padding: 16px 18px;
        background: #132f4c !important;
        margin-bottom: 12px;
    }

    /* Grounding panel */
    .g-head {
        font-size: 13px;
        font-weight: 600;
        color: #3dd598 !important;
        margin-bottom: 12px;
        padding-bottom: 10px;
        border-bottom: 0.5px solid #2a4a6b !important;
    }

    .g-label {
        font-size: 11px;
        color: #8ba3c0 !important;
        margin-top: 12px;
    }

    .g-value {
        font-size: 13px;
        font-weight: 600;
        color: #e3f0ff !important;
    }

    .g-text {
        font-size: 12px;
        color: #b8ccea !important;
        line-height: 1.5;
    }

    .g-sql {
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace !important;
        font-size: 11px;
        background: #0a1929 !important;
        border: 0.5px solid #2a4a6b !important;
        border-radius: 8px;
        padding: 10px 12px;
        color: #b8ccea !important;
        line-height: 1.7;
        margin-top: 5px;
    }

    .g-divider {
        height: 0.5px;
        background: #2a4a6b !important;
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
        background: #132f4c !important;
        border: 1px solid #2a4a6b !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 16px rgba(0,0,0,0.5) !important;
        padding: 14px 16px !important;
    }

    .g-row {
        display: flex;
        justify-content: space-between;
        font-size: 12px;
        padding: 5px 0;
    }

    .g-row .name {
        color: #8ba3c0 !important;
    }

    .g-row .val {
        font-weight: 600;
        color: #e3f0ff !important;
    }

    /* Ontology */
    .onto-group-title {
        font-size: 11px;
        color: #8ba3c0 !important;
        margin: 14px 0 8px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .onto-item {
        display: flex;
        justify-content: space-between;
        font-size: 13px;
        color: #b8ccea !important;
        padding: 4px 0;
    }

    .onto-item .count {
        color: #8ba3c0 !important;
        font-size: 12px;
    }

    .metric-card {
        border: 0.5px solid #2a4a6b !important;
        border-radius: 10px;
        padding: 10px 12px;
        margin-bottom: 10px;
        background: #132f4c !important;
    }

    .metric-card .m-name {
        font-size: 13px;
        font-weight: 600;
        color: #e3f0ff !important;
    }

    .metric-card .m-def {
        font-size: 12px;
        color: #b8ccea !important;
        margin-top: 2px;
        line-height: 1.4;
    }

    .metric-card .m-view {
        font-size: 11px;
        color: #8ba3c0 !important;
        margin-top: 5px;
        font-family: ui-monospace, Menlo, Consolas, monospace !important;
    }

    /* Scrollable message area */
    .st-key-msg_scroll {
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
        padding: 0 8px 90px !important;
        max-width: 820px !important;
        margin: 0 auto !important;
    }

    /* Chat composer */
    [data-testid="stChatInput"] {
        border-radius: 14px !important;
        border: 1px solid #2a4a6b !important;
        background: #132f4c !important;
    }

    [data-testid="stChatInput"] textarea {
        font-size: 14px !important;
        background: #132f4c !important;
        color: #e3f0ff !important;
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
        background: #0a1929 !important;
    }

    /* Fixed sidebar */
    [data-testid="stSidebar"] {
        width: 300px !important;
        min-width: 300px !important;
    }

    [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarCollapse"],
    [data-testid="collapsedControl"] {
        display: none !important;
    }

    /* Fixed composer */
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
            #0a1929 62%,
            rgba(10,25,41,0)
        ) !important;
        padding: 10px 2rem 16px !important;
    }

    /* Composer bar */
    .st-key-composer_bar {
        border: 1px solid #2a4a6b !important;
        border-radius: 20px !important;
        background: #132f4c !important;
        padding: 0 10px !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.4) !important;
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
        background: transparent !important;
        color: #e3f0ff !important;
    }

    .st-key-composer_bar [data-testid="stTextInput"] input::placeholder {
        color: #5f7a99 !important;
    }

    /* Audio recorder */
    .st-key-composer_bar [data-testid="stAudioInput"] {
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
        padding: 0 !important;
        min-height: 0 !important;
    }

    .st-key-composer_bar [data-testid="stAudioInput"] > div {
        min-height: 0 !important;
        padding: 0 !important;
        background: transparent !important;
    }

    /* Hide waveform / timecode */
    .st-key-composer_bar [data-testid="stAudioInputWaveSurfer"],
    .st-key-composer_bar [data-testid="stAudioInputWaveformTimeCode"],
    .st-key-composer_bar [data-testid="stAudioInput"] time,
    .st-key-composer_bar [data-testid="stAudioInput"] [class*="waveform" i],
    .st-key-composer_bar [data-testid="stAudioInput"] canvas {
        display: none !important;
    }

    .st-key-composer_bar [data-testid="stAudioInput"] {
        max-width: 44px !important;
        overflow: hidden !important;
    }

    .st-key-composer_bar [data-testid="stButton"] button {
        border-radius: 18px !important;
        height: 32px !important;
        min-height: 32px !important;
        margin: 0 !important;
        padding: 0 14px !important;
    }

    /* Dataframes */
    [data-testid="stDataFrame"] {
        background: #132f4c !important;
        border-radius: 8px !important;
    }

    /* Alerts */
    [data-testid="stAlert"] {
        background: #132f4c !important;
        border: 1px solid #2a4a6b !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Session State ---
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


# KPIs are read from the GOVERNED snapshot view, not computed in the UI. This
# makes CORE the single source of truth (the "UI never calculates a metric" rule
# holds fully) — the app only reads and formats V_DASHBOARD_KPI_SNAPSHOT.
KPI_SNAPSHOT_QUERY = (
    "SELECT METRIC_ID, METRIC_NAME, METRIC_VALUE, UNIT "
    "FROM SUPPLY_CHAIN_ONTOLOGY.CORE.V_DASHBOARD_KPI_SNAPSHOT"
)
KPI_COLOR = {
    "outbound_otd": "#5fb3e8", "inbound_otd": "#e8c76b", "fill_rate": "#3dd598",
    "dii": "#f08a4b", "landed_cost": "#9d93e8",
}

# Persona shapes emphasis, not values: same governed metrics, reordered so the
# lens each role cares about leads. It does NOT change any metric's value.
# accent_soft/accent_hover are dark-mode tinted surfaces (not light tints).
PERSONA_CONFIG = {
    "Planning": {
        "kpi_order": ["outbound_otd", "fill_rate", "dii", "landed_cost", "inbound_otd"],
        "accent": "#4d9fff", "accent_dark": "#79b8ff", "accent_soft": "#1a2a3c", "accent_hover": "#213449",
        "bg": "#0d1620", "card": "#17222f", "elevated": "#1f2d3d", "border": "#2a3a4d", "sidebar": "#0a111a",
    },
    "Procurement": {
        "kpi_order": ["inbound_otd", "landed_cost", "dii", "fill_rate", "outbound_otd"],
        "accent": "#3dd598", "accent_dark": "#6ee0b0", "accent_soft": "#1a2622", "accent_hover": "#20302b",
        "bg": "#121816", "card": "#1c231f", "elevated": "#242d28", "border": "#2f3a34", "sidebar": "#0d1211",
    },
    "Logistics": {
        "kpi_order": ["outbound_otd", "inbound_otd", "fill_rate", "dii", "landed_cost"],
        "accent": "#ff6b5a", "accent_dark": "#ff9080", "accent_soft": "#2a1a1d", "accent_hover": "#352024",
        "bg": "#1a1315", "card": "#241b1e", "elevated": "#2f2328", "border": "#3f2d33", "sidebar": "#120d0e",
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
    """Read the governed KPI snapshot. Returns ({metric_id: {name,value,color}}, errors)."""
    errors = []
    metrics: dict[str, dict] = {}
    try:
        cols, rows = run_sql(KPI_SNAPSHOT_QUERY)
        idx = {c: i for i, c in enumerate(cols)}
        for r in rows:
            mid = r[idx["METRIC_ID"]]
            metrics[mid] = {
                "name": r[idx["METRIC_NAME"]],
                "value": _fmt_kpi(r[idx["METRIC_VALUE"]], r[idx["UNIT"]]),
                "color": KPI_COLOR.get(mid, "#8a909b"),
            }
    except Exception as e:
        errors.append(f"KPI snapshot: {type(e).__name__}: {e}")
    return metrics, errors


# Persona dashboards. Each chart is a (title, sql, render-kind, x, y, color) spec.
# All SQL reads the governed CORE views (fully qualified so it resolves in SiS
# regardless of session context). render kinds: line | bar | bar_v | components.
_O = "SUPPLY_CHAIN_ONTOLOGY.CORE.VW_ON_TIME_DELIVERY_OUTBOUND"
_I = "SUPPLY_CHAIN_ONTOLOGY.CORE.VW_ON_TIME_DELIVERY_INBOUND"
_F = "SUPPLY_CHAIN_ONTOLOGY.CORE.VW_FILL_RATE"
_D = "SUPPLY_CHAIN_ONTOLOGY.CORE.VW_DAYS_OF_INVENTORY"
_L = "SUPPLY_CHAIN_ONTOLOGY.CORE.VW_LANDED_COST"
_SUP = "SUPPLY_CHAIN_ONTOLOGY.RAW.DIM_SUPPLIER"
_LANDED_NUM = "(UNIT_PRICE_USD*QUANTITY_SHIPPED+FREIGHT_COST_USD+INSURANCE_COST_USD+CUSTOMS_COST_USD)"
_LANDED_NUM_L = "(l.UNIT_PRICE_USD*l.QUANTITY_SHIPPED+l.FREIGHT_COST_USD+l.INSURANCE_COST_USD+l.CUSTOMS_COST_USD)"

PERSONA_DASHBOARD_QUERIES = {
    "Planning": [
        ("Outbound OTD trend", f"SELECT DELIVERY_MONTH AS MONTH, ROUND(AVG(IS_ON_TIME)*100,2) AS OTD_PCT FROM {_O} GROUP BY DELIVERY_MONTH ORDER BY DELIVERY_MONTH", "line", "MONTH", "OTD_PCT", "#4d9fff"),
        ("Fill rate trend", f"SELECT ORDER_MONTH AS MONTH, ROUND(SUM(QUANTITY_SHIPPED)/NULLIF(SUM(QUANTITY_ORDERED),0)*100,2) AS FILL_PCT FROM {_F} GROUP BY ORDER_MONTH ORDER BY ORDER_MONTH", "line", "MONTH", "FILL_PCT", "#3dd598"),
        ("Days of inventory by plant", f"SELECT PLANT_ID, ROUND(SUM(ON_HAND_QTY)/NULLIF(SUM(AVG_DAILY_USAGE),0),1) AS DII FROM {_D} GROUP BY PLANT_ID ORDER BY DII DESC", "bar", "PLANT_ID", "DII", "#f08a4b"),
        ("Days of inventory by part category", f"SELECT PART_CATEGORY_ID, ROUND(SUM(ON_HAND_QTY)/NULLIF(SUM(AVG_DAILY_USAGE),0),1) AS DII FROM {_D} GROUP BY PART_CATEGORY_ID ORDER BY DII DESC", "bar", "PART_CATEGORY_ID", "DII", "#9d93e8"),
        ("Inventory on hand by plant", f"SELECT PLANT_ID, SUM(ON_HAND_QTY) AS ON_HAND FROM {_D} GROUP BY PLANT_ID ORDER BY ON_HAND DESC", "bar", "PLANT_ID", "ON_HAND", "#5fb3e8"),
    ],
    "Procurement": [
        ("Inbound OTD by supplier (worst 12)", f"SELECT s.SUPPLIER_NAME AS SUPPLIER, ROUND(AVG(i.IS_ON_TIME)*100,2) AS OTD_PCT FROM {_I} i LEFT JOIN {_SUP} s ON i.SUPPLIER_ID=s.SUPPLIER_ID GROUP BY s.SUPPLIER_NAME ORDER BY OTD_PCT ASC LIMIT 12", "bar", "SUPPLIER", "OTD_PCT", "#e8c76b"),
        ("Landed cost by supplier (top 12)", f"SELECT s.SUPPLIER_NAME AS SUPPLIER, ROUND(SUM({_LANDED_NUM_L})/NULLIF(SUM(l.QUANTITY_SHIPPED),0),2) AS LANDED FROM {_L} l LEFT JOIN {_SUP} s ON l.SUPPLIER_ID=s.SUPPLIER_ID GROUP BY s.SUPPLIER_NAME ORDER BY LANDED DESC LIMIT 12", "bar", "SUPPLIER", "LANDED", "#9d93e8"),
        ("Landed cost by part (top 12)", f"SELECT PART_ID, ROUND(SUM({_LANDED_NUM})/NULLIF(SUM(QUANTITY_SHIPPED),0),2) AS LANDED FROM {_L} GROUP BY PART_ID ORDER BY LANDED DESC LIMIT 12", "bar", "PART_ID", "LANDED", "#f08a4b"),
        ("Landed cost breakdown", f"SELECT SUM(UNIT_PRICE_USD*QUANTITY_SHIPPED) AS PRODUCT, SUM(FREIGHT_COST_USD) AS FREIGHT, SUM(INSURANCE_COST_USD) AS INSURANCE, SUM(CUSTOMS_COST_USD) AS CUSTOMS FROM {_L}", "components", "", "", "#4d9fff"),
        ("Supplier reliability vs inbound OTD", f"SELECT s.SUPPLIER_RELIABILITY_SCORE AS RELIABILITY, ROUND(AVG(i.IS_ON_TIME)*100,2) AS OTD_PCT FROM {_I} i JOIN {_SUP} s ON i.SUPPLIER_ID=s.SUPPLIER_ID GROUP BY s.SUPPLIER_RELIABILITY_SCORE HAVING COUNT(*)>0 ORDER BY RELIABILITY", "scatter", "RELIABILITY", "OTD_PCT", "#3dd598"),
    ],
    "Logistics": [
        ("Outbound OTD trend", f"SELECT DELIVERY_MONTH AS MONTH, ROUND(AVG(IS_ON_TIME)*100,2) AS OTD_PCT FROM {_O} GROUP BY DELIVERY_MONTH ORDER BY DELIVERY_MONTH", "line", "MONTH", "OTD_PCT", "#4d9fff"),
        ("OTD by plant", f"SELECT PLANT_ID, ROUND(AVG(IS_ON_TIME)*100,2) AS OTD_PCT FROM {_O} GROUP BY PLANT_ID ORDER BY OTD_PCT ASC", "bar", "PLANT_ID", "OTD_PCT", "#5fb3e8"),
        ("OTD by carrier (worst 12)", f"SELECT CARRIER, ROUND(AVG(IS_ON_TIME)*100,2) AS OTD_PCT FROM {_O} GROUP BY CARRIER ORDER BY OTD_PCT ASC LIMIT 12", "bar", "CARRIER", "OTD_PCT", "#e8c76b"),
        ("OTD by mode", f"SELECT MODE, ROUND(AVG(IS_ON_TIME)*100,2) AS OTD_PCT FROM {_O} GROUP BY MODE ORDER BY OTD_PCT ASC", "bar", "MODE", "OTD_PCT", "#3dd598"),
        ("Late shipments by carrier (top 12)", f"SELECT CARRIER, SUM(CASE WHEN IS_ON_TIME=0 THEN 1 ELSE 0 END) AS LATE FROM {_O} GROUP BY CARRIER ORDER BY LATE DESC LIMIT 12", "bar", "CARRIER", "LATE", "#f08a4b"),
        ("Delivery delay distribution", f"SELECT CASE WHEN DELIVERY_DATE<=PROMISED_DELIVERY_DATE THEN 'On time' WHEN DATEDIFF('day',PROMISED_DELIVERY_DATE,DELIVERY_DATE)=1 THEN '1 day late' WHEN DATEDIFF('day',PROMISED_DELIVERY_DATE,DELIVERY_DATE) BETWEEN 2 AND 3 THEN '2-3 days late' WHEN DATEDIFF('day',PROMISED_DELIVERY_DATE,DELIVERY_DATE) BETWEEN 4 AND 7 THEN '4-7 days late' ELSE '8+ days late' END AS BUCKET, COUNT(*) AS SHIPMENTS FROM {_O} GROUP BY 1", "bar_v", "BUCKET", "SHIPMENTS", "#4d9fff"),
    ],
}

_NON_NUMERIC = {"MONTH", "PLANT_ID", "SUPPLIER", "PART_ID", "PART_CATEGORY_ID", "CARRIER", "MODE", "BUCKET"}


@st.cache_data(ttl=300, show_spinner=False)
def load_df(sql: str) -> pd.DataFrame:
    try:
        cols, rows = run_sql(sql)
        df = pd.DataFrame(rows, columns=cols)
        for c in df.columns:
            if c not in _NON_NUMERIC:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df
    except Exception:
        return pd.DataFrame()


def line_chart(df: pd.DataFrame, x: str, y: str, color: str, y_title: str):
    lo = max(0, float(df[y].min()) - 2)
    hi = float(df[y].max()) + 2
    return (
        alt.Chart(df).mark_line(point=True, color=color, strokeWidth=2.5)
        .encode(
            x=alt.X(f"{x}:N", title=None, axis=alt.Axis(labelAngle=-45)),
            y=alt.Y(f"{y}:Q", scale=alt.Scale(domain=[lo, hi]), title=y_title),
            tooltip=[x, y],
        )
        .properties(height=240)
    )


def bar_chart(df: pd.DataFrame, x: str, y: str, color: str, y_title: str, horizontal: bool = True):
    if horizontal:
        enc = dict(
            y=alt.Y(f"{x}:N", sort="-x", title=None),
            x=alt.X(f"{y}:Q", title=y_title),
        )
    else:
        enc = dict(
            x=alt.X(f"{x}:N", sort="-y", title=None),
            y=alt.Y(f"{y}:Q", title=y_title),
        )
    return (
        alt.Chart(df).mark_bar(color=color, cornerRadius=3)
        .encode(tooltip=[x, y], **enc)
        .properties(height=260)
    )


def scatter_chart(df: pd.DataFrame, x: str, y: str, color: str, y_title: str):
    return (
        alt.Chart(df).mark_circle(size=90, color=color, opacity=0.7)
        .encode(
            x=alt.X(f"{x}:Q", title=x.replace("_", " ").title()),
            y=alt.Y(f"{y}:Q", title=y_title),
            tooltip=[x, y],
        )
        .properties(height=260)
    )


def render_persona_chart(spec, idx: int = 0) -> None:
    title, sql, kind, x, y, color = spec
    with st.container(key=f"chartcard_{idx}", border=True):
        st.markdown(f'<div class="chart-card-title">{title}</div>', unsafe_allow_html=True)
        df = load_df(sql)
        if df.empty:
            st.caption("No data.")
            return
        if kind == "line":
            st.altair_chart(line_chart(df, x, y, color, y), width="stretch")
        elif kind == "bar":
            st.altair_chart(bar_chart(df, x, y, color, y, horizontal=True), width="stretch")
        elif kind == "bar_v":
            st.altair_chart(bar_chart(df, x, y, color, y, horizontal=False), width="stretch")
        elif kind == "scatter":
            st.altair_chart(scatter_chart(df, x, y, color, y), width="stretch")
        elif kind == "components":
            comp = pd.DataFrame({
                "Component": ["Product", "Freight", "Insurance", "Customs"],
                "USD": [df.iloc[0]["PRODUCT"], df.iloc[0]["FREIGHT"], df.iloc[0]["INSURANCE"], df.iloc[0]["CUSTOMS"]],
            })
            st.altair_chart(bar_chart(comp, "Component", "USD", color, "Total USD"), width="stretch")


def _rows_as_dicts(cols: list[str], rows: list[list]) -> list[dict]:
    return [dict(zip(cols, r)) for r in rows]


def _payload_text(payload) -> str:
    """Extract plain text from a thread message_payload (a JSON string or dict)."""
    try:
        obj = json.loads(payload) if isinstance(payload, str) else payload
    except (json.JSONDecodeError, TypeError):
        return str(payload)
    if isinstance(obj, dict):
        texts = [c.get("text", "") for c in obj.get("content", []) if c.get("type") == "text"]
        joined = " ".join(t for t in texts if t).strip()
        return joined or ""
    return str(payload)


@st.cache_data(ttl=600, show_spinner=False)
def thread_title(thread_id: int) -> str:
    """Title for an unnamed thread: its first user message, truncated."""
    try:
        for m in get_thread_messages(thread_id, page_size=20):
            if m.get("role") == "user":
                text = _payload_text(m.get("message_payload")).strip()
                if text:
                    return text[:48] + ("…" if len(text) > 48 else "")
    except Exception:
        pass
    return "Untitled conversation"


def restore_thread(thread_id: int) -> None:
    """Load a past thread's messages into the chat and set up resume state.

    Assistant messages are reparsed through the same _extract_from_response_payload
    used for live answers, so restored turns rebuild their chart/table/SQL/grounding
    (the stored message_payload has the same content-item shape as a live response).
    """
    msgs = get_thread_messages(thread_id)
    chat = []
    last_assistant_id = 0
    for m in msgs:
        role = m.get("role")
        payload = m.get("message_payload")
        if role == "user":
            text = _payload_text(payload)
            if text:
                chat.append({"role": "user", "content": text})
        elif role == "assistant":
            res = AgentResult()
            try:
                obj = json.loads(payload) if isinstance(payload, str) else payload
                _extract_from_response_payload(obj, res)
            except (json.JSONDecodeError, TypeError):
                res.answer_text = _payload_text(payload)
            chat.append({"role": "assistant", "result": res, "question": None})
            if m.get("message_id"):
                last_assistant_id = m["message_id"]
    st.session_state.chat_messages = chat
    st.session_state.thread_id = thread_id
    st.session_state.parent_message_id = last_assistant_id


def submit_question(prompt: str) -> None:
    """Queue a question (typed or transcribed). It's streamed on the next run so
    the answer types out live; see the pending-question handler in the chat view."""
    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    st.session_state.pending_question = prompt


def finalize_answer(prompt: str, result: AgentResult) -> None:
    """After a streamed answer completes, update thread state + store the message."""
    if result.assistant_message_id is not None:
        st.session_state.parent_message_id = result.assistant_message_id
    if st.session_state.get("just_created_thread") and st.session_state.thread_id:
        try:
            set_thread_name(st.session_state.thread_id, prompt)
        except Exception:
            pass
        st.session_state.just_created_thread = False
    st.session_state.chat_messages.append({"role": "assistant", "result": result, "question": prompt})


@st.cache_data(ttl=600, show_spinner=False)
def load_ontology_entities() -> list[dict]:
    cols, rows = run_sql(
        "SELECT entity_id, display_name, entity_type, description, source_object, business_domain "
        "FROM V_ONTOLOGY_ENTITIES WHERE is_active = 'true' ORDER BY entity_type, display_name"
    )
    return _rows_as_dicts(cols, rows)


@st.cache_data(ttl=600, show_spinner=False)
def load_ontology_graph() -> list[dict]:
    cols, rows = run_sql(
        "SELECT source_id, source_name, edge_label, target_id, target_name, "
        "relationship_type, cardinality FROM V_ONTOLOGY_GRAPH"
    )
    return _rows_as_dicts(cols, rows)


@st.cache_data(ttl=600, show_spinner=False)
def load_ontology_attributes(entity_id: str) -> list[dict]:
    safe = entity_id.replace("'", "''")
    cols, rows = run_sql(
        "SELECT attribute_name, display_name, data_type, is_key, is_foreign_key "
        f"FROM V_ONTOLOGY_ATTRIBUTES WHERE entity_id = '{safe}' AND is_active = 'true'"
    )
    return _rows_as_dicts(cols, rows)


@st.cache_data(ttl=600, show_spinner=False)
def load_entity_metrics(entity_id: str) -> list[dict]:
    safe = entity_id.replace("'", "''")
    cols, rows = run_sql(
        "SELECT metric_name, definition, canonical_object, role "
        f"FROM V_ONTOLOGY_ENTITY_METRICS WHERE entity_id = '{safe}'"
    )
    return _rows_as_dicts(cols, rows)


@st.cache_data(ttl=600, show_spinner=False)
def load_metric_catalog() -> list[dict]:
    cols, rows = run_sql(
        "SELECT display_name, description, definition, grain, canonical_object, business_domain "
        "FROM V_ONTOLOGY_METRICS WHERE is_active = 'true' ORDER BY display_name"
    )
    return _rows_as_dicts(cols, rows)


def _autoscroll_chat(nonce: int) -> None:
    """Pin the fixed-height message pane to the bottom on new content.

    Rendered via st.html (inline in the app DOM), so the script scrolls the
    st.container(height=...) box directly. The nonce (message count) changes the
    script body each turn, forcing Streamlit to re-run it as the conversation grows.
    """
    st.html(
        f"""
        <script>
        (function() {{
            const nonce = {nonce};
            function scroll() {{
                const el = document.querySelector('.st-key-msg_scroll');
                if (!el) return;
                el.scrollTop = el.scrollHeight;
                el.querySelectorAll('*').forEach(c => {{
                    if (c.scrollHeight > c.clientHeight) c.scrollTop = c.scrollHeight;
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
    st.markdown(f'<div class="user-row"><div class="user-bubble">{text}</div></div>', unsafe_allow_html=True)


def render_assistant(res: AgentResult) -> None:
    if res.error:
        st.error(res.error)
        return

    if res.tools_used:
        tools = "".join(
            f'<div class="trace-step"><span class="trace-check">✓</span>Used tool → <code>{t}</code></div>'
            for t in res.tools_used
        )
        st.markdown(
            '<div class="trace-box">'
            '<div class="trace-head">✨ <span class="title">Agent trace</span>'
            f'<span class="meta">· {len(res.tools_used)} tool(s)</span></div>'
            f'<div class="trace-body">{tools}</div></div>',
            unsafe_allow_html=True,
        )

    if res.answer_text:
        st.markdown(res.answer_text)

    if getattr(res, "chart_spec", None):
        st.vega_lite_chart(res.chart_spec, width="stretch")

    if res.result_rows and res.column_names:
        df = pd.DataFrame(res.result_rows, columns=res.column_names)
        st.dataframe(df, width="stretch", hide_index=True)


# --- Sidebar (fixed, full-height, collapsible): navigation + persona ---
with st.sidebar:
    st.markdown(
        '<div class="brand-container"><div class="brand-logo">SC</div>Supply chain copilot</div>',
        unsafe_allow_html=True,
    )
    if st.button("+ New chat", key="new_chat", width="stretch"):
        st.session_state.chat_messages = []
        st.session_state.thread_id = None
        st.session_state.parent_message_id = 0
        st.session_state.active_view = "Chat"
        st.rerun()
    st.markdown('<div class="nav-group">Navigation</div>', unsafe_allow_html=True)
    view_icons = {
        "Dashboard": ":material/dashboard: Dashboard",
        "Chat": ":material/chat: Chat",
        "Ontology explorer": ":material/account_tree: Ontology explorer",
    }
    views = ["Dashboard", "Chat", "Ontology explorer"]
    cur = st.session_state.active_view if st.session_state.active_view in views else "Chat"
    view = st.radio(
        "View", views, index=views.index(cur),
        format_func=lambda v: view_icons[v],
        label_visibility="collapsed", key="nav_radio",
    )
    if view != st.session_state.active_view:
        st.session_state.active_view = view
        st.rerun()

    if st.session_state.active_view != "Ontology explorer":
        st.markdown('<div class="nav-group">Viewing as</div>', unsafe_allow_html=True)
        persona_icons = {
            "Planning": ":material/bar_chart: Planning",
            "Procurement": ":material/shopping_cart: Procurement",
            "Logistics": ":material/local_shipping: Logistics",
        }
        personas = ["Planning", "Procurement", "Logistics"]
        persona = st.radio(
            "Viewing as", personas,
            index=personas.index(st.session_state.active_persona),
            format_func=lambda p: persona_icons[p],
            label_visibility="collapsed", key="persona_radio",
        )
        if persona and persona != st.session_state.active_persona:
            st.session_state.active_persona = persona
            st.rerun()

    st.markdown('<div class="nav-group">Conversations</div>', unsafe_allow_html=True)
    try:
        threads = list_threads()
    except Exception:
        threads = []
    if not threads:
        st.markdown('<div class="hist-empty">No past conversations yet</div>', unsafe_allow_html=True)
    for t in threads:
        tid = t.get("thread_id")
        name = t.get("thread_name") or thread_title(tid)
        is_active = tid == st.session_state.thread_id
        btn_key = "hist_active" if is_active else f"hist_{tid}"
        c_open, c_del = st.columns([5, 1], gap="small", vertical_alignment="center")
        with c_open:
            if st.button(name, key=btn_key, width="stretch"):
                with st.spinner("Loading conversation…"):
                    restore_thread(tid)
                st.session_state.active_view = "Chat"
                st.rerun()
        with c_del:
            if st.button("🗑", key=f"del_{tid}", help="Delete conversation"):
                try:
                    delete_thread(tid)
                except Exception:
                    pass
                if is_active:
                    st.session_state.chat_messages = []
                    st.session_state.thread_id = None
                    st.session_state.parent_message_id = 0
                st.rerun()

# --- Persona theming overlay: repaints both the surface palette (bg/card/border)
# AND the accent on every rerun. Base CSS at the top of the file holds the
# default (Planning navy) palette; this block overrides it for the active
# persona. Shadow uses the accent at low opacity because dark-on-dark shadows
# vanish regardless of hue family.
_pc = PERSONA_CONFIG.get(st.session_state.active_persona, PERSONA_CONFIG["Planning"])
_A, _AD, _AS, _AH = _pc["accent"], _pc["accent_dark"], _pc["accent_soft"], _pc["accent_hover"]
_BG, _CARD, _ELEV, _BORDER, _SIDE = _pc["bg"], _pc["card"], _pc["elevated"], _pc["border"], _pc["sidebar"]
_hex = _A.lstrip("#")
_glow = f"rgba({int(_hex[0:2],16)},{int(_hex[2:4],16)},{int(_hex[4:6],16)},0.25)"
_bg_hex = _BG.lstrip("#")
_bg_rgba0 = f"rgba({int(_bg_hex[0:2],16)},{int(_bg_hex[2:4],16)},{int(_bg_hex[4:6],16)},0)"
st.markdown(
    f"""<style>
    /* Surface overrides — every bg/border selector from the base CSS, re-tinted. */
    .stApp, body {{ background-color: {_BG} !important; }}
    [data-testid="stAppViewContainer"] {{ background: {_BG} !important; }}
    [data-testid="stMain"] {{ background: {_BG} !important; }}
    [data-testid="stSidebar"] {{ background: {_SIDE} !important; border-right-color: {_BORDER} !important; }}
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {{ background: {_ELEV} !important; }}
    [data-testid="stBottomBlockContainer"] {{ background: {_BG} !important; }}
    .st-key-composer_dock {{ background: linear-gradient(to top, {_BG} 62%, {_bg_rgba0}) !important; }}
    .kpi-inner {{ background: {_CARD} !important; border-color: {_BORDER} !important; }}
    .band-rule {{ background: {_BORDER} !important; }}
    [class*="st-key-chartcard_"] {{ background: {_CARD} !important; border-color: {_BORDER} !important; }}
    .chart-card-title {{ background: {_ELEV} !important; border-bottom-color: {_BORDER} !important; }}
    .st-key-main_row > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(2),
    .st-key-main_row > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(3) {{
        border-left-color: {_BORDER} !important;
    }}
    .st-key-main_row > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child:nth-last-child(2) {{
        background-color: {_CARD} !important; border-right-color: {_BORDER} !important;
    }}
    [data-testid="stButton"] button:hover {{ background: {_ELEV} !important; }}
    .trace-box {{ background: {_CARD} !important; border-color: {_BORDER} !important; }}
    .trace-head {{ background: {_ELEV} !important; border-bottom-color: {_BORDER} !important; }}
    .trace-step code {{ background: {_ELEV} !important; }}
    .answer-card {{ background: {_CARD} !important; border-color: {_BORDER} !important; }}
    .g-head, .g-divider {{ border-bottom-color: {_BORDER} !important; }}
    .g-sql {{ background: {_BG} !important; border-color: {_BORDER} !important; }}
    .st-key-onto_detail_card {{ background: {_CARD} !important; border-color: {_BORDER} !important; }}
    .metric-card {{ background: {_CARD} !important; border-color: {_BORDER} !important; }}
    [data-testid="stChatInput"] {{ background: {_CARD} !important; border-color: {_BORDER} !important; }}
    [data-testid="stChatInput"] textarea {{ background: {_CARD} !important; }}
    [data-testid="stPills"] button {{ background: {_CARD} !important; border-color: {_BORDER} !important; }}
    [data-testid="stDataFrame"] {{ background: {_CARD} !important; }}
    [data-testid="stAlert"] {{ background: {_CARD} !important; border-color: {_BORDER} !important; }}

    /* Accent overrides — persona-colored interactive affordances. */
    .brand-logo {{ background: {_A} !important; color: {_BG} !important; }}
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {{ background: {_AS} !important; }}
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p {{ color: {_A} !important; }}
    .st-key-new_chat button {{ background: {_CARD} !important; color: {_A} !important; border-color: {_A} !important; }}
    .st-key-new_chat button:hover {{ background: {_AH} !important; color: {_AD} !important; border-color: {_AD} !important; }}
    [data-testid="stSidebar"] .st-key-hist_active button {{ background: {_AS} !important; color: {_A} !important; }}
    .user-bubble {{ background: {_A} !important; color: {_BG} !important; }}
    .st-key-composer_bar {{ background: {_CARD} !important; border: 1.5px solid {_A} !important; box-shadow: 0 2px 14px {_glow} !important; }}
    a, a:visited {{ color: {_A} !important; }}
    </style>""",
    unsafe_allow_html=True,
)

# --- Dashboard View: KPI strip + persona-specific governed charts ---
if st.session_state.active_view == "Dashboard":
    persona = st.session_state.active_persona
    specs = PERSONA_DASHBOARD_QUERIES.get(persona, PERSONA_DASHBOARD_QUERIES["Planning"])

    with st.spinner("Loading dashboard…"):
        kpis, kpi_errors = load_kpis()

    order = PERSONA_CONFIG.get(persona, PERSONA_CONFIG["Planning"])["kpi_order"]
    ordered_ids = [m for m in order if m in kpis] + [m for m in kpis if m not in order]
    kpi_cols = st.columns(len(ordered_ids) or 1)
    for col, mid in zip(kpi_cols, ordered_ids):
        m = kpis[mid]
        with col:
            st.markdown(
                f'<div class="kpi-inner" style="border-left-color:{m["color"]};">'
                f'<div class="label">{m["name"]}</div><div class="value">{m["value"]}</div></div>',
                unsafe_allow_html=True,
            )
    if kpi_errors:
        with st.expander("⚠ KPI load errors"):
            for err in kpi_errors:
                st.write(err)

    st.markdown('<div class="band-rule"></div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="dash-heading">📊 {persona} Reports &amp; Analytics</div>',
        unsafe_allow_html=True,
    )

    # Render the active persona's charts in a 2-column grid of cards.
    for i in range(0, len(specs), 2):
        left, right = st.columns(2, gap="medium")
        with left:
            render_persona_chart(specs[i], idx=i)
        if i + 1 < len(specs):
            with right:
                render_persona_chart(specs[i + 1], idx=i + 1)

# --- Main Chat Layout ---
elif st.session_state.active_view == "Chat":
    main = st.container(key="main_row")
    with main:
        col_chat = st.container()

        with col_chat:
            if not st.session_state.chat_messages:
                st.markdown(
                    '<div class="empty-greeting"><span class="big">How can I help with your supply chain?</span>'
                    'Ask about suppliers, plants, orders, inventory…</div>',
                    unsafe_allow_html=True,
                )
            else:
                # Fixed-height scrollable message pane: tall content (charts,
                # tables) scrolls inside this box instead of pushing the page down.
                with st.container(height=640, key="msg_scroll"):
                    for msg in st.session_state.chat_messages:
                        if msg["role"] == "user":
                            render_user(msg["content"])
                        else:
                            render_assistant(msg["result"])

                    # A queued question streams its answer live here, then is
                    # finalized (stored) and the page reruns to settle.
                    pending = st.session_state.get("pending_question")
                    if pending:
                        st.session_state.pending_question = None
                        if st.session_state.thread_id is None:
                            try:
                                st.session_state.thread_id = create_thread()
                                st.session_state.parent_message_id = 0
                                st.session_state.just_created_thread = True
                            except Exception:
                                st.session_state.thread_id = None
                        # Give the agent the active persona as context so it can
                        # frame the answer for that role. The user still sees their
                        # original question in the chat; only the agent gets the wrap.
                        persona = st.session_state.active_persona
                        contextual = (
                            f"The user is viewing the supply chain as the {persona} persona. "
                            f"Frame the answer for that role.\n\nUser question:\n{pending}"
                        )
                        gen = ask_agent_stream(
                            contextual,
                            thread_id=st.session_state.thread_id,
                            parent_message_id=st.session_state.parent_message_id,
                        )
                        st.write_stream(gen)
                        result = getattr(ask_agent_stream, "result", AgentResult())
                        finalize_answer(pending, result)
                        st.rerun()

                st.markdown('<div id="chat-scroll-anchor"></div>', unsafe_allow_html=True)
                _autoscroll_chat(len(st.session_state.chat_messages))

# --- Ontology View (live metadata from V_ONTOLOGY_* views) ---
else:
    try:
        entities = load_ontology_entities()
        graph = load_ontology_graph()
    except Exception as e:
        entities, graph = [], []
        st.error(f"Could not load ontology metadata: {e}")

    ENTITY_TYPE_COLOR = {
        "MASTER": "#5fb3e8", "REFERENCE": "#9d93e8", "FACT": "#ff6b5a",
        "LOGISTICS": "#3dd598", "TRANSACTION": "#f08a4b", "ASSET": "#e879b5",
        "BRIDGE": "#8ba3c0",
    }

    st.markdown('<div class="dash-heading">🕸 Ontology explorer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="g-text">Click any entity in the map to see its attributes, relationships, and metrics.</div>',
        unsafe_allow_html=True,
    )

    # Interactive flow graph with a floating detail card overlaid on its top-right.
    # Node colors encode entity type; the layered layout mirrors the L-to-R ERD.
    if graph:
        name_by_id = {e["ENTITY_ID"]: e["DISPLAY_NAME"] for e in entities}
        type_by_id = {e["ENTITY_ID"]: e["ENTITY_TYPE"] for e in entities}
        node_ids = set()
        for edge in graph:
            node_ids.add(edge["SOURCE_ID"])
            node_ids.add(edge["TARGET_ID"])

        _node_bg = PERSONA_CONFIG.get(st.session_state.active_persona, PERSONA_CONFIG["Planning"])["card"]
        flow_nodes = []
        for nid in node_ids:
            col = ENTITY_TYPE_COLOR.get(type_by_id.get(nid, ""), "#8A909B")
            flow_nodes.append(StreamlitFlowNode(
                id=nid, pos=(0, 0),
                data={"content": name_by_id.get(nid, nid)},
                node_type="default", source_position="right", target_position="left",
                selectable=True,
                style={"background": _node_bg, "border": f"2px solid {col}",
                       "borderRadius": "8px", "fontSize": "12px", "color": "#e3f0ff",
                       "padding": "6px 10px"},
            ))
        flow_edges = [
            StreamlitFlowEdge(
                id=f"{edge['SOURCE_ID']}-{edge['TARGET_ID']}-{i}",
                source=edge["SOURCE_ID"], target=edge["TARGET_ID"],
                label=edge.get("EDGE_LABEL", ""), animated=False, edge_type="smoothstep",
            )
            for i, edge in enumerate(graph)
        ]

        if "onto_flow_state" not in st.session_state:
            st.session_state.onto_flow_state = StreamlitFlowState(flow_nodes, flow_edges)

        stage = st.container(key="onto_stage")
        with stage:
            new_state = streamlit_flow(
                "onto_flow", st.session_state.onto_flow_state,
                layout=LayeredLayout(direction="right", node_node_spacing=40, node_layer_spacing=110),
                fit_view=True, height=600, get_node_on_click=True,
                show_minimap=True, show_controls=True, hide_watermark=True,
            )

            eid = st.session_state.active_entity
            ent = next((e for e in entities if e["ENTITY_ID"] == eid), None) if eid else None
            with st.container(key="onto_detail_card"):
                if not ent:
                    st.markdown(
                        '<div class="onto-group-title">Entity detail</div>'
                        '<div class="g-text">Click a node to see its attributes, relationships, and metrics.</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    parts = [
                        f'<div class="onto-group-title">{ent["ENTITY_TYPE"].title()}</div>',
                        f'<div class="g-value">{ent["DISPLAY_NAME"]}</div>',
                        f'<div class="g-text">{ent.get("DESCRIPTION", "")}</div>',
                        f'<div class="g-label">Source</div>',
                        f'<div class="g-text">{ent.get("SOURCE_OBJECT", "")}</div>',
                    ]
                    attrs = load_ontology_attributes(eid)
                    if attrs:
                        parts.append('<div class="g-label">Attributes</div>')
                        parts += [
                            f'<div class="onto-item"><span>{a["DISPLAY_NAME"] or a["ATTRIBUTE_NAME"]}'
                            f'{" 🔑" if str(a.get("IS_KEY")).lower() == "true" else ""}</span>'
                            f'<span class="count">{a["DATA_TYPE"]}</span></div>'
                            for a in attrs
                        ]
                    rels = [g for g in graph if g["SOURCE_ID"] == eid or g["TARGET_ID"] == eid]
                    if rels:
                        parts.append('<div class="g-label">Relationships</div>')
                        parts += [
                            f'<div class="onto-item"><span>{r["SOURCE_NAME"]} {r["EDGE_LABEL"]} {r["TARGET_NAME"]}</span>'
                            f'<span class="count">{r.get("CARDINALITY", "")}</span></div>'
                            for r in rels
                        ]
                    ems = load_entity_metrics(eid)
                    if ems:
                        parts.append('<div class="g-label">Metrics</div>')
                        parts += [
                            f'<div class="onto-item"><span>{m["METRIC_NAME"]}</span>'
                            f'<span class="count">{m.get("ROLE", "")}</span></div>'
                            for m in ems
                        ]
                    st.markdown("".join(parts), unsafe_allow_html=True)

        if new_state and new_state.selected_id and new_state.selected_id != st.session_state.active_entity:
            st.session_state.active_entity = new_state.selected_id
            st.rerun()
    else:
        st.info("No relationship data available.")

    st.markdown('<div class="band-rule"></div>', unsafe_allow_html=True)
    st.markdown('<div class="onto-group-title">Governed metric catalog</div>', unsafe_allow_html=True)
    _metric_colors = ["#5fb3e8", "#e8c76b", "#3dd598", "#f08a4b", "#9d93e8"]
    try:
        catalog = load_metric_catalog()
    except Exception:
        catalog = []
    cat_cols = st.columns(len(catalog) or 1)
    for col, m, color in zip(cat_cols, catalog, _metric_colors * 3):
        with col:
            st.markdown(
                f'<div class="metric-card" style="border-left:3px solid {color};">'
                f'<div class="m-name">{m["DISPLAY_NAME"]}</div>'
                f'<div class="m-def">{m.get("DEFINITION", "")}</div>'
                f'<div class="m-view">{m.get("CANONICAL_OBJECT", "")}</div></div>',
                unsafe_allow_html=True,
            )

# Composer at TOP LEVEL. Voice transcribes INTO the text box (review-then-send).
# Streamlit rule that drives this design: a widget's value lives in
# st.session_state[key]. To pre-fill or clear it, set that KEY *before* the
# widget is created this run — never assign the widget's return to a parallel var
# (that gets overwritten by the widget's own retained state on rerun).
if st.session_state.active_view == "Chat":
    if "composer_input" not in st.session_state:
        st.session_state.composer_input = ""
    if "last_audio_file_id" not in st.session_state:
        st.session_state.last_audio_file_id = None
    if "pending_clear" not in st.session_state:
        st.session_state.pending_clear = False

    # Apply a clear requested by the previous run, before the widget renders.
    if st.session_state.pending_clear:
        st.session_state.composer_input = ""
        st.session_state.pending_clear = False

    composer_dock = st.container(key="composer_dock")
    with composer_dock, st.container(key="composer_bar"):
        text_col, mic_col, send_col = st.columns([8, 1.4, 1.2], vertical_alignment="center")

        with mic_col:
            audio = st.audio_input("Speak", label_visibility="collapsed", key="voice_in")

        # Dedup by the recording's stable file_id (id() changes every rerun, causing
        # the repeated-transcription loop). Transcribe once, write into the box KEY.
        if audio is not None:
            fid = getattr(audio, "file_id", None) or getattr(audio, "name", None)
            if fid != st.session_state.last_audio_file_id:
                st.session_state.last_audio_file_id = fid
                try:
                    with st.spinner("Transcribing…"):
                        spoken = transcribe_audio(audio.getvalue())
                    if spoken:
                        st.session_state.composer_input = spoken
                        st.rerun()
                    else:
                        st.warning("No speech detected. Try again.")
                except Exception as e:
                    st.error(f"Transcription failed: {e}")

        with text_col:
            # on_change fires when the user presses Enter in the text box,
            # so Enter submits just like clicking Send.
            st.text_input(
                "Ask",
                placeholder="Ask about suppliers, plants, orders, inventory…",
                label_visibility="collapsed", key="composer_input",
                on_change=lambda: st.session_state.update(enter_pressed=True),
            )
        with send_col:
            send = st.button("Send", type="primary", width="stretch")

    submitted = send or st.session_state.pop("enter_pressed", False)
    if submitted and st.session_state.composer_input.strip():
        q = st.session_state.composer_input.strip()
        st.session_state.pending_clear = True  # clear the box on next run
        submit_question(q)
        st.rerun()
