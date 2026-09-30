import streamlit as st
import pandas as pd
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


# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Supply Chain Copilot",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# GLOBAL CSS
# ============================================================================

st.markdown(
    """
    <style>

    /* ------------------------------------------------------------------ */
    /* General                                                           */
    /* ------------------------------------------------------------------ */

    .block-container {
        padding-top: 1rem;
        padding-bottom: 7rem;
    }

    .stApp {
        background-color: #ffffff;
    }

    /* ------------------------------------------------------------------ */
    /* Sidebar                                                           */
    /* ------------------------------------------------------------------ */

    section[data-testid="stSidebar"] {
        border-right: 1px solid #e5e7eb;
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1rem;
    }

    /* ------------------------------------------------------------------ */
    /* Dashboard                                                         */
    /* ------------------------------------------------------------------ */

    .dashboard-title {
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 2px;
    }

    .dashboard-subtitle {
        color: #6b7280;
        margin-bottom: 18px;
    }

    .kpi-card {
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 16px;
        background: #ffffff;
        min-height: 105px;
    }

    .kpi-label {
        color: #6b7280;
        font-size: 13px;
        margin-bottom: 7px;
    }

    .kpi-value {
        font-size: 27px;
        font-weight: 700;
        line-height: 1.1;
    }

    /* ------------------------------------------------------------------ */
    /* Chat                                                              */
    /* ------------------------------------------------------------------ */

    .chat-header {
        font-size: 27px;
        font-weight: 700;
        margin-bottom: 2px;
    }

    .chat-subtitle {
        color: #6b7280;
        margin-bottom: 15px;
    }

    .chat-message-user {
        background: #f3f4f6;
        border-radius: 12px;
        padding: 11px 14px;
        margin: 8px 0;
        margin-left: 18%;
    }

    .chat-message-assistant {
        padding: 10px 5px;
        margin: 8px 0;
        margin-right: 8%;
    }

    .message-role {
        font-size: 11px;
        font-weight: 700;
        color: #6b7280;
        margin-bottom: 4px;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }

    /* ------------------------------------------------------------------ */
    /* Ontology                                                          */
    /* ------------------------------------------------------------------ */

    .ontology-title {
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 2px;
    }

    .ontology-subtitle {
        color: #6b7280;
        margin-bottom: 18px;
    }

    .ontology-card {
        padding: 14px 16px;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        margin-bottom: 10px;
        background: #ffffff;
    }

    .entity-title {
        font-size: 22px;
        font-weight: 700;
    }

    .muted {
        color: #6b7280;
    }

    .edge {
        padding: 8px 10px;
        border-radius: 7px;
        background: #f5f6f8;
        margin-bottom: 6px;
        font-size: 13px;
    }

    .metric-card {
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 10px 12px;
        margin-bottom: 8px;
    }

    .metric-card .m-name {
        font-size: 13px;
        font-weight: 600;
    }

    .metric-card .m-def {
        font-size: 12px;
        color: #6b7280;
        margin-top: 3px;
        line-height: 1.4;
    }

    .metric-card .m-view {
        font-size: 10px;
        color: #9ca3af;
        margin-top: 5px;
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }

    /* ------------------------------------------------------------------ */
    /* Persona                                                           */
    /* ------------------------------------------------------------------ */

    .persona-caption {
        color: #6b7280;
        font-size: 12px;
        margin-bottom: 5px;
    }

    /* ------------------------------------------------------------------ */
    /* Composer                                                          */
    /* ------------------------------------------------------------------ */

    div[data-testid="stVerticalBlock"]:has(.composer-bar) {
        border-radius: 14px;
    }

    .composer-bar {
        border: 1px solid #dfe3e8;
        border-radius: 14px;
        padding: 8px;
        background: #ffffff;
        box-shadow: 0 3px 14px rgba(0, 0, 0, 0.06);
    }

    .composer-bar input {
        border: none !important;
        box-shadow: none !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# SESSION STATE
# ============================================================================

DEFAULT_SESSION_STATE = {
    "active_view": "Dashboard",
    "active_persona": "Planning",
    "chat_messages": [],
    "thread_id": None,
    "parent_message_id": None,
    "active_entity": None,
    "pending_question": None,
    "composer_input": "",
    "pending_clear": False,
}

for key, value in DEFAULT_SESSION_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================================
# PERSONA CONFIG
# ============================================================================

PERSONAS = {
    "Planning": {
        "icon": "📊",
        "description": "Inventory, demand and fulfillment",
        "color": "#378ADD",
    },
    "Procurement": {
        "icon": "🛒",
        "description": "Suppliers, inbound and landed cost",
        "color": "#1D9E75",
    },
    "Logistics": {
        "icon": "🚚",
        "description": "Shipments, delivery and transportation",
        "color": "#D85A30",
    },
}


# ============================================================================
# SQL HELPERS
# ============================================================================

def load_df(sql: str) -> pd.DataFrame:
    """
    Execute SQL through cortex_agent.run_sql() and return a DataFrame.
    """
    try:
        result = run_sql(sql)

        if isinstance(result, pd.DataFrame):
            df = result.copy()

        elif isinstance(result, list):
            df = pd.DataFrame(result)

        elif isinstance(result, dict):
            if "data" in result:
                df = pd.DataFrame(result["data"])
            elif "rows" in result:
                df = pd.DataFrame(result["rows"])
            else:
                df = pd.DataFrame([result])

        else:
            df = pd.DataFrame()

        if not df.empty:
            df.columns = [str(c).upper() for c in df.columns]

        return df

    except Exception as exc:
        st.error(f"SQL query failed: {exc}")
        return pd.DataFrame()


# ============================================================================
# KPI SNAPSHOT
# ============================================================================

KPI_SQL = """
SELECT
    METRIC_ID,
    METRIC_NAME,
    METRIC_VALUE,
    UNIT
FROM SUPPLY_CHAIN_ONTOLOGY.CORE.V_DASHBOARD_KPI_SNAPSHOT
ORDER BY METRIC_ID
"""


def load_kpis() -> pd.DataFrame:
    return load_df(KPI_SQL)


def format_kpi_value(value, unit=None) -> str:
    if pd.isna(value):
        return "—"

    try:
        numeric = float(value)
    except Exception:
        return str(value)

    unit = str(unit or "").upper()

    if "%" in unit:
        return f"{numeric:.1f}%"

    if unit in {"DAYS", "DAY"}:
        return f"{numeric:.1f}"

    if unit in {"USD", "$"}:
        return f"${numeric:,.0f}"

    if abs(numeric) >= 1_000_000:
        return f"{numeric / 1_000_000:.1f}M"

    if abs(numeric) >= 1_000:
        return f"{numeric / 1_000:.1f}K"

    return f"{numeric:,.1f}"


# ============================================================================
# DASHBOARD QUERIES
# ============================================================================

DASHBOARD_QUERIES = {

    "Planning": {
        "Outbound OTD Trend": """
            SELECT
                DELIVERY_MONTH,
                AVG(IS_ON_TIME) * 100 AS OTD_PCT
            FROM SUPPLY_CHAIN_ONTOLOGY.CORE.VW_ON_TIME_DELIVERY_OUTBOUND
            GROUP BY DELIVERY_MONTH
            ORDER BY DELIVERY_MONTH
        """,

        "Fill Rate Trend": """
            SELECT
                ORDER_MONTH,
                AVG(FILL_RATE) * 100 AS FILL_RATE_PCT
            FROM SUPPLY_CHAIN_ONTOLOGY.CORE.VW_FILL_RATE
            GROUP BY ORDER_MONTH
            ORDER BY ORDER_MONTH
        """,

        "Days of Inventory by Plant": """
            SELECT
                PLANT_ID,
                AVG(DII) AS DII
            FROM SUPPLY_CHAIN_ONTOLOGY.CORE.VW_DAYS_OF_INVENTORY
            GROUP BY PLANT_ID
            ORDER BY DII DESC
        """,

        "Days of Inventory by Part Category": """
            SELECT
                PART_CATEGORY,
                AVG(DII) AS DII
            FROM SUPPLY_CHAIN_ONTOLOGY.CORE.VW_DAYS_OF_INVENTORY
            GROUP BY PART_CATEGORY
            ORDER BY DII DESC
        """,

        "Inventory On Hand by Plant": """
            SELECT
                PLANT_ID,
                SUM(INVENTORY_ON_HAND) AS INVENTORY_ON_HAND
            FROM SUPPLY_CHAIN_ONTOLOGY.CORE.VW_DAYS_OF_INVENTORY
            GROUP BY PLANT_ID
            ORDER BY INVENTORY_ON_HAND DESC
        """,
    },

    "Procurement": {
        "Inbound OTD by Supplier": """
            SELECT
                SUPPLIER_ID,
                AVG(IS_ON_TIME) * 100 AS INBOUND_OTD_PCT
            FROM SUPPLY_CHAIN_ONTOLOGY.CORE.VW_ON_TIME_DELIVERY_INBOUND
            GROUP BY SUPPLIER_ID
            ORDER BY INBOUND_OTD_PCT
        """,

        "Landed Cost by Supplier": """
            SELECT
                SUPPLIER_ID,
                SUM(LANDED_COST) AS LANDED_COST
            FROM SUPPLY_CHAIN_ONTOLOGY.CORE.VW_LANDED_COST
            GROUP BY SUPPLIER_ID
            ORDER BY LANDED_COST DESC
        """,

        "Landed Cost by Part": """
            SELECT
                PART_ID,
                SUM(LANDED_COST) AS LANDED_COST
            FROM SUPPLY_CHAIN_ONTOLOGY.CORE.VW_LANDED_COST
            GROUP BY PART_ID
            ORDER BY LANDED_COST DESC
        """,

        "Landed Cost Breakdown": """
            SELECT
                SUM(PURCHASE_COST) AS PURCHASE_COST,
                SUM(FREIGHT_COST) AS FREIGHT_COST,
                SUM(INSURANCE_COST) AS INSURANCE_COST,
                SUM(CUSTOMS_COST) AS CUSTOMS_COST
            FROM SUPPLY_CHAIN_ONTOLOGY.CORE.VW_LANDED_COST
        """,

        "Supplier Reliability vs Inbound OTD": """
            SELECT
                SUPPLIER_ID,
                AVG(IS_ON_TIME) * 100 AS INBOUND_OTD_PCT,
                COUNT(*) AS SHIPMENTS
            FROM SUPPLY_CHAIN_ONTOLOGY.CORE.VW_ON_TIME_DELIVERY_INBOUND
            GROUP BY SUPPLIER_ID
            ORDER BY INBOUND_OTD_PCT
        """,
    },

    "Logistics": {
        "Outbound OTD Trend": """
            SELECT
                DELIVERY_MONTH,
                AVG(IS_ON_TIME) * 100 AS OTD_PCT
            FROM SUPPLY_CHAIN_ONTOLOGY.CORE.VW_ON_TIME_DELIVERY_OUTBOUND
            GROUP BY DELIVERY_MONTH
            ORDER BY DELIVERY_MONTH
        """,

        "OTD by Plant": """
            SELECT
                PLANT_ID,
                AVG(IS_ON_TIME) * 100 AS OTD_PCT
            FROM SUPPLY_CHAIN_ONTOLOGY.CORE.VW_ON_TIME_DELIVERY_OUTBOUND
            GROUP BY PLANT_ID
            ORDER BY OTD_PCT
        """,

        "OTD by Carrier": """
            SELECT
                CARRIER,
                AVG(IS_ON_TIME) * 100 AS OTD_PCT
            FROM SUPPLY_CHAIN_ONTOLOGY.CORE.VW_ON_TIME_DELIVERY_OUTBOUND
            GROUP BY CARRIER
            ORDER BY OTD_PCT
        """,

        "OTD by Mode": """
            SELECT
                TRANSPORT_MODE,
                AVG(IS_ON_TIME) * 100 AS OTD_PCT
            FROM SUPPLY_CHAIN_ONTOLOGY.CORE.VW_ON_TIME_DELIVERY_OUTBOUND
            GROUP BY TRANSPORT_MODE
            ORDER BY OTD_PCT
        """,

        "Late Shipments by Carrier": """
            SELECT
                CARRIER,
                COUNT(*) AS LATE_SHIPMENTS
            FROM SUPPLY_CHAIN_ONTOLOGY.CORE.VW_ON_TIME_DELIVERY_OUTBOUND
            WHERE IS_ON_TIME = 0
            GROUP BY CARRIER
            ORDER BY LATE_SHIPMENTS DESC
        """,

        "Delivery Delay Distribution": """
            SELECT
                DELAY_DAYS,
                COUNT(*) AS SHIPMENTS
            FROM SUPPLY_CHAIN_ONTOLOGY.CORE.VW_ON_TIME_DELIVERY_OUTBOUND
            GROUP BY DELAY_DAYS
            ORDER BY DELAY_DAYS
        """,
    },
}


# ============================================================================
# CHART HELPERS
# ============================================================================

def make_line_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
):
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        st.info("No data available for this chart.")
        return

    chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X(x_col, title=None),
            y=alt.Y(y_col, title=None),
            tooltip=list(df.columns),
        )
        .properties(
            title=title,
            height=320,
        )
        .interactive()
    )

    st.altair_chart(chart, width="stretch")


def make_bar_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    horizontal: bool = False,
):
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        st.info("No data available for this chart.")
        return

    if horizontal:
        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X(y_col, title=None),
                y=alt.Y(
                    x_col,
                    sort="-x",
                    title=None,
                ),
                tooltip=list(df.columns),
            )
            .properties(
                title=title,
                height=max(300, min(650, len(df) * 28)),
            )
            .interactive()
        )
    else:
        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X(x_col, title=None),
                y=alt.Y(y_col, title=None),
                tooltip=list(df.columns),
            )
            .properties(
                title=title,
                height=320,
            )
            .interactive()
        )

    st.altair_chart(chart, width="stretch")


def make_scatter_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
):
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        st.info("No data available for this chart.")
        return

    chart = (
        alt.Chart(df)
        .mark_circle(size=100)
        .encode(
            x=alt.X(x_col, title=None),
            y=alt.Y(y_col, title=None),
            tooltip=list(df.columns),
        )
        .properties(
            title=title,
            height=350,
        )
        .interactive()
    )

    st.altair_chart(chart, width="stretch")


# ============================================================================
# PERSONA DASHBOARD CHART RENDERING
# ============================================================================

def render_persona_chart(title: str, sql: str):
    df = load_df(sql)

    if df.empty:
        st.info(f"No data available for **{title}**.")
        return

    cols = list(df.columns)

    if len(cols) < 2:
        st.dataframe(df, width="stretch", hide_index=True)
        return

    numeric_cols = df.select_dtypes(
        include=["number"]
    ).columns.tolist()

    if not numeric_cols:
        st.dataframe(df, width="stretch", hide_index=True)
        return

    y_col = numeric_cols[-1]
    x_col = cols[0]

    lower_title = title.lower()

    if "trend" in lower_title:
        make_line_chart(
            df,
            x_col,
            y_col,
            title,
        )

    elif "scatter" in lower_title or "reliability" in lower_title:
        if len(numeric_cols) >= 2:
            make_scatter_chart(
                df,
                numeric_cols[0],
                numeric_cols[1],
                title,
            )
        else:
            make_bar_chart(
                df,
                x_col,
                y_col,
                title,
                horizontal=True,
            )

    elif "breakdown" in lower_title:
        # Convert one-row cost breakdown into a long format.
        if len(df) == 1:
            row = df.iloc[0]
            chart_df = pd.DataFrame(
                {
                    "CATEGORY": list(df.columns),
                    "VALUE": [
                        row[col]
                        for col in df.columns
                    ],
                }
            )

            make_bar_chart(
                chart_df,
                "CATEGORY",
                "VALUE",
                title,
                horizontal=True,
            )
        else:
            st.dataframe(
                df,
                width="stretch",
                hide_index=True,
            )

    else:
        make_bar_chart(
            df,
            x_col,
            y_col,
            title,
            horizontal=True,
        )


# ============================================================================
# ONTOLOGY LOADERS
# ============================================================================

@st.cache_data(ttl=300)
def load_ontology_entities() -> pd.DataFrame:
    return load_df(
        """
        SELECT *
        FROM SUPPLY_CHAIN_ONTOLOGY.CORE.V_ONTOLOGY_ENTITIES
        ORDER BY DISPLAY_NAME
        """
    )


@st.cache_data(ttl=300)
def load_ontology_graph() -> pd.DataFrame:
    return load_df(
        """
        SELECT *
        FROM SUPPLY_CHAIN_ONTOLOGY.CORE.V_ONTOLOGY_GRAPH
        """
    )


@st.cache_data(ttl=300)
def load_ontology_attributes() -> pd.DataFrame:
    return load_df(
        """
        SELECT *
        FROM SUPPLY_CHAIN_ONTOLOGY.CORE.V_ONTOLOGY_ATTRIBUTES
        ORDER BY ENTITY_ID, DISPLAY_NAME
        """
    )


@st.cache_data(ttl=300)
def load_entity_metrics() -> pd.DataFrame:
    return load_df(
        """
        SELECT *
        FROM SUPPLY_CHAIN_ONTOLOGY.CORE.V_ONTOLOGY_ENTITY_METRICS
        ORDER BY ENTITY_ID, DISPLAY_NAME
        """
    )


@st.cache_data(ttl=300)
def load_metric_catalog() -> pd.DataFrame:
    return load_df(
        """
        SELECT *
        FROM SUPPLY_CHAIN_ONTOLOGY.CORE.DIM_METRIC_CATALOG
        ORDER BY METRIC_NAME
        """
    )


# ============================================================================
# CHAT HELPERS
# ============================================================================

def _rows_as_dicts(value):
    if value is None:
        return []

    if isinstance(value, pd.DataFrame):
        return value.to_dict("records")

    if isinstance(value, list):
        return value

    if isinstance(value, dict):
        if "rows" in value:
            return value["rows"]

        if "data" in value:
            return value["data"]

        return [value]

    return []


def _payload_text(payload) -> str:
    """
    Best-effort extraction of displayable text from Cortex responses.
    """

    if payload is None:
        return ""

    if isinstance(payload, str):
        return payload

    if isinstance(payload, dict):

        for key in (
            "text",
            "content",
            "message",
            "response",
            "answer",
        ):
            value = payload.get(key)

            if isinstance(value, str):
                return value

            if isinstance(value, list):
                pieces = []

                for item in value:
                    if isinstance(item, str):
                        pieces.append(item)
                    elif isinstance(item, dict):
                        text = item.get("text")

                        if text:
                            pieces.append(str(text))

                if pieces:
                    return "".join(pieces)

        return ""

    if isinstance(payload, list):
        pieces = []

        for item in payload:
            text = _payload_text(item)

            if text:
                pieces.append(text)

        return "".join(pieces)

    return ""


def thread_title(prompt: str) -> str:
    clean = " ".join(prompt.split())

    if len(clean) <= 48:
        return clean

    return clean[:45].rstrip() + "..."


def restore_thread(thread_id: str):
    """
    Restore an existing Cortex Agent thread into the local chat state.
    """

    try:
        messages = get_thread_messages(thread_id)

        restored = []

        for message in messages or []:
            role = message.get("role")

            if role not in {"user", "assistant"}:
                continue

            content = message.get("content")

            text = _payload_text(content)

            if not text:
                text = _payload_text(message)

            if text:
                restored.append(
                    {
                        "role": role,
                        "content": text,
                    }
                )

        st.session_state.chat_messages = restored
        st.session_state.thread_id = thread_id
        st.session_state.parent_message_id = None

    except Exception as exc:
        st.error(f"Unable to restore conversation: {exc}")


def submit_question(prompt: str) -> None:
    """
    Queue a question for Cortex Agent streaming.
    """

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
    """
    Finalize the assistant response after streaming completes.
    """

    text = ""

    if result is not None:

        if isinstance(result, dict):
            text = _payload_text(result)

        else:
            for attr in (
                "text",
                "content",
                "response",
                "answer",
            ):
                value = getattr(
                    result,
                    attr,
                    None,
                )

                if isinstance(value, str) and value.strip():
                    text = value
                    break

    if not text:
        text = "I couldn't generate a response."

    st.session_state.chat_messages.append(
        {
            "role": "assistant",
            "content": text,
        }
    )

    st.session_state.pending_question = None


# ============================================================================
# CHAT RENDERING
# ============================================================================

def render_user(content: str):
    st.markdown(
        f"""
        <div class="chat-message-user">
            <div class="message-role">You</div>
            {content}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_assistant(content: str):
    st.markdown(
        f"""
        <div class="chat-message-assistant">
            <div class="message-role">Supply Chain Copilot</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(content)


def _autoscroll_chat():
    st.markdown(
        """
        <script>
        setTimeout(function() {
            const elements =
                window.parent.document.querySelectorAll(
                    '[data-testid="stVerticalScroll"]'
                );

            if (elements.length > 0) {
                const el = elements[elements.length - 1];
                el.scrollTop = el.scrollHeight;
            }
        }, 100);
        </script>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:

    st.markdown(
        "## 🔗 Supply Chain Copilot"
    )

    st.caption(
        "Governed conversational analytics"
    )

    st.divider()

    st.markdown("### Navigation")

    nav_options = [
        "Dashboard",
        "Chat",
        "Ontology",
    ]

    selected_view = st.radio(
        "Navigation",
        nav_options,
        index=nav_options.index(
            st.session_state.active_view
        ),
        label_visibility="collapsed",
    )

    if selected_view != st.session_state.active_view:
        st.session_state.active_view = selected_view

    st.divider()

    st.markdown(
        '<div class="persona-caption">VIEWING AS</div>',
        unsafe_allow_html=True,
    )

    persona = st.selectbox(
        "Persona",
        list(PERSONAS.keys()),
        index=list(PERSONAS.keys()).index(
            st.session_state.active_persona
        ),
        label_visibility="collapsed",
    )

    if persona != st.session_state.active_persona:
        st.session_state.active_persona = persona

    persona_info = PERSONAS[persona]

    st.caption(
        f'{persona_info["icon"]} {persona_info["description"]}'
    )

    st.divider()

    # ------------------------------------------------------------------------
    # Conversations
    # ------------------------------------------------------------------------

    st.markdown("### Conversations")

    if st.button(
        "＋ New conversation",
        width="stretch",
    ):
        try:
            thread = create_thread()

            if isinstance(thread, dict):
                thread_id = (
                    thread.get("thread_id")
                    or thread.get("id")
                )
            else:
                thread_id = thread

            st.session_state.thread_id = thread_id

        except Exception:
            st.session_state.thread_id = None

        st.session_state.parent_message_id = None
        st.session_state.chat_messages = []
        st.session_state.pending_question = None
        st.rerun()

    try:
        threads = list_threads()

        if threads:

            for thread in threads[:20]:

                if not isinstance(thread, dict):
                    continue

                tid = (
                    thread.get("thread_id")
                    or thread.get("id")
                )

                title = (
                    thread.get("name")
                    or thread.get("title")
                    or "Conversation"
                )

                if not tid:
                    continue

                if st.button(
                    str(title),
                    key=f"thread_{tid}",
                    width="stretch",
                ):
                    restore_thread(tid)
                    st.session_state.active_view = "Chat"
                    st.rerun()

    except Exception:
        st.caption("No saved conversations available.")


# ============================================================================
# PERSONA CSS ACCENT
# ============================================================================

persona_color = PERSONAS[
    st.session_state.active_persona
]["color"]

st.markdown(
    f"""
    <style>
    .persona-accent {{
        color: {persona_color};
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

    st.markdown(
        '<div class="dashboard-title">Supply Chain Dashboard</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="dashboard-subtitle">
            {PERSONAS[persona]["icon"]}
            {persona} perspective · governed supply-chain metrics
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------------
    # KPI cards
    # ------------------------------------------------------------------------

    kpis = load_kpis()

    if kpis.empty:

        st.info(
            "No KPI snapshot is currently available."
        )

    else:

        # Display up to six KPIs.
        kpi_count = min(len(kpis), 6)

        columns = st.columns(
            kpi_count
        )

        for idx in range(kpi_count):

            row = kpis.iloc[idx]

            metric_name = row.get(
                "METRIC_NAME",
                "Metric",
            )

            metric_value = format_kpi_value(
                row.get("METRIC_VALUE"),
                row.get("UNIT"),
            )

            with columns[idx]:

                st.markdown(
                    f"""
                    <div class="kpi-card">
                        <div class="kpi-label">
                            {metric_name}
                        </div>
                        <div class="kpi-value">
                            {metric_value}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.divider()

    # ------------------------------------------------------------------------
    # Persona dashboard
    # ------------------------------------------------------------------------

    queries = DASHBOARD_QUERIES.get(
        persona,
        {},
    )

    if not queries:

        st.info(
            f"No dashboard configuration exists for {persona}."
        )

    else:

        chart_items = list(
            queries.items()
        )

        for i in range(
            0,
            len(chart_items),
            2,
        ):

            row_items = chart_items[
                i:i + 2
            ]

            cols = st.columns(
                len(row_items)
            )

            for col, (
                title,
                sql,
            ) in zip(
                cols,
                row_items,
            ):

                with col:

                    st.markdown(
                        f"#### {title}"
                    )

                    render_persona_chart(
                        title,
                        sql,
                    )


# ============================================================================
# CHAT VIEW
# ============================================================================

if st.session_state.active_view == "Chat":

    st.markdown(
        '<div class="chat-header">Supply Chain Copilot</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="chat-subtitle">
            {PERSONAS[st.session_state.active_persona]["icon"]}
            {st.session_state.active_persona} ·
            Ask questions about suppliers, plants, orders,
            inventory, shipments and governed metrics.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------------
    # Chat history
    # ------------------------------------------------------------------------

    chat_container = st.container(
        height=540,
        border=False,
    )

    with chat_container:

        for message in st.session_state.chat_messages:

            role = message.get("role")
            content = message.get(
                "content",
                "",
            )

            if role == "user":
                render_user(content)

            elif role == "assistant":
                render_assistant(content)

    # ------------------------------------------------------------------------
    # Pending question / Cortex streaming
    # ------------------------------------------------------------------------

    pending = st.session_state.get(
        "pending_question"
    )

    if pending:

        contextual = (
            f"Current persona: "
            f"{st.session_state.active_persona}\n\n"
            f"User question: {pending}"
        )

        with chat_container:

            st.markdown(
                """
                <div class="chat-message-assistant">
                    <div class="message-role">
                        Supply Chain Copilot
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            try:

                gen = ask_agent_stream(
                    contextual,
                    thread_id=st.session_state.thread_id,
                    parent_message_id=(
                        st.session_state.parent_message_id
                    ),
                )

                st.write_stream(gen)

                # ask_agent_stream stores the final AgentResult
                # on the function object.
                result = getattr(
                    ask_agent_stream,
                    "result",
                    AgentResult(),
                )

                finalize_answer(
                    pending,
                    result,
                )

            except Exception as exc:

                st.session_state.chat_messages.append(
                    {
                        "role": "assistant",
                        "content": (
                            "I encountered an error while "
                            f"processing your request:\n\n"
                            f"`{exc}`"
                        ),
                    }
                )

                st.session_state.pending_question = None

        st.rerun()

    _autoscroll_chat()


# ============================================================================
# ONTOLOGY VIEW
# ============================================================================

if st.session_state.active_view == "Ontology":

    st.markdown(
        '<div class="ontology-title">🔗 Ontology Explorer</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="ontology-subtitle">
            Explore governed business entities, relationships,
            attributes and canonical metrics across the supply-chain
            semantic layer.
        </div>
        """,
        unsafe_allow_html=True,
    )

    entities = load_ontology_entities()
    graph = load_ontology_graph()
    attributes = load_ontology_attributes()
    entity_metrics = load_entity_metrics()
    metrics = load_metric_catalog()

    # ------------------------------------------------------------------------
    # Normalize
    # ------------------------------------------------------------------------

    for df in (
        entities,
        graph,
        attributes,
        entity_metrics,
        metrics,
    ):
        if not df.empty:
            df.columns = [
                str(c).upper()
                for c in df.columns
            ]

    # ------------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------------

    c1, c2, c3 = st.columns(
        [2, 1, 1]
    )

    with c1:

        search = st.text_input(
            "Search ontology",
            placeholder=(
                "Supplier, Product, Landed Cost..."
            ),
        ).strip().lower()

    with c2:

        if (
            not entities.empty
            and "ENTITY_TYPE" in entities.columns
        ):

            types = [
                "All"
            ] + sorted(
                entities[
                    "ENTITY_TYPE"
                ]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

        else:
            types = ["All"]

        entity_type = st.selectbox(
            "Entity type",
            types,
        )

    with c3:

        if (
            not entities.empty
            and "BUSINESS_DOMAIN" in entities.columns
        ):

            domains = [
                "All"
            ] + sorted(
                entities[
                    "BUSINESS_DOMAIN"
                ]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

        else:
            domains = ["All"]

        domain = st.selectbox(
            "Business domain",
            domains,
        )

    # ------------------------------------------------------------------------
    # Filter entities
    # ------------------------------------------------------------------------

    filtered = entities.copy()

    if (
        not filtered.empty
        and entity_type != "All"
        and "ENTITY_TYPE" in filtered.columns
    ):
        filtered = filtered[
            filtered["ENTITY_TYPE"]
            == entity_type
        ]

    if (
        not filtered.empty
        and domain != "All"
        and "BUSINESS_DOMAIN" in filtered.columns
    ):
        filtered = filtered[
            filtered["BUSINESS_DOMAIN"]
            == domain
        ]

    if search and not filtered.empty:

        masks = []

        if "DISPLAY_NAME" in filtered.columns:
            masks.append(
                filtered[
                    "DISPLAY_NAME"
                ]
                .fillna("")
                .astype(str)
                .str.lower()
                .str.contains(
                    search,
                    na=False,
                )
            )

        if "DESCRIPTION" in filtered.columns:
            masks.append(
                filtered[
                    "DESCRIPTION"
                ]
                .fillna("")
                .astype(str)
                .str.lower()
                .str.contains(
                    search,
                    na=False,
                )
            )

        if "ENTITY_ID" in filtered.columns:
            masks.append(
                filtered[
                    "ENTITY_ID"
                ]
                .fillna("")
                .astype(str)
                .str.lower()
                .str.contains(
                    search,
                    na=False,
                )
            )

        if masks:
            combined = masks[0]

            for mask in masks[1:]:
                combined = combined | mask

            filtered = filtered[
                combined
            ]

    # ------------------------------------------------------------------------
    # Summary strip
    # ------------------------------------------------------------------------

    a, b, c, d = st.columns(4)

    with a:
        st.metric(
            "Entities",
            len(entities),
        )

    with b:
        st.metric(
            "Graph records",
            len(graph),
        )

    with c:
        st.metric(
            "Metrics",
            len(metrics),
        )

    with d:
        st.metric(
            "Filtered entities",
            len(filtered),
        )

    st.divider()

    # ------------------------------------------------------------------------
    # Entity selection
    # ------------------------------------------------------------------------

    left, middle, right = st.columns(
        [1.05, 1.9, 1.15]
    )

    with left:

        st.subheader("Entities")

        if filtered.empty:

            st.info(
                "No entities match the current filters."
            )

            selected_id = None

        else:

            labels = {}

            for _, row in filtered.iterrows():

                entity_id = row.get(
                    "ENTITY_ID"
                )

                display_name = row.get(
                    "DISPLAY_NAME",
                    entity_id,
                )

                entity_type_value = row.get(
                    "ENTITY_TYPE",
                    "",
                )

                labels[
                    entity_id
                ] = (
                    f"{display_name} · "
                    f"{entity_type_value}"
                )

            existing_selection = (
                st.session_state.active_entity
            )

            if (
                existing_selection
                not in labels
            ):
                existing_selection = (
                    list(labels.keys())[0]
                )

            selected_id = st.radio(
                "Select an entity",
                options=list(
                    labels.keys()
                ),
                index=list(
                    labels.keys()
                ).index(
                    existing_selection
                ),
                format_func=lambda x: labels[x],
                label_visibility="collapsed",
            )

            st.session_state.active_entity = (
                selected_id
            )

    # ------------------------------------------------------------------------
    # Relationship graph
    # ------------------------------------------------------------------------

    with middle:

        st.subheader(
            "Relationship Graph"
        )

        if selected_id is None:

            st.info(
                "Select an entity to explore its relationships."
            )

        else:

            selected_rows = entities[
                entities["ENTITY_ID"]
                == selected_id
            ]

            if selected_rows.empty:

                st.info(
                    "Selected entity could not be found."
                )

            else:

                selected_row = (
                    selected_rows.iloc[0]
                )

                selected_name = (
                    selected_row.get(
                        "DISPLAY_NAME",
                        selected_id,
                    )
                )

                st.markdown(
                    f"**{selected_name}**"
                )

                # Build graph records.
                outgoing = pd.DataFrame()
                incoming = pd.DataFrame()

                if not graph.empty:

                    if (
                        "FROM_ENTITY_ID"
                        in graph.columns
                    ):
                        outgoing = graph[
                            graph[
                                "FROM_ENTITY_ID"
                            ]
                            == selected_id
                        ]

                    if (
                        "TO_ENTITY_ID"
                        in graph.columns
                    ):
                        incoming = graph[
                            graph[
                                "TO_ENTITY_ID"
                            ]
                            == selected_id
                        ]

                if (
                    outgoing.empty
                    and incoming.empty
                ):

                    st.info(
                        "No relationships defined."
                    )

                else:

                    # --------------------------------------------------------
                    # Streamlit Flow graph
                    # --------------------------------------------------------

                    nodes = []
                    edges = []

                    related_ids = {
                        selected_id
                    }

                    if not outgoing.empty:

                        for _, row in outgoing.iterrows():

                            related_ids.add(
                                row.get(
                                    "TO_ENTITY_ID"
                                )
                            )

                    if not incoming.empty:

                        for _, row in incoming.iterrows():

                            related_ids.add(
                                row.get(
                                    "FROM_ENTITY_ID"
                                )
                            )

                    related_entities = (
                        entities[
                            entities[
                                "ENTITY_ID"
                            ].isin(
                                related_ids
                            )
                        ]
                        if not entities.empty
                        else pd.DataFrame()
                    )

                    positions = {}

                    position_values = [
                        (50, 50),
                        (350, 50),
                        (50, 250),
                        (350, 250),
                        (50, 450),
                        (350, 450),
                        (200, 650),
                    ]

                    for idx, entity_id in enumerate(
                        related_ids
                    ):

                        if entity_id is None:
                            continue

                        row_matches = (
                            related_entities[
                                related_entities[
                                    "ENTITY_ID"
                                ]
                                == entity_id
                            ]
                            if not related_entities.empty
                            else pd.DataFrame()
                        )

                        if row_matches.empty:

                            display_name = str(
                                entity_id
                            )

                        else:

                            display_name = row_matches.iloc[
                                0
                            ].get(
                                "DISPLAY_NAME",
                                entity_id,
                            )

                        x, y = position_values[
                            idx % len(
                                position_values
                            )
                        ]

                        positions[
                            entity_id
                        ] = (
                            x,
                            y,
                        )

                        nodes.append(
                            StreamlitFlowNode(
                                str(entity_id),
                                (
                                    str(
                                        display_name
                                    )
                                ),
                                (
                                    x,
                                    y,
                                ),
                                style={
                                    "background":
                                        (
                                            persona_color
                                            if entity_id
                                            == selected_id
                                            else "#ffffff"
                                        ),
                                    "color":
                                        (
                                            "#ffffff"
                                            if entity_id
                                            == selected_id
                                            else "#111827"
                                        ),
                                    "border":
                                        (
                                            "1px solid "
                                            + persona_color
                                        ),
                                    "borderRadius":
                                        "10px",
                                    "padding":
                                        "10px",
                                },
                            )
                        )

                    # Outgoing edges
                    if not outgoing.empty:

                        for idx, row in outgoing.iterrows():

                            source = row.get(
                                "FROM_ENTITY_ID"
                            )

                            target = row.get(
                                "TO_ENTITY_ID"
                            )

                            if (
                                source is None
                                or target is None
                            ):
                                continue

                            relationship = row.get(
                                "RELATIONSHIP_NAME",
                                "",
                            )

                            edges.append(
                                StreamlitFlowEdge(
                                    f"edge_out_{idx}",
                                    str(source),
                                    str(target),
                                    label=str(
                                        relationship
                                    ),
                                )
                            )

                    # Incoming edges
                    if not incoming.empty:

                        for idx, row in incoming.iterrows():

                            source = row.get(
                                "FROM_ENTITY_ID"
                            )

                            target = row.get(
                                "TO_ENTITY_ID"
                            )

                            if (
                                source is None
                                or target is None
                            ):
                                continue

                            relationship = row.get(
                                "RELATIONSHIP_NAME",
                                "",
                            )

                            edges.append(
                                StreamlitFlowEdge(
                                    f"edge_in_{idx}",
                                    str(source),
                                    str(target),
                                    label=str(
                                        relationship
                                    ),
                                )
                            )

                    if nodes:

                        flow_state = StreamlitFlowState(
                            nodes,
                            edges,
                        )

                        streamlit_flow(
                            f"ontology_flow_{selected_id}",
                            flow_state,
                            layout=LayeredLayout(
                                direction="LR"
                            ),
                            fit_view=True,
                            height=470,
                            enable_node_menu=False,
                            enable_edge_menu=False,
                            enable_pane_menu=False,
                            hide_watermark=True,
                        )

    # ------------------------------------------------------------------------
    # Entity details
    # ------------------------------------------------------------------------

    with right:

        st.subheader(
            "Entity Details"
        )

        if selected_id is None:

            st.info(
                "Select an entity."
            )

        else:

            selected_rows = entities[
                entities["ENTITY_ID"]
                == selected_id
            ]

            if selected_rows.empty:

                st.info(
                    "Entity details unavailable."
                )

            else:

                row = selected_rows.iloc[0]

                st.markdown(
                    f"""
                    <div class="ontology-card">
                        <div class="entity-title">
                            {row.get("DISPLAY_NAME", selected_id)}
                        </div>
                        <div class="muted">
                            {row.get("ENTITY_TYPE", "")}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                description = row.get(
                    "DESCRIPTION"
                )

                if pd.notna(description):
                    st.markdown(
                        f"**Description**  \n{description}"
                    )

                business_domain = row.get(
                    "BUSINESS_DOMAIN"
                )

                if pd.notna(business_domain):
                    st.markdown(
                        f"**Business domain:** "
                        f"{business_domain}"
                    )

                source_object = row.get(
                    "SOURCE_OBJECT"
                )

                if pd.notna(source_object):
                    st.markdown(
                        f"**Source:** `{source_object}`"
                    )

    # ------------------------------------------------------------------------
    # Attributes and metrics
    # ------------------------------------------------------------------------

    st.divider()

    detail_left, detail_right = st.columns(
        [1.2, 1]
    )

    with detail_left:

        st.subheader(
            "Attributes"
        )

        if attributes.empty:

            st.info(
                "No ontology attributes available."
            )

        else:

            entity_attributes = attributes[
                attributes["ENTITY_ID"]
                == selected_id
            ]

            if entity_attributes.empty:

                st.info(
                    "No attributes defined for this entity."
                )

            else:

                display_cols = [
                    col
                    for col in [
                        "DISPLAY_NAME",
                        "ATTRIBUTE_NAME",
                        "DATA_TYPE",
                        "DESCRIPTION",
                        "IS_KEY",
                        "IS_FOREIGN_KEY",
                    ]
                    if col
                    in entity_attributes.columns
                ]

                st.dataframe(
                    entity_attributes[
                        display_cols
                    ],
                    width="stretch",
                    hide_index=True,
                )

    with detail_right:

        st.subheader(
            "Governed Metrics"
        )

        if entity_metrics.empty:

            st.info(
                "No governed metrics mapped to this entity."
            )

        else:

            entity_metric_rows = entity_metrics[
                entity_metrics["ENTITY_ID"]
                == selected_id
            ]

            if entity_metric_rows.empty:

                st.info(
                    "No governed metrics mapped to this entity."
                )

            else:

                for _, metric in entity_metric_rows.iterrows():

                    metric_name = metric.get(
                        "DISPLAY_NAME",
                        metric.get(
                            "METRIC_NAME",
                            "Metric",
                        ),
                    )

                    definition = metric.get(
                        "DEFINITION",
                        metric.get(
                            "DESCRIPTION",
                            "",
                        ),
                    )

                    canonical_object = metric.get(
                        "CANONICAL_OBJECT",
                        "",
                    )

                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="m-name">
                                {metric_name}
                            </div>
                            <div class="m-def">
                                {definition or ""}
                            </div>
                            <div class="m-view">
                                {canonical_object or ""}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


# ============================================================================
# TEXT-ONLY CHAT COMPOSER
# ============================================================================
#
# Voice/audio/transcription has intentionally been removed.
#
# The composer follows:
#
#   User types question
#          ↓
#   Enter OR Send
#          ↓
#   submit_question()
#          ↓
#   Cortex Agent streaming
#
# There is intentionally NO:
#
#   st.audio_input()
#   transcribe_audio()
#   voice_in
#   last_audio_file_id
#   audio transcription state
#   audio CSS
#
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

    with composer_dock:

        with st.container(
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
                    on_change=lambda: (
                        st.session_state.update(
                            enter_pressed=True
                        )
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

        q = (
            st.session_state
            .composer_input
            .strip()
        )

        # Clear the input on the next Streamlit run.
        st.session_state.pending_clear = True

        submit_question(q)

        st.rerun()