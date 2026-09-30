import json

import altair as alt
import pandas as pd
import streamlit as st

from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowEdge, StreamlitFlowNode
from streamlit_flow.layouts import LayeredLayout

from cortex_agent import (
    AgentResult,
    ask_agent_stream,
    create_thread,
    delete_thread,
    get_thread_messages,
    list_threads,
    run_sql,
    set_thread_name,
)


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Supply Chain Copilot",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# CSS
# =============================================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 6rem;
        max-width: 1600px;
    }

    .main-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.1rem;
    }

    .subtitle {
        color: #777;
        margin-bottom: 1.2rem;
    }

    .kpi-card {
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 12px;
        padding: 18px;
        min-height: 125px;
        background: rgba(128,128,128,0.04);
    }

    .kpi-title {
        font-size: 0.85rem;
        color: #777;
        margin-bottom: 8px;
    }

    .kpi-value {
        font-size: 1.8rem;
        font-weight: 700;
    }

    .kpi-unit {
        font-size: 0.8rem;
        color: #888;
    }

    .persona-card {
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 12px;
        padding: 14px;
        margin-bottom: 8px;
    }

    .entity-title {
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 4px;
    }

    .muted {
        color: #777;
    }

    .edge {
        padding: 8px 10px;
        border-bottom: 1px solid rgba(128,128,128,0.15);
    }

    .chat-user {
        background: rgba(70,120,200,0.08);
        border-radius: 12px;
        padding: 12px 15px;
        margin: 6px 0;
    }

    .chat-assistant {
        background: rgba(128,128,128,0.08);
        border-radius: 12px;
        padding: 12px 15px;
        margin: 6px 0;
    }

    div[data-testid="stChatMessage"] {
        border-radius: 12px;
    }

    .section-title {
        font-size: 1.25rem;
        font-weight: 650;
        margin-bottom: 0.5rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# SESSION STATE
# =============================================================================

DEFAULT_STATE = {
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


for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =============================================================================
# CONSTANTS
# =============================================================================

PERSONAS = {
    "Planning": {
        "icon": "📊",
        "description": (
            "Inventory, service levels, fulfillment and planning performance."
        ),
    },
    "Procurement": {
        "icon": "🛒",
        "description": (
            "Supplier performance, inbound delivery and landed cost."
        ),
    },
    "Logistics": {
        "icon": "🚚",
        "description": (
            "Outbound delivery, carriers, modes and shipment performance."
        ),
    },
}


# =============================================================================
# GENERIC SQL HELPERS
# =============================================================================

def sql_df(statement: str) -> pd.DataFrame:
    """
    Execute SQL through cortex_agent.run_sql() and return a DataFrame.

    Authentication is handled entirely inside cortex_agent.py.
    """

    try:
        columns, rows = run_sql(statement)

        if not columns:
            return pd.DataFrame()

        return pd.DataFrame(rows, columns=columns)

    except Exception as exc:
        st.error(f"SQL query failed: {exc}")
        return pd.DataFrame()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Snowflake column names for easier UI handling."""

    if df.empty:
        return df

    out = df.copy()
    out.columns = [
        str(col).strip().upper()
        for col in out.columns
    ]

    return out


# =============================================================================
# DASHBOARD QUERIES
# =============================================================================

def load_kpi_snapshot() -> pd.DataFrame:
    return normalize_columns(
        sql_df(
            """
            SELECT
                METRIC_ID,
                METRIC_NAME,
                METRIC_VALUE,
                UNIT
            FROM SUPPLY_CHAIN_ONTOLOGY.CORE.V_DASHBOARD_KPI_SNAPSHOT
            ORDER BY METRIC_ID
            """
        )
    )


def load_planning_data() -> dict[str, pd.DataFrame]:

    return {
        "otd_trend": normalize_columns(
            sql_df(
                """
                SELECT
                    DATE_TRUNC('MONTH', DELIVERY_DATE) AS MONTH,
                    AVG(IS_ON_TIME) AS OTD
                FROM SUPPLY_CHAIN_ONTOLOGY.CORE.VW_ON_TIME_DELIVERY_OUTBOUND
                GROUP BY 1
                ORDER BY 1
                """
            )
        ),

        "fill_rate": normalize_columns(
            sql_df(
                """
                SELECT
                    DATE_TRUNC('MONTH', ORDER_DATE) AS MONTH,
                    AVG(FILL_RATE) AS FILL_RATE
                FROM SUPPLY_CHAIN_ONTOLOGY.CORE.VW_FILL_RATE
                GROUP BY 1
                ORDER BY 1
                """
            )
        ),

        "dii_plant": normalize_columns(
            sql_df(
                """
                SELECT
                    PLANT,
                    AVG(DAYS_OF_INVENTORY) AS DII
                FROM SUPPLY_CHAIN_ONTOLOGY.CORE.VW_DAYS_OF_INVENTORY
                GROUP BY PLANT
                ORDER BY DII DESC
                """
            )
        ),

        "dii_category": normalize_columns(
            sql_df(
                """
                SELECT
                    PART_CATEGORY,
                    AVG(DAYS_OF_INVENTORY) AS DII
                FROM SUPPLY_CHAIN_ONTOLOGY.CORE.VW_DAYS_OF_INVENTORY
                GROUP BY PART_CATEGORY
                ORDER BY DII DESC
                """
            )
        ),

        "inventory_plant": normalize_columns(
            sql_df(
                """
                SELECT
                    PLANT,
                    SUM(INVENTORY_ON_HAND) AS INVENTORY_ON_HAND
                FROM SUPPLY_CHAIN_ONTOLOGY.CORE.VW_DAYS_OF_INVENTORY
                GROUP BY PLANT
                ORDER BY INVENTORY_ON_HAND DESC
                """
            )
        ),
    }


def load_procurement_data() -> dict[str, pd.DataFrame]:

    return {
        "supplier_otd": normalize_columns(
            sql_df(
                """
                SELECT
                    SUPPLIER_NAME,
                    AVG(IS_ON_TIME) AS INBOUND_OTD
                FROM SUPPLY_CHAIN_ONTOLOGY.CORE.VW_ON_TIME_DELIVERY_INBOUND
                GROUP BY SUPPLIER_NAME
                ORDER BY INBOUND_OTD
                """
            )
        ),

        "landed_supplier": normalize_columns(
            sql_df(
                """
                SELECT
                    SUPPLIER_NAME,
                    AVG(LANDED_COST) AS LANDED_COST
                FROM SUPPLY_CHAIN_ONTOLOGY.CORE.VW_LANDED_COST
                GROUP BY SUPPLIER_NAME
                ORDER BY LANDED_COST DESC
                """
            )
        ),

        "landed_part": normalize_columns(
            sql_df(
                """
                SELECT
                    PART_NAME,
                    AVG(LANDED_COST) AS LANDED_COST
                FROM SUPPLY_CHAIN_ONTOLOGY.CORE.VW_LANDED_COST
                GROUP BY PART_NAME
                ORDER BY LANDED_COST DESC
                """
            )
        ),

        "landed_breakdown": normalize_columns(
            sql_df(
                """
                SELECT
                    SUPPLIER_NAME,
                    SUM(MATERIAL_COST) AS MATERIAL_COST,
                    SUM(FREIGHT_COST) AS FREIGHT_COST,
                    SUM(DUTY_COST) AS DUTY_COST,
                    SUM(OTHER_COST) AS OTHER_COST
                FROM SUPPLY_CHAIN_ONTOLOGY.CORE.VW_LANDED_COST
                GROUP BY SUPPLIER_NAME
                ORDER BY SUPPLIER_NAME
                """
            )
        ),

        "supplier_reliability": normalize_columns(
            sql_df(
                """
                SELECT
                    SUPPLIER_NAME,
                    AVG(IS_ON_TIME) AS INBOUND_OTD,
                    COUNT(*) AS SHIPMENT_COUNT
                FROM SUPPLY_CHAIN_ONTOLOGY.CORE.VW_ON_TIME_DELIVERY_INBOUND
                GROUP BY SUPPLIER_NAME
                ORDER BY INBOUND_OTD
                """
            )
        ),
    }


def load_logistics_data() -> dict[str, pd.DataFrame]:

    return {
        "otd_trend": normalize_columns(
            sql_df(
                """
                SELECT
                    DATE_TRUNC('MONTH', DELIVERY_DATE) AS MONTH,
                    AVG(IS_ON_TIME) AS OTD
                FROM SUPPLY_CHAIN_ONTOLOGY.CORE.VW_ON_TIME_DELIVERY_OUTBOUND
                GROUP BY 1
                ORDER BY 1
                """
            )
        ),

        "plant_otd": normalize_columns(
            sql_df(
                """
                SELECT
                    PLANT,
                    AVG(IS_ON_TIME) AS OTD
                FROM SUPPLY_CHAIN_ONTOLOGY.CORE.VW_ON_TIME_DELIVERY_OUTBOUND
                GROUP BY PLANT
                ORDER BY OTD
                """
            )
        ),

        "carrier_otd": normalize_columns(
            sql_df(
                """
                SELECT
                    CARRIER,
                    AVG(IS_ON_TIME) AS OTD
                FROM SUPPLY_CHAIN_ONTOLOGY.CORE.VW_ON_TIME_DELIVERY_OUTBOUND
                GROUP BY CARRIER
                ORDER BY OTD
                """
            )
        ),

        "mode_otd": normalize_columns(
            sql_df(
                """
                SELECT
                    TRANSPORT_MODE,
                    AVG(IS_ON_TIME) AS OTD
                FROM SUPPLY_CHAIN_ONTOLOGY.CORE.VW_ON_TIME_DELIVERY_OUTBOUND
                GROUP BY TRANSPORT_MODE
                ORDER BY OTD
                """
            )
        ),

        "late_carriers": normalize_columns(
            sql_df(
                """
                SELECT
                    CARRIER,
                    COUNT(*) AS LATE_SHIPMENTS
                FROM SUPPLY_CHAIN_ONTOLOGY.CORE.VW_ON_TIME_DELIVERY_OUTBOUND
                WHERE IS_ON_TIME = 0
                GROUP BY CARRIER
                ORDER BY LATE_SHIPMENTS DESC
                """
            )
        ),

        "delay_distribution": normalize_columns(
            sql_df(
                """
                SELECT
                    DELAY_DAYS,
                    COUNT(*) AS SHIPMENT_COUNT
                FROM SUPPLY_CHAIN_ONTOLOGY.CORE.VW_ON_TIME_DELIVERY_OUTBOUND
                WHERE IS_ON_TIME = 0
                GROUP BY DELAY_DAYS
                ORDER BY DELAY_DAYS
                """
            )
        ),
    }


# =============================================================================
# ONTOLOGY LOADERS
# =============================================================================

def load_ontology_entities() -> pd.DataFrame:

    return normalize_columns(
        sql_df(
            """
            SELECT *
            FROM SUPPLY_CHAIN_ONTOLOGY.CORE.V_ONTOLOGY_ENTITIES
            ORDER BY DISPLAY_NAME
            """
        )
    )


def load_ontology_relationships() -> pd.DataFrame:

    return normalize_columns(
        sql_df(
            """
            SELECT *
            FROM SUPPLY_CHAIN_ONTOLOGY.CORE.V_ONTOLOGY_GRAPH
            """
        )
    )


def load_ontology_attributes() -> pd.DataFrame:

    return normalize_columns(
        sql_df(
            """
            SELECT *
            FROM SUPPLY_CHAIN_ONTOLOGY.CORE.V_ONTOLOGY_ATTRIBUTES
            """
        )
    )


def load_entity_metrics() -> pd.DataFrame:

    return normalize_columns(
        sql_df(
            """
            SELECT *
            FROM SUPPLY_CHAIN_ONTOLOGY.CORE.V_ONTOLOGY_ENTITY_METRICS
            """
        )
    )


def load_ontology_metrics() -> pd.DataFrame:

    return normalize_columns(
        sql_df(
            """
            SELECT *
            FROM SUPPLY_CHAIN_ONTOLOGY.CORE.V_ONTOLOGY_METRICS
            """
        )
    )


# =============================================================================
# CHART HELPERS
# =============================================================================

def line_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
) -> None:

    if df.empty:
        st.info("No data available.")
        return

    chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X(x, title=x.replace("_", " ").title()),
            y=alt.Y(y, title=y.replace("_", " ").title()),
            tooltip=list(df.columns),
        )
        .properties(
            title=title,
            height=320,
        )
        .interactive()
    )

    st.altair_chart(
        chart,
        use_container_width=True,
    )


def bar_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
) -> None:

    if df.empty:
        st.info("No data available.")
        return

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(x, title=x.replace("_", " ").title()),
            y=alt.Y(y, title=y.replace("_", " ").title()),
            tooltip=list(df.columns),
        )
        .properties(
            title=title,
            height=320,
        )
        .interactive()
    )

    st.altair_chart(
        chart,
        use_container_width=True,
    )


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:

    st.markdown(
        "## 🔗 Supply Chain Copilot"
    )

    st.caption(
        "Governed conversational analytics"
    )

    st.divider()

    st.markdown("### Workspace")

    selected_view = st.radio(
        "View",
        [
            "Dashboard",
            "Ontology Explorer",
            "Copilot",
        ],
        index=[
            "Dashboard",
            "Ontology Explorer",
            "Copilot",
        ].index(
            st.session_state.active_view
        ),
    )

    if selected_view != st.session_state.active_view:
        st.session_state.active_view = selected_view
        st.rerun()

    st.divider()

    st.markdown("### Persona")

    persona = st.radio(
        "Persona",
        list(PERSONAS.keys()),
        index=list(PERSONAS.keys()).index(
            st.session_state.active_persona
        ),
    )

    if persona != st.session_state.active_persona:
        st.session_state.active_persona = persona
        st.rerun()

    p = PERSONAS[persona]

    st.caption(
        f"{p['icon']} {p['description']}"
    )

    st.divider()

    st.caption(
        "Database: SUPPLY_CHAIN_ONTOLOGY"
    )

    st.caption(
        "Schema: CORE"
    )


# =============================================================================
# HEADER
# =============================================================================

st.markdown(
    '<div class="main-title">Supply Chain Copilot</div>',
    unsafe_allow_html=True,
)

st.markdown(
    f'<div class="subtitle">'
    f'{PERSONAS[st.session_state.active_persona]["icon"]} '
    f'{st.session_state.active_persona} workspace'
    f'</div>',
    unsafe_allow_html=True,
)


# =============================================================================
# DASHBOARD
# =============================================================================

def render_dashboard() -> None:

    st.markdown(
        '<div class="section-title">Supply Chain Health</div>',
        unsafe_allow_html=True,
    )

    kpis = load_kpi_snapshot()

    if kpis.empty:
        st.warning(
            "No KPI snapshot data was returned."
        )
    else:

        cols = st.columns(
            min(max(len(kpis), 1), 5)
        )

        for i, (_, row) in enumerate(
            kpis.head(5).iterrows()
        ):

            metric_name = row.get(
                "METRIC_NAME",
                "Metric",
            )

            metric_value = row.get(
                "METRIC_VALUE",
                "",
            )

            unit = row.get(
                "UNIT",
                "",
            )

            with cols[i % len(cols)]:

                st.markdown(
                    f"""
                    <div class="kpi-card">
                        <div class="kpi-title">
                            {metric_name}
                        </div>
                        <div class="kpi-value">
                            {metric_value}
                        </div>
                        <div class="kpi-unit">
                            {unit}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.divider()

    current_persona = st.session_state.active_persona

    if current_persona == "Planning":

        data = load_planning_data()

        c1, c2 = st.columns(2)

        with c1:
            line_chart(
                data["otd_trend"],
                "MONTH",
                "OTD",
                "Outbound OTD Trend",
            )

        with c2:
            line_chart(
                data["fill_rate"],
                "MONTH",
                "FILL_RATE",
                "Fill Rate Trend",
            )

        c3, c4 = st.columns(2)

        with c3:
            bar_chart(
                data["dii_plant"],
                "PLANT",
                "DII",
                "Days of Inventory by Plant",
            )

        with c4:
            bar_chart(
                data["dii_category"],
                "PART_CATEGORY",
                "DII",
                "Days of Inventory by Part Category",
            )

        st.subheader(
            "Inventory on Hand by Plant"
        )

        bar_chart(
            data["inventory_plant"],
            "PLANT",
            "INVENTORY_ON_HAND",
            "Inventory on Hand",
        )

    elif current_persona == "Procurement":

        data = load_procurement_data()

        c1, c2 = st.columns(2)

        with c1:
            bar_chart(
                data["supplier_otd"],
                "SUPPLIER_NAME",
                "INBOUND_OTD",
                "Inbound OTD by Supplier",
            )

        with c2:
            bar_chart(
                data["landed_supplier"],
                "SUPPLIER_NAME",
                "LANDED_COST",
                "Landed Cost by Supplier",
            )

        c3, c4 = st.columns(2)

        with c3:
            bar_chart(
                data["landed_part"],
                "PART_NAME",
                "LANDED_COST",
                "Landed Cost by Part",
            )

        with c4:
            bar_chart(
                data["supplier_reliability"],
                "SUPPLIER_NAME",
                "INBOUND_OTD",
                "Supplier Reliability",
            )

        st.subheader(
            "Landed Cost Breakdown"
        )

        breakdown = data["landed_breakdown"]

        if breakdown.empty:
            st.info(
                "No landed cost breakdown available."
            )
        else:

            melt = breakdown.melt(
                id_vars=["SUPPLIER_NAME"],
                var_name="COST_TYPE",
                value_name="COST",
            )

            chart = (
                alt.Chart(melt)
                .mark_bar()
                .encode(
                    x=alt.X(
                        "SUPPLIER_NAME",
                        title="Supplier",
                    ),
                    y=alt.Y(
                        "COST",
                        title="Cost",
                    ),
                    color=alt.Color(
                        "COST_TYPE",
                        title="Cost Type",
                    ),
                    tooltip=list(melt.columns),
                )
                .properties(
                    height=360,
                )
                .interactive()
            )

            st.altair_chart(
                chart,
                use_container_width=True,
            )

    else:

        data = load_logistics_data()

        c1, c2 = st.columns(2)

        with c1:
            line_chart(
                data["otd_trend"],
                "MONTH",
                "OTD",
                "Outbound OTD Trend",
            )

        with c2:
            bar_chart(
                data["plant_otd"],
                "PLANT",
                "OTD",
                "OTD by Plant",
            )

        c3, c4 = st.columns(2)

        with c3:
            bar_chart(
                data["carrier_otd"],
                "CARRIER",
                "OTD",
                "OTD by Carrier",
            )

        with c4:
            bar_chart(
                data["mode_otd"],
                "TRANSPORT_MODE",
                "OTD",
                "OTD by Transport Mode",
            )

        c5, c6 = st.columns(2)

        with c5:
            bar_chart(
                data["late_carriers"],
                "CARRIER",
                "LATE_SHIPMENTS",
                "Late Shipments by Carrier",
            )

        with c6:
            bar_chart(
                data["delay_distribution"],
                "DELAY_DAYS",
                "SHIPMENT_COUNT",
                "Delivery Delay Distribution",
            )


# =============================================================================
# ONTOLOGY EXPLORER
# =============================================================================

def render_ontology() -> None:

    st.markdown(
        '<div class="section-title">'
        'Ontology Explorer'
        '</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Explore governed entities, relationships, attributes and metrics."
    )

    entities = load_ontology_entities()
    relationships = load_ontology_relationships()
    attributes = load_ontology_attributes()
    entity_metrics = load_entity_metrics()
    metrics = load_ontology_metrics()

    if entities.empty:
        st.warning(
            "No ontology entities were returned."
        )
        return

    entities = entities.copy()

    if "ENTITY_ID" not in entities.columns:
        st.error(
            "V_ONTOLOGY_ENTITIES does not contain ENTITY_ID."
        )
        return

    if "DISPLAY_NAME" not in entities.columns:
        entities["DISPLAY_NAME"] = entities["ENTITY_ID"]

    if "ENTITY_TYPE" not in entities.columns:
        entities["ENTITY_TYPE"] = ""

    entity_types = sorted(
        entities["ENTITY_TYPE"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    c_filter1, c_filter2 = st.columns(2)

    with c_filter1:

        type_filter = st.multiselect(
            "Entity type",
            entity_types,
        )

    with c_filter2:

        search = st.text_input(
            "Search entity",
            placeholder="Supplier, Plant, Part...",
        )

    filtered = entities.copy()

    if type_filter:
        filtered = filtered[
            filtered["ENTITY_TYPE"].isin(
                type_filter
            )
        ]

    if search.strip():

        q = search.strip().lower()

        filtered = filtered[
            filtered.astype(str)
            .apply(
                lambda col: col.str.lower().str.contains(
                    q,
                    na=False,
                )
            )
            .any(axis=1)
        ]

    if filtered.empty:
        st.info(
            "No entities match the current filters."
        )
        return

    labels = {
        row["ENTITY_ID"]: (
            f'{row["DISPLAY_NAME"]} · '
            f'{row["ENTITY_TYPE"]}'
        )
        for _, row in filtered.iterrows()
    }

    entity_ids = list(labels.keys())

    current_entity = st.session_state.active_entity

    if (
        current_entity not in entity_ids
    ):
        current_entity = entity_ids[0]
        st.session_state.active_entity = current_entity

    selected_id = st.selectbox(
        "Entity",
        entity_ids,
        index=entity_ids.index(
            current_entity
        ),
        format_func=lambda x: labels[x],
    )

    st.session_state.active_entity = selected_id

    st.divider()

    left, middle, right = st.columns(
        [0.25, 0.45, 0.30]
    )

    # -------------------------------------------------------------------------
    # LEFT — ENTITY LIST
    # -------------------------------------------------------------------------

    with left:

        st.subheader("Entities")

        for entity_id in entity_ids:

            label = labels[entity_id]

            if st.button(
                label,
                key=f"entity_{entity_id}",
                use_container_width=True,
            ):
                st.session_state.active_entity = entity_id
                st.rerun()

    # -------------------------------------------------------------------------
    # MIDDLE — RELATIONSHIPS
    # -------------------------------------------------------------------------

    with middle:

        st.subheader("Relationship Graph")

        selected_name = entities.loc[
            entities["ENTITY_ID"] == selected_id,
            "DISPLAY_NAME",
        ].iloc[0]

        st.markdown(
            f"**{selected_name}**"
        )

        if relationships.empty:

            st.info(
                "No relationships registered."
            )

        else:

            outgoing = relationships[
                relationships.get(
                    "FROM_ENTITY_ID",
                    pd.Series(dtype=str),
                )
                == selected_id
            ]

            incoming = relationships[
                relationships.get(
                    "TO_ENTITY_ID",
                    pd.Series(dtype=str),
                )
                == selected_id
            ]

            if outgoing.empty and incoming.empty:

                st.info(
                    "No relationships defined."
                )

            else:

                for _, row in outgoing.iterrows():

                    rel = row.get(
                        "RELATIONSHIP_NAME",
                        "relationship",
                    )

                    target = row.get(
                        "TO_ENTITY_ID",
                        "",
                    )

                    cardinality = row.get(
                        "CARDINALITY",
                        "",
                    )

                    st.markdown(
                        f"""
                        <div class="edge">
                            → <b>{rel}</b> → {target}
                            <span class="muted">
                                ({cardinality})
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                for _, row in incoming.iterrows():

                    rel = row.get(
                        "RELATIONSHIP_NAME",
                        "relationship",
                    )

                    source = row.get(
                        "FROM_ENTITY_ID",
                        "",
                    )

                    cardinality = row.get(
                        "CARDINALITY",
                        "",
                    )

                    st.markdown(
                        f"""
                        <div class="edge">
                            ← <b>{rel}</b> ← {source}
                            <span class="muted">
                                ({cardinality})
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    # -------------------------------------------------------------------------
    # RIGHT — DETAILS
    # -------------------------------------------------------------------------

    with right:

        st.subheader("Details")

        entity_rows = entities[
            entities["ENTITY_ID"] == selected_id
        ]

        if not entity_rows.empty:

            entity = entity_rows.iloc[0]

            st.markdown(
                f'<div class="entity-title">'
                f'{entity.get("DISPLAY_NAME", selected_id)}'
                f'</div>',
                unsafe_allow_html=True,
            )

            st.caption(
                str(
                    entity.get(
                        "ENTITY_TYPE",
                        "",
                    )
                )
            )

            description = entity.get(
                "DESCRIPTION",
                "",
            )

            if pd.notna(description):
                st.write(description)

            source = entity.get(
                "SOURCE_OBJECT",
                "",
            )

            if pd.notna(source) and source:

                st.markdown("**Source**")

                st.code(
                    str(source),
                    language="text",
                )

            domain = entity.get(
                "BUSINESS_DOMAIN",
                "",
            )

            if pd.notna(domain) and domain:

                st.markdown(
                    "**Business domain**"
                )

                st.write(domain)

            st.markdown(
                "**Attributes**"
            )

            if attributes.empty:

                st.caption(
                    "No attributes registered yet."
                )

            else:

                attrs = attributes[
                    attributes.get(
                        "ENTITY_ID",
                        pd.Series(dtype=str),
                    )
                    == selected_id
                ]

                if attrs.empty:

                    st.caption(
                        "No attributes registered yet."
                    )

                else:

                    for _, attr in attrs.iterrows():

                        attr_name = attr.get(
                            "ATTRIBUTE_NAME",
                            "",
                        )

                        data_type = attr.get(
                            "DATA_TYPE",
                            "",
                        )

                        flags = []

                        if attr.get(
                            "IS_KEY",
                            False,
                        ):
                            flags.append("KEY")

                        if attr.get(
                            "IS_FOREIGN_KEY",
                            False,
                        ):
                            flags.append("FK")

                        flag_text = (
                            " · "
                            + " / ".join(flags)
                            if flags
                            else ""
                        )

                        st.markdown(
                            f"`{attr_name}` · "
                            f"{data_type}"
                            f"{flag_text}"
                        )

    # -------------------------------------------------------------------------
    # GOVERNED METRICS
    # -------------------------------------------------------------------------

    st.divider()

    st.subheader(
        "Governed Metrics"
    )

    if entity_metrics.empty:

        st.caption(
            "No entity-metric mappings available."
        )

    else:

        em = entity_metrics[
            entity_metrics.get(
                "ENTITY_ID",
                pd.Series(dtype=str),
            )
            == selected_id
        ]

        if em.empty:

            st.caption(
                "No governed metrics mapped to this entity."
            )

        else:

            st.dataframe(
                em,
                use_container_width=True,
                hide_index=True,
            )

    # -------------------------------------------------------------------------
    # METRIC CATALOG
    # -------------------------------------------------------------------------

    with st.expander(
        "Metric Catalog",
        expanded=False,
    ):

        if metrics.empty:

            st.info(
                "No governed metrics available."
            )

        else:

            st.dataframe(
                metrics,
                use_container_width=True,
                hide_index=True,
            )


# =============================================================================
# CHAT
# =============================================================================

def initialize_thread() -> bool:

    if st.session_state.thread_id is not None:
        return True

    try:

        thread_id = create_thread()

        st.session_state.thread_id = thread_id
        st.session_state.parent_message_id = None

        return True

    except Exception as exc:

        st.error(
            f"Unable to create Cortex thread: {exc}"
        )

        return False


def load_existing_thread(
    thread_id: int,
) -> None:

    try:

        messages = get_thread_messages(
            thread_id
        )

        chat = []

        for message in messages:

            role = message.get(
                "role",
                "assistant",
            )

            content = message.get(
                "content",
                [],
            )

            text_parts = []

            if isinstance(content, list):

                for item in content:

                    if (
                        isinstance(item, dict)
                        and item.get("type") == "text"
                    ):
                        text_parts.append(
                            item.get(
                                "text",
                                "",
                            )
                        )

            elif isinstance(content, str):

                text_parts.append(content)

            text_value = "\n".join(
                x for x in text_parts if x
            )

            if text_value:

                chat.append(
                    {
                        "role": role,
                        "content": text_value,
                    }
                )

        st.session_state.chat_messages = chat
        st.session_state.thread_id = thread_id

    except Exception as exc:

        st.error(
            f"Unable to load thread: {exc}"
        )


def render_chat_history() -> None:

    for message in st.session_state.chat_messages:

        role = message.get(
            "role",
            "assistant",
        )

        content = message.get(
            "content",
            "",
        )

        with st.chat_message(
            "user" if role == "user"
            else "assistant"
        ):
            st.markdown(content)


def submit_question(
    question: str,
) -> None:

    question = question.strip()

    if not question:
        return

    if not initialize_thread():
        return

    contextual_question = (
        f"Current user persona: "
        f"{st.session_state.active_persona}.\n\n"
        f"User question:\n{question}"
    )

    st.session_state.chat_messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        response_placeholder = st.empty()

        generated_text = ""

        try:

            generator = ask_agent_stream(
                contextual_question,
                thread_id=st.session_state.thread_id,
                parent_message_id=(
                    st.session_state.parent_message_id
                ),
            )

            for chunk in generator:

                generated_text += chunk

                response_placeholder.markdown(
                    generated_text
                )

            result = getattr(
                ask_agent_stream,
                "result",
                AgentResult(),
            )

            if result.error:

                response_placeholder.error(
                    result.error
                )

                generated_text = (
                    f"Sorry, I couldn't complete "
                    f"that request.\n\n"
                    f"**Error:** {result.error}"
                )

            if result.assistant_message_id:

                st.session_state.parent_message_id = (
                    result.assistant_message_id
                )

            if result.thread_id:

                st.session_state.thread_id = (
                    result.thread_id
                )

            if result.generated_sql:

                with st.expander(
                    "Generated SQL",
                    expanded=False,
                ):

                    st.code(
                        result.generated_sql,
                        language="sql",
                    )

            if result.result_rows:

                with st.expander(
                    "Query Results",
                    expanded=False,
                ):

                    result_df = pd.DataFrame(
                        result.result_rows,
                        columns=(
                            result.column_names
                            if result.column_names
                            else None
                        ),
                    )

                    st.dataframe(
                        result_df,
                        use_container_width=True,
                        hide_index=True,
                    )

            if result.chart_spec:

                try:

                    chart = alt.Chart.from_dict(
                        result.chart_spec
                    )

                    st.altair_chart(
                        chart,
                        use_container_width=True,
                    )

                except Exception:
                    pass

            if result.warnings:

                for warning in result.warnings:
                    st.warning(warning)

        except Exception as exc:

            generated_text = (
                f"Sorry, something went wrong: {exc}"
            )

            response_placeholder.error(
                generated_text
            )

    st.session_state.chat_messages.append(
        {
            "role": "assistant",
            "content": generated_text,
        }
    )


def render_copilot() -> None:

    st.markdown(
        '<div class="section-title">'
        'Supply Chain Copilot'
        '</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        f"Persona: "
        f"{PERSONAS[st.session_state.active_persona]['icon']} "
        f"{st.session_state.active_persona}"
    )

    # -------------------------------------------------------------------------
    # THREAD CONTROLS
    # -------------------------------------------------------------------------

    with st.expander(
        "Conversation history",
        expanded=False,
    ):

        try:

            threads = list_threads()

        except Exception as exc:

            threads = []

            st.caption(
                f"Unable to load previous conversations: {exc}"
            )

        if threads:

            thread_options = {
                t.get(
                    "thread_id"
                ): (
                    t.get(
                        "thread_name"
                    )
                    or f"Conversation {t.get('thread_id')}"
                )
                for t in threads
                if t.get("thread_id") is not None
            }

            current_thread = (
                st.session_state.thread_id
            )

            selected_thread = st.selectbox(
                "Conversation",
                list(thread_options.keys()),
                index=(
                    list(thread_options.keys()).index(
                        current_thread
                    )
                    if current_thread in thread_options
                    else 0
                ),
                format_func=lambda x: thread_options[x],
            )

            c1, c2 = st.columns(2)

            with c1:

                if st.button(
                    "Load",
                    use_container_width=True,
                ):

                    load_existing_thread(
                        selected_thread
                    )

                    st.rerun()

            with c2:

                if st.button(
                    "Delete",
                    use_container_width=True,
                ):

                    try:

                        delete_thread(
                            selected_thread
                        )

                        if (
                            selected_thread
                            == st.session_state.thread_id
                        ):

                            st.session_state.thread_id = None
                            st.session_state.parent_message_id = None
                            st.session_state.chat_messages = []

                        st.rerun()

                    except Exception as exc:

                        st.error(
                            f"Unable to delete thread: {exc}"
                        )

        else:

            st.caption(
                "No previous conversations."
            )

        if st.button(
            "＋ New conversation",
            use_container_width=True,
        ):

            st.session_state.thread_id = None
            st.session_state.parent_message_id = None
            st.session_state.chat_messages = []

            st.rerun()

    # -------------------------------------------------------------------------
    # CHAT HISTORY
    # -------------------------------------------------------------------------

    render_chat_history()

    # -------------------------------------------------------------------------
    # SUGGESTIONS
    # -------------------------------------------------------------------------

    if not st.session_state.chat_messages:

        st.markdown(
            "### Try asking"
        )

        suggestions = [
            "What is our outbound OTD?",
            "Show outbound OTD by plant.",
            "Which suppliers have poor inbound OTD?",
            "What is the fill rate by plant?",
            "Which parts have more than 45 days of inventory?",
            "Why has DII increased?",
            "What is landed cost by supplier?",
            "Prove that OTD is consistent across personas.",
            "Give me a supply-chain health summary.",
        ]

        suggestion_cols = st.columns(3)

        for i, suggestion in enumerate(
            suggestions
        ):

            with suggestion_cols[
                i % 3
            ]:

                if st.button(
                    suggestion,
                    key=f"suggestion_{i}",
                    use_container_width=True,
                ):

                    st.session_state.pending_question = (
                        suggestion
                    )

                    st.rerun()

    # -------------------------------------------------------------------------
    # INPUT
    # -------------------------------------------------------------------------

    prompt = st.chat_input(
        "Ask a supply-chain question..."
    )

    if prompt:

        submit_question(prompt)
        st.rerun()

    if st.session_state.pending_question:

        question = (
            st.session_state.pending_question
        )

        st.session_state.pending_question = None

        submit_question(question)
        st.rerun()


# =============================================================================
# ROUTER
# =============================================================================

if st.session_state.active_view == "Dashboard":

    render_dashboard()

elif st.session_state.active_view == "Ontology Explorer":

    render_ontology()

else:

    render_copilot()