"""
Cortex Agent client for Supply Chain Copilot.

Authentication architecture
---------------------------
1. Direct SQL:
   - Snowflake Python Connector
   - Username + password
   - No PAT
   - No private key

2. Cortex Agent / Thread REST APIs:
   - Key-pair JWT authentication
   - No PAT
   - Required for Streamlit Community Cloud until OAuth is configured

3. Voice:
   - Disabled

Configuration
-------------
[snowflake]
account = "YOUR_ACCOUNT"
account_url = "https://YOUR_ACCOUNT.snowflakecomputing.com"
user = "YOUR_USERNAME"
password = "YOUR_PASSWORD"
warehouse = "YOUR_WAREHOUSE"
database = "SUPPLY_CHAIN_ONTOLOGY"
schema = "CORE"
agent = "SUPPLY_CHAIN_COPILOT"

private_key_passphrase = "..."  # optional
"""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
import requests
import snowflake.connector
import streamlit as st

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    PublicFormat,
    load_pem_private_key,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

def _cfg() -> dict:
    """
    Read Snowflake configuration from Streamlit secrets.

    SQL authentication uses:
        account
        user
        password

    Cortex Agent REST authentication additionally uses:
        private_key
        private_key_passphrase
    """

    s = st.secrets["snowflake"]

    return {
        "account": s["account"],
        "account_url": s["account_url"].rstrip("/"),
        "user": s["user"],
        "password": s["password"],
        "warehouse": s["warehouse"],
        "database": s["database"],
        "schema": s["schema"],
        "agent": s["agent"],
        "private_key": s.get("private_key"),
        "private_key_passphrase": s.get(
            "private_key_passphrase",
            None,
        ),
    }


# =============================================================================
# KEY-PAIR AUTHENTICATION
# =============================================================================
#
# This section is ONLY used by:
#   - Cortex Agent REST API
#   - Cortex thread REST API
#
# Direct SQL does NOT use this key.
# Direct SQL uses username/password below.
# =============================================================================

@st.cache_resource
def _private_key():
    """
    Load RSA private key for Cortex Agent REST authentication.

    This is intentionally NOT used for run_sql().
    """

    c = _cfg()

    pem_data = c.get("private_key")

    if not pem_data:
        raise RuntimeError(
            "Cortex Agent REST authentication requires "
            "'private_key' in [snowflake] secrets."
        )

    if isinstance(pem_data, str):
        pem_data = pem_data.encode("utf-8")

    passphrase = c.get(
        "private_key_passphrase"
    )

    if passphrase:
        passphrase = passphrase.encode("utf-8")

    return load_pem_private_key(
        pem_data,
        password=passphrase,
        backend=default_backend(),
    )


@st.cache_resource
def _public_key_fingerprint() -> str:
    """
    Return Snowflake SHA256 public-key fingerprint.
    """

    private_key = _private_key()

    public_key_raw = (
        private_key
        .public_key()
        .public_bytes(
            Encoding.DER,
            PublicFormat.SubjectPublicKeyInfo,
        )
    )

    sha256hash = hashlib.sha256()
    sha256hash.update(public_key_raw)

    return (
        "SHA256:"
        + base64.b64encode(
            sha256hash.digest()
        ).decode("utf-8")
    )


def _qualified_username() -> str:
    """
    Build Snowflake ACCOUNT.USER identifier used by JWT.
    """

    c = _cfg()

    account = c["account"].strip()

    if ".global" not in account.lower():

        if "." in account:
            account = account.split(
                ".",
                1,
            )[0]

    account = (
        account
        .replace(".", "-")
        .upper()
    )

    user = c["user"].upper()

    return f"{account}.{user}"


@st.cache_data(ttl=3000)
def _bearer() -> tuple[str, str]:
    """
    Generate short-lived Snowflake key-pair JWT.

    Used ONLY for Cortex REST APIs.
    """

    private_key = _private_key()

    qualified_username = (
        _qualified_username()
    )

    public_key_fp = (
        _public_key_fingerprint()
    )

    now = datetime.now(
        timezone.utc
    )

    lifetime = timedelta(
        minutes=59
    )

    payload = {
        "iss": (
            qualified_username
            + "."
            + public_key_fp
        ),
        "sub": qualified_username,
        "iat": now,
        "exp": now + lifetime,
    }

    token = jwt.encode(
        payload,
        key=private_key,
        algorithm="RS256",
    )

    if isinstance(token, bytes):
        token = token.decode("utf-8")

    return (
        token,
        "KEYPAIR_JWT",
    )


# =============================================================================
# CORTEX AGENT REST CONFIGURATION
# =============================================================================

def _run_url(c: dict) -> str:

    return (
        f"{c['account_url']}/api/v2/databases/"
        f"{c['database']}/schemas/"
        f"{c['schema']}/agents/"
        f"{c['agent']}:run"
    )


def _headers(
    c: dict,
    streaming: bool,
) -> dict:

    token, token_type = _bearer()

    return {
        "Authorization": (
            f"Bearer {token}"
        ),
        "Content-Type": "application/json",
        "Accept": (
            "text/event-stream"
            if streaming
            else "application/json"
        ),
        "X-Snowflake-Authorization-Token-Type": (
            token_type
        ),
    }


# =============================================================================
# RESULT OBJECT
# =============================================================================

@dataclass
class AgentResult:

    answer_text: str = ""

    generated_sql: str = ""

    query_id: Optional[str] = None

    result_rows: list[list] = field(
        default_factory=list
    )

    column_names: list[str] = field(
        default_factory=list
    )

    tools_used: list[str] = field(
        default_factory=list
    )

    citations: list[dict] = field(
        default_factory=list
    )

    warnings: list[str] = field(
        default_factory=list
    )

    chart_spec: Optional[dict] = None

    thread_id: Optional[int] = None

    assistant_message_id: Optional[int] = None

    error: Optional[str] = None


# =============================================================================
# RESULT SET PARSER
# =============================================================================

def _load_result_set(
    rs: dict,
    result: AgentResult,
) -> None:

    if not rs:
        return

    data = (
        rs.get("data", [])
        or result.result_rows
    )

    if data:
        result.result_rows = data

    metadata = rs.get(
        "resultSetMetaData",
        {},
    )

    row_type = metadata.get(
        "rowType",
        [],
    )

    columns = []

    for column in row_type:

        if isinstance(column, dict):

            columns.append(
                column.get(
                    "name",
                    "",
                )
            )

    if columns:
        result.column_names = columns


# =============================================================================
# RESPONSE PARSER
# =============================================================================

def _extract_from_response_payload(
    data: dict,
    result: AgentResult,
) -> None:
    """
    Extract useful information from Cortex Agent response.
    """

    if not isinstance(data, dict):
        return

    # -------------------------------------------------------------------------
    # Content
    # -------------------------------------------------------------------------

    for item in data.get(
        "content",
        [],
    ):

        if not isinstance(item, dict):
            continue

        item_type = item.get(
            "type"
        )

        # ---------------------------------------------------------------------
        # Text
        # ---------------------------------------------------------------------

        if item_type == "text":

            text_value = item.get(
                "text",
                "",
            )

            if text_value.strip():
                result.answer_text = (
                    text_value
                )

            for annotation in item.get(
                "annotations",
                [],
            ):

                result.citations.append(
                    {
                        "title": annotation.get(
                            "doc_title"
                        ),
                        "text": annotation.get(
                            "text"
                        ),
                        "type": annotation.get(
                            "type"
                        ),
                    }
                )

        # ---------------------------------------------------------------------
        # Table
        # ---------------------------------------------------------------------

        elif item_type == "table":

            _load_result_set(
                item.get(
                    "result_set",
                    {},
                ),
                result,
            )

            if item.get(
                "query_id"
            ):
                result.query_id = (
                    item["query_id"]
                )

        # ---------------------------------------------------------------------
        # Chart
        # ---------------------------------------------------------------------

        elif item_type == "chart":

            chart = item.get(
                "chart",
                {},
            )

            spec = chart.get(
                "chart_spec"
            )

            if isinstance(
                spec,
                str,
            ):

                try:

                    result.chart_spec = (
                        json.loads(spec)
                    )

                except json.JSONDecodeError:
                    pass

            elif isinstance(
                spec,
                dict,
            ):

                result.chart_spec = spec

        # ---------------------------------------------------------------------
        # Tool use
        # ---------------------------------------------------------------------

        elif item_type == "tool_use":

            tool_use = item.get(
                "tool_use",
                item,
            )

            name = (
                tool_use.get("name")
                or tool_use.get("type")
                or "tool"
            )

            result.tools_used.append(
                name
            )

        # ---------------------------------------------------------------------
        # Tool result
        # ---------------------------------------------------------------------

        elif item_type == "tool_result":

            tool_result = item.get(
                "tool_result",
                item,
            )

            for sub_content in tool_result.get(
                "content",
                [],
            ):

                if not isinstance(
                    sub_content,
                    dict,
                ):
                    continue

                payload = sub_content.get(
                    "json"
                )

                if not isinstance(
                    payload,
                    dict,
                ):
                    continue

                if payload.get("sql"):
                    result.generated_sql = (
                        payload["sql"]
                    )

                if payload.get("query_id"):
                    result.query_id = (
                        payload["query_id"]
                    )

                if payload.get(
                    "result_set"
                ):

                    _load_result_set(
                        payload[
                            "result_set"
                        ],
                        result,
                    )

    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------

    metadata = data.get(
        "metadata",
        {},
    )

    if metadata.get(
        "thread_id"
    ) is not None:

        result.thread_id = (
            metadata["thread_id"]
        )

    if metadata.get(
        "assistant_message_id"
    ) is not None:

        result.assistant_message_id = (
            metadata[
                "assistant_message_id"
            ]
        )

    # -------------------------------------------------------------------------
    # Warnings
    # -------------------------------------------------------------------------

    for warning in data.get(
        "warnings",
        [],
    ):

        if isinstance(
            warning,
            dict,
        ):

            message = warning.get(
                "message",
                "",
            )

            if message:
                result.warnings.append(
                    message
                )


# =============================================================================
# SSE PARSER
# =============================================================================

def _parse_sse(
    response: requests.Response,
) -> AgentResult:

    result = AgentResult()

    response.encoding = "utf-8"

    event_type = None

    for raw in response.iter_lines(
        decode_unicode=True
    ):

        if not raw:

            event_type = None
            continue

        if raw.startswith(
            "event:"
        ):

            event_type = raw[
                len("event:"):
            ].strip()

            continue

        if not raw.startswith(
            "data:"
        ):
            continue

        payload = raw[
            len("data:"):
        ].strip()

        try:

            data = json.loads(
                payload
            )

        except json.JSONDecodeError:

            continue

        # ---------------------------------------------------------------------
        # Analyst delta
        # ---------------------------------------------------------------------

        if (
            event_type
            == "response.tool_result.analyst.delta"
        ):

            delta = data.get(
                "delta",
                {},
            )

            if delta.get("sql"):
                result.generated_sql = (
                    delta["sql"]
                )

            if delta.get("query_id"):
                result.query_id = (
                    delta["query_id"]
                )

            if delta.get(
                "result_set"
            ):

                _load_result_set(
                    delta[
                        "result_set"
                    ],
                    result,
                )

        # ---------------------------------------------------------------------
        # Text
        # ---------------------------------------------------------------------

        elif (
            event_type
            == "response.text.delta"
        ):

            text_value = data.get(
                "text",
                "",
            )

            if text_value:
                result.answer_text += (
                    text_value
                )

        # ---------------------------------------------------------------------
        # Table
        # ---------------------------------------------------------------------

        elif (
            event_type
            == "response.table"
        ):

            if data.get(
                "query_id"
            ):

                result.query_id = (
                    data["query_id"]
                )

            _load_result_set(
                data.get(
                    "result_set",
                    {},
                ),
                result,
            )

        # ---------------------------------------------------------------------
        # Chart
        # ---------------------------------------------------------------------

        elif (
            event_type
            == "response.chart"
        ):

            spec = data.get(
                "chart_spec"
            )

            if isinstance(
                spec,
                str,
            ):

                try:

                    result.chart_spec = (
                        json.loads(spec)
                    )

                except json.JSONDecodeError:
                    pass

            elif isinstance(
                spec,
                dict,
            ):

                result.chart_spec = spec

        # ---------------------------------------------------------------------
        # Tool
        # ---------------------------------------------------------------------

        elif (
            event_type
            == "response.tool_use"
        ):

            result.tools_used.append(
                data.get(
                    "name"
                )
                or data.get(
                    "type",
                    "tool",
                )
            )

        # ---------------------------------------------------------------------
        # Warning
        # ---------------------------------------------------------------------

        elif (
            event_type
            == "response.warning"
        ):

            message = data.get(
                "message",
                "",
            )

            if message:
                result.warnings.append(
                    message
                )

        # ---------------------------------------------------------------------
        # Full response
        # ---------------------------------------------------------------------

        elif (
            event_type
            == "response"
        ):

            _extract_from_response_payload(
                data,
                result,
            )

    return result


# =============================================================================
# ASK AGENT
# =============================================================================

def ask_agent(
    question: str,
    *,
    thread_id: int | None = None,
    parent_message_id: int | None = None,
    stream: bool = False,
    timeout: int = 120,
) -> AgentResult:
    """
    Send a question to Cortex Agent.
    """

    try:

        c = _cfg()

    except Exception as exc:

        return AgentResult(
            error=(
                "Snowflake configuration error: "
                f"{exc}"
            )
        )

    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": question,
                    }
                ],
            }
        ],
        "stream": stream,
    }

    if thread_id is not None:

        body["thread_id"] = (
            thread_id
        )

        body["parent_message_id"] = (
            parent_message_id
            if parent_message_id is not None
            else 0
        )

    try:

        response = requests.post(
            _run_url(c),
            headers=_headers(
                c,
                stream,
            ),
            json=body,
            stream=stream,
            timeout=timeout,
        )

        response.raise_for_status()

    except requests.HTTPError as exc:

        detail = ""

        try:
            detail = response.text[
                :1000
            ]

        except Exception:
            pass

        return AgentResult(
            error=(
                "Agent request failed "
                f"({response.status_code}): "
                f"{detail or exc}"
            )
        )

    except requests.RequestException as exc:

        return AgentResult(
            error=(
                "Could not reach Snowflake: "
                f"{exc}"
            )
        )

    if stream:

        return _parse_sse(
            response
        )

    result = AgentResult()

    try:

        payload = response.json()

    except ValueError as exc:

        result.error = (
            "Snowflake returned an invalid "
            f"JSON response: {exc}"
        )

        return result

    _extract_from_response_payload(
        payload,
        result,
    )

    return result


# =============================================================================
# STREAMING AGENT
# =============================================================================

def ask_agent_stream(
    question: str,
    *,
    thread_id: int | None = None,
    parent_message_id: int | None = None,
    timeout: int = 300,
):
    """
    Generator used by Streamlit for live Agent streaming.

    The final AgentResult is exposed as:
        ask_agent_stream.result
    """

    result = AgentResult()

    try:

        c = _cfg()

    except Exception as exc:

        result.error = (
            "Snowflake configuration error: "
            f"{exc}"
        )

        ask_agent_stream.result = (
            result
        )

        return

    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": question,
                    }
                ],
            }
        ],
        "stream": True,
    }

    if thread_id is not None:

        body["thread_id"] = (
            thread_id
        )

        body["parent_message_id"] = (
            parent_message_id
            if parent_message_id is not None
            else 0
        )

    try:

        response = requests.post(
            _run_url(c),
            headers=_headers(
                c,
                True,
            ),
            json=body,
            stream=True,
            timeout=timeout,
        )

        response.raise_for_status()

    except requests.HTTPError as exc:

        detail = ""

        try:
            detail = response.text[
                :1000
            ]

        except Exception:
            pass

        result.error = (
            "Agent request failed "
            f"({response.status_code}): "
            f"{detail or exc}"
        )

        ask_agent_stream.result = (
            result
        )

        return

    except requests.RequestException as exc:

        result.error = (
            "Could not reach Snowflake: "
            f"{exc}"
        )

        ask_agent_stream.result = (
            result
        )

        return

    response.encoding = "utf-8"

    event_type = None

    for raw in response.iter_lines(
        decode_unicode=True
    ):

        if not raw:

            event_type = None
            continue

        if raw.startswith(
            "event:"
        ):

            event_type = raw[
                len("event:"):
            ].strip()

            continue

        if not raw.startswith(
            "data:"
        ):

            continue

        payload = raw[
            len("data:"):
        ].strip()

        try:

            data = json.loads(
                payload
            )

        except json.JSONDecodeError:

            continue

        # ---------------------------------------------------------------------
        # Text delta
        # ---------------------------------------------------------------------

        if (
            event_type
            == "response.text.delta"
        ):

            chunk = data.get(
                "text",
                "",
            )

            if chunk:

                result.answer_text += (
                    chunk
                )

                yield chunk

        # ---------------------------------------------------------------------
        # Analyst result
        # ---------------------------------------------------------------------

        elif (
            event_type
            == "response.tool_result.analyst.delta"
        ):

            delta = data.get(
                "delta",
                {},
            )

            if delta.get("sql"):
                result.generated_sql = (
                    delta["sql"]
                )

            if delta.get("query_id"):
                result.query_id = (
                    delta["query_id"]
                )

            if delta.get(
                "result_set"
            ):

                _load_result_set(
                    delta[
                        "result_set"
                    ],
                    result,
                )

        # ---------------------------------------------------------------------
        # Table
        # ---------------------------------------------------------------------

        elif (
            event_type
            == "response.table"
        ):

            if data.get(
                "query_id"
            ):

                result.query_id = (
                    data["query_id"]
                )

            _load_result_set(
                data.get(
                    "result_set",
                    {},
                ),
                result,
            )

        # ---------------------------------------------------------------------
        # Chart
        # ---------------------------------------------------------------------

        elif (
            event_type
            == "response.chart"
        ):

            spec = data.get(
                "chart_spec"
            )

            if isinstance(
                spec,
                str,
            ):

                try:

                    result.chart_spec = (
                        json.loads(spec)
                    )

                except json.JSONDecodeError:
                    pass

            elif isinstance(
                spec,
                dict,
            ):

                result.chart_spec = spec

        # ---------------------------------------------------------------------
        # Tool
        # ---------------------------------------------------------------------

        elif (
            event_type
            == "response.tool_use"
        ):

            result.tools_used.append(
                data.get(
                    "name"
                )
                or data.get(
                    "type",
                    "tool",
                )
            )

        # ---------------------------------------------------------------------
        # Warning
        # ---------------------------------------------------------------------

        elif (
            event_type
            == "response.warning"
        ):

            message = data.get(
                "message",
                "",
            )

            if message:
                result.warnings.append(
                    message
                )

        # ---------------------------------------------------------------------
        # Full response
        # ---------------------------------------------------------------------

        elif (
            event_type
            == "response"
        ):

            _extract_from_response_payload(
                data,
                result,
            )

    ask_agent_stream.result = (
        result
    )


# =============================================================================
# DIRECT SQL
# =============================================================================
#
# IMPORTANT:
#
# This section does NOT use:
#   - PAT
#   - private key
#   - JWT
#   - SQL REST API
#
# It uses:
#   Snowflake Python Connector
#   username + password
# =============================================================================

@st.cache_resource
def _get_snowflake_connection():
    """
    Create Snowflake connection using username/password.

    This is completely independent of Cortex Agent JWT authentication.
    """

    c = _cfg()

    return snowflake.connector.connect(
        account=c["account"],
        user=c["user"],
        password=c["password"],
        warehouse=c["warehouse"],
        database=c["database"],
        schema=c["schema"],
    )


def run_sql(
    statement: str,
    timeout: int = 60,
) -> tuple[list[str], list[list]]:
    """
    Execute SQL using Snowflake Python Connector.

    Authentication:
        username + password

    No PAT.
    No private key.
    No JWT.
    """

    conn = _get_snowflake_connection()

    cursor = conn.cursor()

    try:

        cursor.execute(
            statement,
            timeout=timeout,
        )

        if cursor.description is None:

            return (
                [],
                [],
            )

        columns = [
            description[0]
            for description
            in cursor.description
        ]

        rows = cursor.fetchall()

        return (
            columns,
            [
                list(row)
                for row in rows
            ],
        )

    finally:

        cursor.close()


# =============================================================================
# GOVERNANCE
# =============================================================================

def call_governance(
    metric_query: str,
) -> dict:
    """
    Return governed metric definition JSON.
    """

    safe = metric_query.replace(
        "'",
        "''",
    )

    _, rows = run_sql(
        f"""
        SELECT GET_METRIC_GOVERNANCE(
            '{safe}'
        )
        """
    )

    if (
        rows
        and rows[0]
        and rows[0][0]
    ):

        try:

            value = rows[0][0]

            if isinstance(
                value,
                str,
            ):

                return json.loads(
                    value
                )

            if isinstance(
                value,
                dict,
            ):

                return value

        except (
            json.JSONDecodeError,
            TypeError,
        ):

            pass

    return {}


def check_persona_consistency(
    metric_query: str,
    plant: str | None = None,
    period: str | None = None,
) -> dict:
    """
    Return persona-consistency proof JSON.
    """

    safe = metric_query.replace(
        "'",
        "''",
    )

    if plant:

        safe_plant = plant.replace(
            "'",
            "''",
        )

        plant_sql = (
            f"'{safe_plant}'"
        )

    else:

        plant_sql = "NULL"

    if period:

        safe_period = period.replace(
            "'",
            "''",
        )

        period_sql = (
            f"'{safe_period}'"
        )

    else:

        period_sql = "NULL"

    _, rows = run_sql(
        f"""
        SELECT CHECK_PERSONA_CONSISTENCY(
            '{safe}',
            {plant_sql},
            {period_sql}
        )
        """
    )

    if (
        rows
        and rows[0]
        and rows[0][0]
    ):

        try:

            value = rows[0][0]

            if isinstance(
                value,
                str,
            ):

                return json.loads(
                    value
                )

            if isinstance(
                value,
                dict,
            ):

                return value

        except (
            json.JSONDecodeError,
            TypeError,
        ):

            pass

    return {}


# =============================================================================
# THREAD MANAGEMENT
# =============================================================================

ORIGIN_APP = "sc_copilot"


def _thread_headers() -> dict:

    token, token_type = (
        _bearer()
    )

    return {
        "Authorization": (
            f"Bearer {token}"
        ),
        "Content-Type": "application/json",
        "Accept": "application/json",
        "X-Snowflake-Authorization-Token-Type": (
            token_type
        ),
    }


def create_thread() -> int:
    """
    Create Cortex conversation thread.
    """

    c = _cfg()

    response = requests.post(
        (
            f"{c['account_url']}"
            "/api/v2/cortex/threads"
        ),
        headers=_thread_headers(),
        json={
            "origin_application": ORIGIN_APP
        },
        timeout=30,
    )

    response.raise_for_status()

    payload = response.json()

    return int(
        payload["thread_id"]
    )


def list_threads() -> list[dict]:
    """
    List existing Cortex threads.
    """

    c = _cfg()

    response = requests.get(
        (
            f"{c['account_url']}"
            "/api/v2/cortex/threads"
        ),
        headers=_thread_headers(),
        params={
            "origin_application": ORIGIN_APP
        },
        timeout=30,
    )

    response.raise_for_status()

    payload = response.json()

    if isinstance(
        payload,
        dict,
    ):

        threads = payload.get(
            "threads",
            [],
        )

    elif isinstance(
        payload,
        list,
    ):

        threads = payload

    else:

        threads = []

    threads.sort(
        key=lambda item: item.get(
            "updated_on",
            0,
        ),
        reverse=True,
    )

    return threads


def get_thread_messages(
    thread_id: int,
    page_size: int = 100,
) -> list[dict]:
    """
    Fetch thread messages in chronological order.
    """

    c = _cfg()

    response = requests.get(
        (
            f"{c['account_url']}"
            f"/api/v2/cortex/threads/{thread_id}"
        ),
        headers=_thread_headers(),
        params={
            "page_size": page_size,
            "message_type": "conversation",
        },
        timeout=30,
    )

    response.raise_for_status()

    payload = response.json()

    if isinstance(
        payload,
        dict,
    ):

        messages = payload.get(
            "messages",
            [],
        )

    else:

        messages = []

    messages.reverse()

    return messages


def set_thread_name(
    thread_id: int,
    name: str,
) -> None:
    """
    Set Cortex thread name.
    """

    c = _cfg()

    response = requests.post(
        (
            f"{c['account_url']}"
            f"/api/v2/cortex/threads/{thread_id}"
        ),
        headers=_thread_headers(),
        json={
            "thread_name": name[:64]
        },
        timeout=30,
    )

    response.raise_for_status()


def delete_thread(
    thread_id: int,
) -> None:
    """
    Delete Cortex thread.
    """

    c = _cfg()

    response = requests.delete(
        (
            f"{c['account_url']}"
            f"/api/v2/cortex/threads/{thread_id}"
        ),
        headers=_thread_headers(),
        timeout=30,
    )

    response.raise_for_status()


# =============================================================================
# VOICE INPUT
# =============================================================================
#
# Intentionally disabled.
#
# No:
#   - audio upload
#   - Snowflake voice stage
#   - AI_TRANSCRIBE
#   - Snowpark voice session
#   - transcribe_audio()
#
# =============================================================================