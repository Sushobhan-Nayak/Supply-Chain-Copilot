"""Cortex Agent client for the Supply Chain Copilot.

Uses:
    - Snowflake Python Connector with key-pair authentication for SQL.
    - Snowflake REST API with short-lived key-pair JWTs for Cortex Agent
      and Cortex thread APIs.

No PAT is used anywhere in this file.

Configuration is read from st.secrets.
Private key material should never be committed to source control.

Voice input is currently disabled.
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
import streamlit as st
import snowflake.connector

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    PublicFormat,
    load_pem_private_key,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

def _cfg() -> dict:
    """Read Snowflake configuration from Streamlit secrets."""

    s = st.secrets["snowflake"]

    return {
        "account": s["account"],
        "account_url": s["account_url"].rstrip("/"),
        "user": s["user"],
        "warehouse": s["warehouse"],
        "database": s["database"],
        "schema": s["schema"],
        "agent": s["agent"],
        "private_key": s["private_key"],
        "private_key_passphrase": s.get(
            "private_key_passphrase",
            None,
        ),
    }


# ============================================================================
# KEY-PAIR AUTHENTICATION
# ============================================================================

@st.cache_resource
def _private_key():
    """Load the RSA private key from Streamlit secrets."""

    c = _cfg()

    pem_data = c["private_key"]

    if isinstance(pem_data, str):
        pem_data = pem_data.encode("utf-8")

    passphrase = c.get("private_key_passphrase")

    if passphrase:
        passphrase = passphrase.encode("utf-8")

    return load_pem_private_key(
        pem_data,
        password=passphrase,
        backend=default_backend(),
    )


@st.cache_resource
def _public_key_fingerprint() -> str:
    """Return Snowflake's SHA256 public-key fingerprint."""

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
    """Return ACCOUNT.USER in Snowflake's required uppercase format."""

    c = _cfg()

    account = c["account"].strip()

    # Snowflake's JWT format requires the account identifier
    # without region/cloud-provider suffixes.
    if ".global" not in account.lower():

        if "." in account:
            account = account.split(".", 1)[0]

        elif "-" in account:
            # This branch only applies when an account locator
            # style identifier is supplied.
            #
            # For normal organization-account identifiers such as
            # MYORG-MYACCOUNT, the value is kept intact.
            pass

    account = account.replace(".", "-").upper()

    user = c["user"].upper()

    return f"{account}.{user}"


@st.cache_data(ttl=3000)
def _bearer() -> tuple[str, str]:
    """Generate a short-lived Snowflake key-pair JWT.

    Snowflake key-pair JWTs can be valid for at most one hour.
    We use 59 minutes here.
    """

    private_key = _private_key()

    qualified_username = _qualified_username()
    public_key_fp = _public_key_fingerprint()

    now = datetime.now(timezone.utc)

    lifetime = timedelta(minutes=59)

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


# ============================================================================
# CORTEX AGENT REST CONFIGURATION
# ============================================================================

def _run_url(c: dict) -> str:
    return (
        f"{c['account_url']}/api/v2/databases/"
        f"{c['database']}/schemas/{c['schema']}/"
        f"agents/{c['agent']}:run"
    )


def _headers(
    c: dict,
    streaming: bool,
) -> dict:
    token, token_type = _bearer()

    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": (
            "text/event-stream"
            if streaming
            else "application/json"
        ),
        "X-Snowflake-Authorization-Token-Type": token_type,
    }


# ============================================================================
# RESULT OBJECT
# ============================================================================

@dataclass
class AgentResult:
    answer_text: str = ""
    generated_sql: str = ""
    query_id: Optional[str] = None
    result_rows: list[list] = field(default_factory=list)
    column_names: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    citations: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    chart_spec: Optional[dict] = None
    thread_id: Optional[int] = None
    assistant_message_id: Optional[int] = None
    error: Optional[str] = None


# ============================================================================
# RESPONSE PARSING
# ============================================================================

def _extract_from_response_payload(
    data: dict,
    result: AgentResult,
) -> None:
    """Extract useful information from a Cortex Agent response."""

    # The agent can emit multiple text blocks.
    # The final text block is treated as the final answer.
    for item in data.get("content", []):

        itype = item.get("type")

        # --------------------------------------------------------------------
        # Text
        # --------------------------------------------------------------------

        if itype == "text":

            txt = item.get("text", "")

            if txt.strip():
                result.answer_text = txt

            for ann in item.get("annotations", []):

                result.citations.append(
                    {
                        "title": ann.get("doc_title"),
                        "text": ann.get("text"),
                        "type": ann.get("type"),
                    }
                )

        # --------------------------------------------------------------------
        # Table
        # --------------------------------------------------------------------

        elif itype == "table":

            _load_result_set(
                item.get("result_set", {}),
                result,
            )

            if item.get("query_id"):
                result.query_id = item["query_id"]

        # --------------------------------------------------------------------
        # Chart
        # --------------------------------------------------------------------

        elif itype == "chart":

            spec = (
                item.get("chart", {})
                .get("chart_spec")
            )

            if isinstance(spec, str):

                try:
                    result.chart_spec = json.loads(spec)
                except json.JSONDecodeError:
                    pass

            elif isinstance(spec, dict):

                result.chart_spec = spec

        # --------------------------------------------------------------------
        # Tool use
        # --------------------------------------------------------------------

        elif itype == "tool_use":

            tu = item.get(
                "tool_use",
                item,
            )

            result.tools_used.append(
                tu.get("name")
                or tu.get("type", "tool")
            )

        # --------------------------------------------------------------------
        # Tool result
        # --------------------------------------------------------------------

        elif itype == "tool_result":

            tr = item.get(
                "tool_result",
                item,
            )

            for sub in tr.get("content", []):

                j = (
                    sub.get("json")
                    if isinstance(sub, dict)
                    else None
                )

                if isinstance(j, dict):

                    if j.get("sql"):
                        result.generated_sql = j["sql"]

                    if j.get("query_id"):
                        result.query_id = j["query_id"]

                    if j.get("result_set"):
                        _load_result_set(
                            j["result_set"],
                            result,
                        )

    # ------------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------------

    meta = data.get(
        "metadata",
        {},
    )

    if meta.get("thread_id") is not None:
        result.thread_id = meta["thread_id"]

    if meta.get("assistant_message_id") is not None:
        result.assistant_message_id = (
            meta["assistant_message_id"]
        )

    # ------------------------------------------------------------------------
    # Warnings
    # ------------------------------------------------------------------------

    for w in data.get("warnings", []):

        msg = w.get("message", "")

        if msg:
            result.warnings.append(msg)


def _load_result_set(
    rs: dict,
    result: AgentResult,
) -> None:
    """Load a Snowflake result set into AgentResult."""

    if not rs:
        return

    result.result_rows = (
        rs.get("data", [])
        or result.result_rows
    )

    cols = [
        c["name"]
        for c in rs.get(
            "resultSetMetaData",
            {},
        ).get(
            "rowType",
            [],
        )
    ]

    if cols:
        result.column_names = cols


# ============================================================================
# SSE PARSER
# ============================================================================

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

        if raw.startswith("event:"):

            event_type = raw[
                len("event:"):
            ].strip()

            continue

        if not raw.startswith("data:"):
            continue

        try:

            data = json.loads(
                raw[
                    len("data:"):
                ].strip()
            )

        except json.JSONDecodeError:
            continue

        # --------------------------------------------------------------------
        # Analyst tool result
        # --------------------------------------------------------------------

        if (
            event_type
            == "response.tool_result.analyst.delta"
        ):

            delta = data.get(
                "delta",
                {},
            )

            if delta.get("sql"):
                result.generated_sql = delta["sql"]

            if delta.get("query_id"):
                result.query_id = delta["query_id"]

            if delta.get("result_set"):

                _load_result_set(
                    delta["result_set"],
                    result,
                )

        # --------------------------------------------------------------------
        # Table
        # --------------------------------------------------------------------

        elif event_type == "response.table":

            if data.get("query_id"):
                result.query_id = data["query_id"]

            _load_result_set(
                data.get(
                    "result_set",
                    {},
                ),
                result,
            )

        # --------------------------------------------------------------------
        # Chart
        # --------------------------------------------------------------------

        elif event_type == "response.chart":

            spec = data.get(
                "chart_spec"
            )

            if isinstance(spec, str):

                try:
                    result.chart_spec = json.loads(spec)
                except json.JSONDecodeError:
                    pass

            elif isinstance(spec, dict):

                result.chart_spec = spec

        # --------------------------------------------------------------------
        # Tool use
        # --------------------------------------------------------------------

        elif event_type == "response.tool_use":

            result.tools_used.append(
                data.get("name")
                or data.get("type", "tool")
            )

        # --------------------------------------------------------------------
        # Warning
        # --------------------------------------------------------------------

        elif event_type == "response.warning":

            result.warnings.append(
                data.get(
                    "message",
                    "",
                )
            )

        # --------------------------------------------------------------------
        # Full response
        # --------------------------------------------------------------------

        elif event_type == "response":

            _extract_from_response_payload(
                data,
                result,
            )

    return result


# ============================================================================
# ASK AGENT
# ============================================================================

def ask_agent(
    question: str,
    *,
    thread_id: int | None = None,
    parent_message_id: int | None = None,
    stream: bool = False,
    timeout: int = 120,
) -> AgentResult:
    """Send a question to the Cortex Agent."""

    try:

        c = _cfg()

    except (
        KeyError,
        FileNotFoundError,
    ):

        return AgentResult(
            error=(
                "Snowflake secrets are not configured. "
                "Please configure [snowflake] in "
                "Streamlit secrets."
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

        body["thread_id"] = thread_id

        body["parent_message_id"] = (
            parent_message_id
            if parent_message_id is not None
            else 0
        )

    try:

        resp = requests.post(
            _run_url(c),
            headers=_headers(
                c,
                stream,
            ),
            json=body,
            stream=stream,
            timeout=timeout,
        )

        resp.raise_for_status()

    except requests.HTTPError as e:

        detail = ""

        try:
            detail = resp.text[:400]
        except Exception:
            pass

        return AgentResult(
            error=(
                f"Agent request failed "
                f"({resp.status_code}): "
                f"{detail or e}"
            )
        )

    except requests.RequestException as e:

        return AgentResult(
            error=f"Could not reach Snowflake: {e}"
        )

    if stream:

        return _parse_sse(resp)

    result = AgentResult()

    _extract_from_response_payload(
        resp.json(),
        result,
    )

    return result


# ============================================================================
# STREAMING AGENT
# ============================================================================

def ask_agent_stream(
    question: str,
    *,
    thread_id: int | None = None,
    parent_message_id: int | None = None,
    timeout: int = 300,
):
    """Generator for live Cortex Agent answer streaming."""

    try:

        c = _cfg()

    except (
        KeyError,
        FileNotFoundError,
    ):

        ask_agent_stream.result = AgentResult(
            error="Snowflake secrets are not configured."
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

        body["thread_id"] = thread_id

        body["parent_message_id"] = (
            parent_message_id
            if parent_message_id is not None
            else 0
        )

    result = AgentResult()

    try:

        resp = requests.post(
            _run_url(c),
            headers=_headers(
                c,
                True,
            ),
            json=body,
            stream=True,
            timeout=timeout,
        )

        resp.raise_for_status()

    except requests.RequestException as e:

        result.error = (
            f"Could not reach Snowflake: {e}"
        )

        ask_agent_stream.result = result

        return

    resp.encoding = "utf-8"

    event_type = None

    for raw in resp.iter_lines(
        decode_unicode=True
    ):

        if not raw:
            event_type = None
            continue

        if raw.startswith("event:"):

            event_type = raw[
                len("event:"):
            ].strip()

            continue

        if not raw.startswith("data:"):
            continue

        try:

            data = json.loads(
                raw[
                    len("data:"):
                ].strip()
            )

        except json.JSONDecodeError:
            continue

        # --------------------------------------------------------------------
        # Text streaming
        # --------------------------------------------------------------------

        if event_type == "response.text.delta":

            chunk = data.get(
                "text",
                "",
            )

            if chunk:
                yield chunk

        # --------------------------------------------------------------------
        # Analyst result
        # --------------------------------------------------------------------

        elif (
            event_type
            == "response.tool_result.analyst.delta"
        ):

            delta = data.get(
                "delta",
                {},
            )

            if delta.get("sql"):
                result.generated_sql = delta["sql"]

            if delta.get("query_id"):
                result.query_id = delta["query_id"]

            if delta.get("result_set"):

                _load_result_set(
                    delta["result_set"],
                    result,
                )

        # --------------------------------------------------------------------
        # Table
        # --------------------------------------------------------------------

        elif event_type == "response.table":

            if data.get("query_id"):
                result.query_id = data["query_id"]

            _load_result_set(
                data.get(
                    "result_set",
                    {},
                ),
                result,
            )

        # --------------------------------------------------------------------
        # Chart
        # --------------------------------------------------------------------

        elif event_type == "response.chart":

            spec = data.get(
                "chart_spec"
            )

            if isinstance(spec, str):

                try:
                    result.chart_spec = json.loads(spec)
                except json.JSONDecodeError:
                    pass

            elif isinstance(spec, dict):

                result.chart_spec = spec

        # --------------------------------------------------------------------
        # Full response
        # --------------------------------------------------------------------

        elif event_type == "response":

            _extract_from_response_payload(
                data,
                result,
            )

    ask_agent_stream.result = result


# ============================================================================
# DIRECT SQL
# ============================================================================

@st.cache_resource
def _get_snowflake_connection():
    """Create a Snowflake connector connection using key-pair auth."""

    c = _cfg()

    return snowflake.connector.connect(
        account=c["account"],
        user=c["user"],
        authenticator="SNOWFLAKE_JWT",
        private_key=c["private_key"],
        private_key_file_pwd=c["private_key_passphrase"],
        warehouse=c["warehouse"],
        database=c["database"],
        schema=c["schema"],
    )


def run_sql(
    statement: str,
    timeout: int = 60,
) -> tuple[list[str], list[list]]:
    """Execute SQL through the Snowflake Python Connector.

    Authentication uses Snowflake key-pair authentication.
    No PAT or SQL REST API is used.
    """

    conn = _get_snowflake_connection()

    cur = conn.cursor()

    try:

        cur.execute(
            statement,
            timeout=timeout,
        )

        columns = [
            desc[0]
            for desc in cur.description
        ]

        rows = cur.fetchall()

        return (
            columns,
            [list(row) for row in rows],
        )

    finally:

        cur.close()


# ============================================================================
# GOVERNANCE
# ============================================================================

def call_governance(
    metric_query: str,
) -> dict:
    """Return governed metric definition JSON."""

    safe = metric_query.replace(
        "'",
        "''",
    )

    _, rows = run_sql(
        f"SELECT GET_METRIC_GOVERNANCE('{safe}')"
    )

    if rows and rows[0] and rows[0][0]:

        try:

            return json.loads(
                rows[0][0]
            )

        except (
            json.JSONDecodeError,
            TypeError,
        ):
            return {}

    return {}


def check_persona_consistency(
    metric_query: str,
    plant: str | None = None,
    period: str | None = None,
) -> dict:
    """Return persona-consistency proof JSON."""

    safe = metric_query.replace(
        "'",
        "''",
    )

    p = (
        f"'{plant}'"
        if plant
        else "NULL"
    )

    pe = (
        f"'{period}'"
        if period
        else "NULL"
    )

    _, rows = run_sql(
        f"""
        SELECT CHECK_PERSONA_CONSISTENCY(
            '{safe}',
            {p},
            {pe}
        )
        """
    )

    if rows and rows[0] and rows[0][0]:

        try:

            return json.loads(
                rows[0][0]
            )

        except (
            json.JSONDecodeError,
            TypeError,
        ):
            return {}

    return {}


# ============================================================================
# THREAD MANAGEMENT
# ============================================================================

ORIGIN_APP = "sc_copilot"


def _thread_headers() -> dict:

    token, token_type = _bearer()

    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "X-Snowflake-Authorization-Token-Type": token_type,
    }


def create_thread() -> int:

    c = _cfg()

    resp = requests.post(
        f"{c['account_url']}/api/v2/cortex/threads",
        headers=_thread_headers(),
        json={
            "origin_application": ORIGIN_APP
        },
        timeout=30,
    )

    resp.raise_for_status()

    return int(
        resp.json()["thread_id"]
    )


def list_threads() -> list[dict]:

    c = _cfg()

    resp = requests.get(
        f"{c['account_url']}/api/v2/cortex/threads",
        headers=_thread_headers(),
        params={
            "origin_application": ORIGIN_APP
        },
        timeout=30,
    )

    resp.raise_for_status()

    threads = resp.json() or []

    threads.sort(
        key=lambda t: t.get(
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
    """Fetch thread messages in chronological order."""

    c = _cfg()

    resp = requests.get(
        f"{c['account_url']}/api/v2/cortex/threads/{thread_id}",
        headers=_thread_headers(),
        params={
            "page_size": page_size,
            "message_type": "conversation",
        },
        timeout=30,
    )

    resp.raise_for_status()

    msgs = resp.json().get(
        "messages",
        [],
    )

    msgs.reverse()

    return msgs


def set_thread_name(
    thread_id: int,
    name: str,
) -> None:

    c = _cfg()

    resp = requests.post(
        f"{c['account_url']}/api/v2/cortex/threads/{thread_id}",
        headers=_thread_headers(),
        json={
            "thread_name": name[:64]
        },
        timeout=30,
    )

    resp.raise_for_status()


def delete_thread(
    thread_id: int,
) -> None:

    c = _cfg()

    resp = requests.delete(
        f"{c['account_url']}/api/v2/cortex/threads/{thread_id}",
        headers=_thread_headers(),
        timeout=30,
    )

    resp.raise_for_status()


# ============================================================================
# VOICE INPUT — DISABLED
# ============================================================================
#
# Voice functionality has been intentionally removed.
#
# There is no:
#   - audio upload
#   - Snowflake voice stage
#   - AI_TRANSCRIBE
#   - transcribe_audio()
#   - Snowpark voice session
#