"""Cortex Agent REST client for the Supply Chain Copilot.

Talks to a deployed Snowflake Cortex Agent over the v2 REST API and parses the
response into a flat AgentResult.

Configuration is read from st.secrets (see .streamlit/secrets.toml).
Secrets never live in code so this file is safe to commit.

Voice input is currently disabled.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

import requests
import streamlit as st


# ============================================================================
# CONFIGURATION
# ============================================================================

def _cfg() -> dict:
    """Read Snowflake REST configuration from Streamlit secrets."""

    s = st.secrets["snowflake"]

    return {
        "account_url": s["account_url"].rstrip("/"),
        "database": s["database"],
        "schema": s["schema"],
        "agent": s["agent"],
        "pat": s["pat"],
    }


def _bearer() -> tuple[str, str]:
    """Return the PAT and Snowflake authorization token type."""

    s = st.secrets["snowflake"]

    return (
        s["pat"],
        "PROGRAMMATIC_ACCESS_TOKEN",
    )


def _run_url(c: dict) -> str:
    return (
        f"{c['account_url']}/api/v2/databases/{c['database']}"
        f"/schemas/{c['schema']}/agents/{c['agent']}:run"
    )


def _headers(c: dict, streaming: bool) -> dict:
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

            spec = item.get(
                "chart",
                {},
            ).get(
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

        if event_type == "response.tool_result.analyst.delta":

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

    except (KeyError, FileNotFoundError):

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

    except (KeyError, FileNotFoundError):

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

        elif event_type == "response.tool_result.analyst.delta":

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

def run_sql(
    statement: str,
    timeout: int = 60,
) -> tuple[list[str], list[list]]:
    """Execute SQL through the Snowflake SQL REST API."""

    c = _cfg()

    url = (
        f"{c['account_url']}"
        f"/api/v2/statements"
    )

    headers = {
        "Authorization": (
            f"Bearer {c['pat']}"
        ),
        "Content-Type": "application/json",
        "Accept": "application/json",
        "X-Snowflake-Authorization-Token-Type": (
            "PROGRAMMATIC_ACCESS_TOKEN"
        ),
    }

    s = st.secrets["snowflake"]

    body = {
        "statement": statement,
        "warehouse": s.get("warehouse"),
        "database": c["database"],
        "schema": c["schema"],
        "timeout": timeout,
    }

    resp = requests.post(
        url,
        headers=headers,
        json=body,
        timeout=timeout + 30,
    )

    resp.raise_for_status()

    data = resp.json()

    cols = [
        col["name"]
        for col in data.get(
            "resultSetMetaData",
            {},
        ).get(
            "rowType",
            [],
        )
    ]

    rows = (
        data.get("data", [])
        or []
    )

    return cols, rows


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
# Voice functionality has been intentionally commented out.
#
# Previous implementation:
#
# _VOICE_STAGE = "VOICE_AUDIO"
# _snowpark_session = None
#
#
# def _get_session():
#     global _snowpark_session
#
#     if _snowpark_session is not None:
#         return _snowpark_session
#
#     if IN_SIS:
#         _snowpark_session = _ACTIVE_SESSION
#     else:
#         from snowflake.snowpark import Session
#
#         s = st.secrets["snowflake"]
#
#         _snowpark_session = Session.builder.configs({
#             "account": s["account"],
#             "user": s["user"],
#             "authenticator": "programmatic_access_token",
#             "token": s["pat"],
#             "warehouse": s.get("warehouse"),
#             "database": s["database"],
#             "schema": s["schema"],
#         }).create()
#
#     _snowpark_session.sql(
#         f"CREATE STAGE IF NOT EXISTS {_VOICE_STAGE} "
#         "DIRECTORY=(ENABLE=true) "
#         "ENCRYPTION=(TYPE='SNOWFLAKE_SSE')"
#     ).collect()
#
#     return _snowpark_session
#
#
# def transcribe_audio(audio_bytes: bytes) -> str:
#     import io
#     import time as _time
#
#     session = _get_session()
#
#     fname = (
#         f"voice_{int(_time.time() * 1000)}.wav"
#     )
#
#     session.file.put_stream(
#         io.BytesIO(audio_bytes),
#         f"@{_VOICE_STAGE}/{fname}",
#         auto_compress=False,
#         overwrite=True,
#     )
#
#     safe = fname.replace("'", "''")
#
#     try:
#         rows = session.sql(
#             f"""
#             SELECT AI_TRANSCRIBE(
#                 TO_FILE('@{_VOICE_STAGE}', '{safe}')
#             ) AS T
#             """
#         ).collect()
#
#         return json.loads(
#             rows[0]["T"]
#         ).get(
#             "text",
#             "",
#         ).strip()
#
#     finally:
#         try:
#             session.sql(
#                 f"REMOVE @{_VOICE_STAGE}/{safe}"
#             ).collect()
#         except Exception:
#             pass