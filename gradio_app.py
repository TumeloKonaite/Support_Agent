import json
import os
from typing import Any
from urllib import error, request
from uuid import uuid4

import gradio as gr

DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"
API_BASE_URL_ENV_VAR = "SUPPORT_API_BASE_URL"
REQUEST_TIMEOUT_SECONDS = 60


def _generate_session_id() -> str:
    """Return a fresh client-managed session id."""
    return str(uuid4())


def _resolve_session_id(session_id: str | None) -> str:
    """Return the provided session id or generate a new one."""
    resolved = (session_id or "").strip()
    return resolved or _generate_session_id()


def _get_api_base_url() -> str:
    """Resolve the FastAPI base URL from environment when available."""
    configured = os.getenv(API_BASE_URL_ENV_VAR, DEFAULT_API_BASE_URL).strip()
    return configured.rstrip("/")


def _post_chat_message(message: str, session_id: str) -> dict[str, Any]:
    """Call the existing FastAPI chat endpoint."""
    payload = json.dumps({"message": message, "session_id": session_id}).encode("utf-8")
    chat_request = request.Request(
        url=f"{_get_api_base_url()}/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(chat_request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Chat request failed with HTTP {exc.code}. Response body: {details}"
        ) from exc
    except error.URLError as exc:
        raise RuntimeError(
            "Could not reach the support API. Start the FastAPI app and verify "
            f"{API_BASE_URL_ENV_VAR} points to the correct base URL."
        ) from exc


def _format_citations(citations: list[dict[str, Any]]) -> str:
    """Render citations as Markdown for the metadata panel."""
    if not citations:
        return "None"

    lines = []
    for citation in citations:
        label = citation.get("label", "Unknown citation")
        source = citation.get("source")
        chunk_id = citation.get("chunk_id", "unknown")
        suffix = f" ({source})" if source else ""
        lines.append(f"- {label}{suffix} [`{chunk_id}`]")
    return "\n".join(lines)


def _build_metadata_markdown(response_payload: dict[str, Any]) -> str:
    """Render high-signal response metadata for the latest assistant turn."""
    grounding_status = response_payload.get("grounding_status", "unknown")
    used_context = response_payload.get("used_context", False)
    fallback_reason = response_payload.get("fallback_reason") or "None"
    citations = response_payload.get("citations", [])

    return "\n".join(
        [
            f"**Grounding status:** `{grounding_status}`",
            f"**Used context:** `{used_context}`",
            f"**Fallback reason:** `{fallback_reason}`",
            "**Citations:**",
            _format_citations(citations),
        ]
    )


def _append_turn(
    history: list[dict[str, str]] | None,
    user_message: str,
    assistant_message: str,
) -> list[dict[str, str]]:
    """Return updated chat history for the Gradio chatbot."""
    updated_history = _normalize_history(history)
    updated_history.append({"role": "user", "content": user_message})
    updated_history.append({"role": "assistant", "content": assistant_message})
    return updated_history


def _normalize_history(history: Any) -> list[dict[str, str]]:
    """Convert mixed Gradio history shapes into message dictionaries."""
    normalized: list[dict[str, str]] = []
    pending_user_message: str | None = None

    for item in history or []:
        if isinstance(item, dict):
            role = item.get("role")
            content = item.get("content")
            if isinstance(role, str) and isinstance(content, str):
                normalized.append({"role": role, "content": content})
            continue

        if (
            isinstance(item, (list, tuple))
            and len(item) == 2
            and all(isinstance(value, str) for value in item)
        ):
            pending_user_message, assistant_message = item
            normalized.append({"role": "user", "content": pending_user_message})
            normalized.append({"role": "assistant", "content": assistant_message})

    return normalized


def _format_assistant_message(response_payload: dict[str, Any]) -> str:
    """Render the assistant response, emphasizing guardrail fallbacks."""
    assistant_message = response_payload.get("response", "").strip()
    if not assistant_message:
        assistant_message = "The API returned an empty response."

    if response_payload.get("grounding_status") == "fallback":
        fallback_reason = response_payload.get("fallback_reason") or "unspecified"
        return f"[Fallback: {fallback_reason}] {assistant_message}"

    return assistant_message


def submit_message(
    message: str,
    session_id: str,
    history: list[dict[str, str]] | None,
) -> tuple[list[dict[str, str]], list[dict[str, str]], str, str, dict[str, Any], str]:
    """Handle a single chat turn through the existing API."""
    cleaned_message = message.strip()
    resolved_session_id = _resolve_session_id(session_id)
    if not cleaned_message:
        normalized_history = _normalize_history(history)
        return (
            normalized_history,
            normalized_history,
            resolved_session_id,
            "Enter a message to start the conversation.",
            {},
            "",
        )

    try:
        response_payload = _post_chat_message(cleaned_message, resolved_session_id)
        assistant_message = _format_assistant_message(response_payload)
    except RuntimeError as exc:
        error_message = f"Request error: {exc}"
        response_payload = {
            "response": error_message,
            "citations": [],
            "used_context": False,
            "grounding_status": "client_error",
            "fallback_reason": "api_unavailable",
        }
        assistant_message = _format_assistant_message(response_payload)

    history = _append_turn(history, cleaned_message, assistant_message)
    metadata_markdown = _build_metadata_markdown(response_payload)
    return (
        history,
        history,
        resolved_session_id,
        metadata_markdown,
        response_payload,
        "",
    )


def new_conversation() -> tuple[list[dict[str, str]], list[dict[str, str]], str, str, dict[str, Any], str]:
    """Reset the conversation state and issue a fresh session id."""
    new_session_id = _generate_session_id()
    return [], [], new_session_id, "No messages yet.", {}, ""


with gr.Blocks(title="Support Assistant Demo") as demo:
    gr.Markdown(
        "\n".join(
            [
                "# Support Assistant Demo",
                "Thin Gradio client for the existing FastAPI chat API.",
            ]
        )
    )

    history_state = gr.State([])

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Conversation", height=520)
            message_input = gr.Textbox(
                label="Message",
                placeholder="Ask a support question...",
                lines=4,
            )
            with gr.Row():
                send_button = gr.Button("Send", variant="primary")
                new_chat_button = gr.Button("New conversation")
        with gr.Column(scale=2):
            session_id_input = gr.Textbox(
                label="Session ID",
                value=_generate_session_id(),
                info="Reuse a value to continue an existing conversation.",
            )
            api_base_url = gr.Textbox(
                label="API Base URL",
                value=_get_api_base_url(),
                interactive=False,
            )
            gr.Markdown("### Latest response details")
            metadata_output = gr.Markdown(value="No messages yet.")
            with gr.Accordion("Debug response payload", open=False):
                raw_payload = gr.JSON(label="Raw API response")

    send_button.click(
        fn=submit_message,
        inputs=[message_input, session_id_input, history_state],
        outputs=[
            chatbot,
            history_state,
            session_id_input,
            metadata_output,
            raw_payload,
            message_input,
        ],
    )

    message_input.submit(
        fn=submit_message,
        inputs=[message_input, session_id_input, history_state],
        outputs=[
            chatbot,
            history_state,
            session_id_input,
            metadata_output,
            raw_payload,
            message_input,
        ],
    )

    new_chat_button.click(
        fn=new_conversation,
        inputs=[],
        outputs=[
            chatbot,
            history_state,
            session_id_input,
            metadata_output,
            raw_payload,
            message_input,
        ],
    )


if __name__ == "__main__":
    demo.launch()
