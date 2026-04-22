from collections.abc import AsyncIterator
import logging
import re
from uuid import uuid4

from src.app.api.schemas.chat import ChatCitation, ChatRequest, ChatResponse
from src.app.domain.support.guardrails import SupportGuardrailPolicy
from src.app.domain.support.models import (
    ChatSession,
    ConversationTurn,
    PromptBuildInput,
    SupportAnswer,
)
from src.app.domain.support.observability import log_support_event
from src.app.domain.support.prompt_builder import SupportPromptBuilder
from src.app.domain.support.retrieval import RetrievalDecision, SupportRetrieval
from src.app.domain.support.router import RouteDecision, RouteType, SupportRouter
from src.app.infrastructure.llm.openai_client import LLMClient
from src.app.infrastructure.storage.conversation_store import ConversationStore

logger = logging.getLogger(__name__)
_CITATION_RE = re.compile(r"\[\d+\]")
TOOL_PLACEHOLDER_MESSAGE = (
    "I can help answer questions from our support knowledge, but I cannot perform "
    "that action yet. Please contact the human support team for help with that request."
)


class SupportService:
    """Application service responsible for the chat request lifecycle."""

    def __init__(
        self,
        conversation_store: ConversationStore,
        openai_client: LLMClient,
        prompt_builder: SupportPromptBuilder,
        retrieval_pipeline: SupportRetrieval,
        router: SupportRouter,
        guardrail_policy: SupportGuardrailPolicy | None = None,
    ) -> None:
        self._conversation_store = conversation_store
        self._openai_client = openai_client
        self._prompt_builder = prompt_builder
        self._retrieval_pipeline = retrieval_pipeline
        self._router = router
        self._guardrail_policy = guardrail_policy or SupportGuardrailPolicy()

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Process a chat request and return the assistant response."""
        session = await self._load_session(request.session_id)
        guarded_answer = self._prepare_answer(session, request.message)
        if guarded_answer.grounding_status == "fallback":
            response = guarded_answer.message
        else:
            messages = self._build_messages(
                session,
                request.message,
                guarded_answer=guarded_answer,
            )
            response = await self._openai_client.complete(messages)
            guarded_answer = self._finalize_grounded_answer(guarded_answer, response)
        await self._persist_conversation(
            session.session_id,
            session.history,
            request.message,
            guarded_answer.message if guarded_answer.message else response,
        )
        return self._to_chat_response(guarded_answer)

    async def stream_chat(self, request: ChatRequest) -> AsyncIterator[str]:
        """Process a chat request and yield the assistant response as a stream."""
        session = await self._load_session(request.session_id)
        guarded_answer = self._prepare_answer(session, request.message)
        if guarded_answer.grounding_status == "fallback":
            yield guarded_answer.message
            await self._persist_conversation(
                session.session_id,
                session.history,
                request.message,
                guarded_answer.message,
            )
            return

        messages = self._build_messages(
            session,
            request.message,
            guarded_answer=guarded_answer,
        )
        response_chunks: list[str] = []
        async for chunk in self._openai_client.stream_complete(messages):
            response_chunks.append(chunk)
            yield chunk

        response = "".join(response_chunks)
        finalized_answer = self._finalize_grounded_answer(guarded_answer, response)
        suffix = finalized_answer.message[len(response):]
        if suffix:
            yield suffix
        await self._persist_conversation(
            session.session_id,
            session.history,
            request.message,
            finalized_answer.message,
        )

    async def _load_session(self, session_id: str | None) -> ChatSession:
        """Resolve the chat session and load any stored conversation history."""
        resolved_session_id = session_id or str(uuid4())
        history = self._conversation_store.load(resolved_session_id)
        return ChatSession(session_id=resolved_session_id, history=history)

    def _prepare_answer(
        self,
        session: ChatSession,
        user_message: str,
    ) -> SupportAnswer:
        """Resolve retrieval and routing before the model is invoked."""
        retrieval_decision = self._retrieval_pipeline.run(
            user_message,
            request_id=session.session_id,
        )
        route_decision = self._router.decide(user_message, retrieval_decision)
        answer = self._build_routed_answer(retrieval_decision, route_decision)
        log_support_event(
            logger,
            event="support.request.model_input",
            payload={
                "request_id": session.session_id,
                "history_turn_count": len(session.history),
                "retrieval_used_fallback": retrieval_decision.used_fallback,
                "retrieval_confidence": retrieval_decision.confidence_score,
                "retrieval_decision_reason": retrieval_decision.decision_reason,
                "retrieved_context_count": len(retrieval_decision.retrieved_context),
                "route": route_decision.route.value,
                "route_reason": route_decision.reason,
                "guardrail_status": answer.grounding_status,
                "guardrail_fallback_reason": answer.fallback_reason,
            },
        )
        if answer.grounding_status == "fallback":
            log_support_event(
                logger,
                event="support.guardrails.fallback",
                payload={
                    "request_id": session.session_id,
                    "fallback_reason": answer.fallback_reason,
                    "route": route_decision.route.value,
                    "route_reason": route_decision.reason,
                    "retrieval_confidence": retrieval_decision.confidence_score,
                    "retrieval_decision_reason": retrieval_decision.decision_reason,
                },
            )
        return answer

    def _build_routed_answer(
        self,
        retrieval_decision: RetrievalDecision,
        route_decision: RouteDecision,
    ) -> SupportAnswer:
        if route_decision.route is RouteType.TOOL:
            return SupportAnswer(
                message=TOOL_PLACEHOLDER_MESSAGE,
                grounding_status="fallback",
                fallback_reason="tool_not_supported",
            )

        guardrail_decision = self._guardrail_policy.evaluate(retrieval_decision)
        return guardrail_decision.answer

    def _build_messages(
        self,
        session: ChatSession,
        user_message: str,
        *,
        guarded_answer: SupportAnswer,
    ) -> list[ConversationTurn]:
        """Build the model input from stored history and the new user message."""
        prompt = self._prompt_builder.build(
            PromptBuildInput(
                history=session.history,
                user_message=user_message,
                request_id=session.session_id,
                retrieved_context=guarded_answer.context_chunks,
            )
        )
        return [
            ConversationTurn(role="system", content=prompt.system_prompt),
            ConversationTurn(role="user", content=prompt.user_prompt),
        ]

    def _finalize_grounded_answer(
        self,
        guarded_answer: SupportAnswer,
        response: str,
    ) -> SupportAnswer:
        message = response.strip()
        if guarded_answer.citations and not _CITATION_RE.search(message):
            citation_labels = " ".join(citation.label for citation in guarded_answer.citations)
            message = f"{message}\n\nSources: {citation_labels}"
        return SupportAnswer(
            message=message,
            citations=guarded_answer.citations,
            context_chunks=guarded_answer.context_chunks,
            used_context=guarded_answer.used_context,
            grounding_status=guarded_answer.grounding_status,
            fallback_reason=guarded_answer.fallback_reason,
        )

    def _to_chat_response(self, answer: SupportAnswer) -> ChatResponse:
        return ChatResponse(
            response=answer.message,
            citations=[
                ChatCitation(
                    chunk_id=citation.chunk_id,
                    label=citation.label,
                    source=citation.source,
                )
                for citation in answer.citations
            ],
            used_context=answer.used_context,
            grounding_status=answer.grounding_status,
            fallback_reason=answer.fallback_reason,
        )

    async def _persist_conversation(
        self,
        session_id: str,
        history: list[ConversationTurn],
        user_message: str,
        assistant_response: str,
    ) -> None:
        """Persist the updated conversation."""
        messages = [
            *history,
            ConversationTurn(role="user", content=user_message),
            ConversationTurn(role="assistant", content=assistant_response),
        ]
        self._conversation_store.save(session_id, messages)
