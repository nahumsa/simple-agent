"""Middleware hooks for SimpleAgentLoop behavior."""

from __future__ import annotations

from agent_core.interfaces import AgentSession
from agent_core.types import LLMResult, ToolCall
from frameworks.barebones.doom_loop import check_for_doom_loop


class AgentMiddleware:
    """Base middleware with no-op hooks."""

    async def before_llm_call(self, session: AgentSession) -> None:
        return None

    async def after_llm_call(self, session: AgentSession, result: LLMResult) -> None:
        return None

    async def before_tool_call(self, session: AgentSession, tool_call: ToolCall) -> None:
        return None

    async def after_tool_call(
        self,
        session: AgentSession,
        tool_call: ToolCall,
        output: str,
        success: bool,
    ) -> None:
        return None


class ContextCompactionMiddleware(AgentMiddleware):
    """Compact context before LLM calls when the context requests it."""

    async def before_llm_call(self, session: AgentSession) -> None:
        if session.context.needs_compaction:
            await session.context.compact()


class DoomLoopGuardMiddleware(AgentMiddleware):
    """Inject a corrective prompt when repeated loops are detected."""

    async def before_llm_call(self, session: AgentSession) -> None:
        doom_prompt = check_for_doom_loop(session.context.messages())
        if not doom_prompt:
            return

        session.context.add_user_message(doom_prompt)
        await session.emit(
            "tool_log",
            {"tool": "system", "log": "Repetition guard activated."},
        )
