"""Simplified orchestration loop for a tool-using LLM agent."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence

from agent_core.interfaces import AgentSession
from agent_core.types import AgentState, LLMResult, ToolCall
from frameworks.barebones.middleware import (
    AgentMiddleware,
    ContextCompactionMiddleware,
    DoomLoopGuardMiddleware,
)


class SimpleAgentLoop:
    """Small orchestration loop for a tool-using LLM agent."""

    def __init__(
        self,
        session: AgentSession,
        middleware: Sequence[AgentMiddleware] | None = None,
    ) -> None:
        self.session = session
        self.middleware = list(middleware) if middleware is not None else [
            ContextCompactionMiddleware(),
            DoomLoopGuardMiddleware(),
        ]

    async def run(self, submission_queue: asyncio.Queue) -> None:
        await self.session.emit("ready", {"message": "Agent ready"})

        while self.session.running:
            submission = await submission_queue.get()

            if submission.type == "user_input":
                await self.run_turn(submission.text)
            elif submission.type == "undo":
                self.session.context.undo_last_turn()
                await self.session.emit("undo_complete", {})
            elif submission.type == "shutdown":
                await self.shutdown()
                break
            else:
                await self.session.emit("error", {"error": "Unknown submission type"})

    async def run_turn(self, user_text: str | None = None) -> str | None:
        self.session.state = AgentState.RUNNING_TURN

        if user_text:
            self.session.context.add_user_message(user_text)

        final_response = None

        for _ in range(self.session.config.max_iterations):
            if self.session.cancelled:
                await self.cleanup_after_cancel()
                return None

            await self.before_llm_call()
            llm_result = await self.call_llm()
            await self.after_llm_call(llm_result)

            if llm_result.content:
                await self.session.emit(
                    "assistant_message",
                    {"content": llm_result.content},
                )

            if not llm_result.tool_calls:
                if llm_result.content:
                    self.session.context.add_assistant_message(llm_result.content)
                    final_response = llm_result.content
                break

            self.session.context.add_assistant_tool_calls(llm_result)

            await self.execute_tools(llm_result.tool_calls)

        self.session.state = AgentState.READY
        await self.session.emit("turn_complete", {"final_response": final_response})
        return final_response

    async def call_llm(self) -> LLMResult:
        response = await self.session.llm.complete(
            messages=self.session.context.messages(),
            tools=self.session.tools.specs(),
        )

        tool_calls = []
        for raw_call in response.tool_calls or []:
            try:
                args = json.loads(raw_call.arguments)
            except (json.JSONDecodeError, TypeError, ValueError):
                self.session.context.add_tool_result(
                    raw_call.id,
                    raw_call.name,
                    "Malformed JSON arguments",
                    success=False,
                )
                continue

            tool_calls.append(
                ToolCall(
                    id=raw_call.id,
                    name=raw_call.name,
                    args=args,
                    raw_arguments=raw_call.arguments,
                    extra_content=getattr(raw_call, "extra_content", None),
                )
            )

        return LLMResult(content=response.content, tool_calls=tool_calls)

    async def execute_tools(self, tool_calls: list[ToolCall]) -> None:
        tasks = [self.execute_one_tool(tool_call) for tool_call in tool_calls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for tool_call, result in zip(tool_calls, results):
            if isinstance(result, BaseException):
                output = str(result)
                success = False
            else:
                output, success = result

            self.session.context.add_tool_result(
                tool_call.id,
                tool_call.name,
                output,
                success=success,
            )
            await self.emit_tool_output(tool_call, output, success)
            await self.after_tool_call(tool_call, output, success)

    async def execute_one_tool(self, tool_call: ToolCall) -> tuple[str, bool]:
        await self.before_tool_call(tool_call)
        await self.session.emit(
            "tool_call",
            {
                "tool_call_id": tool_call.id,
                "tool": tool_call.name,
                "arguments": tool_call.args,
            },
        )
        return await self.session.tools.call(
            tool_call.name,
            tool_call.args,
            session=self.session,
            tool_call_id=tool_call.id,
        )

    async def emit_tool_output(
        self,
        tool_call: ToolCall,
        output: str,
        success: bool,
    ) -> None:
        await self.session.emit(
            "tool_output",
            {
                "tool_call_id": tool_call.id,
                "tool": tool_call.name,
                "output": output,
                "success": success,
            },
        )

    async def before_llm_call(self) -> None:
        for middleware in self.middleware:
            await middleware.before_llm_call(self.session)

    async def after_llm_call(self, result: LLMResult) -> None:
        for middleware in self.middleware:
            await middleware.after_llm_call(self.session, result)

    async def before_tool_call(self, tool_call: ToolCall) -> None:
        for middleware in self.middleware:
            await middleware.before_tool_call(self.session, tool_call)

    async def after_tool_call(
        self,
        tool_call: ToolCall,
        output: str,
        success: bool,
    ) -> None:
        for middleware in self.middleware:
            await middleware.after_tool_call(self.session, tool_call, output, success)

    async def inject_doom_loop_prompt_if_needed(self) -> None:
        for middleware in self.middleware:
            if isinstance(middleware, DoomLoopGuardMiddleware):
                await middleware.before_llm_call(self.session)

    async def compact_if_needed(self) -> None:
        for middleware in self.middleware:
            if isinstance(middleware, ContextCompactionMiddleware):
                await middleware.before_llm_call(self.session)

    async def cleanup_after_cancel(self) -> None:
        await self.session.tools.cancel_running()
        await self.session.emit("interrupted", {})
        self.session.state = AgentState.READY

    async def shutdown(self) -> None:
        self.session.state = AgentState.SHUTTING_DOWN
        self.session.running = False
        await self.session.save()
        await self.session.emit("shutdown", {})
