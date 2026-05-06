"""Simplified orchestration loop for a tool-using LLM agent."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from agent.doom_loop import check_for_doom_loop
from agent.types import LLMResult, ToolCall


class SimpleAgentLoop:
    """Small orchestration loop for a tool-using LLM agent."""

    def __init__(self, session: Any):
        self.session = session

    async def run(self, submission_queue: asyncio.Queue) -> None:
        await self.session.emit("ready", {"message": "Agent ready"})

        while self.session.running:
            submission = await submission_queue.get()

            if submission.type == "user_input":
                await self.run_turn(submission.text)
            elif submission.type == "approval":
                await self.handle_approval(submission.approvals)
            elif submission.type == "undo":
                self.session.context.undo_last_turn()
                await self.session.emit("undo_complete", {})
            elif submission.type == "shutdown":
                await self.shutdown()
                break
            else:
                await self.session.emit("error", {"error": "Unknown submission type"})

    async def run_turn(self, user_text: str | None = None) -> str | None:
        if user_text:
            self.session.context.add_user_message(user_text)

        final_response = None

        for _ in range(self.session.config.max_iterations):
            if self.session.cancelled:
                await self.cleanup_after_cancel()
                return None

            await self.compact_if_needed()
            await self.inject_doom_loop_prompt_if_needed()

            llm_result = await self.call_llm()

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

            auto_tools, approval_tools = self.split_by_approval(llm_result.tool_calls)

            if auto_tools:
                await self.execute_tools(auto_tools)

            if approval_tools:
                self.session.pending_approval = approval_tools
                await self.emit_approval_required(approval_tools)
                return None

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

            tool_calls.append(ToolCall(id=raw_call.id, name=raw_call.name, args=args))

        return LLMResult(content=response.content, tool_calls=tool_calls)

    async def execute_tools(self, tool_calls: list[ToolCall]) -> None:
        tasks = [self.execute_one_tool(tool_call) for tool_call in tool_calls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for tool_call, result in zip(tool_calls, results):
            if isinstance(result, Exception):
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

    async def execute_one_tool(self, tool_call: ToolCall) -> tuple[str, bool]:
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

    def split_by_approval(
        self,
        tool_calls: list[ToolCall],
    ) -> tuple[list[ToolCall], list[ToolCall]]:
        auto_tools = []
        approval_tools = []

        for tool_call in tool_calls:
            if self.needs_approval(tool_call):
                approval_tools.append(tool_call)
            else:
                auto_tools.append(tool_call)

        return auto_tools, approval_tools

    def needs_approval(self, tool_call: ToolCall) -> bool:
        if tool_call.name == "sandbox_create":
            return tool_call.args.get("hardware") != "cpu-basic"

        if tool_call.name == "hf_jobs":
            return tool_call.args.get("operation") in {"run", "uv", "schedule"}

        if tool_call.name in {"hf_repo_files", "hf_repo_git"}:
            return tool_call.args.get("operation") in {
                "upload",
                "delete",
                "merge_pr",
                "delete_branch",
                "create_repo",
            }

        return False

    async def handle_approval(self, approvals: list[dict[str, Any]]) -> None:
        pending: list[ToolCall] = self.session.pending_approval or []
        self.session.pending_approval = None

        approved_ids = {
            approval["tool_call_id"]
            for approval in approvals
            if approval.get("approved")
        }

        approved_tools = []
        for tool_call in pending:
            if tool_call.id in approved_ids:
                approved_tools.append(tool_call)
            else:
                self.session.context.add_tool_result(
                    tool_call.id,
                    tool_call.name,
                    "Tool call rejected by user",
                    success=False,
                )
                await self.emit_tool_output(
                    tool_call,
                    "Tool call rejected by user",
                    False,
                )

        if approved_tools:
            await self.execute_tools(approved_tools)

        # Continue the model loop with the new tool results.
        await self.run_turn(None)

    async def emit_approval_required(self, approval_tools: list[ToolCall]) -> None:
        await self.session.emit(
            "approval_required",
            {
                "tools": [
                    {
                        "id": tool.id,
                        "tool": tool.name,
                        "arguments": tool.args,
                    }
                    for tool in approval_tools
                ]
            },
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

    async def inject_doom_loop_prompt_if_needed(self) -> None:
        doom_prompt = check_for_doom_loop(self.session.context.messages())
        if not doom_prompt:
            return

        self.session.context.add_user_message(doom_prompt)
        await self.session.emit(
            "tool_log",
            {"tool": "system", "log": "Repetition guard activated."},
        )

    async def compact_if_needed(self) -> None:
        if self.session.context.needs_compaction:
            await self.session.context.compact()

    async def cleanup_after_cancel(self) -> None:
        await self.session.tools.cancel_running()
        await self.session.emit("interrupted", {})

    async def shutdown(self) -> None:
        self.session.running = False
        await self.session.save()
        await self.session.emit("shutdown", {})
