"""Tool registries for the simple agent."""

from __future__ import annotations

from pathlib import Path

from agent.types import JsonObject, ToolSpec


class NoTools:
    """Empty tool registry for plain chat mode."""

    def specs(self) -> list[ToolSpec]:
        return []

    async def call(
        self,
        name: str,
        args: JsonObject,
        *,
        session: object,
        tool_call_id: str,
    ) -> tuple[str, bool]:
        return f"Unknown tool: {name}", False

    async def cancel_running(self) -> None:
        return None


class ChallengeDataTools:
    """Tools for browsing extracted Coding Challenges markdown files."""

    def __init__(self, data_dir: Path | str = "data/extracted_data") -> None:
        self.data_dir = Path(data_dir)
        self.index_file = self.data_dir / "index.md"

    def initial_context(self) -> str | None:
        """Return the challenge index so it can be added to the agent context."""
        try:
            index = self.index_file.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return None

        return (
            "The following Coding Challenges index is available. "
            "Use the read_file tool to inspect any referenced markdown file in detail.\n\n"
            f"{index}"
        )

    def specs(self) -> list[ToolSpec]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "read_challenge_index",
                    "description": "Read the Coding Challenges markdown index.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": (
                        "Read a markdown file from data/extracted_data. "
                        "Use paths from the challenge index, for example "
                        "001-challenge-wc.md."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Relative markdown path to read from data/extracted_data.",
                            }
                        },
                        "required": ["path"],
                        "additionalProperties": False,
                    },
                },
            },
        ]

    async def call(
        self,
        name: str,
        args: JsonObject,
        *,
        session: object,
        tool_call_id: str,
    ) -> tuple[str, bool]:
        if name == "read_challenge_index":
            return self._read_file(self.index_file)

        if name == "read_file":
            requested_path = args.get("path")
            if not isinstance(requested_path, str) or not requested_path.strip():
                return "Missing required string argument: path", False
            return self._read_relative_markdown(requested_path)

        return f"Unknown tool: {name}", False

    async def cancel_running(self) -> None:
        return None

    def _read_relative_markdown(self, requested_path: str) -> tuple[str, bool]:
        relative_path = Path(requested_path)
        if relative_path.is_absolute():
            return "Path must be relative to data/extracted_data", False
        if relative_path.suffix != ".md":
            return "Only markdown files can be read", False

        path = self.data_dir / relative_path
        try:
            data_root = self.data_dir.resolve(strict=True)
            resolved_path = path.resolve(strict=True)
        except FileNotFoundError:
            return f"File not found: {requested_path}", False

        if data_root not in resolved_path.parents and resolved_path != data_root:
            return "Path escapes data/extracted_data", False

        return self._read_file(resolved_path)

    @staticmethod
    def _read_file(path: Path) -> tuple[str, bool]:
        try:
            return path.read_text(encoding="utf-8"), True
        except FileNotFoundError:
            return f"File not found: {path}", False
        except OSError as exc:
            return f"Could not read {path}: {exc}", False
