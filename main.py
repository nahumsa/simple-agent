"""Thin CLI entrypoint for the educational chatbot.

Core framework-neutral code lives in ``agent_core``. Framework-specific
implementations live under ``frameworks``.
"""

from cli import main

if __name__ == "__main__":
    main()
