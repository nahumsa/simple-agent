"""Public compatibility module and CLI entrypoint for the educational agent.

The implementation lives in the ``agent`` package:
- agent/types.py: shared dataclasses
- agent/doom_loop.py: repeated tool-call detection
- agent/interfaces.py: protocol definitions
- agent/loop.py: SimpleAgentLoop orchestration
- cli.py: command-line chat interface
"""

from cli import main

if __name__ == "__main__":
    main()
