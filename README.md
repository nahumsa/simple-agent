# AI Powered Chatbot

<p align="center">
  <a href="https://github.com/nahumsa/simple-agent/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/nahumsa/simple-agent/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://mypy-lang.org/"><img alt="mypy" src="https://img.shields.io/badge/type%20checked-mypy-blue.svg"></a>
  <a href="https://docs.astral.sh/ruff/"><img alt="Ruff" src="https://img.shields.io/badge/linting-ruff-261230.svg"></a>
</p>

A small, readable Python agent for answering questions about [Coding Challenges](https://codingchallenges.fyi/).

This project is intentionally lightweight. There is no framework hiding the interesting parts: the chat loop, LLM client, conversation memory, tool calling, and CLI wiring are all plain Python.
It is a good project if you want to learn how an LLM-powered agent works from end to end.

The agent starts a terminal chat session, loads a system prompt, gives the model access to the local Coding Challenges markdown files, and lets the model call tools such as `read_challenge_index` and `read_file` before answering.

There are some patterns which are heavily inpired by [ml-intern](https://github.com/huggingface/ml-intern/tree/main).

---

## What the agent can do

- Chat with a local Ollama model, OpenAI, or an OpenAI-compatible endpoint.
- Answer questions about the Coding Challenges dataset in `data/extracted_data/`.
- Let the model browse challenge markdown files through safe read-only tools.
- Keep conversation history in memory during the current terminal session.
- Detect repeated tool-call loops and nudge the model to try a different approach.
- Run without an API key in `echo` mode for a quick smoke test.

The tools are deliberately limited: the agent can only read markdown files from `data/extracted_data/`.
It cannot edit files, run shell commands, or read arbitrary paths on your machine.

---

## Requirements

- Python 3.12 or newer
- [`uv`](https://docs.astral.sh/uv/) for dependency and virtual environment management
- One of the following LLM options:
  - [Ollama](https://ollama.com/) running locally, or
  - an OpenAI API key, or
  - any OpenAI-compatible chat completions server (gemini, for instance)

The runtime code only uses the Python standard library. `uv` is still recommended because it installs the dev tools used by this repository: `pytest`, `ruff`, and `mypy`.

---

## Quick start

From the project root:

```bash
uv sync --dev
uv run python main.py --provider echo
```

You should see:

```text
Chat started with echo:gemma4:latest. Type /exit or /quit to stop.
you:
```

Type a message:

```text
you: hello
assistant: Echo: hello
```

Exit with either command:

```text
/exit
```

or

```text
/quit
```

`echo` mode is only a startup test. It does not call an actual model and it does not use the challenge-reading tools. To run the real agent, use Ollama or OpenAI as shown below.

---

## Running with Ollama

Ollama is the default provider in this project.

1. Install Ollama from <https://ollama.com/>.
2. Make sure the Ollama server is running.
3. Pull the model you want to use.

The configured default model is `gemma4:latest`:

```bash
ollama pull gemma4:latest
```

If you prefer another model, pull that model and pass it with `--model`:

```bash
ollama pull llama3.1:8b
uv run python main.py --provider ollama --model llama3.1:8b
```

If your local Ollama server is listening at the default address, this also works:

```bash
uv run python main.py
```

By default the agent sends requests to:

```text
http://localhost:11434/v1/chat/completions
```

That is Ollama's OpenAI-compatible API route.

---

## Running with OpenAI

Set your API key and pass the OpenAI base URL:

```bash
export OPENAI_API_KEY="your-api-key"
uv run python main.py \
  --provider openai \
  --model gpt-4o-mini \
  --base-url https://api.openai.com/v1
```

You can also use `LLM_API_KEY` instead of `OPENAI_API_KEY`:

```bash
export LLM_API_KEY="your-api-key"
```

Important: the default base URL is the local Ollama URL. When using OpenAI, remember to set `--base-url https://api.openai.com/v1` or `LLM_BASE_URL=https://api.openai.com/v1`.

---

## Running with another OpenAI-compatible server

For a local server that exposes `/v1/chat/completions`, point the agent at that server:

```bash
uv run python main.py \
  --provider ollama \
  --model your-model-name \
  --base-url http://localhost:1234/v1
```

Use `--provider ollama` for local OpenAI-compatible servers that do not require an API key. The Ollama provider sends a harmless placeholder key when no key is configured.

If your server requires a key, use:

```bash
uv run python main.py \
  --provider openai \
  --api-key your-api-key \
  --model your-model-name \
  --base-url http://localhost:1234/v1
```

---

## How to chat with the agent

Once the prompt shows `you:`, ask about a challenge or ask for help choosing one:

```text
you: What are good beginner challenges if I like command-line tools?
```

```text
you: Explain the wc challenge and what I should build first.
```

```text
you: Compare the Redis and Memcached challenges. Which one should I try first?
```

```text
you: Find a challenge that teaches parsing and give me a plan for solving it.
```

When the model needs details, it can call the built-in tools:

- `read_challenge_index` reads `data/extracted_data/index.md`.
- `read_file` reads a specific markdown file from `data/extracted_data/`, for example `001-challenge-wc.md`.

The CLI prints final assistant messages directly. Tool output is stored in the conversation context so the model can use it in the next step.

---

## Configuration

Every CLI option can be passed directly when starting the agent:

```bash
uv run python main.py --help
```

Common options:

| Option | What it does | Default |
| --- | --- | --- |
| `--provider` | LLM provider: `ollama`, `openai`, or `echo` | `ollama` |
| `--model` | Chat model name | `gemma4:latest` |
| `--base-url` | OpenAI-compatible API base URL | `http://localhost:11434/v1` |
| `--api-key` | API key for OpenAI-compatible providers | unset |
| `--request-timeout-seconds` | HTTP request timeout | `60` |
| `--max-iterations` | Max LLM/tool steps per user turn | `8` |
| `--system-prompt-file` | Markdown file used as the system prompt | `prompts/system_prompt_tool.md` |

You can also configure the agent with environment variables:

| Environment variable | Purpose |
| --- | --- |
| `LLM_PROVIDER` | `ollama`, `openai`, or `echo` |
| `LLM_MODEL` | Model name |
| `LLM_BASE_URL` | OpenAI-compatible base URL |
| `LLM_API_KEY` | API key used by the LLM client |
| `OPENAI_API_KEY` | Alternative API key variable |
| `LLM_REQUEST_TIMEOUT_SECONDS` | Request timeout in seconds |
| `SYSTEM_PROMPT_FILE` | Path to a markdown system prompt |
| `DEBUG` | Set to `true`, `1`, `yes`, or `on` for verbose logs |

CLI arguments take precedence over environment variables.

Example `.env`-style setup for Ollama:

```bash
export LLM_PROVIDER=ollama
export LLM_MODEL=llama3.1:8b
export LLM_BASE_URL=http://localhost:11434/v1
uv run python main.py
```

Example setup for OpenAI:

```bash
export LLM_PROVIDER=openai
export LLM_MODEL=gpt-4o-mini
export LLM_BASE_URL=https://api.openai.com/v1
export OPENAI_API_KEY="your-api-key"
uv run python main.py
```

---

## System prompt

The default prompt lives at:

```text
prompts/system_prompt_tool.md
```

It tells the model to act as a Coding Challenges assistant. On startup, the CLI also appends the challenge index from:

```text
data/extracted_data/index.md
```

That gives the model a map of the available files before the first user message.

To try a custom prompt:

```bash
uv run python main.py --system-prompt-file prompts/my_prompt.md
```

or:

```bash
export SYSTEM_PROMPT_FILE=prompts/my_prompt.md
uv run python main.py
```

---

## How the agent works

At a high level, one turn looks like this:

1. The user enters a message in the terminal.
2. `SimpleAgentLoop` adds that message to `InMemoryContext`.
3. The loop sends the conversation plus tool specs to the configured LLM.
4. If the model replies normally, the CLI prints the assistant message.
5. If the model asks to call a tool, the loop runs the tool and stores the result.
6. The loop calls the model again with the tool result included.
7. The process repeats until the model gives a final answer or `--max-iterations` is reached.

The LLM client in `agent/llms.py` talks to any service that supports the OpenAI chat completions shape. No OpenAI SDK is required.

---

## Updating the challenge markdown files

The source challenge data is stored in:

```text
data/challenge-data.json
```

The markdown files used by the agent are generated into:

```text
data/extracted_data/
```

Regenerate them with:

```bash
make split-challenge-data
```

or directly:

```bash
uv run python scripts/split_challenge_data.py
```

---

## Development commands

Install everything:

```bash
make install
```

Run tests:

```bash
make test
```

Run Ruff:

```bash
make ruff
```

Run mypy:

```bash
make mypy
```

Run the full local CI check:

```bash
make ci
```

The GitHub Actions workflow runs the same checks on Python 3.12, 3.13, and 3.14.

---

## Troubleshooting

### `Connection refused` or the request hangs with Ollama

Make sure Ollama is running and that the base URL is correct:

```bash
ollama serve
uv run python main.py --provider ollama --base-url http://localhost:11434/v1
```

### `model not found`

Pull the model first, or choose a model you already have:

```bash
ollama pull llama3.1:8b
uv run python main.py --provider ollama --model llama3.1:8b
```

### OpenAI tries to call localhost

Set the OpenAI base URL explicitly:

```bash
uv run python main.py \
  --provider openai \
  --model gpt-4o-mini \
  --base-url https://api.openai.com/v1
```

### The model answers without reading files

Tool use depends on the model. Use a model that supports OpenAI-style tool/function calling. If a smaller local model ignores tools, try a stronger model or ask more directly, for example:

```text
Read 001-challenge-wc.md and explain the first implementation step.
```

### I want to see the raw LLM requests and responses

Enable debug logging:

```bash
DEBUG=true uv run python main.py --provider ollama
```

### `System prompt file not found`

Check the path passed to `--system-prompt-file` or the `SYSTEM_PROMPT_FILE` environment variable. Paths are resolved relative to your current working directory unless you pass an absolute path.

---

## Notes

This is an educational agent, not a production chatbot service.
Conversation history is kept only in memory, there is no web UI, and the built-in tools are read-only on purpose.
That makes the code easier to understand and safer to run while learning how agents are put together.
