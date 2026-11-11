# repo-reviewer

An AI agent that reviews a codebase. Point it at a local folder or a GitHub repo and it explores the code on its own (reading files, searching for risky patterns), then writes a prioritized review with findings tied to real files and line numbers.

Built on the Claude API with tool use, in a few hundred lines of Python.

```bash
repo-review psf/requests --out review.md
```

## How it works

```
            ┌──────────────────────────── agent loop (max N turns) ───────────────────────────┐
 task ──▶   │  Claude ──tool_use──▶ RepoTools (read-only, sandboxed) ──tool_result──▶ Claude  │
            │     │                  repo_overview · list_files · read_file · search_code   │
            │     └── submit_review (structured JSON) ──▶ Review ──▶ Markdown / JSON report  │
            └─────────────────────────────────────────────────────────────────────────────────┘
```

1. The model gets four read-only tools plus a `submit_review` tool whose JSON schema defines the report (summary, architecture, strengths, findings with severity, category, file, line, suggestion).
2. Each turn, every `tool_use` block is executed and its result is returned in one message, matched by `tool_use_id`. Tool errors (bad path, bad regex) are sent back with `is_error: true`, so the model can correct itself instead of the run crashing.
3. On the final allowed turn the agent sets `tool_choice` to force `submit_review`, so a run always ends with a report and never loops forever.
4. The structured result is sorted by severity and rendered to Markdown or JSON, with turn count, tool usage and token totals in the footer.

## Design decisions

- **Structured output through a tool.** Having the model "call" `submit_review` gives validated JSON instead of parsing free text.
- **Sandboxed, read-only tools.** Every path is resolved and rejected if it escapes the repo root. There are no write or shell tools, so reviewing an untrusted repo is safe.
- **Bounded cost.** File reads, listings and search results are capped, dependency and build folders are skipped, and the turn limit bounds total spend.
- **Testable without an API key.** The agent takes any client object, so the tests drive the full loop with a scripted fake Claude.

## Setup

Requires Python 3.10+ and an [Anthropic API key](https://console.anthropic.com/).

```bash
git clone https://github.com/<you>/repo-reviewer.git
cd repo-reviewer
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
export ANTHROPIC_API_KEY=your-key-here              # Windows: set ANTHROPIC_API_KEY=...
```

## Usage

```bash
repo-review .                                  # review the current folder, print Markdown
repo-review https://github.com/psf/requests    # shallow-clones to a temp dir, then reviews
repo-review owner/repo --out review.md         # save the report
repo-review . --focus "security and error handling"
repo-review . --json                           # machine-readable output
repo-review . --max-turns 15 --model claude-sonnet-5-5
```

Progress is printed to stderr as the agent works:

```
  · repo_overview()
  · read_file(path='README.md')
  · search_code(pattern='(api_key|secret|password)\\s*=')
  · review submitted after 9 turns
```

## Tests

```bash
pytest -q
```

Covers the tools (sandboxing, ignore rules, ranges, search) and the agent loop (parallel tool calls, error recovery, forced final submit, nudging, turn limit, report rendering). CI runs them on every push.

## Project layout

```
src/repo_reviewer/
  agent.py    agent loop, system prompt, submit_review schema
  tools.py    sandboxed read-only repo tools + JSON schemas
  report.py   Markdown / JSON rendering
  cli.py      command line, GitHub cloning
tests/        pytest suite with a fake Claude client
```

## Ideas for next steps

- `--diff` mode to review only the changes on a branch (pull-request review)
- Post the review as a GitHub PR comment from a GitHub Action
- Prompt caching on the growing conversation to cut cost on long runs
- An eval set of repos with known issues to measure precision and recall

## License

MIT
