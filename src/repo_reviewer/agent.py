"""The agent loop: Claude explores the repo with tools, then submits a structured review."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .tools import TOOL_SCHEMAS, RepoTools, ToolError

DEFAULT_MODEL = "claude-sonnet-5-5"

SYSTEM_PROMPT = """You are a senior software engineer doing a code review of an unfamiliar repository.

You can only read the repo through the tools provided. Work like a careful reviewer:
1. Start with repo_overview, then read the README and main config files.
2. Find the entry points and the most important modules; read them.
3. Use search_code to check for common problems: hard-coded secrets, unsafe input
   handling, SQL built from strings, missing error handling, TODO/FIXME clusters,
   disabled TLS verification, and missing tests.
4. Be economical: skim large files in ranges, and stop exploring once you have enough
   evidence. You have a limited number of turns.

When finished, call submit_review exactly once. Every finding must cite a real file
(and line, when you have one) that you actually looked at. Do not invent problems;
fewer well-evidenced findings beat many speculative ones. Include strengths too."""

SUBMIT_TOOL: dict[str, Any] = {
    "name": "submit_review",
    "description": "Submit the final review. Call exactly once, after exploring.",
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {"type": "string", "description": "2-4 sentences: what the project is and its overall health."},
            "architecture": {"type": "string", "description": "Short description of structure and main components."},
            "strengths": {"type": "array", "items": {"type": "string"}},
            "findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "severity": {"type": "string", "enum": ["high", "medium", "low"]},
                        "category": {"type": "string", "enum": ["security", "bug", "reliability", "performance", "maintainability", "testing", "docs"]},
                        "file": {"type": "string"},
                        "line": {"type": "integer"},
                        "title": {"type": "string"},
                        "detail": {"type": "string"},
                        "suggestion": {"type": "string"},
                    },
                    "required": ["severity", "category", "file", "title", "detail", "suggestion"],
                },
            },
        },
        "required": ["summary", "architecture", "strengths", "findings"],
    },
}

SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


@dataclass
class Finding:
    severity: str
    category: str
    file: str
    title: str
    detail: str
    suggestion: str
    line: int | None = None


@dataclass
class Review:
    summary: str
    architecture: str
    strengths: list[str]
    findings: list[Finding]
    turns: int = 0
    tool_calls: dict[str, int] = field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""

    @classmethod
    def from_tool_input(cls, data: dict[str, Any], **meta: Any) -> "Review":
        findings = []
        for f in data.get("findings", []):
            findings.append(Finding(
                severity=f.get("severity", "low"),
                category=f.get("category", "maintainability"),
                file=f.get("file", "?"),
                line=f.get("line"),
                title=f.get("title", ""),
                detail=f.get("detail", ""),
                suggestion=f.get("suggestion", ""),
            ))
        findings.sort(key=lambda f: (SEVERITY_ORDER.get(f.severity, 3), f.file, f.line or 0))
        return cls(
            summary=data.get("summary", ""),
            architecture=data.get("architecture", ""),
            strengths=list(data.get("strengths", [])),
            findings=findings,
            **meta,
        )


class AgentError(RuntimeError):
    pass


def _block_to_dict(block: Any) -> dict[str, Any]:
    if block.type == "text":
        return {"type": "text", "text": block.text}
    if block.type == "tool_use":
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    raise AgentError(f"Unexpected content block type: {block.type}")


class ReviewAgent:
    """Runs a manual tool-use loop against the Claude Messages API."""

    def __init__(
        self,
        client: Any,
        tools: RepoTools,
        model: str = DEFAULT_MODEL,
        max_turns: int = 25,
        max_tokens: int = 4096,
        on_event: Callable[[str], None] | None = None,
    ) -> None:
        if max_turns < 2:
            raise ValueError("max_turns must be at least 2")
        self.client = client
        self.tools = tools
        self.model = model
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.on_event = on_event or (lambda _msg: None)

    def run(self, focus: str | None = None) -> Review:
        task = "Review this repository and submit your review."
        if focus:
            task += f"\nPay particular attention to: {focus}"
        messages: list[dict[str, Any]] = [{"role": "user", "content": task}]
        in_tok = out_tok = 0

        for turn in range(1, self.max_turns + 1):
            last_turn = turn == self.max_turns
            kwargs: dict[str, Any] = dict(
                model=self.model,
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT,
                tools=[*TOOL_SCHEMAS, SUBMIT_TOOL],
                messages=messages,
            )
            if last_turn:
                # Out of budget: force the model to wrap up with what it has.
                kwargs["tool_choice"] = {"type": "tool", "name": "submit_review"}
                self.on_event("turn limit reached, asking for the final review")

            resp = self.client.messages.create(**kwargs)
            in_tok += getattr(resp.usage, "input_tokens", 0)
            out_tok += getattr(resp.usage, "output_tokens", 0)
            messages.append({"role": "assistant", "content": [_block_to_dict(b) for b in resp.content]})

            tool_uses = [b for b in resp.content if b.type == "tool_use"]
            if not tool_uses:
                self.on_event("model replied without a tool call, nudging it to submit")
                messages.append({"role": "user", "content": "Please call submit_review with your findings now."})
                continue

            results = []
            for tu in tool_uses:
                if tu.name == "submit_review":
                    self.on_event(f"review submitted after {turn} turns")
                    return Review.from_tool_input(
                        tu.input,
                        turns=turn,
                        tool_calls=dict(self.tools.calls),
                        input_tokens=in_tok,
                        output_tokens=out_tok,
                        model=self.model,
                    )
                self.on_event(f"{tu.name}({_short_args(tu.input)})")
                try:
                    output, is_error = self.tools.dispatch(tu.name, tu.input), False
                except ToolError as e:
                    output, is_error = f"Error: {e}", True
                results.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": output,
                    "is_error": is_error,
                })
            messages.append({"role": "user", "content": results})

        raise AgentError(f"No review submitted within {self.max_turns} turns.")


def _short_args(args: dict[str, Any]) -> str:
    text = ", ".join(f"{k}={v!r}" for k, v in args.items())
    return text if len(text) <= 80 else text[:77] + "..."
