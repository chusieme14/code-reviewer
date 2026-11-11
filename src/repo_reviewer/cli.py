"""Command-line entry point: `repo-review <path-or-github-url>`."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .agent import DEFAULT_MODEL, AgentError, ReviewAgent
from .report import to_json, to_markdown
from .tools import RepoTools

GITHUB_RE = re.compile(r"^(?:https?://github\.com/)?([\w.-]+)/([\w.-]+?)(?:\.git)?/?$")


def parse_target(target: str) -> tuple[str, str]:
    """Return ("local", path) or ("github", clone_url)."""
    if Path(target).exists():
        return "local", target
    m = GITHUB_RE.match(target.strip())
    if m:
        owner, repo = m.groups()
        return "github", f"https://github.com/{owner}/{repo}.git"
    raise ValueError(f"'{target}' is not an existing folder or a GitHub repo (owner/repo or URL).")


@contextmanager
def checkout(target: str) -> Iterator[Path]:
    kind, where = parse_target(target)
    if kind == "local":
        yield Path(where)
        return
    if not shutil.which("git"):
        raise SystemExit("git is required to review a GitHub repo.")
    tmp = tempfile.mkdtemp(prefix="repo-review-")
    try:
        print(f"cloning {where} ...", file=sys.stderr)
        subprocess.run(["git", "clone", "--depth", "1", "--quiet", where, tmp], check=True)
        yield Path(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="repo-review", description="AI code review agent powered by Claude.")
    ap.add_argument("target", help="Local folder, GitHub URL, or owner/repo")
    ap.add_argument("-o", "--out", help="Write the report to this file (default: stdout)")
    ap.add_argument("--json", action="store_true", help="Output JSON instead of Markdown")
    ap.add_argument("--focus", help="Something to pay extra attention to, e.g. 'security'")
    ap.add_argument("--model", default=os.getenv("REVIEW_MODEL", DEFAULT_MODEL))
    ap.add_argument("--max-turns", type=int, default=25)
    ap.add_argument("-q", "--quiet", action="store_true", help="Hide progress output")
    args = ap.parse_args(argv)

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY first (see .env.example).", file=sys.stderr)
        return 2

    import anthropic  # imported late so --help works without the SDK configured

    log = (lambda _m: None) if args.quiet else (lambda m: print(f"  · {m}", file=sys.stderr))

    try:
        with checkout(args.target) as root:
            name = Path(args.target.rstrip("/")).name.removesuffix(".git") or root.name
            agent = ReviewAgent(
                client=anthropic.Anthropic(),
                tools=RepoTools(root),
                model=args.model,
                max_turns=args.max_turns,
                on_event=log,
            )
            review = agent.run(focus=args.focus)
    except (ValueError, AgentError, subprocess.CalledProcessError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except anthropic.APIError as e:
        print(f"Claude API error: {e}", file=sys.stderr)
        return 1

    text = to_json(review) if args.json else to_markdown(review, name)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
