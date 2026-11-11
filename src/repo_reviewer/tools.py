"""Read-only, sandboxed file tools the agent can call.

Every path the model asks for is resolved against the repo root and rejected if
it escapes it, so the agent can only ever *read* inside the repo under review.
Outputs are size-capped to keep the context window (and the bill) under control.
"""

from __future__ import annotations

import fnmatch
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

IGNORED_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv", "env",
    "dist", "build", ".next", ".nuxt", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    ".idea", ".vscode", "target", "vendor", "coverage", ".tox", ".gradle",
}

BINARY_EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".bmp", ".pdf", ".zip", ".gz",
    ".tar", ".tgz", ".7z", ".rar", ".jar", ".class", ".so", ".dll", ".dylib", ".exe",
    ".bin", ".woff", ".woff2", ".ttf", ".otf", ".eot", ".mp3", ".mp4", ".mov", ".wav",
    ".pyc", ".db", ".sqlite", ".lock",
}

KEY_FILES = [
    "README.md", "README", "pyproject.toml", "setup.py", "requirements.txt",
    "package.json", "tsconfig.json", "go.mod", "Cargo.toml", "pom.xml",
    "Dockerfile", "docker-compose.yml", ".env.example", "Makefile",
]

MAX_READ_CHARS = 40_000
MAX_LIST_ENTRIES = 400
MAX_SEARCH_RESULTS = 60


class ToolError(Exception):
    """Raised for bad tool input. The message is sent back to the model as an error."""


class RepoTools:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise ValueError(f"Not a directory: {self.root}")
        self.calls: Counter[str] = Counter()

    # ---- helpers -----------------------------------------------------------

    def _resolve(self, rel: str) -> Path:
        path = (self.root / (rel or ".")).resolve()
        if path != self.root and not path.is_relative_to(self.root):
            raise ToolError(f"Path '{rel}' is outside the repository.")
        return path

    def _rel(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix() or "."

    def _iter_files(self, start: Path) -> Iterator[Path]:
        for dirpath, dirnames, filenames in os.walk(start):
            dirnames[:] = sorted(d for d in dirnames if d not in IGNORED_DIRS)
            for name in sorted(filenames):
                p = Path(dirpath) / name
                if p.suffix.lower() not in BINARY_EXTS:
                    yield p

    # ---- tools -------------------------------------------------------------

    def repo_overview(self) -> str:
        files = list(self._iter_files(self.root))
        by_ext = Counter(p.suffix.lower() or "(none)" for p in files)
        top = sorted(
            (p.name + ("/" if p.is_dir() else ""))
            for p in self.root.iterdir()
            if p.name not in IGNORED_DIRS
        )
        present = [k for k in KEY_FILES if (self.root / k).is_file()]
        lines = [
            f"Repository: {self.root.name}",
            f"Text files: {len(files)}",
            "Top level: " + ", ".join(top),
            "Key files: " + (", ".join(present) or "none found"),
            "Files by extension: " + ", ".join(f"{e} {n}" for e, n in by_ext.most_common(12)),
        ]
        return "\n".join(lines)

    def list_files(self, path: str = ".", pattern: str | None = None) -> str:
        start = self._resolve(path)
        if not start.is_dir():
            raise ToolError(f"'{path}' is not a directory.")
        out = []
        for p in self._iter_files(start):
            rel = self._rel(p)
            if pattern and not fnmatch.fnmatch(rel, pattern) and not fnmatch.fnmatch(p.name, pattern):
                continue
            out.append(f"{rel}  ({p.stat().st_size} bytes)")
            if len(out) >= MAX_LIST_ENTRIES:
                out.append(f"... truncated at {MAX_LIST_ENTRIES} entries; narrow with path or pattern")
                break
        return "\n".join(out) or "No files found."

    def read_file(self, path: str, start_line: int = 1, end_line: int | None = None) -> str:
        p = self._resolve(path)
        if not p.is_file():
            raise ToolError(f"'{path}' is not a file. Use list_files to find valid paths.")
        if p.suffix.lower() in BINARY_EXTS:
            raise ToolError(f"'{path}' looks like a binary file.")
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        start = max(1, int(start_line))
        end = len(lines) if end_line is None else min(len(lines), int(end_line))
        if start > max(len(lines), 1):
            raise ToolError(f"'{path}' has only {len(lines)} lines.")
        numbered, size = [], 0
        for i in range(start, end + 1):
            row = f"{i:>5}  {lines[i - 1]}"
            size += len(row) + 1
            if size > MAX_READ_CHARS:
                numbered.append(f"... truncated at line {i - 1}; read the rest with start_line={i}")
                break
            numbered.append(row)
        return f"{self._rel(p)} (lines {start}-{end} of {len(lines)})\n" + "\n".join(numbered)

    def search_code(self, pattern: str, glob: str | None = None) -> str:
        try:
            rx = re.compile(pattern)
        except re.error as e:
            raise ToolError(f"Invalid regex: {e}") from e
        hits = []
        for p in self._iter_files(self.root):
            rel = self._rel(p)
            if glob and not fnmatch.fnmatch(rel, glob) and not fnmatch.fnmatch(p.name, glob):
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for n, line in enumerate(text.splitlines(), 1):
                if rx.search(line):
                    hits.append(f"{rel}:{n}: {line.strip()[:200]}")
                    if len(hits) >= MAX_SEARCH_RESULTS:
                        hits.append(f"... stopped at {MAX_SEARCH_RESULTS} matches; refine the pattern or glob")
                        return "\n".join(hits)
        return "\n".join(hits) or "No matches."

    # ---- dispatch ----------------------------------------------------------

    def dispatch(self, name: str, args: dict[str, Any]) -> str:
        fn = {
            "repo_overview": self.repo_overview,
            "list_files": self.list_files,
            "read_file": self.read_file,
            "search_code": self.search_code,
        }.get(name)
        if fn is None:
            raise ToolError(f"Unknown tool '{name}'.")
        self.calls[name] += 1
        try:
            return fn(**args)
        except TypeError as e:
            raise ToolError(f"Bad arguments for {name}: {e}") from e


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "repo_overview",
        "description": "Summary of the repository: top-level entries, key config files present, and file counts by extension. Call this first.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "list_files",
        "description": "List text files (recursively) under a directory, with sizes. Dependency, build and VCS folders are skipped.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory relative to repo root. Default '.'."},
                "pattern": {"type": "string", "description": "Optional glob such as '*.py' or 'src/**/*.ts'."},
            },
        },
    },
    {
        "name": "read_file",
        "description": "Read a file with line numbers. Use start_line/end_line for large files.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path relative to repo root."},
                "start_line": {"type": "integer", "minimum": 1},
                "end_line": {"type": "integer", "minimum": 1},
            },
            "required": ["path"],
        },
    },
    {
        "name": "search_code",
        "description": "Regex search across the repo. Returns 'path:line: text' matches. Good for finding secrets, TODOs, call sites, or risky APIs.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Python regular expression."},
                "glob": {"type": "string", "description": "Optional file filter such as '*.py'."},
            },
            "required": ["pattern"],
        },
    },
]
