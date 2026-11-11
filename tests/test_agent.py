"""Agent-loop tests with a scripted fake Claude client (no network, no API key)."""

from types import SimpleNamespace as NS

import pytest

from repo_reviewer.agent import AgentError, ReviewAgent
from repo_reviewer.cli import parse_target
from repo_reviewer.report import to_markdown
from repo_reviewer.tools import RepoTools


def tool_use(id_, name, inp):
    return NS(type="tool_use", id=id_, name=name, input=inp)


def text(t):
    return NS(type="text", text=t)


def response(*blocks):
    return NS(content=list(blocks), usage=NS(input_tokens=100, output_tokens=20))


SUBMIT = {
    "summary": "A tiny demo app.",
    "architecture": "Single module.",
    "strengths": ["Small and readable"],
    "findings": [
        {"severity": "low", "category": "maintainability", "file": "src/app.py", "line": 4,
         "title": "Unresolved TODO", "detail": "Errors are not handled.", "suggestion": "Add try/except."},
        {"severity": "high", "category": "security", "file": "src/app.py", "line": 2,
         "title": "Hard-coded API key", "detail": "Secret committed to source.", "suggestion": "Use env vars."},
    ],
}


class FakeClient:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        # Snapshot messages: the agent keeps appending to the same list.
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.script.pop(0)


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("x = 1\nAPI_KEY = 'sk'\n")
    return tmp_path


def test_full_loop_runs_tools_and_returns_sorted_review(repo):
    client = FakeClient([
        response(text("Let me look around."), tool_use("t1", "repo_overview", {})),
        response(tool_use("t2", "read_file", {"path": "src/app.py"}),
                 tool_use("t3", "search_code", {"pattern": "API_KEY"})),
        response(tool_use("t4", "submit_review", SUBMIT)),
    ])
    events = []
    review = ReviewAgent(client, RepoTools(repo), on_event=events.append).run()

    assert review.turns == 3
    assert [f.severity for f in review.findings] == ["high", "low"]  # sorted by severity
    assert review.tool_calls == {"repo_overview": 1, "read_file": 1, "search_code": 1}
    assert review.input_tokens == 300 and review.output_tokens == 60

    # Both tool results from turn 2 went back in one user message, matched by id.
    results = client.calls[2]["messages"][-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["t2", "t3"]
    assert "API_KEY" in results[0]["content"]
    assert all(not r["is_error"] for r in results)


def test_tool_errors_are_reported_back_not_raised(repo):
    client = FakeClient([
        response(tool_use("t1", "read_file", {"path": "../secret"})),
        response(tool_use("t2", "submit_review", SUBMIT)),
    ])
    ReviewAgent(client, RepoTools(repo)).run()
    result = client.calls[1]["messages"][-1]["content"][0]
    assert result["is_error"] is True and "outside" in result["content"]


def test_last_turn_forces_submit(repo):
    client = FakeClient([
        response(tool_use("t1", "repo_overview", {})),
        response(tool_use("t2", "submit_review", SUBMIT)),
    ])
    ReviewAgent(client, RepoTools(repo), max_turns=2).run()
    assert "tool_choice" not in client.calls[0]
    assert client.calls[1]["tool_choice"] == {"type": "tool", "name": "submit_review"}


def test_text_only_reply_gets_nudged(repo):
    client = FakeClient([
        response(text("I think it's fine.")),
        response(tool_use("t1", "submit_review", SUBMIT)),
    ])
    review = ReviewAgent(client, RepoTools(repo)).run()
    assert review.turns == 2
    assert "submit_review" in client.calls[1]["messages"][-1]["content"]


def test_gives_up_if_never_submitted(repo):
    client = FakeClient([response(text("hmm")) for _ in range(3)])
    with pytest.raises(AgentError):
        ReviewAgent(client, RepoTools(repo), max_turns=3).run()


def test_markdown_report(repo):
    client = FakeClient([response(tool_use("t1", "submit_review", SUBMIT))])
    md = to_markdown(ReviewAgent(client, RepoTools(repo)).run(), "demo")
    assert md.startswith("# Code review: demo")
    assert "1 high · 0 medium · 1 low" in md
    assert md.index("Hard-coded API key") < md.index("Unresolved TODO")
    assert "`src/app.py:2`" in md


@pytest.mark.parametrize("target,url", [
    ("anthropics/anthropic-sdk-python", "https://github.com/anthropics/anthropic-sdk-python.git"),
    ("https://github.com/psf/requests", "https://github.com/psf/requests.git"),
    ("https://github.com/psf/requests.git", "https://github.com/psf/requests.git"),
])
def test_parse_github_targets(target, url):
    assert parse_target(target) == ("github", url)


def test_parse_local_and_invalid(tmp_path):
    assert parse_target(str(tmp_path)) == ("local", str(tmp_path))
    with pytest.raises(ValueError):
        parse_target("not a repo at all")
