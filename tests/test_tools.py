import pytest

from repo_reviewer.tools import RepoTools, ToolError


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "README.md").write_text("# Demo\nA tiny app.\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text(
        "import os\n"
        "API_KEY = 'sk-hardcoded'\n"
        "def main():\n"
        "    # TODO: handle errors\n"
        "    return os.getenv('X')\n"
    )
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "junk.js").write_text("API_KEY = 1")
    (tmp_path / "logo.png").write_bytes(b"\x89PNG")
    return tmp_path


def test_overview_mentions_key_files(repo):
    out = RepoTools(repo).repo_overview()
    assert "README.md" in out
    assert "node_modules" not in out


def test_list_files_skips_ignored_and_binary(repo):
    out = RepoTools(repo).list_files()
    assert "src/app.py" in out
    assert "node_modules" not in out
    assert "logo.png" not in out


def test_list_files_pattern(repo):
    out = RepoTools(repo).list_files(pattern="*.py")
    assert "src/app.py" in out and "README.md" not in out


def test_read_file_with_range(repo):
    out = RepoTools(repo).read_file("src/app.py", start_line=2, end_line=3)
    assert "lines 2-3 of 5" in out
    assert "API_KEY" in out and "import os" not in out


def test_read_file_rejects_path_escape(repo):
    with pytest.raises(ToolError, match="outside"):
        RepoTools(repo).read_file("../../etc/passwd")


def test_read_missing_file(repo):
    with pytest.raises(ToolError, match="not a file"):
        RepoTools(repo).read_file("nope.py")


def test_search_finds_secret_but_not_in_ignored_dirs(repo):
    out = RepoTools(repo).search_code(r"API_KEY\s*=")
    assert "src/app.py:2:" in out
    assert "node_modules" not in out


def test_search_bad_regex(repo):
    with pytest.raises(ToolError, match="Invalid regex"):
        RepoTools(repo).search_code("(")


def test_dispatch_counts_calls_and_rejects_unknown(repo):
    tools = RepoTools(repo)
    tools.dispatch("list_files", {})
    tools.dispatch("list_files", {"pattern": "*.md"})
    assert tools.calls["list_files"] == 2
    with pytest.raises(ToolError):
        tools.dispatch("delete_everything", {})
    with pytest.raises(ToolError, match="Bad arguments"):
        tools.dispatch("read_file", {"wrong": 1})
