#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = ["textual>=1.0", "rapidfuzz>=3.0"]
# ///
"""Browse and search Claude Code and VS Code Copilot chat sessions."""

import argparse
import dataclasses
import hashlib
import io
import json
import os
import re
import subprocess
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, Literal
from urllib.parse import unquote, urlparse

from rapidfuzz import fuzz, utils
from rich.console import Console
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import DataTable, Footer, Input, RichLog, Select
from textual.worker import get_current_worker

CACHE_VERSION = 2
INJECTED_CONTEXT_TAGS = ("<ide_opened_file", "<ide_selection")
FUZZY_SCORE_CUTOFF = 80
TITLE_WIDTH = 40
PROJECT_WIDTH = 25
COPILOT_SESSION_GLOBS = (
    "workspaceStorage/*/chatSessions/*",
    "globalStorage/emptyWindowChatSessions/*",
)
COPILOT_SESSION_SUFFIXES = (".json", ".jsonl")
MARKDOWN_FILE_LINK = re.compile(r"\[([^\]]*)\]\(file://([^)#]*)[^)]*\)")

type JsonObject = dict[str, Any]
type Source = Literal["claude", "copilot"]
type Role = Literal["you", "assistant", "tool"]

SPEAKERS: dict[Source, str] = {"claude": "Claude", "copilot": "Copilot"}


class SessionFormatError(Exception):
    pass


@dataclass(frozen=True)
class SessionSummary:
    source: Source
    session_id: str
    path: str
    cwd: str
    branch: str
    title: str
    last_activity: str
    prompts: tuple[str, ...]

    @property
    def last_activity_time(self) -> datetime:
        return datetime.fromisoformat(self.last_activity)

    @property
    def fuzzy_text(self) -> str:
        return f"{self.title} {self.cwd} {self.branch} {self.source}"

    @classmethod
    def from_json(cls, data: JsonObject) -> "SessionSummary":
        return cls(
            source=data["source"],
            session_id=data["session_id"],
            path=data["path"],
            cwd=data["cwd"],
            branch=data["branch"],
            title=data["title"],
            last_activity=data["last_activity"],
            prompts=tuple(data["prompts"]),
        )


@dataclass(frozen=True)
class TranscriptEntry:
    role: Role
    text: str


@dataclass(frozen=True)
class SessionIndex:
    sessions: list[SessionSummary]
    reparsed: int


def require[T](record: JsonObject, key: str, expected: type[T], location: str) -> T:
    value = record.get(key)
    if not isinstance(value, expected):
        raise SessionFormatError(f"{location}: expected {expected.__name__} field {key!r}, got {value!r}")
    return value


def iter_records(path: Path) -> Iterator[tuple[str, JsonObject]]:
    with path.open("rb") as file:
        for line_number, line in enumerate(file, start=1):
            location = f"{path}:{line_number}"
            if not line.endswith(b"\n"):
                # Claude Code may still be appending this line.
                return
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise SessionFormatError(f"{location}: malformed JSON") from error
            if not isinstance(record, dict):
                raise SessionFormatError(f"{location}: expected a JSON object, got {record!r}")
            yield location, record


def is_human_prompt(record: JsonObject) -> bool:
    origin = record.get("origin")
    return isinstance(origin, dict) and origin.get("kind") == "human"


def prompt_text(record: JsonObject, location: str) -> str:
    content = require(record, "message", dict, location).get("content")
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        raise SessionFormatError(f"{location}: expected str or list message content, got {content!r}")
    texts = [
        require(block, "text", str, location)
        for block in content
        if require(block, "type", str, location) == "text"
        and not require(block, "text", str, location).lstrip().startswith(INJECTED_CONTEXT_TAGS)
    ]
    return "\n\n".join(texts).strip()


def first_line(text: str) -> str:
    stripped = text.strip()
    return stripped.splitlines()[0] if stripped else ""


def truncate(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"


def truncate_start(text: str, width: int) -> str:
    return text if len(text) <= width else "…" + text[-(width - 1) :]


def parse_claude_session(path: Path) -> SessionSummary | None:
    cwd: str | None = None
    branch = ""
    last_activity = ""
    ai_title = ""
    custom_title = ""
    prompts: list[str] = []
    for location, record in iter_records(path):
        match require(record, "type", str, location):
            case "ai-title":
                ai_title = require(record, "aiTitle", str, location)
            case "custom-title":
                custom_title = require(record, "customTitle", str, location)
            case "user" | "assistant" as kind:
                cwd = require(record, "cwd", str, location)
                branch = require(record, "gitBranch", str, location)
                last_activity = require(record, "timestamp", str, location)
                if kind == "user" and is_human_prompt(record):
                    text = prompt_text(record, location)
                    if text:
                        prompts.append(text)
            case _:
                pass
    if cwd is None:
        return None
    title = custom_title or ai_title or (first_line(prompts[0]) if prompts else "(no prompt)")
    return SessionSummary(
        source="claude",
        session_id=path.stem,
        path=str(path),
        cwd=cwd,
        branch=branch,
        title=title,
        last_activity=last_activity,
        prompts=tuple(prompts),
    )


def tool_call_summary(block: JsonObject, location: str) -> str:
    name = require(block, "name", str, location)
    tool_input = require(block, "input", dict, location)
    argument = next((value for value in tool_input.values() if isinstance(value, str)), "")
    return f"{name}({truncate(first_line(argument), 100)})"


def load_claude_transcript(path: Path) -> list[TranscriptEntry]:
    entries: list[TranscriptEntry] = []
    for location, record in iter_records(path):
        kind = require(record, "type", str, location)
        if kind == "user" and is_human_prompt(record):
            text = prompt_text(record, location)
            if text:
                entries.append(TranscriptEntry("you", text))
        elif kind == "assistant":
            content = require(require(record, "message", dict, location), "content", list, location)
            for block in content:
                match require(block, "type", str, location):
                    case "text":
                        entries.append(TranscriptEntry("assistant", require(block, "text", str, location)))
                    case "tool_use":
                        entries.append(TranscriptEntry("tool", tool_call_summary(block, location)))
                    case _:
                        pass
    return entries


def replay_mutation_log(path: Path) -> JsonObject | None:
    state: JsonObject | None = None
    for location, record in iter_records(path):
        kind = require(record, "kind", int, location)
        if kind == 0:
            state = require(record, "v", dict, location)
            continue
        if state is None:
            raise SessionFormatError(f"{location}: mutation before initial state")
        *parents, leaf = require(record, "k", list, location)
        node: Any = state
        for key in parents:
            node = node[key]
        match kind:
            case 1:
                node[leaf] = record.get("v")
            case 2:
                items = node[leaf]
                start = record.get("i")
                if start is not None:
                    del items[start:]
                items.extend(record.get("v", []))
            case _:
                raise SessionFormatError(f"{location}: unknown mutation kind {kind}")
    return state


def load_copilot_state(path: Path) -> JsonObject | None:
    if path.suffix == ".jsonl":
        return replay_mutation_log(path)
    return json.loads(path.read_text(encoding="utf-8"))


def uri_to_path(uri: str) -> str:
    parsed = urlparse(uri)
    return unquote(parsed.path) if parsed.scheme == "file" else uri


def copilot_workspace(path: Path) -> str:
    if path.parent.name == "emptyWindowChatSessions":
        return ""
    workspace_file = path.parent.parent / "workspace.json"
    match json.loads(workspace_file.read_text(encoding="utf-8")):
        case {"folder": str(uri)} | {"workspace": str(uri)}:
            return uri_to_path(uri)
        case other:
            raise SessionFormatError(f"{workspace_file}: unexpected workspace {other!r}")


def copilot_prompt(request: JsonObject, location: str) -> str:
    return require(require(request, "message", dict, location), "text", str, location).strip()


def parse_copilot_session(path: Path) -> SessionSummary | None:
    state = load_copilot_state(path)
    if state is None:
        return None
    location = str(path)
    requests = require(state, "requests", list, location)
    if not requests:
        return None
    prompts = [prompt for prompt in (copilot_prompt(request, location) for request in requests) if prompt]
    last_timestamp_ms = require(requests[-1], "timestamp", int, location)
    title = state.get("customTitle") or (first_line(prompts[0]) if prompts else "(no prompt)")
    return SessionSummary(
        source="copilot",
        session_id=path.stem,
        path=str(path),
        cwd=copilot_workspace(path),
        branch="",
        title=title,
        last_activity=datetime.fromtimestamp(last_timestamp_ms / 1000, UTC).isoformat(),
        prompts=tuple(prompts),
    )


def inline_reference_name(part: JsonObject, location: str) -> str:
    match part:
        case {"name": str(name)} | {"inlineReference": {"name": str(name)}}:
            return name
        case {"inlineReference": {"path": str(path)}}:
            return Path(path).name
        case _:
            raise SessionFormatError(f"{location}: unexpected inline reference {part!r}")


def copilot_tool_summary(part: JsonObject, location: str) -> str:
    match part.get("invocationMessage"):
        case str(message) | {"value": str(message)}:
            readable = MARKDOWN_FILE_LINK.sub(lambda link: link[1] or shorten_home(unquote(link[2])), message)
            return f"{require(part, 'toolId', str, location)}({truncate(first_line(readable), 100)})"
        case other:
            raise SessionFormatError(f"{location}: unexpected tool invocation message {other!r}")


def copilot_response_entries(parts: list[JsonObject], location: str) -> Iterator[TranscriptEntry]:
    chunks: list[str] = []
    for part in parts:
        match part.get("kind"):
            case None:
                chunks.append(require(part, "value", str, location))
            case "inlineReference":
                chunks.append(f"`{inline_reference_name(part, location)}`")
            case "codeblockUri":
                chunks.append(require(require(part, "uri", dict, location), "path", str, location))
            case "toolInvocationSerialized":
                if "".join(chunks).strip():
                    yield TranscriptEntry("assistant", "".join(chunks).strip())
                chunks = []
                yield TranscriptEntry("tool", copilot_tool_summary(part, location))
            case _:
                pass
    if "".join(chunks).strip():
        yield TranscriptEntry("assistant", "".join(chunks).strip())


def load_copilot_transcript(path: Path) -> list[TranscriptEntry]:
    state = load_copilot_state(path)
    location = str(path)
    if state is None:
        raise SessionFormatError(f"{location}: empty session")
    entries: list[TranscriptEntry] = []
    for request in require(state, "requests", list, location):
        prompt = copilot_prompt(request, location)
        if prompt:
            entries.append(TranscriptEntry("you", prompt))
        entries.extend(copilot_response_entries(require(request, "response", list, location), location))
    return entries


PARSERS: dict[Source, Callable[[Path], SessionSummary | None]] = {
    "claude": parse_claude_session,
    "copilot": parse_copilot_session,
}
TRANSCRIPT_LOADERS: dict[Source, Callable[[Path], list[TranscriptEntry]]] = {
    "claude": load_claude_transcript,
    "copilot": load_copilot_transcript,
}


def session_files(claude_root: Path, vscode_root: Path) -> list[tuple[Path, Source]]:
    claude: list[tuple[Path, Source]] = [(path, "claude") for path in claude_root.glob("*/*.jsonl")]
    copilot: list[tuple[Path, Source]] = []
    for pattern in COPILOT_SESSION_GLOBS:
        for path in vscode_root.glob(pattern):
            if path.suffix in COPILOT_SESSION_SUFFIXES:
                copilot.append((path, "copilot"))
    return sorted(claude + copilot)


def latest_copy_per_session(sessions: list[SessionSummary]) -> list[SessionSummary]:
    # VS Code copies a chat into every workspace it is moved to, under the same session ID.
    oldest_first = sorted(sessions, key=lambda session: (session.last_activity_time, len(session.prompts)))
    return list({session.session_id: session for session in oldest_first}.values())


def cache_file_for(*roots: Path) -> Path:
    digest = hashlib.sha256("\0".join(map(str, roots)).encode()).hexdigest()[:16]
    return Path.home() / ".cache" / "csesh" / f"index-{digest}.json"


def load_index(claude_root: Path, vscode_root: Path) -> SessionIndex:
    cache_file = cache_file_for(claude_root, vscode_root)
    cached: dict[str, JsonObject] = {}
    if cache_file.exists():
        stored = json.loads(cache_file.read_text(encoding="utf-8"))
        if stored["version"] == CACHE_VERSION:
            cached = stored["entries"]

    entries: dict[str, JsonObject] = {}
    reparsed = 0
    for path, source in session_files(claude_root, vscode_root):
        stat = path.stat()
        entry = cached.get(str(path))
        if entry is None or entry["mtime_ns"] != stat.st_mtime_ns or entry["size"] != stat.st_size:
            summary = PARSERS[source](path)
            entry = {
                "mtime_ns": stat.st_mtime_ns,
                "size": stat.st_size,
                "summary": None if summary is None else dataclasses.asdict(summary),
            }
            reparsed += 1
        entries[str(path)] = entry

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_file.with_suffix(".tmp")
    temporary.write_text(json.dumps({"version": CACHE_VERSION, "entries": entries}), encoding="utf-8")
    temporary.replace(cache_file)

    sessions = latest_copy_per_session(
        [SessionSummary.from_json(entry["summary"]) for entry in entries.values() if entry["summary"]]
    )
    sessions.sort(key=lambda session: session.last_activity_time, reverse=True)
    return SessionIndex(sessions=sessions, reparsed=reparsed)


def fuzzy_matches(sessions: list[SessionSummary], query: str) -> list[SessionSummary]:
    words = utils.default_process(query).split()
    if not words:
        return sessions
    scored: list[tuple[float, SessionSummary]] = []
    for session in sessions:
        text = utils.default_process(session.fuzzy_text)
        score = min(fuzz.partial_ratio(word, text) for word in words)
        if score >= FUZZY_SCORE_CUTOFF:
            scored.append((score, session))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [session for _, session in scored]


def prompt_matches(sessions: list[SessionSummary], pattern: re.Pattern[str]) -> list[SessionSummary]:
    return [session for session in sessions if any(pattern.search(prompt) for prompt in session.prompts)]


def relative_time(moment: datetime, now: datetime) -> str:
    seconds = int((now - moment).total_seconds())
    if seconds < 60:
        return "now"
    if seconds < 3600:
        return f"{seconds // 60}m"
    if seconds < 86400:
        return f"{seconds // 3600}h"
    if seconds < 30 * 86400:
        return f"{seconds // 86400}d"
    return ">30d"


def shorten_home(path: str) -> str:
    home = str(Path.home())
    if path == home or path.startswith(home + "/"):
        return "~" + path[len(home) :]
    return path


def project_label(cwd: str) -> str:
    return shorten_home(cwd) if cwd else "(no folder)"


# Each column's first letter is its sort key.
SORT_KEYS: dict[str, Callable[[SessionSummary], Any]] = {
    "when": lambda session: session.last_activity_time,
    "agent": lambda session: session.source,
    "title": lambda session: session.title.casefold(),
    "project": lambda session: project_label(session.cwd).casefold(),
    "branch": lambda session: session.branch.casefold(),
}


def transcript_lines(session: SessionSummary, transcript: list[TranscriptEntry]) -> Iterator[Text]:
    yield Text(session.title, style="bold")
    branch = f" @ {session.branch}" if session.branch else ""
    yield Text(f"{project_label(session.cwd)}{branch} · {session.session_id}", style="dim")
    speaker: Role | None = None
    for entry in transcript:
        if entry.role == "tool":
            yield Text(f"  → {entry.text}", style="dim")
            continue
        if entry.role != speaker:
            yield Text("")
            yield Text("You" if entry.role == "you" else SPEAKERS[session.source], style="bold cyan")
            speaker = entry.role
        yield Text(entry.text)


class SessionBrowser(App[None]):
    CSS = """
    #filters { height: auto; }
    #query { width: 1fr; margin-right: 1; }
    #project { width: 50; }
    #sessions { height: 1fr; scrollbar-size-vertical: 1; }
    #preview { height: 1fr; border-top: solid $accent; scrollbar-size-vertical: 1; }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("ctrl+t", "toggle_mode", "Fuzzy / prompt search"),
        Binding("slash", "focus_search", "Search"),
        Binding("escape", "focus_sessions", "Session list", show=False),
        # Priority, so it also quits from the search field, which would otherwise copy.
        Binding("ctrl+c", "quit", "Quit", show=False, priority=True),
        Binding("c", "copy_id", "Copy ID"),
        Binding("o", "open_pager", "Open in less"),
        *(Binding(column[0], f"sort('{column}')", f"Sort by {column}", show=False) for column in SORT_KEYS),
    ]

    def __init__(self, index: SessionIndex) -> None:
        super().__init__()
        self.sessions = index.sessions
        self.reparsed = index.reparsed
        self.sessions_by_id = {session.session_id: session for session in self.sessions}
        self.missing_cwds = {session.cwd for session in self.sessions if session.cwd and not Path(session.cwd).is_dir()}
        self.prompt_search = False
        self.sort: tuple[str, bool] | None = None  # (column, descending)

    def compose(self) -> ComposeResult:
        projects = sorted({session.cwd for session in self.sessions}, key=project_label)
        with Horizontal(id="filters"):
            yield Input(placeholder=self.search_placeholder(), id="query", compact=True)
            yield Select[str](
                [(project_label(cwd), cwd) for cwd in projects],
                prompt="All projects",
                id="project",
                compact=True,
            )
        yield DataTable(id="sessions", cursor_type="row", zebra_stripes=True)
        yield RichLog(id="preview", wrap=True, markup=False, auto_scroll=True)
        yield Footer()

    def on_mount(self) -> None:
        self.refresh_table()
        self.query_one("#query", Input).focus()

    def search_placeholder(self) -> str:
        if self.prompt_search:
            return "Regex over your prompts (ctrl+t: fuzzy title/project/branch)"
        return "Fuzzy title/project/branch (ctrl+t: regex over your prompts)"

    def selected_project(self) -> str | None:
        value = self.query_one("#project", Select).value
        return value if isinstance(value, str) else None

    def filtered_sessions(self) -> list[SessionSummary] | str:
        project = self.selected_project()
        candidates = [session for session in self.sessions if project is None or session.cwd == project]
        query = self.query_one("#query", Input).value.strip()
        if not query:
            return candidates
        if not self.prompt_search:
            return fuzzy_matches(candidates, query)
        try:
            pattern = re.compile(query, re.IGNORECASE)
        except re.error as error:
            return f"invalid regex: {error}"
        return prompt_matches(candidates, pattern)

    def shown_sort(self) -> tuple[str, bool] | None:
        if self.sort is not None:
            return self.sort
        ranked_by_match = not self.prompt_search and self.query_one("#query", Input).value.strip()
        return None if ranked_by_match else ("when", True)

    def column_label(self, column: str, accent: str) -> Text:
        label = Text.assemble((column[0].upper(), f"bold {accent}"), column[1:])
        shown_sort = self.shown_sort()
        if shown_sort is not None and shown_sort[0] == column:
            label.append("▼" if shown_sort[1] else "▲")
        return label

    def refresh_table(self) -> None:
        table = self.query_one("#sessions", DataTable)
        # Rebuild the columns too, so widths shrink back and the sort arrow moves.
        table.clear(columns=True)
        accent = self.get_css_variables()["accent"]
        for column in SORT_KEYS:
            table.add_column(self.column_label(column, accent), key=column)
        shown = self.filtered_sessions()
        status = f"{len(self.sessions)} sessions ({self.reparsed} reparsed)"
        if isinstance(shown, str):
            self.sub_title = f"{status} · {shown}"
            self.query_one("#preview", RichLog).clear()
            return
        if self.sort is not None:
            column, descending = self.sort
            shown = sorted(shown, key=SORT_KEYS[column], reverse=descending)
        now = datetime.now(UTC)
        for session in shown:
            missing = session.cwd in self.missing_cwds
            style = "dim" if missing else ""
            marker = "✗ " if missing else ""
            project = marker + truncate_start(project_label(session.cwd), PROJECT_WIDTH - len(marker))
            table.add_row(
                Text(relative_time(session.last_activity_time, now), style=style),
                Text(session.source, style=style),
                Text(truncate(session.title, TITLE_WIDTH), style=style),
                Text(project, style=style),
                Text(session.branch, style=style),
                key=session.session_id,
            )
        self.sub_title = f"{status} · {len(shown)} shown"
        if shown:
            self.load_preview(shown[0])
        else:
            self.query_one("#preview", RichLog).clear()

    def highlighted_session(self) -> SessionSummary | None:
        table = self.query_one("#sessions", DataTable)
        if table.row_count == 0:
            return None
        row_key, _ = table.coordinate_to_cell_key(table.cursor_coordinate)
        return self.sessions_by_id[str(row_key.value)]

    @on(Input.Changed, "#query")
    def query_changed(self) -> None:
        self.refresh_table()

    @on(Input.Submitted, "#query")
    def query_submitted(self) -> None:
        self.action_focus_sessions()

    @on(Select.Changed, "#project")
    def project_changed(self) -> None:
        self.refresh_table()

    @on(DataTable.HeaderSelected, "#sessions")
    def header_selected(self, event: DataTable.HeaderSelected) -> None:
        self.action_sort(str(event.column_key.value))

    @on(DataTable.RowHighlighted, "#sessions")
    def row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        self.load_preview(self.sessions_by_id[str(event.row_key.value)])

    @on(DataTable.RowSelected, "#sessions")
    def row_selected(self, event: DataTable.RowSelected) -> None:
        self.copy_session_id(self.sessions_by_id[str(event.row_key.value)])

    def action_copy_id(self) -> None:
        session = self.highlighted_session()
        if session is not None:
            self.copy_session_id(session)

    def action_open_pager(self) -> None:
        session = self.highlighted_session()
        if session is None:
            return
        transcript = TRANSCRIPT_LOADERS[session.source](Path(session.path))
        buffer = io.StringIO()
        console = Console(file=buffer, force_terminal=True, width=self.size.width)
        for line in transcript_lines(session, transcript):
            console.print(line)
        with self.suspend():
            # +G opens at the end, where the most recent messages are.
            subprocess.run(["less", "-R", "+G"], input=buffer.getvalue(), text=True, check=True)

    def action_toggle_mode(self) -> None:
        self.prompt_search = not self.prompt_search
        self.query_one("#query", Input).placeholder = self.search_placeholder()
        self.refresh_table()

    def action_sort(self, column: str) -> None:
        first_descending = column == "when"
        current = self.shown_sort()
        if current == (column, first_descending):
            self.sort = (column, not first_descending)
        elif current is not None and current[0] == column:
            self.sort = None
        else:
            self.sort = (column, first_descending)
        self.refresh_table()

    def action_focus_search(self) -> None:
        self.query_one("#query", Input).focus()

    def action_focus_sessions(self) -> None:
        self.query_one("#sessions", DataTable).focus()

    def copy_session_id(self, session: SessionSummary) -> None:
        if os.environ.get("WAYLAND_DISPLAY"):
            # wl-copy forks a clipboard server that inherits any captured pipe and would block us.
            subprocess.run(
                ["wl-copy", session.session_id],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
        else:
            self.copy_to_clipboard(session.session_id)
        self.notify(session.session_id, title="Copied session ID")

    @work(exclusive=True, thread=True, group="preview")
    def load_preview(self, session: SessionSummary) -> None:
        transcript = TRANSCRIPT_LOADERS[session.source](Path(session.path))
        if not get_current_worker().is_cancelled:
            self.call_from_thread(self.show_transcript, session, transcript)

    def show_transcript(self, session: SessionSummary, transcript: list[TranscriptEntry]) -> None:
        log = self.query_one("#preview", RichLog)
        log.clear()
        for line in transcript_lines(session, transcript):
            log.write(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claude-root", type=Path, default=Path.home() / ".claude" / "projects")
    parser.add_argument("--vscode-root", type=Path, default=Path.home() / ".config" / "Code" / "User")
    arguments = parser.parse_args()
    roots: list[Path] = [arguments.claude_root.expanduser().resolve(), arguments.vscode_root.expanduser().resolve()]
    for root in roots:
        if not root.is_dir():
            raise SystemExit(f"{root} is not a directory")
    SessionBrowser(load_index(*roots)).run()


if __name__ == "__main__":
    main()
