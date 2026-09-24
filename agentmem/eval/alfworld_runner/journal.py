"""Flat-file trajectory journal for the ALFWorld long-context baseline.

This is the deliberately-vanilla long-context baseline: training trajectories are
appended as raw text (no encoding, no embedding, no graph). At test time the agent
queries the journal via case-insensitive substring grep. The LLM authors the query
itself; the journal returns up to k matching entries (most recent first).

This is NOT a memory backend in the BaseAlfWorldMemoryBackend sense — it lives
outside that hierarchy because (a) it has no per-task `get_memory_prompt`, only a
mid-episode tool, and (b) it never runs `memory_construction`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

@dataclass
class JournalEntry:
    episode_idx: int
    phase: str                     
    task_desc: str
    task_type: str
    success: bool
    trajectory_text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_idx": self.episode_idx,
            "phase": self.phase,
            "task_desc": self.task_desc,
            "task_type": self.task_type,
            "success": self.success,
            "trajectory_text": self.trajectory_text,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "JournalEntry":
        return cls(
            episode_idx=int(payload.get("episode_idx", -1)),
            phase=str(payload.get("phase", "")),
            task_desc=str(payload.get("task_desc", "")),
            task_type=str(payload.get("task_type", "")),
            success=bool(payload.get("success", False)),
            trajectory_text=str(payload.get("trajectory_text", "")),
        )

@dataclass
class Journal:
    """An append-only flat journal over ALFWorld trajectories.

    `path` is optional; if provided, every `add(...)` is mirrored to disk as a JSONL
    line. `load(path)` is a class-method to reconstruct the in-memory list.
    """

    entries: list[JournalEntry] = field(default_factory=list)
    path: Optional[Path] = None

    def add(
        self,
        *,
        episode_idx: int,
        phase: str,
        task_desc: str,
        task_type: str,
        success: bool,
        trajectory: list[dict[str, Any]],
    ) -> JournalEntry:
        text = _format_trajectory_text(trajectory)
        entry = JournalEntry(
            episode_idx=episode_idx,
            phase=phase,
            task_desc=str(task_desc or "").strip(),
            task_type=str(task_type or "").strip(),
            success=bool(success),
            trajectory_text=text,
        )
        self.entries.append(entry)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
        return entry

    def grep(self, query: str, k: int = 3, *, max_chars_per_entry: int = 2000) -> list[dict[str, Any]]:
        """Case-insensitive substring search over journal entries.

        Match is on the union of `task_desc + "\\n" + trajectory_text`. Returns up to
        `k` matches, most-recent first, each truncated to `max_chars_per_entry`.

        Empty query returns the most recent `k` entries (so the agent can always make
        progress even if it submits nothing useful).
        """
        q = str(query or "").strip().lower()

        def _matches(entry: JournalEntry) -> bool:
            if not q:
                return True
            haystack = f"{entry.task_desc}\n{entry.trajectory_text}".lower()
            return q in haystack

        matched = [e for e in reversed(self.entries) if _matches(e)]
        out: list[dict[str, Any]] = []
        for entry in matched[: max(1, int(k))]:
            text = entry.trajectory_text
            if len(text) > max_chars_per_entry:
                text = text[: max_chars_per_entry - 15].rstrip() + "... [truncated]"
            out.append(
                {
                    "episode_idx": entry.episode_idx,
                    "phase": entry.phase,
                    "task_desc": entry.task_desc,
                    "task_type": entry.task_type,
                    "success": entry.success,
                    "trajectory_text": text,
                }
            )
        return out

    @classmethod
    def load(cls, path: str | Path) -> "Journal":
        p = Path(path)
        entries: list[JournalEntry] = []
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                entries.append(JournalEntry.from_dict(json.loads(line)))
        return cls(entries=entries, path=p)

    def __len__(self) -> int:
        return len(self.entries)

_JOURNAL_QUERY_RE = re.compile(
    r"^\s*journal_search\s*[:\-]\s*(?P<query>.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

def parse_journal_search(text: str, *, tool_name: str = "journal_search") -> Optional[str]:
    """Extract a `<tool_name>: ...` query from an LLM emission, or None.

    Accepted forms (lenient — the LLM may emit any of these):
        journal_search: candle in lamp
        dci_search: mug in coffee machine
        Action: journal_search: pen drawer

    Returns the trimmed query string, or None if no match. If the LLM emits multiple
    matching lines we keep the first.
    """
    if not text:
        return None
    if tool_name == "journal_search":
        pattern = _JOURNAL_QUERY_RE
    else:
        pattern = re.compile(
            rf"^\s*{re.escape(tool_name)}\s*[:\-]\s*(?P<query>.+?)\s*$",
            re.IGNORECASE | re.MULTILINE,
        )
    match = pattern.search(str(text))
    if not match:
        return None
    query = match.group("query").strip().strip("`").strip("'\"").rstrip(".")
    return query or None

def render_journal_results(matches: list[dict[str, Any]], *, tool_name: str = "journal_search") -> str:
    """Render grep results back into a single observation string for the agent."""
    if not matches:
        return f"[{tool_name}] no matching past trajectories found."
    parts: list[str] = [f"[{tool_name}] {len(matches)} match(es), most-recent first:"]
    for idx, m in enumerate(matches):
        header = (
            f"--- Match {idx + 1} | episode {m['episode_idx']} | phase={m['phase']} "
            f"| task_type={m['task_type'] or 'generic'} | success={m['success']} ---"
        )
        parts.append(header)
        if m.get("task_desc"):
            parts.append(f"Task: {m['task_desc']}")
        parts.append(m.get("trajectory_text", ""))
    return "\n".join(parts).strip()

def _format_trajectory_text(trajectory: list[dict[str, Any]]) -> str:
    """Same single-line-per-step format used by the cross-episode memory backends.

    Mirrors `agentmem/eval/alfworld_runner/memory.py:format_trajectory_for_memory`
    so the long-context journal is on equal footing with the structured backends'
    notion of a trajectory.
    """
    lines: list[str] = []
    turn_idx = 0
    for step in trajectory or []:
        action = str(step.get("action") or "").strip()
        observation = str(step.get("observation") or "").strip()
        if not action:
            continue
        lines.append(f"Turn {turn_idx}:")
        lines.append(f"Action: {action}")
        lines.append(f"Observation: {observation}")
        turn_idx += 1
    return "\n".join(lines)
