"""Group-travel environment helpers for the MemoryArena harness.

The public HF snapshot contains task prompts and gold plans, but not the
TravelPlanner-style restaurant/accommodation catalogs used by the original
interactive environment. This module therefore provides two explicit modes:

``catalog``
    Reconstruct a diagnostic candidate catalog from plan values present in the
    task. Candidates are not marked as correct.

``paper_sim``
    Expose only the base traveler's finalized plan slots through the tool
    surface. This avoids current-session gold leakage when the original
    TravelPlanner database is unavailable.

``oracle``
    Return the requested current-session slot values from the gold plan. This
    is an upper-bound/debug mode and must not be used for paper-grade claims.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

DAY_WORDS = {
    "first": 1,
    "1st": 1,
    "one": 1,
    "second": 2,
    "2nd": 2,
    "two": 2,
    "third": 3,
    "3rd": 3,
    "three": 3,
    "fourth": 4,
    "4th": 4,
    "four": 4,
    "fifth": 5,
    "5th": 5,
    "five": 5,
    "sixth": 6,
    "6th": 6,
    "six": 6,
    "seventh": 7,
    "7th": 7,
    "seven": 7,
}

SLOT_ALIASES = {
    "breakfast": "breakfast",
    "lunch": "lunch",
    "dinner": "dinner",
    "accommodation": "accommodation",
    "hotel": "accommodation",
    "stay": "accommodation",
    "lodging": "accommodation",
    "attraction": "attraction",
    "transportation": "transportation",
    "flight": "transportation",
    "drive": "transportation",
    "bus": "transportation",
    "train": "transportation",
}

@dataclass(frozen=True)
class RequestedSlot:
    day: int
    slot: str
    evidence: str

def normalize(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text).lower()).strip()

def day_from_item(item: dict[str, Any]) -> int | None:
    for key in ("days", "day"):
        if key in item:
            try:
                return int(item[key])
            except Exception:
                return None
    return None

def iter_plan_items(plan: Any):
    if isinstance(plan, dict):
        day = day_from_item(plan)
        if day is not None:
            yield plan
        for value in plan.values():
            yield from iter_plan_items(value)
    elif isinstance(plan, list):
        for value in plan:
            yield from iter_plan_items(value)

def max_day_in_task(task: dict[str, Any]) -> int:
    days: list[int] = []
    for source in [task.get("base_person"), *(task.get("answers") or [])]:
        for item in iter_plan_items(source):
            day = day_from_item(item)
            if day is not None:
                days.append(day)
    return max(days, default=0)

def sentence_spans(question: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", question)
    return [p.strip() for p in parts if p.strip()]

def detect_requested_slots(question: str, *, max_day: int) -> list[RequestedSlot]:
    requested: list[RequestedSlot] = []
    for span in sentence_spans(question):
        low = normalize(span)
        days = set()
        for word, day in DAY_WORDS.items():
            if re.search(rf"\b{re.escape(word)}\b", low):
                days.add(day)
        for match in re.finditer(r"\bday\s*(\d+)\b", low):
            days.add(int(match.group(1)))
        slots = {
            canonical
            for alias, canonical in SLOT_ALIASES.items()
            if re.search(rf"\b{re.escape(alias)}\b", low)
        }
        if slots and not days:
            days = set(range(1, max_day + 1))
        for day in sorted(days):
            for slot in sorted(slots):
                requested.append(RequestedSlot(day=day, slot=slot, evidence=span))

    seen: set[tuple[int, str]] = set()
    deduped: list[RequestedSlot] = []
    for item in requested:
        key = (item.day, item.slot)
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped

def collect_slot_catalog(task: dict[str, Any]) -> dict[tuple[int, str], list[str]]:
    values: dict[tuple[int, str], list[str]] = defaultdict(list)
    sources = [task.get("base_person"), *(task.get("answers") or [])]
    for source in sources:
        for item in iter_plan_items(source):
            day = day_from_item(item)
            if day is None:
                continue
            for slot in sorted(set(SLOT_ALIASES.values()) | {"current_city"}):
                value = item.get(slot)
                if value is None or value == "":
                    continue
                value_s = str(value)
                if value_s not in values[(day, slot)]:
                    values[(day, slot)].append(value_s)
    return dict(values)

def collect_base_slot_catalog(task: dict[str, Any]) -> dict[tuple[int, str], list[str]]:
    values: dict[tuple[int, str], list[str]] = defaultdict(list)
    for item in iter_plan_items(task.get("base_person")):
        day = day_from_item(item)
        if day is None:
            continue
        for slot in sorted(set(SLOT_ALIASES.values()) | {"current_city"}):
            value = item.get(slot)
            if value is None or value == "":
                continue
            value_s = str(value)
            if value_s not in values[(day, slot)]:
                values[(day, slot)].append(value_s)
    return dict(values)

def _gold_values_for_session(task: dict[str, Any], session_id: int) -> dict[tuple[int, str], list[str]]:
    answers = task.get("answers") or []
    if session_id >= len(answers):
        return {}
    values: dict[tuple[int, str], list[str]] = defaultdict(list)
    for item in iter_plan_items(answers[session_id]):
        day = day_from_item(item)
        if day is None:
            continue
        for slot in set(SLOT_ALIASES.values()) | {"current_city"}:
            value = item.get(slot)
            if value not in (None, ""):
                values[(day, slot)].append(str(value))
    return dict(values)

def render_group_travel_environment(
    task: dict[str, Any],
    *,
    session_id: int,
    question: str,
    mode: str = "catalog",
    max_candidates_per_slot: int = 12,
) -> str:
    """Render tool observations for one group-travel session.

    ``mode='catalog'`` is a diagnostic approximation of the missing public
    catalogs. ``mode='oracle'`` intentionally exposes current-session feasible
    values and is only for upper-bound debugging.
    """
    if mode in {"", "none", None}:                                    
        return ""
    if mode not in {"catalog", "oracle", "paper_sim"}:
        raise ValueError("group travel environment mode must be one of: none, catalog, oracle, paper_sim")

    requested = detect_requested_slots(question, max_day=max_day_in_task(task))
    if not requested:
        return ""

    if mode == "oracle":
        values = _gold_values_for_session(task, session_id)
    elif mode == "paper_sim":
        values = collect_base_slot_catalog(task)
    else:
        values = collect_slot_catalog(task)
    lines = [
        "Group Travel environment observations:",
        f"Mode: {mode}",
    ]
    if mode == "paper_sim":
        lines.append(
            "Source: base traveler finalized plan only; no current-session gold plan or future traveler plan values are exposed."
        )
    elif mode == "catalog":
        lines.append(
            "Source: reconstructed diagnostic candidate catalog from this HF task snapshot; "
            "candidates are not marked as correct."
        )
    else:
        lines.append(
            "Source: oracle current-session feasible slots from gold plan; use only for debugging, not paper-grade results."
        )
    lines.append("Requested slots and available candidates:")
    for req in requested:
        candidates = [
            value
            for value in values.get((req.day, req.slot), [])
            if value and value != "-"
        ][:max_candidates_per_slot]
        payload = {
            "day": req.day,
            "slot": req.slot,
            "request": req.evidence,
            "candidates": candidates,
        }
        lines.append(json.dumps(payload, ensure_ascii=False))
    return "\n".join(lines)

class ReconstructedGroupTravelEnv:
    """Small deterministic tool environment for one group-travel session.

    The environment exposes requested day/slot candidate values through a
    ``search_slot`` action. In ``catalog`` mode it returns all reconstructed
    candidates for that day/slot. In ``oracle`` mode it returns the current
    session's gold value for the requested slot. The latter simulates a working
    database/search environment for debugging memory integration, not official
    benchmark scoring.
    """

    def __init__(
        self,
        task: dict[str, Any],
        *,
        session_id: int,
        question: str,
        mode: str = "oracle",
        max_candidates_per_slot: int = 12,
    ) -> None:
        if mode not in {"catalog", "oracle", "paper_sim"}:
            raise ValueError("mode must be catalog, oracle, or paper_sim")
        self.task = task
        self.session_id = session_id
        self.question = question
        self.mode = mode
        self.max_candidates_per_slot = max_candidates_per_slot
        self.requested_slots = detect_requested_slots(question, max_day=max_day_in_task(task))
        self._catalog_values = collect_slot_catalog(task)
        self._paper_sim_values = collect_base_slot_catalog(task)
        self._oracle_values = _gold_values_for_session(task, session_id)

    def instructions(self) -> str:
        requested = [
            {"day": item.day, "slot": item.slot, "request": item.evidence}
            for item in self.requested_slots
        ]
        if self.mode == "paper_sim":
            mode_note = (
                "The environment exposes only finalized base-traveler slots and never current-session gold values. "
                "For other database searches, use the request constraints and retrieved memory conservatively."
            )
        elif self.mode == "catalog":
            mode_note = "The environment is a reconstructed candidate catalog from the HF task snapshot."
        else:
            mode_note = "The environment simulates successful TravelPlanner database searches for the current session."
        return (
            "You may query a Group Travel environment before giving the final plan.\n"
            f"{mode_note}\n"
            "Use one JSON object per turn.\n"
            "Tool call format: {\"tool\":\"search_slot\",\"day\":2,\"slot\":\"breakfast\"}\n"
            "Final answer format: {\"final\": <complete updated itinerary JSON>}\n"
            "Requested slots detected for this session:\n"
            f"{json.dumps(requested, ensure_ascii=False, indent=2)}"
        )

    def search_slot(self, *, day: int, slot: str) -> dict[str, Any]:
        slot = SLOT_ALIASES.get(slot.lower(), slot.lower())
        if self.mode == "oracle":
            source = self._oracle_values
        elif self.mode == "paper_sim":
            source = self._paper_sim_values
        else:
            source = self._catalog_values
        candidates = [
            value
            for value in source.get((int(day), slot), [])
            if value and value != "-"
        ][: self.max_candidates_per_slot]
        requested = [
            item.evidence
            for item in self.requested_slots
            if item.day == int(day) and item.slot == slot
        ]
        return {
            "tool": "search_slot",
            "mode": self.mode,
            "day": int(day),
            "slot": slot,
            "request_evidence": requested,
            "candidates": candidates,
        }

    def handle_action(self, action: dict[str, Any]) -> dict[str, Any]:
        tool = str(action.get("tool", "")).lower()
        if tool == "search_slot":
            return self.search_slot(
                day=int(action.get("day", 0)),
                slot=str(action.get("slot", "")),
            )
        if tool == "list_requested_slots":
            return {
                "tool": "list_requested_slots",
                "requested_slots": [
                    {"day": item.day, "slot": item.slot, "request": item.evidence}
                    for item in self.requested_slots
                ],
            }
        return {"error": f"Unknown tool: {tool}", "available_tools": ["search_slot", "list_requested_slots"]}

def parse_tool_or_final(text: str) -> dict[str, Any] | None:
    """Extract a JSON action/final object from model text."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.I | re.DOTALL)
    candidates = [fenced.group(1).strip()] if fenced else []
    start = text.find("{")
    if start >= 0:
        candidates.append(text[start:])
    candidates.append(text)
    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            value, _ = decoder.raw_decode(candidate)
        except Exception:
            continue
        if isinstance(value, dict):
            return value
    return None
