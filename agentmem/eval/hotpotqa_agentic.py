"""HotpotQA corpus-regime long-context and file-browsing helpers.

The methods operate over the shared HippoRAG-2 ``hotpotqa_corpus.json``:

- ``HotpotQALongContextMethod`` — selects bounded context from the corpus and
  lets the outer runner perform one shared QA call.
- ``HotpotQAMemRLMethod`` — a browse loop with optional ``note`` /
  ``recall`` actions that mimic MemRL's read/write style. On single-shot
  HotpotQA the write side is dormant; the loop only exercises the read
  policy. Wiring a real trained MemRL policy is left for future work and
  documented inline.

Both classes follow the same shape as ``HotpotQAPlugMemMethod`` so they slot
into ``HOTPOTQA_METHOD_REGISTRY`` and ``run_hotpotqa50_method`` without
special-casing the dispatcher.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentmem.methods.base import EfficiencyCounters
from agentmem.providers.base import Message
from agentmem.providers.openai_compat import OpenAICompatibleProvider

@dataclass
class CorpusIndex:
    """Title-keyed view of the HippoRAG-2 ``hotpotqa_corpus.json`` corpus."""

    docs_by_title: dict[str, str] = field(default_factory=dict)
    titles: list[str] = field(default_factory=list)

    @classmethod
    def from_json(cls, corpus_path: str | Path) -> "CorpusIndex":
        path = Path(corpus_path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        items = raw if isinstance(raw, list) else [raw]
        docs_by_title: dict[str, str] = {}
        titles: list[str] = []
        for entry in items:
            if not isinstance(entry, dict):
                continue
            title = str(entry.get("title", "")).strip()
            text = str(entry.get("text", "")).strip()
            if not title or not text:
                continue
            if title in docs_by_title:
                docs_by_title[title] += "\n" + text
            else:
                docs_by_title[title] = text
                titles.append(title)
        return cls(docs_by_title=docs_by_title, titles=titles)

class CorpusBrowser:
    """grep / read primitives over a ``CorpusIndex``."""

    def __init__(self, index: CorpusIndex) -> None:
        self.index = index

    def grep(self, pattern: str, *, max_results: int = 10) -> list[dict[str, str]]:
        """Case-insensitive substring/regex search over titles + bodies."""
        if not pattern:
            return []
        try:
            rx = re.compile(pattern, flags=re.IGNORECASE)
        except re.error:
            rx = re.compile(re.escape(pattern), flags=re.IGNORECASE)
        hits: list[dict[str, str]] = []
        for title in self.index.titles:
            text = self.index.docs_by_title.get(title, "")
            m = rx.search(title) or rx.search(text)
            if m:
                start = max(0, m.start() - 80)
                end = min(len(text), m.end() + 200)
                snippet = text[start:end].strip()
                hits.append({"title": title, "snippet": snippet})
                if len(hits) >= max_results:
                    break
        return hits

    def read(self, title: str) -> str:
        """Return the full body of a paragraph by title (best-effort match)."""
        if title in self.index.docs_by_title:
            return self.index.docs_by_title[title]
        norm = title.strip().lower()
        for t, body in self.index.docs_by_title.items():
            if t.strip().lower() == norm:
                return body
        return ""

_BROWSE_SYSTEM = (
    "You are a long-context retrieval agent. You answer multi-hop HotpotQA "
    "questions by browsing a flat 9,811-paragraph Wikipedia corpus through "
    "tool calls and accumulating evidence in your conversation history. "
    "Your strength is the ability to read MANY paragraphs across the chain "
    "of reasoning — be generous with retrieval. Tools:\n"
    "  - grep(pattern): regex/substring search; returns up to 10 (title, snippet) hits.\n"
    "  - read(title): return the full body of a paragraph by title.\n"
    "  - finish(answer): submit your final short answer and stop.\n"
    "\n"
    "Strategy guidance — favor recall over speed:\n"
    "  1. For each entity or relation in the question, run grep with several "
    "alternate phrasings (synonyms, dates, partial names) — at least 3–5 grep "
    "calls per relation.\n"
    "  2. Read the full body of every plausibly-relevant title, not just the "
    "snippets. Bridge questions need the bridging entity confirmed by reading.\n"
    "  3. Keep retrieving until you have read at least 8–15 paragraphs and "
    "you can cite specific facts that resolve every clause of the question. "
    "If your accumulated reads are short, you have not searched enough.\n"
    "  4. Only then call finish, with a short answer (yes/no for boolean, "
    "otherwise an entity or short span — no explanation).\n"
    "\n"
    "Issue exactly one tool call per turn as a single JSON object on its own "
    "line, e.g. {\"tool\":\"grep\",\"pattern\":\"Sofia Coppola\"} or "
    "{\"tool\":\"read\",\"title\":\"Lost in Translation (film)\"}. You have "
    "many turns; use them to accumulate a long context before answering."
)

_MEMRL_BROWSE_SYSTEM = (
    "You answer multi-hop HotpotQA questions by browsing a flat Wikipedia "
    "corpus while maintaining a working note. Tools:\n"
    "  - grep(pattern): regex/substring search, returns up to 10 (title, snippet).\n"
    "  - read(title): full body of a paragraph by title.\n"
    "  - note(text): append `text` to your private working note (used only by you).\n"
    "  - recall(): return your full working note.\n"
    "  - finish(answer): submit a short final answer.\n"
    "\n"
    "Issue exactly one tool call per turn as a single JSON object on its own line. "
    "Use note() to keep facts you have confirmed; recall() to inspect what you have "
    "written; finish() with a short answer when done."
)

_TOOL_CALL_RE = re.compile(r"\{[^{}]*\"tool\"[^{}]*\}", flags=re.DOTALL)

def _parse_tool_call(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    candidates: list[str] = []
    if text.startswith("{"):
        candidates.append(text)
    candidates.extend(_TOOL_CALL_RE.findall(text))
    for c in candidates:
        try:
            obj = json.loads(c)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "tool" in obj:
            return obj
    return None

def _short_observation(payload: Any, *, char_budget: int = 4000) -> str:
    """Render a tool observation. Default budget bumped to 4K chars (~1K tokens)
    so a `read(title)` returns the full paragraph (mean 570 chars) plus headroom
    for multi-paragraph responses, supporting the long-context strategy.
    """
    rendered = json.dumps(payload, ensure_ascii=False)
    if len(rendered) <= char_budget:
        return rendered
    return rendered[:char_budget] + "...[truncated]"

def _run_browse(
    *,
    question: str,
    browser: CorpusBrowser,
    provider: OpenAICompatibleProvider,
    counters: EfficiencyCounters,
    max_turns: int = 10,
    enable_notes: bool = False,
    max_response_tokens: int = 512,
) -> tuple[str, str]:
    """Run the browse loop. Returns (final_answer, transcript)."""
    system = _MEMRL_BROWSE_SYSTEM if enable_notes else _BROWSE_SYSTEM
    history: list[Message] = [
        Message(role="system", content=system),
        Message(role="user", content=f"Question: {question}\n\nBegin browsing."),
    ]
    notebook: list[str] = []
    transcript: list[str] = []
    final_answer: str = ""

    for turn in range(max_turns):
        t0 = time.perf_counter()
        response = provider.chat(
            history,
            temperature=0.0,
            max_tokens=max_response_tokens,
        )
        w_llm = time.perf_counter() - t0
        usage = dict(response.usage or {})
        counters.record_llm_call(
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            phase="query",
        )
        if hasattr(counters, "add_seconds"):
            counters.add_seconds("w_llm", w_llm)
        text = (response.content or "").strip()
        transcript.append(f"AGENT[{turn}]: {text}")
        history.append(Message(role="assistant", content=text))

        call = _parse_tool_call(text)
        if call is None:
            history.append(Message(
                role="user",
                content="Your last message was not a JSON tool call. Reply with one JSON object only, e.g. {\"tool\":\"finish\",\"answer\":\"...\"}.",
            ))
            continue

        tool = str(call.get("tool", "")).lower()
        t_tool = time.perf_counter()
        if tool == "grep":
            hits = browser.grep(str(call.get("pattern", "")), max_results=10)
            obs = _short_observation({"hits": hits})
        elif tool == "read":
            body = browser.read(str(call.get("title", "")))
            obs = _short_observation({"body": body})
        elif tool == "note" and enable_notes:
            notebook.append(str(call.get("text", "")))
            obs = _short_observation({"ok": True, "note_count": len(notebook)})
        elif tool == "recall" and enable_notes:
            obs = _short_observation({"notes": notebook})
        elif tool == "finish":
            final_answer = str(call.get("answer", "")).strip()
            transcript.append(f"FINAL: {final_answer}")
            break
        else:
            obs = _short_observation({"error": f"unknown tool {tool!r}"})
        w_tool = time.perf_counter() - t_tool
        if hasattr(counters, "add_seconds"):
            counters.add_seconds("w_tool", w_tool)
        transcript.append(f"OBS[{turn}]: {obs}")
        history.append(Message(role="user", content=f"Observation: {obs}"))

    return final_answer, "\n".join(transcript)

class HotpotQALongContextMethod:
    """HotpotQA long-context baseline over the shared corpus.

    This is a non-agentic stuff-context path: build stores the shared
    HippoRAG-2 corpus once, answer returns the bounded context selected by
    :class:`agentmem.methods.longcontext.LongContextMethod`, and the outer
    HotpotQA runner performs the single shared QA call.
    """

    name = "longcontext"

    def __init__(
        self,
        llm_model: str = "Qwen/Qwen3-32B",
        llm_base_url: str | None = None,
        llm_api_key: str = "EMPTY",
        corpus_path: str | None = None,
        max_turns: int = 30,
        max_response_tokens: int = 512,
        max_model_length: int = 32768,
        token_counter: Any = None,
        **_kw: Any,
    ) -> None:
        from agentmem.methods.longcontext import LongContextMethod

        self.llm_model = llm_model
        self.llm_base_url = llm_base_url
        self.llm_api_key = llm_api_key
        self.max_turns = max_turns
        self.max_response_tokens = max_response_tokens
        self._token_counter = token_counter
        self._counters = EfficiencyCounters()
        self._provider = OpenAICompatibleProvider(
            api_key=llm_api_key,
            model=llm_model,
            base_url=llm_base_url or "http://localhost:8000/v1",
            max_retries=5,
        )
        path = Path(corpus_path) if corpus_path else _default_corpus_path()
        self._index = CorpusIndex.from_json(path)
        self._browser = CorpusBrowser(self._index)
        corpus_text = "\n\n".join(
            f"Document {idx} — {title}:\n{self._index.docs_by_title.get(title, '')}"
            for idx, title in enumerate(self._index.titles, start=1)
        )
        self._lc_method = LongContextMethod(
            max_model_length=int(os.environ.get("HOTPOTQA_LONGCONTEXT_MAX_MODEL_LENGTH", max_model_length)),
            max_response_tokens=int(os.environ.get("HOTPOTQA_LONGCONTEXT_MAX_RESPONSE_TOKENS", max_response_tokens)),
            token_counter=token_counter,
        )
        self._lc_memory = self._lc_method.build(corpus_text, task="hotpotqa")

    @property
    def counters(self) -> EfficiencyCounters:
        return self._counters

    def reset_counters(self) -> EfficiencyCounters:
        self._counters = EfficiencyCounters()
        self._lc_method.reset_counters()
        return self._counters

    def build_from_sample(self, sample: dict[str, Any], *, sample_id: str) -> Any:
        return {"sample_id": sample_id}

    def build(self, traj_text: str, *, task: str = "") -> Any:
        return {"sample_id": task or "anonymous"}

    def answer(self, memory: Any, question: str) -> str:
        context = self._lc_method.answer(self._lc_memory, question)
        self._counters = self._lc_method.counters
        return context

def _default_corpus_path() -> Path:
    """Default location of the HippoRAG-2 HotpotQA corpus.

    Resolution order:
    1. ``$HOTPOTQA_CORPUS_PATH`` env var
    2. ``vendor/plugmem/bench_data/hotpotqa_hipporag/hotpotqa_corpus.json``
    3. ``datasets/hotpotqa/hotpotqa_corpus.json``
    """
    import os
    env = os.environ.get("HOTPOTQA_CORPUS_PATH")
    if env:
        return Path(env)
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        repo_root / "vendor" / "plugmem" / "bench_data" / "hotpotqa_hipporag" / "hotpotqa_corpus.json",
        repo_root / "datasets" / "hotpotqa" / "hotpotqa_corpus.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "hotpotqa_corpus.json not found. Set HOTPOTQA_CORPUS_PATH or place the file at one of: "
        + ", ".join(str(c) for c in candidates)
    )
