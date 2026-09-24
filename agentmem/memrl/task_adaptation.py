from __future__ import annotations

import re
from collections import OrderedDict
from typing import Any, Iterable, Sequence

from agentmem.eval.metrics import exact_match, token_f1

def compute_memrl_reward(
    prediction: Any,
    reference: Any,
    *,
    scheme: str = "token-f1",
) -> float:
    scheme_key = str(scheme or "token-f1").strip().lower()
    if scheme_key in {"exact", "exact-match", "em"}:
        return float(exact_match(prediction, reference))
    return float(token_f1(prediction, reference))

def build_locomo_memrl_query(question: str, category: int | None) -> str:
    question_text = str(question or "").strip()
    if category == 2:
        hint = "Focus on time, temporal order, dates, and before/after relations."
    elif category == 3:
        hint = "Focus on evidence that supports likely inferences or counterfactual implications."
    elif category == 5:
        hint = "Focus on whether the conversation explicitly mentions the requested fact."
    else:
        hint = "Focus on directly stated grounded dialogue evidence."
    return f"{question_text}\n\nRetrieval hint: {hint}"

def format_locomo_memory_context(selected: Sequence[Any]) -> str:
    direct_sections: list[str] = []
    qa_sections: list[str] = []
    other_sections: list[str] = []

    for idx, cand in enumerate(selected, start=1):
        meta = dict(getattr(cand, "metadata", {}) or {})
        source = str(meta.get("source") or "dialogue").strip().lower()
        content = str(
            meta.get("full_content")
            or meta.get("experience")
            or getattr(cand, "content", "")
            or ""
        ).strip()
        if not content:
            continue

        label = (
            f"[{idx}] sim={float(getattr(cand, 'similarity', 0.0)):.3f} "
            f"util={float(getattr(cand, 'utility', 0.0)):.3f}"
        )
        block = f"{label}\n{content}"
        if source in {"qa_feedback", "qa_memory", "qa"}:
            qa_sections.append(block)
        elif source in {"dialogue", "turn", "event_summary"}:
            direct_sections.append(block)
        else:
            other_sections.append(block)

    parts: list[str] = []
    if direct_sections:
        parts.append("Direct Evidence:\n" + "\n\n".join(direct_sections))
    if qa_sections:
        parts.append("Related Past QA Memories:\n" + "\n\n".join(qa_sections))
    if other_sections:
        parts.append("Other Relevant Memories:\n" + "\n\n".join(other_sections))
    return "\n\n".join(parts)

def build_locomo_qa_memory(
    *,
    question: str,
    prediction: str,
    reference: str | None,
    reward: float,
    category: int | None,
    selected: Sequence[Any],
) -> tuple[str, str, dict[str, Any]]:
    evidence_lines: list[str] = []
    selected_ids: list[str] = []
    for idx, cand in enumerate(selected, start=1):
        meta = dict(getattr(cand, "metadata", {}) or {})
        content = str(
            meta.get("full_content")
            or meta.get("experience")
            or getattr(cand, "content", "")
            or ""
        ).strip()
        if not content:
            continue
        content = content[:800]
        evidence_lines.append(f"[Evidence {idx}]\n{content}")
        memory_id = getattr(cand, "memory_id", None)
        if memory_id is not None:
            selected_ids.append(str(memory_id))

    lines = [
        f"Question: {question}",
        f"Prediction: {prediction}",
    ]
    if reference is not None:
        lines.append(f"Gold: {reference}")
    lines.extend(
        [
            f"Reward: {reward:.4f}",
            "Retrieved Evidence:",
            "\n\n".join(evidence_lines) if evidence_lines else "(none)",
        ]
    )
    metadata = {
        "source": "qa_feedback",
        "category": category,
        "reward": float(reward),
        "selected_memory_ids": selected_ids,
        "success": bool(reward >= 0.999),
    }
    return question, "\n".join(lines), metadata

def chunk_hotpot_documents(sample: dict[str, Any]) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    supporting = sample.get("supporting_facts") or []
    supporting_pairs = {
        (str(item[0]), int(item[1]))
        for item in supporting
        if isinstance(item, (list, tuple)) and len(item) >= 2
    }

    for doc_index, entry in enumerate(sample.get("context", []) or []):
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            continue
        title = str(entry[0] or f"Document {doc_index + 1}").strip() or f"Document {doc_index + 1}"
        sentences = entry[1] if isinstance(entry[1], list) else [entry[1]]
        normalized_sentences = [str(s).strip() for s in sentences if str(s).strip()]
        if not normalized_sentences:
            continue

        full_doc = " ".join(normalized_sentences).strip()
        units.append(
            {
                "task_id": f"doc_{doc_index}",
                "title": title,
                "text": full_doc,
                "chunk_kind": "document",
                "sentence_span": None,
                "is_supporting_fact": any(
                    (title, sent_idx) in supporting_pairs
                    for sent_idx in range(len(normalized_sentences))
                ),
            }
        )

        for sent_idx, sentence in enumerate(normalized_sentences):
            units.append(
                {
                    "task_id": f"doc_{doc_index}_sent_{sent_idx}",
                    "title": title,
                    "text": sentence,
                    "chunk_kind": "sentence",
                    "sentence_span": [sent_idx, sent_idx],
                    "is_supporting_fact": (title, sent_idx) in supporting_pairs,
                }
            )

        if len(normalized_sentences) > 1:
            for sent_idx in range(len(normalized_sentences) - 1):
                span_text = " ".join(normalized_sentences[sent_idx : sent_idx + 2]).strip()
                units.append(
                    {
                        "task_id": f"doc_{doc_index}_span_{sent_idx}_{sent_idx + 1}",
                        "title": title,
                        "text": span_text,
                        "chunk_kind": "sentence_pair",
                        "sentence_span": [sent_idx, sent_idx + 1],
                        "is_supporting_fact": any(
                            (title, idx) in supporting_pairs
                            for idx in range(sent_idx, sent_idx + 2)
                        ),
                    }
                )
    return units

def build_hotpot_memrl_queries(question: str) -> list[str]:
    question_text = str(question or "").strip()
    variants = [
        question_text,
        f"{question_text}\n\nRetrieval hint: find bridge entities, linked pages, and intermediate facts.",
        f"{question_text}\n\nRetrieval hint: find the exact answer span and final supporting sentence.",
    ]
    unique: OrderedDict[str, None] = OrderedDict()
    for item in variants:
        normalized = item.strip()
        if normalized:
            unique[normalized] = None
    return list(unique.keys())

def format_hotpot_context(selected: Sequence[Any]) -> str:
    blocks: list[str] = []
    for idx, cand in enumerate(selected, start=1):
        meta = dict(getattr(cand, "metadata", {}) or {})
        title = str(meta.get("title") or f"Document {idx}")
        chunk_kind = str(meta.get("chunk_kind") or "document")
        sentence_span = meta.get("sentence_span")
        span_text = ""
        if isinstance(sentence_span, (list, tuple)) and len(sentence_span) == 2:
            span_text = f" | sentences {sentence_span[0]}-{sentence_span[1]}"
        content = str(
            meta.get("full_content")
            or meta.get("experience")
            or getattr(cand, "content", "")
            or ""
        ).strip()
        if not content:
            continue
        blocks.append(
            f"[Evidence Block {idx} | {title} | {chunk_kind}{span_text}]\n{content}"
        )
    return "\n\n".join(blocks)

def build_hotpot_answer_prompt(question: str, context: str) -> str:
    return (
        "You are answering a HotpotQA factual question. Use the evidence as your "
        "primary source, but if it is partial, combine it with your world knowledge "
        "and still answer. These are real-world Wikipedia facts.\n\n"
        "Worked examples:\n"
        "Q: Were Scott Derrickson and Ed Wood of the same nationality?\n"
        "Evidence: Scott Derrickson is an American film director.\n"
        "A: yes\n"
        "Rule: if you see evidence for only ONE entity in a yes/no comparison, use "
        "your world knowledge for the other entity; do not default to no.\n\n"
        "Q: What science fantasy YA series is told in first person by several "
        "teenagers and includes complementary books?\n"
        "Evidence: The series is narrated by the teenage Animorphs in first person.\n"
        "A: Animorphs\n\n"
        "Q: What country was the film director born in if the evidence says he was "
        "born in Toronto?\n"
        "Evidence: The director was born in Toronto, Ontario.\n"
        "A: Canada\n\n"
        "Output ONLY the best short answer: 1-5 words, a name/entity/date/number/place "
        "or yes/no. No explanation, no quotes, no apology, no refusal. Never output "
        "'unknown', 'cannot determine', or 'unsupported'.\n\n"
        f"Question: {question}\n\n"
        f"Evidence:\n{context}\n\n"
        f"Question (repeated): {question}\n"
        "Best short answer:"
    )

def build_hotpot_qa_memory(
    *,
    question: str,
    prediction: str,
    reference: str,
    reward: float,
    selected: Sequence[Any],
) -> tuple[str, str, dict[str, Any]]:
    lines = [
        f"Question: {question}",
        f"Prediction: {prediction}",
        f"Gold: {reference}",
        f"Reward: {reward:.4f}",
        "Evidence Summary:",
    ]
    selected_ids: list[str] = []
    titles: list[str] = []
    for cand in selected:
        meta = dict(getattr(cand, "metadata", {}) or {})
        title = str(meta.get("title") or "").strip()
        if title:
            titles.append(title)
        memory_id = getattr(cand, "memory_id", None)
        if memory_id is not None:
            selected_ids.append(str(memory_id))
    lines.append(", ".join(titles) if titles else "(none)")
    metadata = {
        "source": "qa_feedback",
        "reward": float(reward),
        "selected_memory_ids": selected_ids,
        "title_set": titles,
        "success": bool(reward >= 0.999),
    }
    return question, "\n".join(lines), metadata

def supporting_fact_bonus(
    selected: Sequence[Any],
    supporting_facts: Iterable[Any],
    *,
    bonus: float = 0.2,
) -> float:
    pairs = {
        (str(item[0]), int(item[1]))
        for item in supporting_facts
        if isinstance(item, (list, tuple)) and len(item) >= 2
    }
    if not pairs:
        return 0.0

    matched = 0
    for cand in selected:
        meta = dict(getattr(cand, "metadata", {}) or {})
        title = str(meta.get("title") or "").strip()
        span = meta.get("sentence_span")
        if not title or not isinstance(span, (list, tuple)) or len(span) != 2:
            continue
        start, end = int(span[0]), int(span[1])
        if any((title, idx) in pairs for idx in range(start, end + 1)):
            matched += 1
    if matched == 0:
        return 0.0
    return min(float(bonus), float(bonus) * matched / max(len(pairs), 1))

def normalize_alfworld_task_description(task_desc: str, task_type: str = "") -> str:
    text = str(task_desc or "").strip()
    if not text:
        return str(task_type or "").strip()
    normalized = re.sub(r"\s+", " ", text.lower())
    normalized = re.sub(r"you are in the [^.]+\.\s*", "", normalized)
    normalized = re.sub(r"\blook around\b", "", normalized)
    normalized = normalized.strip()
    if task_type:
        return f"{task_type.strip().lower()} | {normalized}"
    return normalized

def format_alfworld_memrl_prompt(
    successful: Sequence[str],
    failed: Sequence[str],
) -> str:
    parts = [
        "Memory system: memrl-style episodic ALFWorld memory",
        (
            "Use successful memories as reusable strategy sketches and failed memories as warnings. "
            "Adapt old plans to the current object names and room state; trust the current observation "
            "if it conflicts with memory."
        ),
    ]
    if successful:
        parts.append("--- SUCCESSFUL MEMORIES ---\n" + "\n\n".join(successful))
    if failed:
        parts.append("--- FAILED MEMORIES ---\n" + "\n\n".join(failed))
    return "\n\n".join(parts)

def build_bigcodebench_task_descriptor(task: dict[str, Any], prompt: str) -> str:
    task_id = str(task.get("task_id") or "unknown").strip()
    entry_point = str(task.get("entry_point") or "").strip()
    libs = task.get("libs")
    if isinstance(libs, (list, tuple)):
        libs_text = ", ".join(str(lib).strip() for lib in libs if str(lib).strip())
    else:
        libs_text = str(libs or "").strip()
    prompt_head = str(prompt or "").strip()
    if len(prompt_head) > 1200:
        prompt_head = prompt_head[:1200].rstrip() + " ..."
    parts = [f"task_id={task_id}"]
    if entry_point:
        parts.append(f"entry_point={entry_point}")
    if libs_text:
        parts.append(f"libs={libs_text}")
    parts.append("prompt:\n" + prompt_head)
    return "\n".join(parts)

def build_bigcodebench_strategy_memory(
    *,
    prompt: str,
    code: str,
    eval_res: dict[str, Any],
    task_description: str,
    outcome: str,
) -> str:
    status = str(eval_res.get("status") or "UNKNOWN").strip()
    error = str(eval_res.get("error") or "").strip()
    code_preview = str(code or "").strip()
    if len(code_preview) > 2500:
        code_preview = code_preview[:2500].rstrip() + "\n# ... truncated ..."

    if outcome.lower() in {"failure", "fail", "failed"}:
        return (
            "[MEMORY TYPE] FAILURE_REFLECTION\n"
            "[TASK]\n"
            f"{task_description}\n\n"
            "[FAILURE MODE]\n"
            f"status={status}\n"
            f"{error or 'Execution or semantic failure.'}\n\n"
            "[ATTEMPTED APPROACH]\n"
            f"{prompt[:800].strip()}\n\n"
            "[FAILED CODE SNIPPET]\n"
            "```python\n"
            f"{code_preview}\n"
            "```"
        ).strip()

    return (
        "[MEMORY TYPE] SUCCESS_PROCEDURE\n"
        "[TASK]\n"
        f"{task_description}\n\n"
        "[STRATEGY]\n"
        "Respect the current function signature, required libraries, and tests. "
        "Reuse only the algorithmic pattern, not stale variable names.\n\n"
        "[WORKING CODE PATTERN]\n"
        "```python\n"
        f"{code_preview}\n"
        "```"
    ).strip()

def format_bigcodebench_memory_context(
    selected_mems: Sequence[dict[str, Any]],
    *,
    memory_budget_tokens: int = 0,
) -> str:
    if not selected_mems:
        return ""

    budget = int(memory_budget_tokens or 0)
    per_item_budget = max(200, budget // max(len(selected_mems), 1)) if budget > 0 else 0
    parts: list[str] = [
        "# Archived Solution Strategies",
        (
            "Use the archived strategies as hints, not as code to copy literally. "
            "Adapt them to the current function signature, imports, and tests."
        ),
        "",
    ]

    for idx, item in enumerate(selected_mems, start=1):
        meta_obj = item.get("metadata")
        meta = dict(meta_obj.model_dump() if hasattr(meta_obj, "model_dump") else meta_obj or {})
        outcome = str(meta.get("outcome") or "unknown").upper()
        task_id = str(meta.get("task_id") or "").strip()
        content = str(item.get("content") or item.get("full_content") or "").strip()
        if per_item_budget > 0 and len(content) > per_item_budget:
            content = content[:per_item_budget].rstrip() + "\n..."
        parts.append(f"## Strategy {idx} [{outcome}]")
        if task_id:
            parts.append(f"Task: {task_id}")
        parts.append(content)
        parts.append("")

    return "\n".join(parts).strip()

_AMABENCH_TURN_RE = re.compile(
    r"Turn\s+(\d+):\s*\n\s*Action:\s*(.*?)\n\s*Observation:\s*(.*?)(?=\nTurn\s+\d+:|\Z)",
    re.DOTALL,
)

def parse_amabench_trajectory(traj_text: str) -> list[dict[str, Any]]:
    """Reverse ``examples/run_amabench.py:trajectory_to_text``.

    Returns a list of ``{turn_idx, action, observation}`` dicts. Robust to
    extra whitespace; missing fields default to ``""``. If the input does
    not match the expected ``Turn N:`` shape (e.g. raw dialogue), returns a
    single synthetic turn so downstream MemRL can still ingest it.
    """
    text = str(traj_text or "")
    if not text.strip():
        return []
    turns: list[dict[str, Any]] = []
    for m in _AMABENCH_TURN_RE.finditer(text):
        turn_idx = int(m.group(1))
        action = m.group(2).strip()
        observation = m.group(3).strip()
        turns.append(
            {"turn_idx": turn_idx, "action": action, "observation": observation}
        )
    if not turns:

        turns.append({"turn_idx": 0, "action": "", "observation": text.strip()})
    return turns

def build_amabench_memrl_query(
    question: str,
    *,
    task_type: str | None = None,
    domain: str | None = None,
) -> str:
    """Add a domain-specific retrieval hint to the question.

    AMABench domains we care about:
      - ``alfworld`` (EMBODIED_AI): focus on object location + state changes
      - ``webarena`` (WEB): focus on UI elements clicked + form fields
      - ``spider2`` (TEXT2SQL): focus on schema, table joins, SQL operators
    """
    question_text = str(question or "").strip()
    tt = str(task_type or "").strip().lower()
    dom = str(domain or "").strip().lower()
    if tt == "alfworld" or "embodied" in dom:
        hint = (
            "Focus on object locations, container contents, state changes, "
            "and which actions caused them."
        )
    elif tt == "webarena" or dom == "web":
        hint = (
            "Focus on UI elements that were clicked, form fields filled, "
            "URLs visited, and rendered page content."
        )
    elif tt == "spider2" or "sql" in dom or "text2sql" in tt:
        hint = (
            "Focus on database schema, table joins, SQL operators issued, "
            "and the rows or counts they returned."
        )
    else:
        hint = (
            "Focus on grounded action/observation evidence directly relevant "
            "to the question."
        )
    return f"{question_text}\n\nRetrieval hint: {hint}"

def format_amabench_memory_context(selected: Sequence[Any]) -> str:
    """Render selected MemRL candidates as an AMABench answer-side context.

    Mirrors ``format_locomo_memory_context`` but groups by AMABench source
    tags (``trajectory_turn`` vs ``qa_feedback``).
    """
    direct_sections: list[str] = []
    qa_sections: list[str] = []
    other_sections: list[str] = []

    for idx, cand in enumerate(selected, start=1):
        meta = dict(getattr(cand, "metadata", {}) or {})
        source = str(meta.get("source") or "trajectory_turn").strip().lower()
        content = str(
            meta.get("full_content")
            or meta.get("experience")
            or getattr(cand, "content", "")
            or ""
        ).strip()
        if not content:
            continue
        label = (
            f"[{idx}] sim={float(getattr(cand, 'similarity', 0.0)):.3f} "
            f"util={float(getattr(cand, 'utility', 0.0)):.3f}"
        )
        block = f"{label}\n{content}"
        if source in {"qa_feedback", "qa_memory", "qa"}:
            qa_sections.append(block)
        elif source in {"trajectory_turn", "turn", "trajectory"}:
            direct_sections.append(block)
        else:
            other_sections.append(block)

    parts: list[str] = []
    if direct_sections:
        parts.append("Trajectory Evidence:\n" + "\n\n".join(direct_sections))
    if qa_sections:
        parts.append("Related Past QA Memories:\n" + "\n\n".join(qa_sections))
    if other_sections:
        parts.append("Other Relevant Memories:\n" + "\n\n".join(other_sections))
    return "\n\n".join(parts)

def build_amabench_qa_memory(
    *,
    question: str,
    prediction: str,
    reference: str | None,
    reward: float,
    task_type: str | None,
    selected: Sequence[Any],
) -> tuple[str, str, dict[str, Any]]:
    """Pack a QA-feedback memory the runtime can write back into the store.

    Output matches ``MemRLRuntimeEngine.add_experience`` expectations
    ``(intent, experience, metadata)``.
    """
    evidence_lines: list[str] = []
    selected_ids: list[str] = []
    for idx, cand in enumerate(selected, start=1):
        meta = dict(getattr(cand, "metadata", {}) or {})
        content = str(
            meta.get("full_content")
            or meta.get("experience")
            or getattr(cand, "content", "")
            or ""
        ).strip()
        if not content:
            continue
        evidence_lines.append(f"[Evidence {idx}]\n{content[:800]}")
        memory_id = getattr(cand, "memory_id", None)
        if memory_id is not None:
            selected_ids.append(str(memory_id))

    lines = [f"Question: {question}", f"Prediction: {prediction}"]
    if reference is not None:
        lines.append(f"Gold: {reference}")
    lines.extend(
        [
            f"Reward: {reward:.4f}",
            "Retrieved Evidence:",
            "\n\n".join(evidence_lines) if evidence_lines else "(none)",
        ]
    )
    metadata = {
        "source": "qa_feedback",
        "task_type": task_type,
        "reward": float(reward),
        "selected_memory_ids": selected_ids,
        "success": bool(reward >= 0.999),
    }
    return question, "\n".join(lines), metadata
