#!/usr/bin/env python3
"""Build a deterministic 200-question MemoryAgentBench subset.

The unit of evaluation in MemoryAgentBench is a question, but the source data
is grouped by haystack/context. This selector balances at the question level
while spreading questions across context-length quartiles inside each of the
four benchmark categories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agentmem.eval.memoryagentbench_runner import load_memoryagentbench

CATEGORIES = ("AR", "TTL", "LRU", "CR")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-questions", type=int, default=200)
    parser.add_argument("--categories", nargs="+", default=list(CATEGORIES), choices=list(CATEGORIES))
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--source-file", default=None, help="Optional local normalized/upstream JSONL instead of HF.")
    parser.add_argument("--output", default="datasets/memoryagentbench/mab_200q.jsonl")
    parser.add_argument("--manifest", default="datasets/memoryagentbench/mab_200q_manifest.json")
    parser.add_argument("--summary-md", default="datasets/memoryagentbench/mab_200q_summary.md")
    parser.add_argument(
        "--max-per-haystack",
        type=int,
        default=3,
        help="Soft cap per haystack; automatically relaxed for small categories.",
    )
    return parser.parse_args()

def approx_tokens(text: str) -> int:
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text or ""))
    except Exception:

        return max(1, math.ceil(len(str(text or "")) / 4))

def stable_hash(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def allocate_counts(total: int, categories: Iterable[str]) -> dict[str, int]:
    cats = list(categories)
    base, rem = divmod(total, len(cats))
    return {cat: base + (1 if i < rem else 0) for i, cat in enumerate(cats)}

def length_quartiles(haystacks: list[dict[str, Any]]) -> dict[str, int]:
    ordered = sorted(
        haystacks,
        key=lambda h: (int(h["context_tokens"]), str(h["haystack_id"])),
    )
    out: dict[str, int] = {}
    n = len(ordered)
    for idx, haystack in enumerate(ordered):
        quartile = min(3, int(idx * 4 / max(n, 1)))
        out[str(haystack["haystack_id"])] = quartile
    return out

def choose_question_indices(num_questions: int, desired: int) -> list[int]:
    """Pick fixed quantile positions from one haystack."""
    if desired <= 0 or num_questions <= 0:
        return []
    if desired >= num_questions:
        return list(range(num_questions))
    if desired == 1:
        return [num_questions // 2]
    picks: list[int] = []
    for rank in range(desired):
        pos = round(rank * (num_questions - 1) / (desired - 1))
        if pos not in picks:
            picks.append(pos)

    for pos in range(num_questions):
        if len(picks) >= desired:
            break
        if pos not in picks:
            picks.append(pos)
    return sorted(picks)

def select_for_category(
    haystacks: list[dict[str, Any]],
    *,
    target: int,
    soft_cap: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not haystacks:
        return [], {"target": target, "selected": 0, "reason": "empty_category"}

    quartiles = length_quartiles(haystacks)
    by_q: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for haystack in haystacks:
        by_q[quartiles[str(haystack["haystack_id"])]].append(haystack)
    for q in range(4):
        by_q[q].sort(key=lambda h: (int(h["context_tokens"]), str(h["haystack_id"])))

    q_targets = allocate_counts(target, [0, 1, 2, 3])
    selected: list[dict[str, Any]] = []
    selected_keys: set[tuple[str, int]] = set()
    per_haystack: Counter[str] = Counter()
    cap = max(1, soft_cap)

    def add_from_quartile(q: int, quota: int, cap_value: int) -> int:
        added = 0
        pool = by_q.get(q, [])
        if not pool:
            return 0

        exhausted = False
        while added < quota and not exhausted:
            exhausted = True
            for haystack in pool:
                hid = str(haystack["haystack_id"])
                if per_haystack[hid] >= cap_value:
                    continue
                qas = haystack.get("qas") or []
                already = {
                    qa_idx for existing_hid, qa_idx in selected_keys if existing_hid == hid
                }
                remaining = [i for i in choose_question_indices(len(qas), len(qas)) if i not in already]
                if not remaining:
                    continue

                idx = remaining[len(remaining) // 2]
                selected_keys.add((hid, idx))
                per_haystack[hid] += 1
                selected.append(_selected_question(haystack, idx, q))
                added += 1
                exhausted = False
                if added >= quota:
                    break
        return added

    for q in range(4):
        add_from_quartile(q, q_targets[q], cap)

    relaxed_cap = cap
    while len(selected) < target:
        before = len(selected)
        relaxed_cap += 1
        for q in range(4):
            if len(selected) >= target:
                break
            add_from_quartile(q, target - len(selected), relaxed_cap)
        if len(selected) == before:
            break

    selected = selected[:target]
    meta = {
        "target": target,
        "selected": len(selected),
        "soft_cap": cap,
        "relaxed_cap": relaxed_cap,
        "haystacks_available": len(haystacks),
        "questions_available": sum(len(h.get("qas") or []) for h in haystacks),
        "selected_haystacks": len({row["haystack_id"] for row in selected}),
        "by_quartile": dict(Counter(str(row["length_quartile"]) for row in selected)),
    }
    return selected, meta

def _selected_question(haystack: dict[str, Any], qa_index: int, quartile: int) -> dict[str, Any]:
    qa = dict((haystack.get("qas") or [])[qa_index])
    return {
        "haystack_id": str(haystack["haystack_id"]),
        "category": haystack["category"],
        "source": haystack.get("source", ""),
        "context_tokens": int(haystack["context_tokens"]),
        "context_chars": int(haystack["context_chars"]),
        "length_quartile": int(quartile),
        "qa_index": int(qa_index),
        "qa_pair_id": str(qa.get("qa_pair_id") or qa.get("question_id") or qa_index),
        "question": qa.get("question", ""),
        "gold": qa.get("gold", ""),
        "question_type": qa.get("question_type", ""),
        "question_date": qa.get("question_date", ""),
        "previous_event": qa.get("previous_event", ""),
    }

def materialize_selected_rows(
    haystacks: list[dict[str, Any]],
    selected: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_hid = {str(h["haystack_id"]): h for h in haystacks}
    selected_by_hid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        selected_by_hid[str(row["haystack_id"])].append(row)

    rows: list[dict[str, Any]] = []
    for hid in sorted(
        selected_by_hid,
        key=lambda item: (
            CATEGORIES.index(by_hid[item]["category"]) if by_hid[item]["category"] in CATEGORIES else 999,
            int(by_hid[item]["context_tokens"]),
            item,
        ),
    ):
        h = by_hid[hid]
        qas = []
        for sel in sorted(selected_by_hid[hid], key=lambda r: int(r["qa_index"])):
            qa = dict((h.get("qas") or [])[int(sel["qa_index"])])
            qa["selection"] = {
                "context_tokens": sel["context_tokens"],
                "context_chars": sel["context_chars"],
                "length_quartile": sel["length_quartile"],
                "qa_index": sel["qa_index"],
            }
            qas.append(qa)
        rows.append(
            {
                "haystack_id": h["haystack_id"],
                "category": h["category"],
                "source": h.get("source", ""),
                "context_text": h.get("context_text", ""),
                "context_tokens": h["context_tokens"],
                "context_chars": h["context_chars"],
                "qas": qas,
                "metadata": h.get("metadata", {}),
                "selection": {
                    "selected_questions": len(qas),
                    "length_quartiles": sorted({qa["selection"]["length_quartile"] for qa in qas}),
                },
            }
        )
    return rows

def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

def write_summary(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# MemoryAgentBench 200Q Selection",
        "",
        f"- Manifest hash: `{manifest['manifest_hash']}`",
        f"- Target questions: `{manifest['target_questions']}`",
        f"- Selected questions: `{manifest['selected_questions']}`",
        f"- Selected haystacks: `{manifest['selected_haystacks']}`",
        "",
        "| Category | Questions | Haystacks | Q0 | Q1 | Q2 | Q3 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cat in CATEGORIES:
        item = manifest["categories"].get(cat, {})
        by_q = item.get("by_quartile", {})
        lines.append(
            f"| {cat} | {item.get('selected', 0)} | {item.get('selected_haystacks', 0)} | "
            f"{by_q.get('0', 0)} | {by_q.get('1', 0)} | {by_q.get('2', 0)} | {by_q.get('3', 0)} |"
        )
    lines.extend(
        [
            "",
            "Selection is deterministic: haystacks are sorted by context length, split into quartiles, and questions are chosen by fixed quantile positions with a per-haystack cap relaxed only when a category has too few haystacks.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def main() -> None:
    args = parse_args()
    categories = [str(c).upper() for c in args.categories]
    target_by_category = allocate_counts(args.target_questions, categories)
    haystacks = load_memoryagentbench(
        categories=categories,
        cache_dir=args.cache_dir,
        test_file=args.source_file,
    )
    for h in haystacks:
        text = str(h.get("context_text") or "")
        h["context_chars"] = len(text)
        h["context_tokens"] = approx_tokens(text)

    all_selected: list[dict[str, Any]] = []
    category_meta: dict[str, Any] = {}
    for cat in categories:
        pool = [h for h in haystacks if h.get("category") == cat]
        selected, meta = select_for_category(
            pool,
            target=target_by_category[cat],
            soft_cap=args.max_per_haystack,
        )
        all_selected.extend(selected)
        category_meta[cat] = meta

    selected_rows = materialize_selected_rows(haystacks, all_selected)
    output_path = Path(args.output)
    manifest_path = Path(args.manifest)
    summary_path = Path(args.summary_md)
    write_jsonl(output_path, selected_rows)

    manifest = {
        "schema_version": 1,
        "dataset": "ai-hyz/MemoryAgentBench",
        "source_file": args.source_file,
        "output": str(output_path),
        "target_questions": args.target_questions,
        "selected_questions": len(all_selected),
        "selected_haystacks": len(selected_rows),
        "categories": category_meta,
        "selection": sorted(
            all_selected,
            key=lambda r: (r["category"], r["length_quartile"], r["context_tokens"], r["haystack_id"], r["qa_index"]),
        ),
    }
    manifest["manifest_hash"] = stable_hash(manifest["selection"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_summary(summary_path, manifest)
    print(json.dumps({k: manifest[k] for k in ("output", "manifest_hash", "selected_questions", "selected_haystacks")}, indent=2))

if __name__ == "__main__":
    main()
