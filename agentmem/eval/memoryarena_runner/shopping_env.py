"""Bundled-shopping environment helpers for the MemoryArena harness.

The public HF MemoryArena snapshot exposes questions and gold ASIN answers,
but not the full WebShop product catalog used by the original interactive
evaluation. This module reconstructs a small deterministic search catalog from
the products that appear in the multi-session task. It is therefore a
simulated-WebShop protocol for reproducible memory diagnostics, not a claim of
official WebShop environment parity.
"""

from __future__ import annotations

import json
import re
import hashlib
from dataclasses import dataclass
from typing import Any

_ASIN_RE = re.compile(r"\bB[A-Z0-9]{9}\b")

def _tokens(text: Any) -> set[str]:
    stop = {
        "and",
        "are",
        "for",
        "from",
        "have",
        "item",
        "need",
        "that",
        "the",
        "this",
        "with",
        "you",
        "your",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(text).lower())
        if len(token) > 2 and token not in stop
    }

@dataclass(frozen=True)
class ProductCandidate:
    session_id: int
    asin: str
    attributes: tuple[str, ...]

    def as_dict(self, *, score: int | None = None) -> dict[str, Any]:
        row: dict[str, Any] = {
            "asin": self.asin,
            "attributes": list(self.attributes),
        }
        if score is not None:
            row["match_score"] = score
        return row

@dataclass(frozen=True)
class PromptOptionCandidate:
    option_id: str
    asin: str
    description: str

    def as_dict(self, *, score: int | None = None) -> dict[str, Any]:
        row: dict[str, Any] = {
            "option_id": self.option_id,
            "asin": self.asin,
            "description": self.description,
        }
        if score is not None:
            row["match_score"] = score
        return row

def collect_product_catalog(task: dict[str, Any]) -> list[ProductCandidate]:
    """Build a task-local product catalog from HF answer records."""
    products: list[ProductCandidate] = []
    seen: set[str] = set()
    for i, answer in enumerate(task.get("answers") or []):
        if not isinstance(answer, dict):
            continue
        asin = answer.get("target_asin") or answer.get("asin")
        if not asin:
            raw = json.dumps(answer, ensure_ascii=False)
            match = _ASIN_RE.search(raw)
            asin = match.group(0) if match else None
        if not asin or asin in seen:
            continue
        attrs = answer.get("attributes") or []
        if isinstance(attrs, str):
            attrs = [attrs]
        products.append(
            ProductCandidate(
                session_id=i,
                asin=str(asin),
                attributes=tuple(str(x) for x in attrs if x is not None),
            )
        )
        seen.add(str(asin))
    return products

def parse_available_options(question: str) -> list[str]:
    """Extract the shuffled prompt options for the current shopping step."""
    if "Available Options" not in question:
        return []
    tail = question.rsplit("Available Options", 1)[-1]
    options: list[str] = []
    for line in tail.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            options.append(stripped[2:].strip())
        elif options and stripped and not stripped.startswith(("**", "#", "Product ")):
            options[-1] = f"{options[-1]} {stripped}"
    return options

def _synthetic_asin(task: dict[str, Any], session_id: int, option_index: int, description: str) -> str:
    seed = f"{task.get('id', task.get('_hf_index', 'task'))}|{session_id}|{option_index}|{description}"
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest().upper()
    return "B" + digest[:9]

def collect_prompt_options(
    task: dict[str, Any],
    *,
    session_id: int,
    question: str,
) -> list[PromptOptionCandidate]:
    """Build candidate products from the current prompt, not from gold ASINs."""
    candidates: list[PromptOptionCandidate] = []
    for index, description in enumerate(parse_available_options(question), start=1):
        candidates.append(
            PromptOptionCandidate(
                option_id=f"option_{index}",
                asin=_synthetic_asin(task, session_id, index, description),
                description=description,
            )
        )
    return candidates

def _infer_target_option(
    task: dict[str, Any],
    *,
    session_id: int,
    question: str,
    gold: Any,
) -> tuple[PromptOptionCandidate | None, dict[str, Any]]:
    """Infer the gold prompt option for judging without exposing it to the agent.

    The public HF snapshot stores the target ASIN and compact attributes, but
    not a full option-to-ASIN map. For paper-simulation runs we therefore score
    against the prompt option whose description best matches the gold
    attributes. This is intentionally a judge-only operation.
    """
    options = collect_prompt_options(task, session_id=session_id, question=question)
    if not options:
        return None, {"reason": "no prompt options parsed"}
    if not isinstance(gold, dict):
        return None, {"reason": "gold is not a dict"}
    attrs = gold.get("attributes") or []
    if isinstance(attrs, str):
        attrs = [attrs]
    target_text = " ".join(str(x) for x in attrs if x is not None)
    target_tokens = _tokens(target_text)
    if not target_tokens:
        return None, {"reason": "gold attributes unavailable"}
    scored = [
        (
            len(target_tokens & _tokens(option.description)),
            len(_tokens(option.description) & target_tokens) / max(1, len(target_tokens)),
            option,
        )
        for option in options
    ]
    scored.sort(key=lambda item: (-item[0], -item[1], item[2].option_id))
    best_score, best_recall, best = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else None
    if best_score <= 0:
        return None, {
            "reason": "no option overlaps gold attributes",
            "gold_attributes": attrs,
        }
    return best, {
        "gold_attributes": attrs,
        "target_option_id": best.option_id,
        "target_overlap": best_score,
        "target_recall": best_recall,
        "second_overlap": second_score,
    }

def render_bundled_shopping_environment(
    task: dict[str, Any],
    *,
    session_id: int,
    question: str,
    mode: str = "catalog",
    max_candidates: int = 12,
) -> str:
    if mode in {"", "none", None}:                                    
        return ""
    env = ReconstructedBundledShoppingEnv(
        task,
        session_id=session_id,
        question=question,
        mode=mode,
        max_candidates=max_candidates,
    )
    return json.dumps(env.search_product(query=question), ensure_ascii=False, indent=2)

class ReconstructedBundledShoppingEnv:
    """Small deterministic product-search environment for one shopping session.

    ``paper_sim`` mode exposes only the current prompt's available options,
    with synthetic ASIN-like IDs. The gold answer is used only by
    ``judge_prediction`` after the model finishes. ``catalog`` and ``oracle``
    are retained for diagnostics and are not paper-comparable.
    """

    def __init__(
        self,
        task: dict[str, Any],
        *,
        session_id: int,
        question: str,
        mode: str = "catalog",
        max_candidates: int = 12,
    ) -> None:
        if mode == "prompt_options":
            mode = "paper_sim"
        if mode not in {"catalog", "oracle", "paper_sim"}:
            raise ValueError("mode must be catalog, oracle, or paper_sim")
        self.task = task
        self.session_id = session_id
        self.question = question
        self.mode = mode
        self.max_candidates = max_candidates
        self.catalog = collect_product_catalog(task)
        self.prompt_options = collect_prompt_options(
            task,
            session_id=session_id,
            question=question,
        )

    def instructions(self) -> str:
        if self.mode == "paper_sim":
            note = (
                "The environment returns only the current prompt's available options, "
                "using synthetic ASIN-like IDs. It does not expose the hidden target."
            )
        elif self.mode == "catalog":
            note = "The environment returns ranked candidates from a reconstructed task-local WebShop catalog."
        else:
            note = "The environment simulates a successful WebShop search with the current target ranked first."
        return (
            "You may query a simulated WebShop search environment before giving the final answer.\n"
            f"{note}\n"
            "Use one JSON object per turn.\n"
            "Tool call format: {\"tool\":\"search_product\",\"query\":\"short product search query\"}\n"
            "Final answer format: {\"final\":\"ASIN: <selected ASIN>\"}\n"
            "Call search_product before the final answer. Select exactly one ASIN for the current subtask."
        )

    def _target_asin(self) -> str | None:
        answers = self.task.get("answers") or []
        if self.session_id >= len(answers) or not isinstance(answers[self.session_id], dict):
            return None
        return answers[self.session_id].get("target_asin") or answers[self.session_id].get("asin")

    def search_product(self, *, query: str | None = None) -> dict[str, Any]:
        query_text = query or self.question
        query_tokens = _tokens(query_text)
        if self.mode == "paper_sim":
            scored_options = [
                (len(query_tokens & _tokens(option.description)), option)
                for option in self.prompt_options
            ]
            scored_options.sort(key=lambda item: (-item[0], item[1].option_id))
            return {
                "tool": "search_product",
                "mode": self.mode,
                "query": query_text,
                "catalog_source": "current prompt available options only; synthetic ASINs; no hidden target marker",
                "candidates": [
                    option.as_dict(score=score)
                    for score, option in scored_options[: self.max_candidates]
                ],
            }

        scored: list[tuple[int, ProductCandidate]] = []
        for product in self.catalog:
            haystack = " ".join(product.attributes)
            score = len(query_tokens & _tokens(haystack))
            scored.append((score, product))

        if self.mode == "oracle":
            target = self._target_asin()
            scored.sort(key=lambda item: (item[1].asin != target, -item[0], item[1].session_id))
        else:
            scored.sort(key=lambda item: (-item[0], item[1].session_id, item[1].asin))

        candidates = [
            product.as_dict(score=score)
            for score, product in scored[: self.max_candidates]
        ]
        return {
            "tool": "search_product",
            "mode": self.mode,
            "query": query_text,
            "catalog_source": "task-local reconstructed products from HF answer ASINs and attributes",
            "candidates": candidates,
        }

    def judge_prediction(self, prediction: str, gold: Any) -> dict[str, Any]:
        if self.mode != "paper_sim":
            from agentmem.eval.memoryarena_runner import judges

            return judges.asin_judge(prediction, gold)
        pred_asin = None
        match = _ASIN_RE.search(prediction or "")
        if match:
            pred_asin = match.group(0)
        target, info = _infer_target_option(
            self.task,
            session_id=self.session_id,
            question=self.question,
            gold=gold,
        )
        target_asin = target.asin if target is not None else None
        return {
            "correct": bool(pred_asin and target_asin and pred_asin == target_asin),
            "pred_asin": pred_asin,
            "target_sim_asin": target_asin,
            **info,
        }

    def handle_action(self, action: dict[str, Any]) -> dict[str, Any]:
        tool = str(action.get("tool", "")).lower()
        if tool == "search_product":
            return self.search_product(query=str(action.get("query") or self.question))
        if tool == "list_catalog":
            if self.mode == "paper_sim":
                return {
                    "tool": "list_catalog",
                    "mode": self.mode,
                    "candidates": [
                        option.as_dict()
                        for option in self.prompt_options[: self.max_candidates]
                    ],
                }
            return {
                "tool": "list_catalog",
                "mode": self.mode,
                "candidates": [product.as_dict() for product in self.catalog[: self.max_candidates]],
            }
        return {
            "error": f"Unknown tool: {tool}",
            "available_tools": ["search_product", "list_catalog"],
        }
