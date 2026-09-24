"""Memory retrieval for AMA-Agent."""

import json
import os
import re
import subprocess
import sys
import tempfile
from typing import Any, Callable, Dict, List, Optional

from rank_bm25 import BM25Okapi

from .prompts import (
    CHECK_STATE_MEM_PROMPT_TEMPLATE,
    CHUNK_SUFFICIENCY_JUDGMENT_PROMPT_TEMPLATE,
    CODE_GENERATION_PROMPT_TEMPLATE,
    render_code_generation_prompt,
)
from .utils import build_graph_node_text, retrieve_with_llm_tools, retrieve_with_qwen

_DEFAULT_RETRIEVE_PREFILTER_K = 32
_DEFAULT_CORPUS_CONTEXT_CHARS = 28_000
_DEFAULT_CORPUS_STATE_CHARS = 6_000

def memory_retrieve(
    memory: Dict[str, Any],
    question: str,
    call_llm_func: Callable,
    top_k: int = 5,
    neighbor_radius: int = 0,
    retrieval_mode: str = "qwen",
    embed_engine: Optional[Callable] = None,
    enable_tools: bool = True,
    model_client: Optional[Any] = None,
    retrieval_prefilter_k: Optional[int] = None,
    causal_graph_max_edges: int = 24,
    tool_retrieval_scope: str = "full",
    tool_retrieval_prefilter_k: Optional[int] = None,
    tool_retrieval_max_results: int = 5,
    context_guardrail: Optional[str] = None,
    prepend_objective: bool = False,
    return_debug: bool = False,
) -> Any:
    """Retrieve relevant context from memory for answering a question.

    Retrieval flow:
    1. Check whether state memory is already sufficient.
    2. Prefer embedding-based turn retrieval when embeddings are available.
    3. Fall back to LLM chunk scoring.
    4. Preserve only question-local causal graph context instead of dumping the
       entire graph into the QA prompt.
    """
    state_mem = memory.get("state_mem", "")
    text_mem = memory.get("text_mem", {})
    graph_mem = memory.get("graph_mem") or memory.get("embed_mem")
    embed_mem = graph_mem
    causal_graph = memory.get("causal_graph")

    task = text_mem.get("task", "")
    trajectory = text_mem.get("trajectory_data", {}).get("trajectory", [])
    retrieval_mode = (retrieval_mode or "qwen").lower()
    prefilter_k = _resolve_retrieve_topk(retrieval_prefilter_k)
    tool_retrieval_scope = (tool_retrieval_scope or "full").lower()

    candidate_node_ids: List[str] = []
    candidate_turn_indices: List[int] = []
    prefilter_active = False
    if graph_mem is not None and embed_engine is not None:
        (
            candidate_node_ids,
            candidate_turn_indices,
            prefilter_active,
        ) = _prefilter_graph_nodes_with_embed(
            question=question,
            graph_mem=graph_mem,
            embed_engine=embed_engine,
            top_k=prefilter_k,
        )

    state_mem_str = _render_prompt_state_memory(
        memory=memory,
        state_mem=state_mem,
        turn_indices=candidate_turn_indices,
        prefilter_active=prefilter_active,
        max_chars=_DEFAULT_CORPUS_STATE_CHARS,
    )

    check_prompt = CHECK_STATE_MEM_PROMPT_TEMPLATE.format(
        state_mem_str=state_mem_str,
        query=question,
    )
    _, check_response = call_llm_func(check_prompt)

    need_retrieval = bool(
        check_response and "NEED_RETRIEVAL" in check_response.upper()
    )

    if not need_retrieval:
        context = _build_context(
            state_mem=state_mem_str,
            task=task,
            causal_graph=None,
            context_guardrail=context_guardrail,
            prepend_objective=prepend_objective,
        )
        debug = {
            "state_mem_sufficient": True,
            "retrieval_mode": retrieval_mode,
            "route_kind": "state_mem_only",
            "selected_node_ids": [],
            "relevant_turn_indices": [],
            "expanded_turn_indices": [],
            "tool_turn_indices": [],
            "answer_candidate": None,
            "code_analysis": None,
            "retrieval_prefilter_k": prefilter_k,
            "causal_graph_max_edges": causal_graph_max_edges,
            "tool_retrieval_scope": tool_retrieval_scope,
            "tool_retrieval_prefilter_k": tool_retrieval_prefilter_k,
            "tool_retrieval_max_results": tool_retrieval_max_results,
            "prepend_objective": prepend_objective,
            "prefilter_active": prefilter_active,
            "prefiltered_node_count": len(candidate_node_ids),
            "prefiltered_turn_count": len(candidate_turn_indices),
            "context_char_count": len(context),
        }
        return (context, debug) if return_debug else context

    relevant_turn_indices: List[int] = []
    selected_node_ids: List[str] = []

    if retrieval_mode == "embed" and graph_mem is not None:
        if prefilter_active and candidate_node_ids:
            selected_node_ids = candidate_node_ids[: min(top_k, len(candidate_node_ids))]
            relevant_turn_indices = _node_ids_to_turn_indices(graph_mem, selected_node_ids)
        elif embed_engine is not None:
            selected_node_ids, relevant_turn_indices = _retrieve_graph_nodes_with_embed(
                question=question,
                graph_mem=graph_mem,
                embed_engine=embed_engine,
                top_k=min(top_k, prefilter_k),
            )
    elif retrieval_mode == "bm25":
        relevant_turn_indices = _retrieve_with_bm25(
            question=question,
            trajectory=_candidate_trajectory(trajectory, candidate_turn_indices),
            top_k=top_k,
        )

    if not relevant_turn_indices:
        _, relevant_turn_indices = retrieve_with_qwen(
            question,
            _candidate_text_mem(text_mem, candidate_turn_indices),
            call_llm_func,
            top_k=top_k,
        )

    expanded_turn_indices = _expand_turn_indices(
        trajectory=trajectory,
        turn_indices=relevant_turn_indices,
        neighbor_radius=neighbor_radius,
    )
    if prefilter_active:
        expanded_turn_indices = _cap_turn_indices(
            turn_indices=expanded_turn_indices,
            preferred_turn_indices=relevant_turn_indices,
            max_turns=prefilter_k,
        )
    relevant_chunks = _extract_chunks(trajectory, expanded_turn_indices)

    route = _judge_retrieval_sufficiency(
        question=question,
        chunks=relevant_chunks,
        call_llm_func=call_llm_func,
    )

    if route["kind"] == "need_graph":
        if selected_node_ids and graph_mem:
            expanded_node_ids = _expand_graph_node_ids(
                graph_mem=graph_mem,
                seed_node_ids=selected_node_ids,
                detail=route.get("detail", ""),
                max_nodes=prefilter_k,
            )
            if expanded_node_ids:
                selected_node_ids = expanded_node_ids
                relevant_turn_indices = sorted(
                    set(relevant_turn_indices) | set(_node_ids_to_turn_indices(graph_mem, selected_node_ids))
                )
        requested_turn_indices = _expand_turns_from_need_graph(
            route.get("detail", ""),
            trajectory=trajectory,
        )
        if requested_turn_indices:
            expanded_turn_indices = sorted(
                set(expanded_turn_indices) | set(requested_turn_indices)
            )
            if prefilter_active:
                expanded_turn_indices = _cap_turn_indices(
                    turn_indices=expanded_turn_indices,
                    preferred_turn_indices=relevant_turn_indices,
                    max_turns=prefilter_k,
                )
            relevant_chunks = _extract_chunks(trajectory, expanded_turn_indices)

    code_analysis = None
    tool_turn_indices: List[int] = []
    if route["kind"] == "need_code":
        code_analysis = _run_need_code_analysis(
            question=question,
            task=task,
            text_mem=text_mem,
            call_llm_func=call_llm_func,
        )

    if route["kind"] != "sufficient" and route["kind"] != "need_code" and enable_tools:

        tool_text_mem = text_mem
        if tool_retrieval_scope == "prefiltered":
            tool_prefilter_k = _resolve_retrieve_topk(
                tool_retrieval_prefilter_k
                if tool_retrieval_prefilter_k is not None
                else prefilter_k
            )
            tool_candidate_turn_indices = candidate_turn_indices
            if graph_mem and candidate_node_ids:
                tool_candidate_turn_indices = _node_ids_to_turn_indices(
                    graph_mem,
                    candidate_node_ids[:tool_prefilter_k],
                )
            tool_text_mem = _candidate_text_mem(text_mem, tool_candidate_turn_indices)

        _, tool_turn_indices = retrieve_with_llm_tools(
            query=question,
            state_mem=state_mem_str,
            text_mem=tool_text_mem,
            call_llm_func=call_llm_func,
            max_results=tool_retrieval_max_results,
        )
        merged_turn_indices = sorted(set(expanded_turn_indices) | set(tool_turn_indices))
        if prefilter_active:
            merged_turn_indices = _cap_turn_indices(
                turn_indices=merged_turn_indices,
                preferred_turn_indices=relevant_turn_indices,
                max_turns=prefilter_k,
            )
        if merged_turn_indices != expanded_turn_indices:
            relevant_chunks = _extract_chunks(trajectory, merged_turn_indices)
            expanded_turn_indices = merged_turn_indices

    filtered_causal_graph = _filter_causal_graph(
        causal_graph=causal_graph,
        turn_indices=expanded_turn_indices,
        max_edges=causal_graph_max_edges,
    )
    retrieved_nodes_text = _format_graph_nodes(graph_mem, selected_node_ids) if selected_node_ids and graph_mem else None

    answer_candidate = None
    if route["kind"] == "sufficient":
        answer_candidate = route.get("answer")

    context = _build_context(
        state_mem=state_mem_str,
        task=task,
        causal_graph=filtered_causal_graph,
        retrieved_nodes_text=retrieved_nodes_text,
        chunks_text=_format_chunks(relevant_chunks),
        chunk_count=len(relevant_chunks),
        answer_candidate=answer_candidate,
        code_analysis=code_analysis,
        context_guardrail=context_guardrail,
        prepend_objective=prepend_objective,
    )
    if prefilter_active:
        context = _truncate_text(
            context,
            int(os.environ.get("AMA_RETRIEVE_CONTEXT_CHARS", _DEFAULT_CORPUS_CONTEXT_CHARS)),
        )
    debug = {
        "state_mem_sufficient": False,
        "retrieval_mode": retrieval_mode,
        "route_kind": route.get("kind"),
        "route_raw": route.get("raw"),
        "selected_node_ids": selected_node_ids,
        "relevant_turn_indices": relevant_turn_indices,
        "expanded_turn_indices": expanded_turn_indices,
        "tool_turn_indices": tool_turn_indices,
        "answer_candidate": answer_candidate,
        "code_analysis_present": bool(code_analysis),
        "retrieved_node_count": len(selected_node_ids),
        "retrieved_turn_count": len(expanded_turn_indices),
        "retrieval_prefilter_k": prefilter_k,
        "causal_graph_max_edges": causal_graph_max_edges,
        "tool_retrieval_scope": tool_retrieval_scope,
        "tool_retrieval_prefilter_k": tool_retrieval_prefilter_k,
        "tool_retrieval_max_results": tool_retrieval_max_results,
        "prepend_objective": prepend_objective,
        "prefilter_active": prefilter_active,
        "prefiltered_node_count": len(candidate_node_ids),
        "prefiltered_turn_count": len(candidate_turn_indices),
        "context_char_count": len(context),
    }
    return (context, debug) if return_debug else context

def _extract_chunks(
    trajectory: List[Dict[str, Any]],
    turn_indices: List[int],
) -> List[Dict[str, Any]]:
    chunks = []
    for turn in trajectory:
        turn_idx = turn.get("turn_idx", -1)
        if turn_idx in turn_indices:
            chunks.append(
                {
                    "turn": turn_idx,
                    "action": turn.get("action", ""),
                    "observation": turn.get("observation", ""),
                }
            )
    chunks.sort(key=lambda item: item["turn"])
    return chunks

def _expand_turn_indices(
    trajectory: List[Dict[str, Any]],
    turn_indices: List[int],
    neighbor_radius: int,
) -> List[int]:
    """Expand retrieved hits with nearby turns to preserve local temporal context."""
    if neighbor_radius <= 0 or not turn_indices:
        return sorted(set(turn_indices))

    ordered_turns = [turn.get("turn_idx", idx) for idx, turn in enumerate(trajectory)]
    turn_to_pos = {turn_idx: pos for pos, turn_idx in enumerate(ordered_turns)}
    expanded_positions = set()

    for turn_idx in turn_indices:
        pos = turn_to_pos.get(turn_idx)
        if pos is None:
            continue
        start = max(0, pos - neighbor_radius)
        end = min(len(ordered_turns), pos + neighbor_radius + 1)
        expanded_positions.update(range(start, end))

    return [ordered_turns[pos] for pos in sorted(expanded_positions)]

def _filter_causal_graph(
    causal_graph: Optional[Any],
    turn_indices: List[int],
    max_edges: int = 24,
) -> Optional[List[Dict[str, Any]]]:
    """Keep only graph edges local to the retrieved evidence.

    The paper describes retrieving top-K nodes and traversing their local
    neighborhoods, not injecting the full graph into every QA prompt. The
    released code path serialized the entire causal graph, which can overwhelm
    the QA context window and drown out the retrieved evidence. Restrict the
    graph to edges directly touching retrieved turns, with a nearest-edge
    fallback when the graph uses sparse turn annotations.
    """
    if not isinstance(causal_graph, list) or not causal_graph:
        return None
    if not turn_indices:
        return None

    turn_set = set(turn_indices)
    matched_edges: list[dict[str, Any]] = []
    fallback_edges: list[tuple[int, dict[str, Any]]] = []

    for edge in causal_graph:
        if not isinstance(edge, dict):
            continue

        edge_turn = edge.get("turn_idx")
        edge_turns = {edge_turn} if isinstance(edge_turn, int) else set()
        if edge_turns & turn_set:
            matched_edges.append(edge)
            continue
        if edge_turns:
            distance = min(
                min(abs(turn - edge_turn) for turn in turn_set)
                for edge_turn in edge_turns
            )
            fallback_edges.append((distance, edge))

    if matched_edges:
        return matched_edges[:max_edges]

    fallback_edges.sort(key=lambda item: item[0])
    nearest_edges = [edge for _, edge in fallback_edges[:max_edges]]
    return nearest_edges or None

def _resolve_retrieve_topk(override: Optional[int] = None) -> int:
    if override is not None:
        return max(1, int(override))
    raw = os.environ.get("AMA_RETRIEVE_TOPK")
    if raw is None:
        return _DEFAULT_RETRIEVE_PREFILTER_K
    try:
        return max(1, int(raw))
    except ValueError:
        return _DEFAULT_RETRIEVE_PREFILTER_K

def _truncate_text(text: Any, max_chars: int) -> str:
    rendered = str(text or "")
    if max_chars <= 0 or len(rendered) <= max_chars:
        return rendered
    return rendered[:max_chars].rstrip() + "\n... [truncated]"

def _render_prompt_state_memory(
    memory: Dict[str, Any],
    state_mem: Any,
    turn_indices: List[int],
    prefilter_active: bool,
    max_chars: int,
) -> str:
    if not prefilter_active:
        return str(state_mem or "")

    reports = memory.get("state_reports") or []
    if isinstance(reports, list) and turn_indices:
        reports_by_turn: Dict[int, Dict[str, Any]] = {}
        for report in reports:
            if not isinstance(report, dict):
                continue
            turn_token = str(report.get("turn_idx", -1)).strip()
            if not turn_token.isdigit():
                continue
            reports_by_turn[int(turn_token)] = report
        parts: List[str] = ["# Question-Local State Memory"]
        for turn_idx in turn_indices:
            turn_token = str(turn_idx).strip()
            if not turn_token.isdigit():
                continue
            report = reports_by_turn.get(int(turn_token))
            if not report:
                continue
            parts.append(f"## Turn {turn_idx}")
            raw_report = str(report.get("raw_report", "") or "").strip()
            if raw_report:
                parts.append(raw_report)
            else:
                action = report.get("action", "")
                observation = report.get("observation", "")
                parts.append(f"Action: {action}\nObservation: {observation}")
            parts.append("")
        rendered = "\n".join(parts).strip()
        if rendered:
            return _truncate_text(rendered, max_chars)

    text_mem = memory.get("text_mem", {})
    trajectory = text_mem.get("trajectory_data", {}).get("trajectory", [])
    chunks = _extract_chunks(trajectory, turn_indices)
    rendered_chunks = _format_chunks(chunks)
    if rendered_chunks and rendered_chunks != "No chunks retrieved.":
        return _truncate_text("# Question-Local State Memory\n" + rendered_chunks, max_chars)

    return _truncate_text(state_mem, max_chars)

def _candidate_trajectory(
    trajectory: List[Dict[str, Any]],
    candidate_turn_indices: List[int],
) -> List[Dict[str, Any]]:
    if not candidate_turn_indices:
        return trajectory
    allowed = {
        int(turn_token)
        for turn_token in (str(turn_idx).strip() for turn_idx in candidate_turn_indices)
        if turn_token.isdigit()
    }
    return [
        turn
        for idx, turn in enumerate(trajectory)
        if str(turn.get("turn_idx", idx)).strip().isdigit()
        and int(str(turn.get("turn_idx", idx)).strip()) in allowed
    ]

def _candidate_text_mem(
    text_mem: Dict[str, Any],
    candidate_turn_indices: List[int],
) -> Dict[str, Any]:
    if not candidate_turn_indices:
        return text_mem

    trajectory_data = dict(text_mem.get("trajectory_data", {}))
    trajectory_data["trajectory"] = _candidate_trajectory(
        trajectory_data.get("trajectory", []),
        candidate_turn_indices,
    )
    narrowed = dict(text_mem)
    narrowed["trajectory_data"] = trajectory_data
    return narrowed

def _cap_turn_indices(
    turn_indices: List[int],
    preferred_turn_indices: List[int],
    max_turns: int,
) -> List[int]:
    if max_turns <= 0 or len(turn_indices) <= max_turns:
        return sorted(set(turn_indices))

    selected: List[int] = []
    seen: set[int] = set()
    for source in (preferred_turn_indices, sorted(turn_indices)):
        for turn_idx in source:
            if turn_idx in seen or turn_idx not in turn_indices:
                continue
            selected.append(turn_idx)
            seen.add(turn_idx)
            if len(selected) >= max_turns:
                return sorted(selected)
    return sorted(selected)

def _judge_retrieval_sufficiency(
    question: str,
    chunks: List[Dict[str, Any]],
    call_llm_func: Callable,
) -> Dict[str, Optional[str]]:
    chunks_text = _format_chunks(chunks)

    sufficiency_prompt = CHUNK_SUFFICIENCY_JUDGMENT_PROMPT_TEMPLATE.format(
        query=question,
        retrieved_chunks=chunks_text,
    )

    _, llm_response = call_llm_func(sufficiency_prompt)
    return _parse_chunk_sufficiency_response(llm_response or "")

def _parse_chunk_sufficiency_response(response: str) -> Dict[str, Optional[str]]:
    stripped = (response or "").strip()
    upper = stripped.upper()

    if upper.startswith("SUFFICIENT"):
        answer_match = re.search(r"ANSWER:\s*(.+)", stripped, flags=re.IGNORECASE | re.DOTALL)
        answer = answer_match.group(1).strip() if answer_match else None
        return {"kind": "sufficient", "detail": None, "answer": answer, "raw": stripped}

    if upper.startswith("NEED_GRAPH"):
        detail = stripped.split(":", 1)[1].strip() if ":" in stripped else ""
        return {"kind": "need_graph", "detail": detail, "answer": None, "raw": stripped}

    if upper.startswith("NEED_CODE"):
        detail = stripped.split(":", 1)[1].strip() if ":" in stripped else ""
        return {"kind": "need_code", "detail": detail, "answer": None, "raw": stripped}

    return {"kind": "need_graph", "detail": stripped, "answer": None, "raw": stripped}

def _expand_turns_from_need_graph(
    detail: str,
    trajectory: List[Dict[str, Any]],
) -> List[int]:
    if not detail:
        return []

    ordered_turns = [turn.get("turn_idx", idx) for idx, turn in enumerate(trajectory)]
    if not ordered_turns:
        return []
    turn_to_pos = {turn_idx: pos for pos, turn_idx in enumerate(ordered_turns)}
    selected: set[int] = set()

    for match in re.finditer(r"turn_(\d+)\s+before=(\d+)\s+after=(\d+)", detail, flags=re.IGNORECASE):
        turn_idx = int(match.group(1))
        before = int(match.group(2))
        after = int(match.group(3))
        pos = turn_to_pos.get(turn_idx)
        if pos is None:
            continue
        start = max(0, pos - before)
        end = min(len(ordered_turns) - 1, pos + after)
        selected.update(ordered_turns[start : end + 1])

    for match in re.finditer(r"turns?\s+(\d+)\s+to\s+(\d+)", detail, flags=re.IGNORECASE):
        start_turn = int(match.group(1))
        end_turn = int(match.group(2))
        if start_turn > end_turn:
            start_turn, end_turn = end_turn, start_turn
        selected.update(
            turn_idx for turn_idx in ordered_turns if start_turn <= turn_idx <= end_turn
        )

    comma_number_matches = re.findall(r"turns?\s+((?:\d+\s*,\s*)+\d+)", detail, flags=re.IGNORECASE)
    for match in comma_number_matches:
        for num in re.findall(r"\d+", match):
            selected.add(int(num))

    return sorted(turn_idx for turn_idx in selected if turn_idx in turn_to_pos)

def _run_need_code_analysis(
    question: str,
    task: str,
    text_mem: Dict[str, Any],
    call_llm_func: Callable,
) -> str:
    trajectory_data = text_mem.get("trajectory_data", {})
    trajectory = trajectory_data.get("trajectory", [])
    trajectory_sample = _format_trajectory_sample(trajectory[:2])
    code_prompt = render_code_generation_prompt(
        query=question,
        task=task,
        trajectory_sample=trajectory_sample,
    )
    _, code_response = call_llm_func(code_prompt)
    generated_code = _extract_python_code(code_response or "")
    if not generated_code:
        return "Code generation failed: no executable Python code found in the model response."

    execution_output = _execute_generated_code(
        generated_code=generated_code,
        trajectory_data=trajectory_data,
    )

    parts = [
        "# Generated Code",
        "```python",
        generated_code,
        "```",
        "",
        "# Execution Output",
        execution_output.strip() or "(no output)",
    ]
    return "\n".join(parts)

def _format_trajectory_sample(trajectory: List[Dict[str, Any]]) -> str:
    if not trajectory:
        return "(trajectory is empty)"
    lines: List[str] = []
    for turn in trajectory:
        turn_idx = turn.get("turn_idx", 0)
        action = turn.get("action", "")
        observation = turn.get("observation", "")
        lines.append(f"Turn {turn_idx}:")
        lines.append(f"  Action: {action}")
        lines.append(f"  Observation: {observation}")
    return "\n".join(lines)

def _extract_python_code(response: str) -> Optional[str]:
    code_block = re.search(r"```python\s*(.*?)```", response, flags=re.IGNORECASE | re.DOTALL)
    if code_block:
        return code_block.group(1).strip()
    generic_block = re.search(r"```\s*(.*?)```", response, flags=re.DOTALL)
    if generic_block:
        return generic_block.group(1).strip()
    stripped = response.strip()
    if stripped:
        return stripped
    return None

def _execute_generated_code(
    generated_code: str,
    trajectory_data: Dict[str, Any],
    timeout_sec: float = 20.0,
) -> str:

    wrapper = (
        "import json\n"
        f"_AMA_TRAJ_JSON_STR = {json.dumps(json.dumps(trajectory_data, ensure_ascii=False))}\n"
        "trajectory_json = json.loads(_AMA_TRAJ_JSON_STR)\n\n"
        f"{generated_code}\n\n"
        "if 'result' in globals():\n"
        "    print('\\n__AMA_RESULT__')\n"
        "    try:\n"
        "        print(json.dumps(result, ensure_ascii=False))\n"
        "    except Exception:\n"
        "        print(str(result))\n"
    )

    with tempfile.TemporaryDirectory(prefix="ama_agent_code_") as tmpdir:
        script_path = os.path.join(tmpdir, "script.py")
        with open(script_path, "w", encoding="utf-8") as fh:
            fh.write(wrapper)
        env = os.environ.copy()
        env["PYTHONNOUSERSITE"] = "1"
        try:
            completed = subprocess.run(
                [sys.executable, script_path],
                cwd=tmpdir,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired:
            return "timeout"

    stdout = (completed.stdout or "").strip()
    stderr = (completed.stderr or "").strip()
    if stderr and stdout:
        return f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
    if stderr:
        return f"STDERR:\n{stderr}"
    return stdout or "(no output)"

def _build_context(
    state_mem: str,
    task: str,
    causal_graph: Optional[Any],
    retrieved_nodes_text: Optional[str] = None,
    chunks_text: Optional[str] = None,
    sufficiency_response: Optional[str] = None,
    chunk_count: int = 0,
    answer_candidate: Optional[str] = None,
    code_analysis: Optional[str] = None,
    context_guardrail: Optional[str] = None,
    prepend_objective: bool = False,
) -> str:
    parts = []
    if prepend_objective and task:
        parts.extend(
            [
                "OBJECTIVE:",
                task,
                "",
            ]
        )

    parts.extend([
        "# State Memory",
        state_mem,
        "",
        "# Task",
        task,
    ])

    if context_guardrail:
        parts.extend(
            [
                "",
                "# Answering Guardrail",
                context_guardrail.strip(),
            ]
        )

    if causal_graph:
        parts.extend(
            [
                "",
                "# Causal Graph",
                json.dumps(causal_graph, indent=2),
            ]
        )

    if retrieved_nodes_text is not None:
        parts.extend(
            [
                "",
                "# Retrieved Graph Nodes",
                retrieved_nodes_text,
            ]
        )

    if chunks_text is not None:
        parts.extend(
            [
                "",
                f"# Retrieved Relevant Information ({chunk_count} chunks)",
                chunks_text,
            ]
        )

    if sufficiency_response is not None:
        parts.extend(
            [
                "",
                "# Sufficiency Assessment",
                sufficiency_response,
            ]
        )

    if answer_candidate:
        parts.extend(
            [
                "",
                "# Direct Answer Candidate",
                answer_candidate,
            ]
        )

    if code_analysis:
        parts.extend(
            [
                "",
                "# Code Analysis",
                code_analysis,
            ]
        )

    return "\n".join(parts)

def _retrieve_with_embed(
    question: str,
    embed_mem: Dict[str, Any],
    embed_engine: Callable,
    top_k: int,
) -> List[int]:
    """Retrieve top-k items by embedding similarity."""
    query_embedding = embed_engine(question)
    similarities = []
    turn_indices = embed_mem.get("turn_indices", [])

    for idx, turn_embedding in enumerate(embed_mem.get("embeddings", [])):
        turn_idx = turn_indices[idx] if idx < len(turn_indices) else idx
        similarities.append((turn_idx, _cosine_similarity(query_embedding, turn_embedding)))

    similarities.sort(key=lambda item: item[1], reverse=True)
    return [turn_idx for turn_idx, _ in similarities[:top_k]]

def _retrieve_graph_nodes_with_embed(
    question: str,
    graph_mem: Dict[str, Any],
    embed_engine: Callable,
    top_k: int,
) -> tuple[List[str], List[int]]:
    """Retrieve top-k graph/state nodes by embedding similarity."""
    query_embedding = embed_engine(question)
    node_ids = graph_mem.get("node_ids", [])
    node_turns = graph_mem.get("node_turns", [])
    node_embeddings = graph_mem.get("embeddings", [])

    scored: List[tuple[str, int, float]] = []
    for idx, node_embedding in enumerate(node_embeddings):
        node_id = node_ids[idx] if idx < len(node_ids) else f"node_{idx}"
        turn_token = str(node_turns[idx] if idx < len(node_turns) else idx).strip()
        if not turn_token.isdigit():
            continue
        turn_idx = int(turn_token)
        scored.append((node_id, turn_idx, _cosine_similarity(query_embedding, node_embedding)))

    scored.sort(key=lambda item: item[2], reverse=True)
    selected = scored[:top_k]
    selected_node_ids = [node_id for node_id, _, _ in selected]
    relevant_turn_indices = sorted({turn_idx for _, turn_idx, _ in selected})
    return selected_node_ids, relevant_turn_indices

def _prefilter_graph_nodes_with_embed(
    question: str,
    graph_mem: Dict[str, Any],
    embed_engine: Callable,
    top_k: int,
) -> tuple[List[str], List[int], bool]:
    """Limit large graph memories to top-K stored node embeddings before LLM use."""
    node_ids = graph_mem.get("node_ids", [])
    node_turns = graph_mem.get("node_turns", [])
    node_embeddings = graph_mem.get("embeddings", [])
    node_count = min(len(node_ids), len(node_embeddings))
    if node_count <= 0:
        return [], [], False
    if node_count <= top_k:
        selected_node_ids = []
        selected_turns = []
        for idx in range(node_count):
            turn_token = str(node_turns[idx] if idx < len(node_turns) else idx).strip()
            if not turn_token.isdigit():
                continue
            selected_node_ids.append(str(node_ids[idx]))
            selected_turns.append(int(turn_token))
        selected_turns = sorted(set(selected_turns))
        return selected_node_ids, selected_turns, False

    query_embedding = embed_engine(question)
    scored: List[tuple[str, int, float]] = []
    for idx, node_embedding in enumerate(node_embeddings[:node_count]):
        node_id = str(node_ids[idx])
        turn_token = str(node_turns[idx] if idx < len(node_turns) else idx).strip()
        if not turn_token.isdigit():
            continue
        turn_idx = int(turn_token)
        scored.append((node_id, turn_idx, _cosine_similarity(query_embedding, node_embedding)))

    scored.sort(key=lambda item: item[2], reverse=True)
    selected = scored[:top_k]
    selected_node_ids = [node_id for node_id, _, _ in selected]
    selected_turns = sorted({turn_idx for _, turn_idx, _ in selected})
    return selected_node_ids, selected_turns, True

def _retrieve_with_bm25(
    question: str,
    trajectory: List[Dict[str, Any]],
    top_k: int,
) -> List[int]:
    if not trajectory:
        return []

    docs = [
        f"Action: {turn.get('action', '')}\nObservation: {turn.get('observation', '')}"
        for turn in trajectory
    ]
    bm25 = BM25Okapi([doc.lower().split() for doc in docs])
    scores = bm25.get_scores(question.lower().split())
    ranked = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
    return [
        trajectory[idx].get("turn_idx", idx)
        for idx in ranked[:top_k]
        if scores[idx] > 0
    ]

def _cosine_similarity(vec1: Any, vec2: Any) -> float:
    if isinstance(vec1, list):
        numerator = sum(float(a) * float(b) for a, b in zip(vec1, vec2))
        norm1 = sum(float(a) * float(a) for a in vec1) ** 0.5
        norm2 = sum(float(b) * float(b) for b in vec2) ** 0.5
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return numerator / (norm1 * norm2)

    import numpy as np

    vec1_arr = np.array(vec1)
    vec2_arr = np.array(vec2)
    norm1 = np.linalg.norm(vec1_arr)
    norm2 = np.linalg.norm(vec2_arr)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1_arr, vec2_arr) / (norm1 * norm2))

def _format_chunks(chunks: List[Dict[str, Any]]) -> str:
    if not chunks:
        return "No chunks retrieved."

    lines = []
    for chunk in chunks:
        turn = chunk.get("turn", 0)
        action = chunk.get("action", "")
        observation = chunk.get("observation", "")[:2000]
        lines.append(f"Turn {turn}:")
        lines.append(f"  Action: {action}")
        lines.append(f"  Observation: {observation}")
        lines.append("")

    return "\n".join(lines)

def _expand_graph_node_ids(
    graph_mem: Dict[str, Any],
    seed_node_ids: List[str],
    detail: str,
    max_nodes: int,
) -> List[str]:
    adjacency = graph_mem.get("adjacency") or {}
    if not adjacency:
        return seed_node_ids[:max_nodes]

    depth = _infer_graph_depth(detail)
    visited = set(seed_node_ids[:max_nodes])
    ordered = list(seed_node_ids[:max_nodes])
    frontier = list(seed_node_ids)
    for _ in range(depth):
        next_frontier: List[str] = []
        for node_id in frontier:
            for neighbor in adjacency.get(node_id, []):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                ordered.append(neighbor)
                next_frontier.append(neighbor)
                if len(ordered) >= max_nodes:
                    return ordered
        if not next_frontier:
            break
        frontier = next_frontier
    return ordered

def _infer_graph_depth(detail: str) -> int:
    numbers = [int(num) for num in re.findall(r"\d+", detail or "")]
    if not numbers:
        return 1
    return max(1, min(3, max(numbers)))

def _node_ids_to_turn_indices(graph_mem: Dict[str, Any], node_ids: List[str]) -> List[int]:
    nodes_by_id = {node.get("node_id"): node for node in graph_mem.get("nodes", [])}
    turn_indices: set[int] = set()
    for node_id in node_ids:
        if node_id not in nodes_by_id:
            continue
        turn_token = str(nodes_by_id[node_id].get("turn_idx", -1)).strip()
        if not turn_token.isdigit():
            continue
        turn_indices.add(int(turn_token))
    return sorted(turn_indices)

def _format_graph_nodes(graph_mem: Dict[str, Any], node_ids: List[str]) -> str:
    nodes_by_id = {node.get("node_id"): node for node in graph_mem.get("nodes", [])}
    parts: List[str] = []
    for node_id in node_ids:
        node = nodes_by_id.get(node_id)
        if not node:
            continue
        parts.append(build_graph_node_text(node))
        parts.append("")
    return "\n".join(parts).strip() or "No graph nodes retrieved."
