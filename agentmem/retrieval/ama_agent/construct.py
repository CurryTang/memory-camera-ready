"""Paper-faithful AMA-Agent memory construction."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from .prompts import MEMORY_CONSTRUCTION_PROMPT_TEMPLATE
from .utils import (
    build_graph_node_text,
    parse_markdown_state_report,
    render_state_memory_summary,
)

def construct_state_memory(
    trajectory_text: str,
    task: str = "",
    call_llm_func: Optional[Callable] = None,
    chunk_size: int = 8192,
    embed_engine: Optional[Callable] = None,
    causal: bool = False,
) -> Dict[str, Any]:
    """Construct a causality graph from adjacent turn-pair state reports.

    The paper appendix describes per-turn analysis over adjacent turn pairs
    ``(o_{t-1}, a_t, o_t)`` using a strict Markdown schema. This builder follows
    that path directly instead of summarizing long chunks.
    """
    del chunk_size                                                                 

    trajectory = _parse_trajectory_text(trajectory_text)
    trajectory_data = {
        "trajectory": trajectory,
        "task": task,
        "episode_id": "episode",
    }
    text_mem = {
        "task": task,
        "trajectory_text": trajectory_text,
        "trajectory_data": trajectory_data,
        "episode_id": "episode",
        "num_turns": len(trajectory),
    }

    reports = _build_turn_reports(
        trajectory=trajectory,
        task=task,
        call_llm_func=call_llm_func,
    )
    graph_mem = _build_graph_memory(reports=reports, embed_engine=embed_engine) if reports else None
    state_mem = render_state_memory_summary(reports)
    causal_graph = graph_mem.get("edges") if (causal and graph_mem) else None

    return {
        "state_mem": state_mem,
        "causal_graph": causal_graph,
        "text_mem": text_mem,
        "embed_mem": graph_mem,
        "graph_mem": graph_mem,
        "state_reports": reports,
        "trajectory": trajectory,
    }

def _parse_trajectory_text(trajectory_text: str) -> List[Dict[str, Any]]:
    trajectory: List[Dict[str, Any]] = []
    lines = trajectory_text.strip().split("\n")

    current_turn: Dict[str, Any] = {}
    current_field: Optional[str] = None
    for line in lines:
        line = line.strip()

        if line.startswith("Turn ") or line.startswith("Step "):
            turn_token = line.split()[1].rstrip(":") if len(line.split()) > 1 else ""
            if not turn_token.isdigit():
                if current_field and line:
                    current_turn[current_field] = current_turn.get(current_field, "") + "\n" + line
                continue
            if current_turn:
                trajectory.append(current_turn)
            current_turn = {"turn_idx": int(turn_token)}
            current_field = None
        elif line.startswith("Action:"):
            current_turn["action"] = line[7:].strip()
            current_field = "action"
        elif line.startswith("Observation:"):
            current_turn["observation"] = line[12:].strip()
            current_field = "observation"
        elif current_field and line:
            current_turn[current_field] = current_turn.get(current_field, "") + "\n" + line

    if current_turn:
        trajectory.append(current_turn)

    if not trajectory:
        for idx, chunk in enumerate(_raw_memory_chunks(trajectory_text)):
            trajectory.append({
                "turn_idx": idx,
                "action": "read_memory_chunk",
                "observation": chunk,
            })

    return trajectory

def _raw_memory_chunks(text: str, max_chars: int = 4000) -> List[str]:
    """Chunk non-Step/Turn benchmark text into observation-sized blocks."""
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for block in str(text or "").split("\n\n"):
        block = block.strip()
        if not block:
            continue
        block_len = len(block)
        if current and current_len + block_len + 2 > max_chars:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0
        if block_len > max_chars:
            for start in range(0, block_len, max_chars):
                part = block[start:start + max_chars].strip()
                if part:
                    chunks.append(part)
            continue
        current.append(block)
        current_len += block_len + 2
    if current:
        chunks.append("\n\n".join(current))
    if not chunks and str(text or "").strip():
        chunks.append(str(text).strip()[:max_chars])
    return chunks

def _build_turn_reports(
    trajectory: List[Dict[str, Any]],
    task: str,
    call_llm_func: Optional[Callable],
) -> List[Dict[str, Any]]:
    reports: List[Dict[str, Any]] = []
    if not call_llm_func:
        return reports

    import os
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    total = len(trajectory)

    workers = max(1, int(os.environ.get("AMA_BUILD_CONCURRENCY", "16")))
    log_every = max(1, total // 50) if total > 100 else 1
    start = time.perf_counter()
    print(
        f"[ama_agent build] _build_turn_reports start: {total} turns, "
        f"workers={workers} (parallel LLM calls)",
        flush=True,
    )

    def _process_one(idx: int) -> Dict[str, Any]:
        turn = trajectory[idx]
        previous_turn = trajectory[idx - 1] if idx > 0 else None
        prompt = MEMORY_CONSTRUCTION_PROMPT_TEMPLATE.format(
            turn_t=_format_turn(turn),
            turn_t_minus_1=_format_turn(previous_turn),
            task=task or "None",
        )
        _, llm_response = call_llm_func(prompt)
        parsed_local = parse_markdown_state_report(llm_response or "")
        raw_report_local = (llm_response or "").strip()
        if not raw_report_local:
            raw_report_local = _fallback_report(turn=turn, previous_turn=previous_turn)
            parsed_local = parse_markdown_state_report(raw_report_local)
        return {
            "turn_idx": turn.get("turn_idx", idx),
            "action": turn.get("action", ""),
            "observation": turn.get("observation", ""),
            "previous_turn_idx": previous_turn.get("turn_idx") if previous_turn else None,
            "raw_report": raw_report_local,
            "parsed": parsed_local,
        }

    ordered: List[Optional[Dict[str, Any]]] = [None] * total
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_process_one, i): i for i in range(total)}
        for fut in as_completed(futures):
            i = futures[fut]
            ordered[i] = fut.result()
            completed += 1
            if completed == 1 or completed % log_every == 0 or completed == total:
                elapsed = time.perf_counter() - start
                rate = completed / max(elapsed, 1e-9)
                eta = (total - completed) / max(rate, 1e-9)
                print(
                    f"[ama_agent build] turn {completed}/{total} "
                    f"({100*completed/total:.1f}%) elapsed={elapsed:.0f}s "
                    f"rate={rate:.2f}/s eta={eta:.0f}s",
                    flush=True,
                )
    reports.extend(r for r in ordered if r is not None)

    elapsed = time.perf_counter() - start
    print(f"[ama_agent build] _build_turn_reports DONE in {elapsed:.0f}s ({total} reports)", flush=True)
    return reports

def _build_graph_memory(
    reports: List[Dict[str, Any]],
    embed_engine: Optional[Callable],
) -> Dict[str, Any]:
    import time
    total = len(reports)
    print(f"[ama_agent build] _build_graph_memory start: {total} reports", flush=True)
    g_start = time.perf_counter()
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    adjacency: Dict[str, List[str]] = {}
    prev_env_node_id: Optional[str] = None
    prev_object_node_ids: Dict[str, str] = {}

    for report in reports:
        turn_token = str(report.get("turn_idx", 0)).strip()
        if not turn_token.isdigit():
            continue
        turn_idx = int(turn_token)
        parsed = report.get("parsed", {})
        objectives = {
            item.get("name", "").strip(): item.get("description", "").strip()
            for item in parsed.get("objectives", [])
            if item.get("name")
        }
        changes = parsed.get("state_changes", {})
        evidence = changes.get("evidence", [])
        env_state = parsed.get("states", {}).get("env_state", [])
        object_states = parsed.get("states", {}).get("object_states", [])

        env_node = {
            "node_id": f"env@{turn_idx}",
            "turn_idx": turn_idx,
            "kind": "env_state",
            "label": "environment",
            "objective_description": "",
            "content": _render_env_node_content(
                turn_idx=turn_idx,
                action=report.get("action", ""),
                env_changed=bool(changes.get("env_changed")),
                evidence=evidence,
                env_state=env_state,
            ),
        }
        nodes.append(env_node)
        adjacency.setdefault(env_node["node_id"], [])

        if prev_env_node_id:
            edge_type = "causal" if changes.get("env_changed") else "temporal"
            _add_edge(
                edges,
                adjacency,
                source=prev_env_node_id,
                target=env_node["node_id"],
                edge_type=edge_type,
                turn_idx=turn_idx,
                detail=report.get("action", ""),
            )
        prev_env_node_id = env_node["node_id"]

        for obj in object_states:
            name = obj.get("name", "").strip() or "unknown_object"
            node_id = f"obj:{name}@{turn_idx}"
            node = {
                "node_id": node_id,
                "turn_idx": turn_idx,
                "kind": "object_state",
                "label": name,
                "objective_description": objectives.get(name, ""),
                "content": _render_object_node_content(
                    turn_idx=turn_idx,
                    name=name,
                    objective_description=objectives.get(name, ""),
                    objective_changed=bool(changes.get("objective_changed")),
                    evidence=evidence,
                    state=obj.get("state", []),
                ),
            }
            nodes.append(node)
            adjacency.setdefault(node_id, [])

            _add_edge(
                edges,
                adjacency,
                source=env_node["node_id"],
                target=node_id,
                edge_type="association",
                turn_idx=turn_idx,
                detail=name,
            )

            prev_object_node_id = prev_object_node_ids.get(name)
            if prev_object_node_id:
                edge_type = "causal" if changes.get("objective_changed") else "temporal"
                _add_edge(
                    edges,
                    adjacency,
                    source=prev_object_node_id,
                    target=node_id,
                    edge_type=edge_type,
                    turn_idx=turn_idx,
                    detail=report.get("action", ""),
                )
            prev_object_node_ids[name] = node_id

    print(f"[ama_agent build] graph built: {len(nodes)} nodes, {len(edges)} edges in {time.perf_counter()-g_start:.0f}s", flush=True)
    node_texts = [build_graph_node_text(node) for node in nodes]
    if embed_engine and node_texts:
        e_start = time.perf_counter()
        print(f"[ama_agent build] embedding {len(node_texts)} node texts", flush=True)
        embeddings = _embed_items(embed_engine, node_texts)
        print(f"[ama_agent build] embeddings DONE in {time.perf_counter()-e_start:.0f}s", flush=True)
    else:
        embeddings = []

    return {
        "nodes": nodes,
        "node_ids": [node["node_id"] for node in nodes],
        "node_turns": [node["turn_idx"] for node in nodes],
        "node_texts": node_texts,
        "embeddings": embeddings,
        "edges": edges,
        "adjacency": adjacency,
    }

def _render_env_node_content(
    turn_idx: int,
    action: str,
    env_changed: bool,
    evidence: List[str],
    env_state: List[Dict[str, str]],
) -> str:
    lines = [
        f"Turn {turn_idx}",
        f"Action: {action}",
        f"Environment changed: {str(env_changed).lower()}",
    ]
    if evidence:
        lines.append("Evidence:")
        lines.extend(f"- {item}" for item in evidence)
    if env_state:
        lines.append("Environment state:")
        lines.extend(f"- {item.get('key', '')}: {item.get('value', '')}" for item in env_state)
    return "\n".join(lines)

def _render_object_node_content(
    turn_idx: int,
    name: str,
    objective_description: str,
    objective_changed: bool,
    evidence: List[str],
    state: List[Dict[str, str]],
) -> str:
    lines = [
        f"Turn {turn_idx}",
        f"Object: {name}",
        f"Objective changed: {str(objective_changed).lower()}",
    ]
    if objective_description:
        lines.append(f"Objective description: {objective_description}")
    if evidence:
        lines.append("Evidence:")
        lines.extend(f"- {item}" for item in evidence)
    if state:
        lines.append("Object state:")
        lines.extend(f"- {item.get('key', '')}: {item.get('value', '')}" for item in state)
    return "\n".join(lines)

def _add_edge(
    edges: List[Dict[str, Any]],
    adjacency: Dict[str, List[str]],
    source: str,
    target: str,
    edge_type: str,
    turn_idx: int,
    detail: str,
) -> None:
    edge = {
        "source": source,
        "target": target,
        "type": edge_type,
        "turn_idx": turn_idx,
        "detail": detail,
    }
    edges.append(edge)
    adjacency.setdefault(source, []).append(target)
    adjacency.setdefault(target, []).append(source)

def _embed_items(embed_engine: Callable, items: List[str], batch: int = 16) -> List[Any]:
    """Call the embedding engine in small batches.

    Long AMABench trajectories produce hundreds of nodes; sending all of
    them in one request overloads sglang/vllm and times out. Batching also
    gives us partial progress on network flakiness.
    """
    if not items:
        return []

    out: List[Any] = []
    for i in range(0, len(items), batch):
        chunk = items[i : i + batch]
        try:
            embeddings = embed_engine(chunk)
        except TypeError:
            embeddings = [embed_engine(it) for it in chunk]
        if (
            len(chunk) == 1
            and isinstance(embeddings, list)
            and embeddings
            and not isinstance(embeddings[0], (list, tuple))
        ):
            embeddings = [embeddings]
        out.extend(embeddings)
    return out

def _format_turn(turn: Optional[Dict[str, Any]]) -> str:
    if not turn:
        return "None"
    return (
        "{\n"
        f'  "turn_idx": {turn.get("turn_idx", 0)},\n'
        f'  "action": {repr(turn.get("action", ""))},\n'
        f'  "observation": {repr(turn.get("observation", ""))}\n'
        "}"
    )

def _fallback_report(
    turn: Dict[str, Any],
    previous_turn: Optional[Dict[str, Any]],
) -> str:
    prev_obs = (previous_turn or {}).get("observation", "")
    curr_obs = turn.get("observation", "")
    env_changed = str(bool(curr_obs and curr_obs != prev_obs)).lower()
    evidence = turn.get("action", "") or curr_obs[:160]
    return (
        "# OBJECTIVES\n"
        "1. inferred_task: no structured objective extracted\n\n"
        "# STATE_CHANGES\n"
        f"env_changed: {env_changed}\n"
        "objective_changed: false\n"
        "evidence:\n"
        f'  - "{evidence}"\n\n'
        "# STATES\n"
        "env_state:\n"
        f'  - observation: {curr_obs.replace(chr(10), " ")}\n'
        "object_states:\n"
        "  - name: inferred_task\n"
        "    state:\n"
        f'      - action: {turn.get("action", "").replace(chr(10), " ")}\n'
    )
