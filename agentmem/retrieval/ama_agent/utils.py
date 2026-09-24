"""Utility functions for the paper-faithful AMA-Agent path."""

import json
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

def parse_markdown_state_report(llm_response: str) -> Dict[str, Any]:
    """Parse the appendix-style per-turn Markdown state report.

    The paper appendix specifies an exact Markdown schema with three sections:
    ``# OBJECTIVES``, ``# STATE_CHANGES``, and ``# STATES``. The parser is
    intentionally tolerant to minor whitespace noise while preserving the same
    logical structure.
    """
    text = (llm_response or "").replace("\r\n", "\n").strip()
    if not text:
        return {
            "objectives": [],
            "state_changes": {"env_changed": False, "objective_changed": False, "evidence": []},
            "states": {"env_state": [], "object_states": []},
            "raw": "",
        }

    objectives_block = _section_body(text, "# OBJECTIVES", "# STATE_CHANGES")
    changes_block = _section_body(text, "# STATE_CHANGES", "# STATES")
    states_block = _section_body(text, "# STATES", None)

    objectives = _parse_objectives(objectives_block)
    env_changed = _parse_bool_field(changes_block, "env_changed")
    objective_changed = _parse_bool_field(changes_block, "objective_changed")
    evidence = _parse_evidence(changes_block)
    env_state, object_states = _parse_states(states_block)

    return {
        "objectives": objectives,
        "state_changes": {
            "env_changed": env_changed,
            "objective_changed": objective_changed,
            "evidence": evidence,
        },
        "states": {
            "env_state": env_state,
            "object_states": object_states,
        },
        "raw": text,
    }

def render_state_memory_summary(reports: List[Dict[str, Any]]) -> str:
    """Render a compact, question-facing state memory summary from per-turn reports."""
    if not reports:
        return ""

    objective_map: Dict[str, str] = {}
    lines: List[str] = ["# OBJECTIVE INVENTORY"]
    for report in reports:
        for item in report.get("parsed", {}).get("objectives", []):
            name = item.get("name", "").strip()
            desc = item.get("description", "").strip()
            if name and name not in objective_map:
                objective_map[name] = desc
    if objective_map:
        for idx, (name, desc) in enumerate(objective_map.items(), start=1):
            suffix = f": {desc}" if desc else ""
            lines.append(f"{idx}. {name}{suffix}")
    else:
        lines.append("1. none: no explicit objective extracted")

    lines.append("")
    lines.append("# TURN REPORTS")
    for report in reports:
        turn_idx = report.get("turn_idx", -1)
        lines.append(f"## Turn {turn_idx}")
        lines.append(report.get("raw_report", "").strip())
        lines.append("")

    return "\n".join(lines).strip()

def build_graph_node_text(node: Dict[str, Any]) -> str:
    """Canonical text used for node embedding and QA context."""
    parts = [
        f"Node ID: {node.get('node_id', '')}",
        f"Turn: {node.get('turn_idx', '')}",
        f"Kind: {node.get('kind', '')}",
        f"Label: {node.get('label', '')}",
    ]
    objective = node.get("objective_description")
    if objective:
        parts.append(f"Objective Description: {objective}")
    content = (node.get("content") or "").strip()
    if content:
        parts.extend(["Content:", content])
    return "\n".join(parts)

def _section_body(text: str, start_marker: str, end_marker: Optional[str]) -> str:
    start = text.find(start_marker)
    if start == -1:
        return ""
    start += len(start_marker)
    if end_marker is None:
        end = len(text)
    else:
        end = text.find(end_marker, start)
        if end == -1:
            end = len(text)
    return text[start:end].strip()

def _parse_objectives(block: str) -> List[Dict[str, str]]:
    objectives: List[Dict[str, str]] = []
    for raw_line in block.splitlines():
        line = raw_line.strip()
        match = re.match(r"^\d+\.\s*(.*?)(?::\s*(.*))?$", line)
        if not match:
            continue
        name = (match.group(1) or "").strip()
        desc = (match.group(2) or "").strip()
        if name:
            objectives.append({"name": name, "description": desc})
    return objectives

def _parse_bool_field(block: str, field_name: str) -> bool:
    match = re.search(rf"^{re.escape(field_name)}:\s*(true|false)\s*$", block, flags=re.IGNORECASE | re.MULTILINE)
    if not match:
        return False
    return match.group(1).lower() == "true"

def _parse_evidence(block: str) -> List[str]:
    evidence: List[str] = []
    in_evidence = False
    for raw_line in block.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if stripped.lower() == "evidence:":
            in_evidence = True
            continue
        if not in_evidence:
            continue
        if not stripped:
            continue
        if not stripped.startswith("-"):
            break
        item = stripped[1:].strip().strip('"').strip("'")
        if item:
            evidence.append(item)
    return evidence

def _parse_states(block: str) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
    env_state: List[Dict[str, str]] = []
    object_states: List[Dict[str, Any]] = []
    mode: Optional[str] = None
    current_object: Optional[Dict[str, Any]] = None
    in_object_state = False

    for raw_line in block.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped == "env_state:":
            mode = "env"
            current_object = None
            in_object_state = False
            continue
        if stripped == "object_states:":
            mode = "objects"
            current_object = None
            in_object_state = False
            continue
        if mode == "env" and stripped.startswith("- "):
            key, value = _parse_key_value_item(stripped[2:])
            if key:
                env_state.append({"key": key, "value": value})
            continue
        if mode != "objects":
            continue
        if stripped.startswith("- name:"):
            if current_object:
                object_states.append(current_object)
            current_object = {"name": stripped.split(":", 1)[1].strip(), "state": []}
            in_object_state = False
            continue
        if stripped == "state:":
            in_object_state = True
            continue
        if current_object is not None and in_object_state and stripped.startswith("- "):
            key, value = _parse_key_value_item(stripped[2:])
            if key:
                current_object["state"].append({"key": key, "value": value})

    if current_object:
        object_states.append(current_object)

    return env_state, object_states

def _parse_key_value_item(text: str) -> Tuple[str, str]:
    if ":" not in text:
        return text.strip(), ""
    key, value = text.split(":", 1)
    return key.strip(), value.strip()

def extract_state_memory_from_response(llm_response: str) -> Optional[str]:
    """Extract state memory content from LLM response after **STATE_MEMORY** marker.

    Exact port of official utils.extract_state_memory_from_response.
    """
    if not llm_response:
        return None

    marker = "**STATE_MEMORY**"
    marker_pos = llm_response.find(marker)

    if marker_pos == -1:

        marker_pos = llm_response.upper().find(marker)
        if marker_pos == -1:
            return None

    state_mem = llm_response[marker_pos + len(marker):].strip()
    return state_mem if state_mem else None

def extract_causal_graph_from_response(
    llm_response: str,
) -> Optional[List[Dict[str, Any]]]:
    """Extract the causal graph JSON array from LLM response after **CAUSAL_GRAPH** marker.

    Exact port of official construct._extract_causal_graph_from_response.
    """
    if not llm_response:
        return None

    marker = "**CAUSAL_GRAPH**"
    pos = llm_response.find(marker)
    if pos == -1:
        pos = llm_response.upper().find(marker)
    if pos == -1:
        return None

    after_marker = llm_response[pos + len(marker):].strip()
    json_array = _extract_json_array(after_marker)
    if json_array is None:
        return None

    try:
        return json.loads(json_array)
    except json.JSONDecodeError:
        return None

def _extract_json_array(text: str) -> Optional[str]:
    """Extract the first complete top-level JSON array from text.

    AMA causal-graph objects contain nested ``entities: [...]`` lists, so a
    simple non-greedy regex stops at the first inner ``]`` and truncates the
    graph. Scan the string and track bracket depth instead.
    """
    start = text.find("[")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for idx in range(start, len(text)):
        ch = text[idx]

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]

    return None

def retrieve_with_qwen(
    query: str,
    text_mem: Dict[str, Any],
    call_llm_func: Callable,
    top_k: int = 5,
) -> Tuple[Dict[str, Any], List[int]]:
    """Retrieve relevant chunks using LLM relevance scoring (0-10).

    Exact port of official utils.retrieve_with_qwen.

    Process:
    1. Chunk trajectory into groups of 5 turns
    2. Score each chunk's relevance to the query (0-10) via LLM
    3. Return top-5 most relevant chunks' turn indices

    Args:
        query: Query string.
        text_mem: Text memory dict with 'trajectory_data' key.
        call_llm_func: ``(prompt: str) -> (_, response: str)`` callable.

    Returns:
        ``(keywords_info, relevant_turn_indices)``
    """
    trajectory = text_mem['trajectory_data']['trajectory']

    chunk_size = 5
    chunks: List[List[Dict[str, Any]]] = []
    for i in range(0, len(trajectory), chunk_size):
        chunk_turns = trajectory[i:i + chunk_size]
        chunks.append(chunk_turns)

    chunk_infos: List[Tuple[int, List[int], str]] = []
    for chunk_idx, chunk_turns in enumerate(chunks):
        chunk_text_parts = []
        turn_indices = []
        for turn in chunk_turns:
            turn_idx = turn.get('turn_idx', 0)
            action = turn.get('action', '')
            observation = turn.get('observation', '')[:300]
            chunk_text_parts.append(
                f"Turn {turn_idx}: Action={action}, Observation={observation}"
            )
            turn_indices.append(turn_idx)
        chunk_infos.append((chunk_idx, turn_indices, "\n".join(chunk_text_parts)))

    chunk_scores: List[Tuple[int, List[int], float]] = []
    for cidx, tidx, ctext in chunk_infos:
        prompt = (
            f"Rate the relevance of the following trajectory chunk to the query "
            f"on a scale of 0-10.\n\n"
            f"Query: {query}\n\n"
            f"Trajectory Chunk:\n{ctext}\n\n"
            f"Consider:\n"
            f"- Does this chunk contain information directly answering the query?\n"
            f"- Are there relevant actions, observations, or state changes?\n"
            f"- How closely do the events relate to what the query is asking?\n\n"
            f"Respond with ONLY a single number from 0 to 10, where:\n"
            f"- 0 = completely irrelevant\n"
            f"- 5 = somewhat relevant\n"
            f"- 10 = highly relevant\n\n"
            f"Score:"
        )
        try:
            _, resp = call_llm_func(prompt)
            m = re.search(r'(\d+(?:\.\d+)?)', resp.strip())
            score = max(0.0, min(10.0, float(m.group(1)))) if m else 0.0
        except Exception:
            score = 0.0
        chunk_scores.append((cidx, tidx, score))

    chunk_scores.sort(key=lambda x: (-x[2], x[0]))
    top_k = min(top_k, len(chunk_scores))

    relevant_turn_indices: List[int] = []
    for i in range(top_k):
        _, turn_indices, score = chunk_scores[i]
        if score > 0:
            relevant_turn_indices.extend(turn_indices)

    relevant_turn_indices = sorted(list(set(relevant_turn_indices)))

    keywords = re.findall(r'\w+', query.lower())
    keywords_info = {
        "keywords": keywords[:5],
        "search_mode": "qwen_relevance",
        "method": "qwen3-4b",
        "num_chunks": len(chunks),
        "top_chunks": top_k,
    }

    return keywords_info, relevant_turn_indices

def cosine_similarity(vec1: Any, vec2: Any) -> float:
    """Cosine similarity between two vectors.

    Port of official utils.cosine_similarity.
    """
    import math

    if isinstance(vec1, list):
        vec1 = [float(x) for x in vec1]
        vec2 = [float(x) for x in vec2]
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)
    else:
        import numpy as np
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        n1 = np.linalg.norm(vec1)
        n2 = np.linalg.norm(vec2)
        if n1 == 0 or n2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (n1 * n2))

def retrieve_with_embed(
    query: str,
    text_mem: Dict[str, Any],
    embed_mem: Dict[str, Any],
    embed_engine: Callable,
) -> Tuple[Dict[str, Any], List[int]]:
    """Retrieve using embedding-based similarity (paper Section 5.2 step 1).

    Port of official utils.retrieve_with_embed (sync version).

    Process:
    1. Embed query using embed_engine
    2. Compute cosine similarity against all turn embeddings
    3. Return top-5 most similar turns

    Args:
        query: Query string.
        text_mem: Text memory with trajectory data.
        embed_mem: Embedding memory from construct_state_memory.
        embed_engine: Callable ``(text) -> list[float]`` returning embeddings.

    Returns:
        ``(keywords_info, relevant_turn_indices)``
    """
    query_embedding = embed_engine(query)

    turn_embeddings = embed_mem['embeddings']
    trajectory = text_mem['trajectory_data']['trajectory']

    similarities: List[Tuple[int, float]] = []
    for i, turn_emb in enumerate(turn_embeddings):
        sim = cosine_similarity(query_embedding, turn_emb)
        turn_idx = trajectory[i].get('turn_idx', i) if i < len(trajectory) else i
        similarities.append((turn_idx, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    top_k = min(5, len(similarities))
    relevant_turn_indices = [similarities[i][0] for i in range(top_k)]

    keywords = re.findall(r'\w+', query.lower())
    keywords_info = {
        "keywords": keywords[:5],
        "search_mode": "embed",
        "method": "embed",
    }

    return keywords_info, relevant_turn_indices

def retrieve_with_llm_tools(
    query: str,
    state_mem: Any,
    text_mem: Dict[str, Any],
    call_llm_func: Callable,
    max_results: int = 5,
) -> Tuple[Dict[str, Any], List[int]]:
    """Retrieve using LLM keyword extraction + traj_find tool (paper Section 5.2 step 3).

    Sync port of official utils.retrieve_with_llm.

    Process:
    1. Ask LLM to extract keywords from query
    2. Use traj_find to search trajectory for each keyword
    3. Return matched turn indices

    Args:
        query: Query string.
        state_mem: State memory (for context in keyword extraction).
        text_mem: Text memory with trajectory data.
        call_llm_func: ``(prompt) -> (_, response)`` callable.

    Returns:
        ``(keywords_info, relevant_turn_indices)``
    """
    from .tools import traj_find

    state_mem_str = json.dumps(state_mem, indent=2) if isinstance(state_mem, dict) else str(state_mem or "")

    keyword_prompt = (
        f"Given the query, extract relevant keywords and search criteria.\n\n"
        f"Query: {query}\n\n"
        f"State Memory Summary:\n{state_mem_str}\n\n"
        f"Extract:\n"
        f"1. Key entities, objects, or actions mentioned\n"
        f"2. Time-related information (turn numbers, ranges)\n"
        f"3. Specific events or patterns to look for\n\n"
        f"Format as JSON:\n"
        f"{{\n"
        f'  "keywords": ["keyword1", "keyword2"],\n'
        f'  "turn_range": {{"start": 1, "end": 5}} or null,\n'
        f'  "search_mode": "keyword" or "action" or "entity"\n'
        f"}}\n\n"
        f"Only output the JSON:"
    )

    _, keyword_response = call_llm_func(keyword_prompt)

    keywords_info: Dict[str, Any] = {}
    if keyword_response:
        try:
            keyword_clean = keyword_response.strip()
            if keyword_clean.startswith("```json"):
                keyword_clean = keyword_clean[7:]
            if keyword_clean.startswith("```"):
                keyword_clean = keyword_clean[3:]
            if keyword_clean.endswith("```"):
                keyword_clean = keyword_clean[:-3]
            keyword_clean = keyword_clean.strip()
            keywords_info = json.loads(keyword_clean)
            keywords_info['method'] = 'llm_tools'
        except Exception:
            keywords_info = {"keywords": [query], "search_mode": "keyword", "method": "llm_tools"}

    trajectory_text_json = json.dumps(text_mem['trajectory_data'])
    keywords = keywords_info.get('keywords', [query])
    relevant_turn_indices: List[int] = []

    for keyword in keywords:
        indices = traj_find(
            trajectory_text_json,
            keyword,
            mode=keywords_info.get('search_mode', 'keyword'),
        )
        relevant_turn_indices.extend(indices)

    relevant_turn_indices = sorted(list(set(relevant_turn_indices)))[:max(1, max_results)]

    return keywords_info, relevant_turn_indices

def extract_final_answer(response: str) -> str:
    """Extract the final answer from model response.

    Exact port of official utils/extract_final_answer.py.
    Looks for ``##Answer:`` marker and takes the first line after it.
    """
    if "##Answer:" in response:
        parts = response.split("##Answer:")
        if len(parts) > 1:
            answer = parts[1].strip()
            answer = answer.split('\n')[0].strip()
            return answer

    return response.strip()
