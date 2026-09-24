"""
Trajectory query tools for AMA-Agent retrieval.

Faithful port of datasets/amabench/code/src/method/ama_agent_core/tool.py
"""

import json
import re
from typing import Any, Dict, List, Optional

def traj_find(
    trajectory_text: str,
    query: str,
    mode: str = "keyword",
) -> List[int]:
    """Find turn indices that match the query.

    Exact port of official tool.traj_find.

    Args:
        trajectory_text: JSON string of trajectory data.
        query: Search term.
        mode: 'keyword', 'regex', 'action', or 'entity'.

    Returns:
        List of matching turn indices.
    """
    trajectory_data = json.loads(trajectory_text)
    trajectory = trajectory_data.get('trajectory', [])

    matched_indices: List[int] = []

    for turn in trajectory:
        turn_idx = turn.get('turn_idx', 0)
        matched = False

        if mode == "keyword":
            query_lower = query.lower()
            action = turn.get('action', '').lower()
            observation = turn.get('observation', '').lower()
            if query_lower in action or query_lower in observation:
                matched = True

        elif mode == "regex":
            pattern = re.compile(query, re.IGNORECASE)
            action = turn.get('action', '')
            observation = turn.get('observation', '')
            if pattern.search(action) or pattern.search(observation):
                matched = True

        elif mode == "action":
            action = turn.get('action', '').lower()
            if query.lower() in action:
                matched = True

        elif mode == "entity":
            action = turn.get('action', '')
            observation = turn.get('observation', '')
            combined_text = action + " " + observation
            if query in combined_text:
                matched = True

        if matched:
            matched_indices.append(turn_idx)

    return matched_indices

def traj_get(
    trajectory_text: str,
    span: Optional[Dict[str, Any]] = None,
    fields: Optional[List[str]] = None,
    auto_compress: bool = False,
) -> str:
    """Get evidence segments from trajectory.

    Exact port of official tool.traj_get.

    Args:
        trajectory_text: JSON string of trajectory data.
        span: ``{'indices': [1,2,3]}`` or ``{'start': 1, 'end': 5}``.
        fields: Fields to include (default: action, observation, action_space).
        auto_compress: Truncate long values.

    Returns:
        Formatted trajectory segment string.
    """
    if fields is None:
        fields = ['action', 'observation', 'action_space']

    trajectory_data = json.loads(trajectory_text)
    trajectory = trajectory_data.get('trajectory', [])

    if span is None:
        selected_turns = trajectory
    elif 'indices' in span:
        indices = span['indices']
        if not isinstance(indices, list):
            indices = [indices]
        selected_turns = [t for t in trajectory if t.get('turn_idx') in indices]
    elif 'start' in span and 'end' in span:
        selected_turns = [
            t for t in trajectory
            if span['start'] <= t.get('turn_idx', 0) <= span['end']
        ]
    else:
        selected_turns = trajectory

    result_lines: List[str] = []
    for turn in selected_turns:
        turn_idx = turn.get('turn_idx', 0)
        result_lines.append(f"Turn {turn_idx}:")
        for field in fields:
            if field in turn:
                value = turn[field]
                if auto_compress and isinstance(value, str) and len(value) > 300:
                    value = value[:300] + "..."
                result_lines.append(f"  {field}: {value}")

    return "\n".join(result_lines)

def execute_tool_call(
    tool_name: str,
    arguments: Dict[str, Any],
    trajectory_text: str,
) -> str:
    """Execute a tool call with given arguments.

    Exact port of official tool.execute_tool_call.
    """
    if tool_name == "traj_find":
        query = arguments.get("query", "")
        mode = arguments.get("mode", "keyword")
        indices = traj_find(trajectory_text, query, mode)
        return json.dumps({"indices": indices, "count": len(indices)})

    elif tool_name == "traj_get":
        span = arguments.get("span")
        fields = arguments.get("fields")
        return traj_get(trajectory_text, span, fields)

    else:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
