import re
from utils import call_gpt, call_qwen, call_dpsk, call_llm_openrouter_api
from memory_structuring.prompt_structuring import (
    GetSubgoalPrompt,
    GetRewardPrompt,
    GetStatePrompt,
    GetTopicPrompt,
    GetSpeakerStancePrompt,
    GetSessionSummaryPrompt,
    GetSemanticPrompt,
    GetProceduralPrompt,
    GetReturnPrompt,
)

def get_subgoal(goal, state_t0, observation_t0, action_t0):
    prompt_obj = GetSubgoalPrompt()
    variables = {"goal": goal, "state": state_t0, "observation": observation_t0, "action": action_t0}
    messages = prompt_obj.render(variables)
    response = call_qwen(messages=[{"role": m.role, "content": m.content} for m in messages])
    pattern = r"### Subgoal\n(.*)"
    match = re.search(pattern, response, re.S)
    subgoal = match.group(1).strip() if match else "<a subgoal>"
    #print(f"Subgoal: {subgoal}")
    return subgoal

def get_reward(goal, state_t0, action_t0, observation_t1):
    prompt_obj = GetRewardPrompt()
    variables = {"goal": goal, "state": state_t0, "action": action_t0, "observation": observation_t1}
    messages = prompt_obj.render(variables)
    response = call_qwen(messages=[{"role": m.role, "content": m.content} for m in messages])
    pattern = r"### Reward\n(.*)"
    match = re.search(pattern, response, re.S)
    reward = match.group(1).strip() if match else "<a reward>"
    #print(f"Reward: {reward}")
    return reward

def get_state(goal, state_t0, action_t0, observation_t1):
    prompt_obj = GetStatePrompt()
    variables = {"goal": goal, "state": state_t0, "action": action_t0, "observation": observation_t1}
    messages = prompt_obj.render(variables)
    response = call_qwen(messages=[{"role": m.role, "content": m.content} for m in messages])
    pattern = r"### State\n(.*)"
    match = re.search(pattern, response, re.S)
    state = match.group(1).strip() if match else "<a state>"
    #print(f"State: {state}")
    return state

def _call_structuring_prompt(prompt_obj, variables):
    messages = prompt_obj.render(variables)
    return call_qwen(messages=[{"role": m.role, "content": m.content} for m in messages])


def _parse_statement_tags(block):
    pattern = r'\*\*Statement:\*\*\s*(.*?)\s*\n\s*\*\*Tags:\*\*\s*(.*?)\s*(?:\n|$)'
    matches = re.findall(pattern, block or "", re.S)
    parsed = []
    for statement, tags in matches:
        clean_statement = statement.strip()
        if not clean_statement:
            continue
        clean_tags = []
        for tag in tags.split(','):
            clean_tag = tag.strip().strip("[]\"'`,:;")
            if clean_tag and clean_tag not in clean_tags:
                clean_tags.append(clean_tag)
        parsed.append((clean_statement, clean_tags))
    return parsed


def _render_dialogue_turns(turns):
    if isinstance(turns, str):
        return turns
    lines = []
    for i, turn in enumerate(turns or []):
        if isinstance(turn, dict):
            speaker = str(turn.get("speaker") or "Unknown").strip() or "Unknown"
            text = str(turn.get("text") or turn.get("content") or turn.get("observation") or "").strip()
            timestamp = str(turn.get("timestamp") or "").strip()
            turn_num = turn.get("turn_num", turn.get("index", i))
        else:
            speaker = "Unknown"
            text = str(turn or "").strip()
            timestamp = ""
            turn_num = i
        if not text:
            continue
        time_part = f" [time={timestamp}]" if timestamp else ""
        lines.append(f"Turn {turn_num}{time_part}: {speaker}: {text}")
    return "\n".join(lines)


def get_topic(dialogue_turns, previous_summary="", mode="conversation"):
    if mode != "conversation":
        return get_subgoal(
            goal="Conversation",
            state_t0=previous_summary,
            observation_t0=_render_dialogue_turns(dialogue_turns),
            action_t0="continue conversation",
        )
    prompt_obj = GetTopicPrompt()
    variables = {
        "dialogue": _render_dialogue_turns(dialogue_turns),
        "previous_summary": previous_summary or "",
    }
    response = _call_structuring_prompt(prompt_obj, variables)
    pattern = r"### Topic\n(.*)"
    match = re.search(pattern, response, re.S)
    return match.group(1).strip() if match else "conversation session"


def get_speaker_stance(dialogue_turns, trajectory_num=0, turn_num=0, time=0, mode="conversation"):
    if mode != "conversation":
        reward = get_reward(
            goal="Conversation",
            state_t0="",
            action_t0=_render_dialogue_turns(dialogue_turns),
            observation_t1="",
        )
        return [{"speaker_stance": reward, "tags": [], "trajectory_num": trajectory_num, "turn_num": turn_num, "time": time}]

    prompt_obj = GetSpeakerStancePrompt()
    variables = {"dialogue": _render_dialogue_turns(dialogue_turns)}
    response = _call_structuring_prompt(prompt_obj, variables)
    pattern = r"### Stances\n(.*)"
    match = re.search(pattern, response, re.S)
    stances = match.group(1).strip() if match else ""
    stance_memory = []
    for statement, tags in _parse_statement_tags(stances):
        stance_memory.append({
            "speaker_stance": statement,
            "tags": tags,
            "trajectory_num": trajectory_num,
            "turn_num": turn_num,
            "time": time,
            "st_ed": "mid",
        })
    return stance_memory


def get_session_summary(previous_summaries, session_turns, mode="conversation"):
    if mode != "conversation":
        return get_state(
            goal="Conversation",
            state_t0=str(previous_summaries or ""),
            action_t0="continue conversation",
            observation_t1=_render_dialogue_turns(session_turns),
        )
    prompt_obj = GetSessionSummaryPrompt()
    variables = {
        "previous_summaries": previous_summaries or "No previous summaries.",
        "dialogue": _render_dialogue_turns(session_turns),
    }
    response = _call_structuring_prompt(prompt_obj, variables)
    pattern = r"### Summary\n(.*)"
    match = re.search(pattern, response, re.S)
    return match.group(1).strip() if match else _render_dialogue_turns(session_turns)


def get_semantic(step, trajectory_num=0, turn_num=0, time=0, mode="trajectory", session_context=""):
    prompt_obj = GetSemanticPrompt()
    variables = {
        "observation": step["observation"],
        "mode": mode,
        "conversation_mode": mode == "conversation",
        "session_context": session_context or "",
    }
    messages = prompt_obj.render(variables)
    # response = call_gpt(messages=[{"role": m.role, "content": m.content} for m in messages])
    # response = call_dpsk(messages=[{"role": m.role, "content": m.content} for m in messages])
    response = call_qwen(messages=[{"role": m.role, "content": m.content} for m in messages])
    # response = call_llm_openrouter_api(model_name="openai/gpt-4o-2024-11-20",messages=[{"role": m.role, "content": m.content} for m in messages])
    pattern = r"### Facts\n(.*)"
    match = re.search(pattern, response, re.S)
    facts = match.group(1).strip() if match else None
    semantic_memory = []
    if not facts == None:
        for idx, (statement, tags) in enumerate(_parse_statement_tags(facts)):
            semantic_memory.append({
                "semantic_memory": statement,
                "tags": tags,
                "trajectory_num" : trajectory_num,
                "turn_num" : turn_num,
                "time": time,
                "st_ed": "mid"
            })
        # if not len(semantic_memory) == 0:
        #     semantic_memory[0]["st_ed"] = "st"
        #     semantic_memory[len(semantic_memory)-1]["st_ed"] = "ed"
    for i in range(len(semantic_memory)):
        print(semantic_memory[i])
    return semantic_memory


def get_conversation_semantic(turns, trajectory_num=0, time=0, session_context="", window_size=12):
    semantic_memory = []
    window_size = max(1, int(window_size or 12))
    normalized_turns = list(turns or [])
    for start in range(0, len(normalized_turns), window_size):
        window = normalized_turns[start:start + window_size]
        if not window:
            continue
        first_turn = window[0]
        if isinstance(first_turn, dict):
            first_turn_num = int(first_turn.get("turn_num", first_turn.get("index", start)) or start)
        else:
            first_turn_num = start
        semantic_memory.extend(get_semantic(
            {"observation": _render_dialogue_turns(window)},
            trajectory_num=trajectory_num,
            turn_num=first_turn_num,
            time=time,
            mode="conversation",
            session_context=session_context,
        ))
    return semantic_memory

def get_return(subgoal: str, procedural_memory: str):
    prompt_obj = GetReturnPrompt()
    variables = {"subgoal": subgoal, "procedural_memory": procedural_memory}
    messages = prompt_obj.render(variables)
    response = call_qwen(messages=[{"role": m.role, "content": m.content} for m in messages])
    pattern = r"### Score\n(.*)"
    match = re.search(pattern, response, re.S)
    _return = match.group(1).strip() if match else 0.0
    return _return

def get_procedural(trajectory: str):
    prompt_obj = GetProceduralPrompt()
    variables = {"trajectory": trajectory}
    messages = prompt_obj.render(variables)
    response = call_qwen(messages=[{"role": m.role, "content": m.content} for m in messages])
    # response = call_llm_openrouter_api(model_name="openai/gpt-4o-2024-11-20",messages=[{"role": m.role, "content": m.content} for m in messages])
    pattern = r"### Goal\n(.*)\n### Experiential Insight"
    goal_match = re.search(pattern, response, re.S)
    goal = goal_match.group(1).strip() if goal_match else "<a goal>"
    pattern = r"### Experiential Insight\n(.*)"
    experience_match = re.search(pattern, response, re.S)
    experience = experience_match.group(1).strip() if experience_match else None
    # _return = get_return(subgoal = goal, procedural_memory = experience)
    _return = 0.0
    res_dict={"procedural_memory": experience, "sub_goal": goal, "return": _return}
    print(res_dict)
    return experience, goal, _return
