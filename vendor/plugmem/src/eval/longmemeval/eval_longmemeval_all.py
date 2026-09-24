import json
import re
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
current_dir = os.path.dirname(os.path.abspath(__file__))
# 计算上上级目录路径
parent_dir = os.path.abspath(os.path.join(current_dir, "../.."))
# 添加到 sys.path
sys.path.append(parent_dir)
from memory_structuring.memory import Memory
from memory_retrieving.memory_graph import MemoryGraph
from memory_retrieving.value_longmemeval import TagEqual, TagRelevant, SemanticEqual, SemanticRelevant, SubgoalEqual, SubgoalRelevant, ProceduralEqual, ProceduralRelevant
from utils import call_qwen, call_gpt


def load_run_prompt() -> str:
    with open("longmemeval_run_prompt.txt", "r") as f:
        return f.read()
run_prompt_template = load_run_prompt()
reasoning_model = os.environ.get("OPENAI_MODEL_NAME", "gpt-4o-mini")
output_dir = os.environ.get("DIR_PATH") or os.path.abspath(os.path.join(current_dir, "../../../data_longmemeval"))
os.environ["DIR_PATH"] = output_dir
for subdir in ("", "episodic_memory", "semantic_memory", "procedural_memory", "subgoal", "tag", "logs"):
    os.makedirs(os.path.join(output_dir, subdir) if subdir else output_dir, exist_ok=True)

def _append_jsonl(path, payload):
    with open(path, "a", encoding="utf-8") as output:
        output.write(json.dumps(payload, ensure_ascii=False) + "\n")

def _record_progress(event, **payload):
    record = {"event": event, **payload}
    _append_jsonl(os.path.join(output_dir, "progress.jsonl"), record)
    print(json.dumps(record, ensure_ascii=False), flush=True)

print("Loading...")
with open("../../LongMemEval/data/longmemeval_s_cleaned.json", "r") as f:
    data = json.load(f)
print("Loading done")
print(f"Using reasoning model: {reasoning_model}")


def _build_memory_from_session(session, time):
    goal = "Answer user's question"
    if session[0]['role'] == 'user':
        memory = Memory(goal=goal, observation=session[0]["content"], time = f"Date: {time}")
        st = 1
    else:
        memory = Memory(goal=goal, observation="User: ...")
        st = 0
    action = None
    for turn in session[st:]:
        if turn["role"] == "assistant":
            action = f"Agent Say: {turn['content']}"
        else:
            if action is None:
                raise ValueError("Encountered user turn before any assistant action in session.")
            memory.append(
                action_t0=action,
                observation_t1=f"User Say: {turn['content']}"
            )
            action = None
    memory.close()
    return memory


worker_count = int(os.getenv("LONGMEMEVAL_SESSION_WORKERS", max(os.cpu_count() or 1, 1)))
cnt = 0

start_idx = int(os.environ.get("LONGMEMEVAL_START_INDEX", "0"))
max_samples = int(os.environ.get("LONGMEMEVAL_MAX_SAMPLES", "500"))
end_idx = min(len(data), start_idx + max_samples)
for n in range(start_idx, end_idx):
    print(n)
    test = data[n]
    question_id = test["question_id"]
    mg = MemoryGraph(
        tag_equal=TagEqual(),
        tag_relevant=TagRelevant(),
        semantic_equal=SemanticEqual(),
        semantic_relevant=SemanticRelevant(), 
        subgoal_equal=SubgoalEqual(),
        subgoal_relevant=SubgoalRelevant(),
        procedural_equal=ProceduralEqual(),
        procedural_relevant=ProceduralRelevant()
    )#1
    question = test["question"]
    sessions = test["haystack_sessions"]
    times = test['haystack_dates']
    task_type = "assistant for user"
    print(f"Loading test {question_id} with {len(sessions)} sessions using {worker_count} workers")
    _record_progress("sample_start", index=n, question_id=question_id, session_count=len(sessions), question_date=test["question_date"])
    retrieve_dir = os.environ.get("LONGMEMEVAL_RETRIEVE_DIR", "").strip()
    cache_path = os.path.join(retrieve_dir, f"retrieve_{question_id}.json") if retrieve_dir else ""
    if cache_path and os.path.exists(cache_path):
        mg.build_mem_from_disk_lme_ver(cache_path)
        print(f"Loaded cached memory graph: {cache_path}")
        _record_progress("memory_graph_loaded", index=n, question_id=question_id, cache_path=cache_path, tag_nodes=len(mg.tag_nodes), semantic_nodes=len(mg.semantic_nodes), procedural_nodes=len(mg.procedural_nodes), subgoal_nodes=len(mg.subgoal_nodes), episodic_nodes=len(mg.episodic_nodes))
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            memories = list(executor.map(_build_memory_from_session, sessions, times))
        print("Memory OK")
        _record_progress("memory_built", index=n, question_id=question_id, session_count=len(memories))
        for memory in memories:
            mg.insert(memory)#5
        print("MG OK")
        _record_progress("memory_graph_ready", index=n, question_id=question_id, tag_nodes=len(mg.tag_nodes), semantic_nodes=len(mg.semantic_nodes), procedural_nodes=len(mg.procedural_nodes), subgoal_nodes=len(mg.subgoal_nodes), episodic_nodes=len(mg.episodic_nodes))
    print("Finish Loading Session")
    goal = "Answer user's question"
    with open("../../../data_longmemeval/retrieve.json", "a",) as input:
        _json = {
            "question_id": question_id,
            "question": question
        }
        input.write(json.dumps(_json) + "\n")
    messages, memory_map, sel_type = mg.retrieve_memory(goal=goal, observation=question, time=f"Date: {test['question_date']}", task_type=task_type)#6
    memory_str = memory_map[sel_type]
    response = call_gpt(messages=messages, model_id=reasoning_model)#7
    information = response
    with open("../../../data_longmemeval/reasoning.jsonl", "a",) as input:
        _json = {
            "question_id": question_id,
            "messages": messages,
            "response": response
        }
        input.write(json.dumps(_json) + "\n")
    prompt_run = run_prompt_template.format(
        information=information,
        question=question,
        time=test['question_date']
    )
    response = call_gpt(
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt_run}
        ],
        model_id=reasoning_model
    )
    with open("../../../data_longmemeval/hypothesis.jsonl", "a",) as input:
        _json = {
            "question_id": question_id,
            "hypothesis": response
        }
        input.write(json.dumps(_json) + "\n")
    trace_payload = {
        "index": n,
        "question_id": question_id,
        "question": question,
        "question_date": test["question_date"],
        "session_count": len(sessions),
        "selected_memory_type": sel_type,
        "selected_memory": memory_str,
        "reasoning_messages": messages,
        "reasoning_response": information,
        "hypothesis": response,
    }
    _append_jsonl(os.path.join(output_dir, "trace.jsonl"), trace_payload)
    _record_progress("sample_done", index=n, question_id=question_id, selected_memory_type=sel_type)
