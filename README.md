# Exploring Cross-Scenario Generality of Agentic Memory Systems

Code for the AACL-IJCNLP 2026 paper. It evaluates twelve memory baselines and **AutoMEM** on five scenarios: multi-session chat, large-corpus retrieval, agentic-trajectory QA, memory stress tests, and long-horizon agentic tasks.

Paper: [arXiv:2606.04315](https://arxiv.org/abs/2606.04315).

## Setup

```bash
pixi install
pixi shell

export OPENAI_API_KEY=EMPTY
export LLM_MODEL=Qwen/Qwen3-32B
export EMBEDDING_MODEL=Qwen/Qwen3-Embedding-4B
export JUDGE_MODEL=Qwen/Qwen3-32B
export LLM_BASE_URL=http://127.0.0.1:30000/v1
export EMBEDDING_BASE_URL=http://127.0.0.1:30001/v1
export JUDGE_BASE_URL=http://127.0.0.1:30000/v1
```

Serve the chat, embedding, and judge models with vLLM or SGLang. ALFWorld uses `Qwen/Qwen2.5-7B-Instruct` as the actor. Set `ALFWORLD_DATA` to the directory that contains `json_2.1.1/`.

## Main table

`examples/run_benchmark.py` is the dispatcher. `agentmem/methods/registry.py` is the method allow-list.

```bash
python examples/run_benchmark.py \
  --benchmark <hotpotqa|locomo|amabench|memoryagentbench|memoryarena> \
  --method <method> \
  --llm-model "$LLM_MODEL" --llm-base-url "$LLM_BASE_URL" --llm-api-key EMPTY \
  --embedding-model "$EMBEDDING_MODEL" \
  --embedding-base-url "$EMBEDDING_BASE_URL" --embedding-api-key EMPTY \
  --output-dir results/<benchmark>/<method> \
  --track-efficiency
```

| Scenario | Benchmark | How to select it |
|---|---|---|
| Personal chat | LoCoMo | `--benchmark locomo` |
| Large-corpus retrieval | HotpotQA | `--benchmark hotpotqa` |
| Trajectory recall | AMABench ALF / Web / Text2SQL | `--benchmark amabench --amabench-domain embodied\|webarena\|text2sql` |
| Memory stress tests | MemoryAgentBench AR, TTL, LRU, CR | build the 200-question manifest, then `--benchmark memoryagentbench --mab-manifest <manifest> --mab-categories AR TTL LRU CR` |
| Agentic tasks | MemoryArena shop / travel | `--benchmark memoryarena --memoryarena-config bundled_shopping` or `group_travel_planner` |
| Agentic tasks | ALFWorld | `scripts/run_alfworld_train_test.py` (below) |

The MemoryAgentBench manifest is built with:

```bash
python scripts/select_memoryagentbench_200q.py \
  --output datasets/memoryagentbench/mab_200q.jsonl \
  --manifest datasets/memoryagentbench/mab_200q_manifest.json
```

ALFWorld is a train-then-test run (256 train episodes, 60 test episodes), not a cell of `run_benchmark.py`:

```bash
export ALFWORLD_DATA=/path/to/alfworld
export RQ1_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
export RQ1_LLM_URL="$LLM_BASE_URL"
export RQ1_EMB_MODEL="$EMBEDDING_MODEL"
export RQ1_EMB_URL="$EMBEDDING_BASE_URL"
export RESULTS_ROOT=results/alfworld
python scripts/run_alfworld_train_test.py
```

## Scoring

QA benchmarks use a Qwen3-32B judge. ALFWorld reports environment success. MemoryArena reports its process score (shop) and corrected slot score (travel) from the runner.

```bash
python scripts/judge_qa_llm_local.py --benchmark <hotpotqa|locomo> \
  --mode <hotpotqa|locomo> --input <answers.jsonl> --output <judged.jsonl> \
  --summary <summary.json> --model "$JUDGE_MODEL" --base-url "$JUDGE_BASE_URL" \
  --api-key EMPTY

python scripts/judge_qa_strict_local.py --input <amabench_answers.jsonl> \
  --output <judged.jsonl> --summary <summary.json> \
  --model "$JUDGE_MODEL" --base-url "$JUDGE_BASE_URL" --api-key EMPTY

python scripts/judge_memoryagentbench_local.py --input <mab_answers.jsonl> \
  --output <judged.jsonl> --summary <summary.json> \
  --model "$JUDGE_MODEL" --base-url "$JUDGE_BASE_URL" --api-key EMPTY
```

## Published methods

These are the systems in the main results table.

| Paper name | `--method` | Code |
|---|---|---|
| Long context | `longcontext` | `agentmem/methods/longcontext.py` |
| SimpleMem | `simplemem` | `agentmem/eval/amabench_runner/methods/simplemem.py` |
| Mem0 | `mem0` | `agentmem/methods/mem0.py` |
| PlugMem | `plugmem` | `agentmem/eval/amabench_runner/methods/plugmem.py` |
| LightMem | `lightmem` | `agentmem/methods/lightmem.py`, `vendor/lightmem/` |
| MemoryOS | `memoryos` | `agentmem/methods/memoryos.py`, `vendor/MemoryOS/` |
| HippoRAGv2 | `hipporag` | `agentmem/eval/amabench_runner/methods/hipporag.py` |
| AMA-Agent | `ama_agent` | `agentmem/eval/amabench_runner/methods/ama_agent.py` |
| DCI-Lite | `dci_lite` | `agentmem/methods/dci_lite.py` |
| DCI-Lite+Sum | `dci_lite_sum` | `agentmem/methods/dci_lite.py` |
| Mem-T | `memt` | `agentmem/eval/amabench_runner/methods/memt.py`, `vendor/memt/` |
| MemRL | `memrl` | `agentmem/methods/memrl.py`, `vendor/memrl/` |
| AutoMEM | `automem` | `agentmem/methods/automem.py` |

## Acknowledgments

This repository vendors snapshots of [MemoryAgentBench](https://github.com/HUST-AI-HYZ/MemoryAgentBench), [MemoryOS](https://github.com/BAI-LAB/MemoryOS), [A-Mem](https://github.com/agiresearch/a-mem), [LightMem](https://github.com/zjunlp/LightMem), [MemRL](https://github.com/MemTensor/MemRL), [Mem-T](https://github.com/yanweiyue/Mem-T), and [PlugMem](https://github.com/TIMAN-group/PlugMem). We thank the authors of those systems.

## Citation

```bibtex
@article{chen2026automem,
  title={Exploring Cross-Scenario Generality of Agentic Memory Systems: Diagnostics and a Strong Baseline},
  author={Chen, Zhikai and Gu, Jialiang and Yin, Junyu and Long, Xianxuan and Zeng, Shenglai and Liu, Xiaoze and Guo, Kai and Zhou, Keren and Tang, Jiliang},
  journal={arXiv preprint arXiv:2606.04315},
  year={2026}
}
```
