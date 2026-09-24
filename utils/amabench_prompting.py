from __future__ import annotations

def get_answer_format_instruction(subset: str) -> str:
    """Return the answer-format instruction for AMABench QA prompts."""
    if subset == "mcq":
        return (
            "Please provide your final answer in the following format:\n"
            "###Answer: (X)\n"
            "where X is the correct option letter (A, B, C, or D)."
        )

    return (
        "Please provide your final answer in the following format:\n"
        "###Answer: [your answer here]"
    )
