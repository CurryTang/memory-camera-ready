"""Paper-appendix AMA-Agent prompts used by the live AMABench path."""

MEMORY_CONSTRUCTION_PROMPT_TEMPLATE = """
You are given an agent trajectory consisting of action and observation pairs.
Your task is to analyze the trajectory turn by turn and produce a Markdown report
that is strictly machine parsable.

Input will provide:
(1) current turn: {turn_t}
(2) previous turn: {turn_t_minus_1}
(3) optional task description: {task}

You must do the following in order.

Section 1: OBJECTIVE INVENTORY
List all objectives that the agent is tracking or manipulating.

Section 2: ENV STATE CHANGE DETECTION
Decide whether the environment state changed at this turn.

Section 3: OBJECT STATE CHANGE DETECTION
Decide whether the object state changed at this turn.

OUTPUT FORMAT
Return a Markdown block with the following exact structure.

# OBJECTIVES
1. <objective_name>: <description>
2. ...

# STATE_CHANGES
env_changed: <true or false>
objective_changed: <true or false>
evidence:
  - "<quote 1>"
  - "<quote 2>"

# STATES
env_state:
  - <key>: <value>
  - ...
object_states:
  - name: <objective_name>
    state:
      - <key>: <value>
      - ...
  - ...
Constraints:
(1) Do not invent facts not supported by the provided turns.
(2) Keep rationales short and grounded.
(3) Use consistent objective names across turns.
"""

CHUNK_SUFFICIENCY_JUDGMENT_PROMPT_TEMPLATE = """
You have retrieved the top ranked most relevant turns from an agent trajectory.
Each turn has a UNIQUE TURN INDEX that you can reference.

Query: {query}

Retrieved Turns:
{retrieved_chunks}

Your Task
Carefully analyze the retrieved turns and determine ONE of the following.

1. SUFFICIENT
The retrieved turns contain enough information to answer the query completely.
If you choose this, you MUST provide the answer immediately in the same response.
Format:
SUFFICIENT
ANSWER: <your complete and accurate answer>

2. NEED_GRAPH
The query can likely be answered by looking at adjacent turns or specific ranges.
Use this when you found relevant information but need surrounding context.

You can specify retrieval in multiple ways.

A. Request adjacent turns
NEED_GRAPH: turn_5 before=2 after=1
NEED_GRAPH: turn_8 before=3 after=0, turn_15 before=0 after=2

B. Request turn ranges
NEED_GRAPH: turns 5 to 10
NEED_GRAPH: turns 3 to 8, turns 15 to 20

C. Request individual turns
NEED_GRAPH: turns 3, 7, 12, 18
NEED_GRAPH: turns 5, 8, 15

3. NEED_CODE
The query requires computational analysis, pattern finding, counting, or aggregation
across the full trajectory and cannot be answered from the retrieved turns alone.
Format:
NEED_CODE: <explain what computation or analysis is needed>

Guidelines
Choose SUFFICIENT only if you can answer completely right now.
Choose NEED_GRAPH when you need immediate context around retrieved turns.
Choose NEED_CODE for trajectory wide computation or when turns do not contain the answer.

Response:
"""

CODE_GENERATION_PROMPT_TEMPLATE = """
Write Python code to extract information from a trajectory to answer the question.
Keep your thinking brief.

Question: {query}

Task: {task}

Trajectory Sample:
{trajectory_sample}

Trajectory JSON Structure
Variable trajectory_json contains:
{
  "trajectory": [
    {
      "turn_idx": 0,
      "action": "...",
      "observation": "..."
    },
    ...
  ],
  "task": "...",
  "episode_id": "..."
}

Requirements
(1) Only use Python standard library unless explicitly allowed otherwise.
(2) The code must be directly executable.
(3) Print intermediate results that justify the final answer.
(4) Reference turn indices in outputs whenever possible.
"""

COMPRESS_PROMPT_TEMPLATE = MEMORY_CONSTRUCTION_PROMPT_TEMPLATE
CAUSAL_PROMPT_TEMPLATE = MEMORY_CONSTRUCTION_PROMPT_TEMPLATE

CHECK_STATE_MEM_PROMPT_TEMPLATE = """You are analyzing whether the compressed state memory contains enough information to answer a question.

State Memory (compressed representation of the trajectory):
{state_mem_str}

Question: {query}

Analyze if the state memory contains sufficient information to answer this question accurately.

Consider:
1. Does the state memory mention the relevant objects/entities in the question?
2. Does it contain the specific information needed (states, relationships, actions)?
3. Is the information detailed enough or just vague references?

Respond with ONLY "SUFFICIENT" or "NEED_RETRIEVAL" followed by a brief reason.

Format:
SUFFICIENT: [reason why state memory is enough]
or
NEED_RETRIEVAL: [what specific information is missing and needs to be retrieved]

Response:"""

def render_code_generation_prompt(*, query: str, task: str, trajectory_sample: str) -> str:
    """Render the appendix-exact code-generation prompt without brace escaping."""
    prompt = CODE_GENERATION_PROMPT_TEMPLATE
    prompt = prompt.replace("{query}", query)
    prompt = prompt.replace("{task}", task)
    prompt = prompt.replace("{trajectory_sample}", trajectory_sample)
    return prompt

TOOL_USE_PROMPT_TEMPLATE = """You are helping retrieve relevant information from a trajectory to answer a question.

**Question:** {query}

**Available Tools:**

You have access to TWO powerful tools to search and retrieve information from the trajectory:

1. **traj_find** - Locates relevant turns
   - Purpose: Search for specific keywords/entities/actions in the trajectory
   - Parameters:
     * query (required): The search term (e.g., "open door", "key", "red box")
     * mode (optional): Search strategy
       - "keyword": Search anywhere in text (default)
       - "action": Search only in action field
       - "entity": Search for specific entity mentions
   - Returns: List of turn indices where the query was found
   - Example: traj_find(query="pick up", mode="action")

2. **traj_get** - Retrieves detailed information
   - Purpose: Get full details from specific turns
   - Parameters:
     * span (required): Which turns to get
       - {{"indices": [1, 2, 3]}} for specific turns
       - {{"start": 1, "end": 5}} for a range
     * fields (optional): What info to include ["action", "observation", "action_space"]
   - Returns: Formatted text with detailed turn information
   - Example: traj_get(span={{"indices": [5, 7, 9]}})

**Recommended Strategy:**
1. Use traj_find to locate turns related to the question
2. Use traj_get to retrieve detailed information from those turns
3. You can call tools multiple times to gather complete information

**Your Task:**
Use these tools strategically to find and retrieve ALL relevant information needed to answer the question thoroughly."""

ANSWER_WITH_RETRIEVAL_PROMPT_TEMPLATE = """Based on the compressed state memory and retrieved detailed information, provide a natural language answer to the query.

Query: {query}

State Memory (compressed):
{state_mem_str}

Retrieved Detailed Information:
{relevant_mem}

CRITICAL: You MUST format your response as follows:
ANSWER: [Your concise, accurate answer here]

Only include the answer after "ANSWER:", nothing else."""

ANSWER_WITHOUT_RETRIEVAL_PROMPT_TEMPLATE = """Based on the compressed state memory, provide a natural language answer to the query.

Query: {query}

State Memory:
{state_mem_str}

CRITICAL: You MUST format your response as follows:
ANSWER: [Your concise, accurate answer here]

Only include the answer after "ANSWER:", nothing else."""
