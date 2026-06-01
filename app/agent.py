"""
app/agent.py — the agentic loop (the upgrade over plain RAG).

Instead of "retrieve once, answer once", GroundTruth runs a small controller:

    retrieve -> grade the best passage
        |-- good enough            -> generate a grounded, cited answer
        |-- weak, retries left     -> rewrite the query and retrieve again
        |-- weak, out of retries   -> ABSTAIN ("not in the docs")

This is what makes the system *reliable*: it tries harder on hard questions and
refuses instead of hallucinating on out-of-scope ones. The same node functions
are reused by the LangGraph version in app/graph.py.
"""

import cohere

from app.config import settings
from app.cohere_utils import call_with_retry
from app.observability import token_usage
from app.retrieve import retrieve

co = cohere.ClientV2(api_key=settings.cohere_api_key)

NOT_FOUND = "I don't have enough information in the docs to answer that."

ANSWER_PROMPT = """You are GroundTruth, a careful documentation assistant. Answer the \
QUESTION in one or two COMPLETE sentences using ONLY the numbered CONTEXT passages. \
Always write the actual answer text — never reply with only a citation. Cite every \
passage you use by its number in square brackets, e.g. [1] or [2]. If the answer is not \
contained in the context, reply with exactly this sentence and nothing else: "{not_found}"

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

REWRITE_PROMPT = """The search results for the question below were weak. Rewrite the \
question into a single, keyword-rich search query that would retrieve the right \
documentation passage. Reply with ONLY the rewritten query, no preamble.

QUESTION: {question}
REWRITTEN QUERY:"""


def _chat(state: dict, prompt: str) -> str:
    """Call Cohere chat and accumulate token usage onto the state."""
    resp = call_with_retry(co.chat, model=settings.chat_model,
                           messages=[{"role": "user", "content": prompt}])
    inp, out = token_usage(resp)
    state["in_tokens"] += inp
    state["out_tokens"] += out
    return resp.message.content[0].text.strip()


def new_state(question: str) -> dict:
    return {
        "question": question,
        "query": question,
        "candidates": [],
        "answer": "",
        "grounded": False,
        "abstained": False,
        "loops": 0,
        "in_tokens": 0,
        "out_tokens": 0,
        "trace": [],
    }


# --- nodes -----------------------------------------------------------------
def step_retrieve(state: dict) -> dict:
    state["candidates"] = retrieve(state["query"])
    top = state["candidates"][0]["score"] if state["candidates"] else 0.0
    state["trace"].append(f"retrieve(query={state['query']!r}) -> top_score={top:.3f}")
    return state


def decide(state: dict) -> str:
    top = state["candidates"][0]["score"] if state["candidates"] else 0.0
    if top >= settings.abstain_threshold:
        return "generate"
    if state["loops"] < settings.max_agent_loops:
        return "rewrite"
    return "abstain"


def step_rewrite(state: dict) -> dict:
    state["loops"] += 1
    state["query"] = _chat(state, REWRITE_PROMPT.format(question=state["question"]))
    state["trace"].append(f"rewrite -> {state['query']!r} (loop {state['loops']})")
    return state


def step_generate(state: dict) -> dict:
    context = "\n\n".join(
        f"[{i + 1}] {c['content']}" for i, c in enumerate(state["candidates"])
    )
    text = _chat(state, ANSWER_PROMPT.format(
        not_found=NOT_FOUND, context=context, question=state["question"]))
    state["answer"] = text
    state["grounded"] = NOT_FOUND.lower() not in text.lower()
    state["abstained"] = not state["grounded"]
    state["trace"].append(f"generate -> grounded={state['grounded']}")
    return state


def step_abstain(state: dict) -> dict:
    state["answer"] = NOT_FOUND
    state["grounded"] = False
    state["abstained"] = True
    state["trace"].append("abstain (no passage cleared the threshold)")
    return state


# --- plain controller (no extra dependencies) ------------------------------
def run_agent(question: str) -> dict:
    state = new_state(question)
    while True:
        step_retrieve(state)
        route = decide(state)
        if route == "generate":
            step_generate(state)
            break
        if route == "rewrite":
            step_rewrite(state)
            continue
        step_abstain(state)
        break
    return _finalize(state)


def _finalize(state: dict) -> dict:
    sources = [
        {"rank": i + 1, "doc": c["doc"], "score": round(c["score"], 3),
         "snippet": c["content"][:300]}
        for i, c in enumerate(state["candidates"])
    ] if state["grounded"] else []
    return {
        "answer": state["answer"],
        "sources": sources,
        "grounded": state["grounded"],
        "abstained": state["abstained"],
        "loops": state["loops"],
        "in_tokens": state["in_tokens"],
        "out_tokens": state["out_tokens"],
        "trace": state["trace"],
    }
