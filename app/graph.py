"""
app/graph.py — the SAME agent, wired as a LangGraph StateGraph.

Why have both this and the plain controller in agent.py?
  - agent.py always runs (no extra dependency) — reliable default.
  - this file expresses the agent as an explicit, inspectable graph, which is
    the production-standard way to build stateful agents (conditional edges,
    checkpointing, human-in-the-loop hooks). Set USE_LANGGRAPH=true to use it.

The graph reuses the node functions from agent.py, so there is one source of truth.
"""

from typing import TypedDict

from app import agent

_compiled = None


class GTState(TypedDict, total=False):
    question: str
    query: str
    candidates: list
    answer: str
    grounded: bool
    abstained: bool
    loops: int
    in_tokens: int
    out_tokens: int
    trace: list


def _build():
    from langgraph.graph import StateGraph, START, END

    g = StateGraph(GTState)
    g.add_node("retrieve", agent.step_retrieve)
    g.add_node("rewrite", agent.step_rewrite)
    g.add_node("generate", agent.step_generate)
    g.add_node("abstain", agent.step_abstain)

    g.add_edge(START, "retrieve")
    g.add_conditional_edges(
        "retrieve",
        agent.decide,
        {"generate": "generate", "rewrite": "rewrite", "abstain": "abstain"},
    )
    g.add_edge("rewrite", "retrieve")
    g.add_edge("generate", END)
    g.add_edge("abstain", END)
    return g.compile()


def run_agent_graph(question: str) -> dict:
    global _compiled
    if _compiled is None:
        _compiled = _build()
    state = agent.new_state(question)
    final = _compiled.invoke(state)
    return agent._finalize(dict(final))
