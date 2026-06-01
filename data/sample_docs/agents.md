# AI Agents

An AI agent is a language model that can take actions through tools, often in a loop, to
accomplish a goal rather than just returning a single response.

## Tools and the ReAct loop

Tool calling (also called function calling) lets the model invoke functions such as search or
a calculator. The ReAct pattern interleaves reasoning and acting: the agent thinks, calls a
tool, observes the result, and repeats.

## Orchestration

In a multi-agent system several specialised agents collaborate. LangGraph models an agent as a
stateful graph with conditional branching and retries. The Model Context Protocol (MCP) is an
open standard for connecting agents to external tools and data sources.
