# Prompt Engineering

A prompt is the instruction and input given to a language model. Small changes in wording can
change the output significantly.

## System prompts and examples

A system prompt sets the model's role and behaviour. Few-shot prompting includes a few worked
examples in the prompt so the model imitates the pattern.

## Reasoning and structure

Chain-of-thought prompting asks the model to reason step by step before answering, which helps
on complex tasks. Structured-output prompting constrains the model to return JSON or a fixed
schema. Clear grounding instructions ("answer only from the context") reduce hallucination.
