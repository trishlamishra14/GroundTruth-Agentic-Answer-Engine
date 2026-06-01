# LLM Safety and Guardrails

Safety techniques keep an LLM system trustworthy and resistant to misuse.

## Common risks and defenses

Prompt injection is an attack where input text tries to override the system instructions.
Guardrails are checks that filter or constrain a model's inputs and outputs. PII redaction
removes sensitive personal information before it is processed.

## Trust behaviours

Grounding answers in retrieved sources and citing them reduces hallucination and makes answers
auditable. Abstention — refusing to answer when the system is not confident — is a key safety
behaviour because a wrong answer is often worse than no answer.
