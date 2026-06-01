# Large Language Models

A large language model (LLM) is a neural network trained on large amounts of text to
predict the next token in a sequence.

## Tokens and context

A token is a chunk of text — on average about four characters, or roughly three quarters
of a word. The context window is the maximum number of tokens a model can consider at once;
text beyond it is not seen by the model.

## Temperature

Temperature controls randomness in generation. A temperature of 0 makes output focused and
nearly deterministic, while higher temperatures produce more diverse and creative output.

## Hallucination

Hallucination is when a model produces text that is fluent and plausible but factually wrong
or unsupported by any source. Grounding the model in retrieved documents reduces hallucination.
