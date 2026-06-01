# Context and Memory

A model only "sees" what fits in its context window, so managing context is essential.

## Fitting the window

The context window limits how much text a model can use at once. Summarization compresses a
long conversation history so it still fits inside the window.

## Long-term memory

Conversation memory stores past turns so a chat stays coherent. Retrieval can act as long-term
memory by fetching relevant past information on demand instead of keeping everything in context.
