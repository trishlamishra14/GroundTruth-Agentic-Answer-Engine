# Retrieval-Augmented Generation

Retrieval-Augmented Generation (RAG) means retrieving relevant documents first, then asking
the model to answer using those documents. It reduces hallucination by grounding the answer
in real sources instead of the model's memory.

## How RAG works

Documents are split into chunks, embedded, and stored. At query time the system retrieves the
top matching chunks and places them in the prompt as context. Chunking with a little overlap
keeps facts from being split across a boundary.

## Grounding and abstention

Because answers are tied to retrieved passages, they can be cited and verified. A good RAG
system abstains — says it does not know — when the retrieved context does not support an answer.
