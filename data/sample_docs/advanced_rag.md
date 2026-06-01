# Advanced RAG Techniques

Beyond basic retrieve-then-generate, several techniques improve RAG quality.

## Retrieval and ranking

Top-k retrieval selects the k most relevant chunks to place in the prompt. Re-ranking then
reorders those chunks by true relevance using a cross-encoder. Query rewriting reformulates a
weak query so it retrieves better passages.

## Harder questions

HyDE (Hypothetical Document Embeddings) generates a hypothetical answer and embeds it to
improve retrieval on difficult queries. GraphRAG builds a knowledge graph over the documents to
answer multi-hop questions that span several sources.
