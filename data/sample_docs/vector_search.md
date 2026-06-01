# Vector Search

A vector database stores embeddings and finds the nearest neighbours to a query vector.

## Approximate nearest neighbour

Exact search is slow on large datasets, so vector databases use approximate nearest neighbour
(ANN) indexes such as HNSW or IVFFlat to trade a little recall for much higher speed.

## Hybrid search and reranking

Hybrid search combines vector (semantic) search with keyword (lexical) search so that both
meaning and exact terms are matched. Reranking uses a cross-encoder that reads the query and
each candidate together to reorder results by true relevance. pgvector adds vector search to
Postgres.
