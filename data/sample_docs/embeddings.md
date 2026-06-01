# Embeddings

An embedding is a vector of numbers that represents the meaning of a piece of text. Texts
with similar meaning have vectors that are close together.

## Measuring similarity

Cosine similarity is the most common way to compare two embeddings. It ranges from -1 to 1,
where a higher value means the texts are more similar in meaning.

## Dimensions and uses

An embedding has a fixed number of dimensions (for example 1024). Embeddings power semantic
search, clustering, classification, and retrieval. Query text and document text are often
embedded in slightly different modes to improve search accuracy.
