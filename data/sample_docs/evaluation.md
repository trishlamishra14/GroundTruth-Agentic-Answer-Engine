# Evaluating LLM Systems

Evaluation measures the quality of an LLM system instead of relying on impressions.

## Core metrics

Faithfulness measures whether an answer is fully supported by the provided context. Retrieval
recall measures whether the correct source document was retrieved. Hallucination rate measures
how often the model invents unsupported claims.

## Tools and process

Ragas is a framework for evaluating RAG systems on metrics like faithfulness, answer relevance,
and context precision. An LLM-as-judge uses a model to grade outputs, though the judge has its
own error rate. Regression testing re-runs a benchmark to catch quality drops before shipping.
