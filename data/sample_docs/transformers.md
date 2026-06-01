# Transformers

The transformer is the neural-network architecture behind modern large language models,
introduced in the paper "Attention Is All You Need".

## Self-attention

Self-attention lets every token attend to every other token in the input, so the model can
weigh which words matter for understanding each position. Multi-head attention runs several
attention operations in parallel to capture different kinds of relationships.

## Position and structure

Because attention has no inherent sense of order, positional encodings are added to give the
model information about token position. Most modern LLMs are decoder-only transformers.
