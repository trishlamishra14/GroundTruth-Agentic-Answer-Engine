# Fine-Tuning

Fine-tuning adapts a pretrained model to a specific task or domain by continuing training on
example data.

## Full fine-tuning vs LoRA

Full fine-tuning updates all of a model's weights and is expensive. LoRA (Low-Rank Adaptation)
instead trains a small number of additional parameters, making adaptation much cheaper and
faster while leaving the base weights frozen.

## RAG vs fine-tuning

RAG adds knowledge at query time by retrieving documents, so it is best when facts change
often. Fine-tuning bakes behaviour, format, or style into the weights, so it is best for how
the model should respond rather than what it should know.
