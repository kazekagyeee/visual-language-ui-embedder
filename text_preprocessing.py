"""Compatibility shim for older imports.

The embedder feeds raw UI text into the LLM tokenizer. Heavy preprocessing
such as stop-word removal, punctuation stripping, digit removal, and
lemmatization changes the prompt semantics and breaks alignment with the
image+text path.
"""


def preprocess_text(text: str) -> str:
    """Return text unchanged, preserving LLM-visible semantics."""
    return text or ""
