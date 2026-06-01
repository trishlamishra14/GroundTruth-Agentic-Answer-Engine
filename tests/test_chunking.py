from app.chunking import chunk_text, split_sentences


def test_split_sentences():
    s = split_sentences("One. Two! Three?")
    assert s == ["One.", "Two!", "Three?"]


def test_chunks_respect_max_chars_roughly():
    text = " ".join(f"Sentence number {i} here." for i in range(50))
    chunks = chunk_text(text, max_chars=120, overlap_sentences=1)
    assert len(chunks) > 1
    # each chunk should be in the right ballpark (allow one sentence of overshoot)
    assert all(len(c) <= 200 for c in chunks)


def test_overlap_carries_context():
    text = "Alpha one. Bravo two. Charlie three. Delta four. Echo five."
    chunks = chunk_text(text, max_chars=25, overlap_sentences=1)
    # consecutive chunks should share a sentence due to overlap
    assert len(chunks) >= 2
