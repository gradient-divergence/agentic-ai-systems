"""Retrieval metrics over a fixed list of questions with one relevant clause each."""


def _check(rankings, relevant):
    if not rankings or len(rankings) != len(relevant):
        raise ValueError("one ranking and relevance label per question required")


def recall_at_k(rankings, relevant, k):
    _check(rankings, relevant)
    if k < 1:
        raise ValueError("positive k required")
    return sum(
        any(hit.clause.citation == label for hit in rows[:k])
        for rows, label in zip(rankings, relevant)
    ) / len(relevant)


def mean_reciprocal_rank(rankings, relevant):
    _check(rankings, relevant)
    return sum(
        next((1 / i for i, hit in enumerate(rows, 1) if hit.clause.citation == label), 0)
        for rows, label in zip(rankings, relevant)
    ) / len(relevant)


def superseded_returned(rankings, as_of):
    if len(rankings) != len(as_of):
        raise ValueError("one time of use per ranking required")
    return sum(not hit.clause.in_force(at) for rows, at in zip(rankings, as_of) for hit in rows)
