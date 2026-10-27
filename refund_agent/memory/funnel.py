"""Count nested question-level survival events, with each stage's own denominator."""

from collections import defaultdict

from demos.ml_data_evaluation_demo import wilson_interval


def stage_flags(questions, first_stage, reranker, assembler):
    flags = []
    for question in questions:
        candidates = first_stage(question)
        ranked = reranker(question, candidates)
        if not {r["id"] for r in ranked} <= {r["id"] for r in candidates}:
            raise ValueError("reranker must read only candidates")
        kept = assembler(question, ranked)
        if not {r["id"] for r in kept} <= {r["id"] for r in ranked}:
            raise ValueError("assembler must read only ranked passages")
        needed = question["needed"]
        flags.append(
            {
                "question": question["id"],
                "wording": question["wording"],
                "ret": any(r["id"] == needed for r in candidates),
                "rank": any(r["id"] == needed for r in ranked),
                "kept": any(r["id"] == needed for r in kept),
                "superseded": any(
                    r["procedure"] == question["procedure"]
                    and r["valid_to"] is not None
                    and r["valid_to"] <= question["at"]
                    for r in kept
                ),
            }
        )
    return flags


# region book:fnd.memory-rag.funnel-02
def funnel(flags, by):
    groups = defaultdict(list)
    for row in flags:
        if row["kept"] > row["rank"] or row["rank"] > row["ret"]:
            raise ValueError("stage flags must be nested")
        groups[row[by]].append(row)
    results = []
    for group, rows in sorted(groups.items()):
        counts = [len(rows)] + [
            sum(r[key] for r in rows) for key in ("ret", "rank", "kept")
        ]
        rates = [
            end / start if start else None
            for start, end in zip(counts, counts[1:])
        ]
        # endregion book:fnd.memory-rag.funnel-02
        results.append(_funnel_row(group, counts, rates))
    return results


def _funnel_row(group, counts, rates):
    intervals = [
        list(wilson_interval(end, start)) if start else None
        for start, end in zip(counts, counts[1:])
    ]
    p_ret, p_rank, p_kept = rates
    product = (
        p_ret * p_rank * p_kept
        if p_ret is not None and p_rank is not None and p_kept is not None
        else None
    )
    return {
        "group": group,
        "counts": counts,
        "rates": rates,
        "intervals": intervals,
        "product": product,
        "end_to_end": counts[-1] / counts[0],
    }
