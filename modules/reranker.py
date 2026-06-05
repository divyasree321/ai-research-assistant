from sentence_transformers import CrossEncoder

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


def rerank(query, papers):

    pairs = []

    for paper in papers:

        pairs.append(
            [
                query,
                str(paper["title"])
                + " "
                + str(paper["abstract"])
            ]
        )

    scores = reranker.predict(
        pairs
    )

    ranked = sorted(
        zip(scores, papers),
        reverse=True
    )

    return [
        p for _, p in ranked[:5]
    ]