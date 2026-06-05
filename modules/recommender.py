from retrieval import retrieve


def recommend(title):

    results = retrieve(
        title,
        top_k=6
    )

    return results[1:]