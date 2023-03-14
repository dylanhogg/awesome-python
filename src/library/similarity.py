from sentence_transformers import SentenceTransformer, util


def _remove_generic_topics(topics):
    remove_topics = ["python", "python2", "python-2", "python3", "python-3", "python-library", "library"]
    for t in remove_topics:
        if t in topics:
            topics.remove(t)
    return topics


def _create_similarity_record(row):
    topics = _remove_generic_topics(row["_topics"])
    description = row["_description"]
    description = description.replace(row["_reponame"] + ": ", "")  # HACK: remove prefixed repo name from description
    description = description.strip().rstrip(".")
    description += ". " + ", ".join(topics)

    return {"repopath": row["_repopath"], "sentence": description, "topics": topics, "category": row["category"]}


def get_lookup_dict(records: dict) -> dict:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    similarity_records = [_create_similarity_record(v) for v in records]
    sentences = [d["sentence"] for d in similarity_records]

    embeddings = model.encode(sentences, show_progress_bar=True)
    cos_sim = util.cos_sim(embeddings, embeddings)  # Returns torch.Tensor

    # Add all pairs to a list with their cosine similarity score
    pairs = []
    for i, _ in enumerate(cos_sim):
        for j, _ in enumerate(cos_sim):
            if i != j:  # Exclude identity
                pairs.append([float(cos_sim[i][j]), i, j])

    # Sort list by the highest cosine similarity score
    pairs = sorted(pairs, key=lambda x: x[0], reverse=True)

    # Construct lookup_dict and pairwise_dataset
    lookup_dict = {}  # NOTE: lookup_dict is used to construct "sim" column in df written to github_data.json
    pairwise_dataset = []  # NOTE: pairwise_dataset is useful for debugging
    sim_cutoff = 0.50  # TODO: review, seems reasonble, below 0.5 is poor
    for idx, (score, i, j) in enumerate(pairs):
        common_topics = list(set(similarity_records[i]["topics"]) & set(similarity_records[j]["topics"]))
        total_topics = len(set(similarity_records[i]["topics"])) + len(set(similarity_records[j]["topics"]))
        sim = float(cos_sim[i][j])
        if sim > sim_cutoff:
            repo1 = similarity_records[i]["repopath"]
            repo2 = similarity_records[j]["repopath"]
            category1 = similarity_records[i]["category"]
            category2 = similarity_records[j]["category"]
            lookup_dict.setdefault(repo1, []).append(
                (repo2, sim, category2, len(common_topics))
            )  # NOTE: fields included in sim column
            pairwise_dataset.append(
                {
                    "idx": idx,
                    "sim": sim,
                    "common_topics": ", ".join(common_topics),
                    "common_topics_count": len(common_topics),
                    "total_topics_count": total_topics,
                    "common_topics_prop": len(common_topics) / total_topics if total_topics > 0 else 0,
                    "repo1": repo1,
                    "repo2": repo2,
                    "category1": category1,
                    "category2": category2,
                    "sent1": similarity_records[i]["sentence"],
                    "sent2": similarity_records[j]["sentence"],
                }
            )

    # Ensure sorted dict value by similarity (item[1])
    lookup_dict = {k: sorted(v, key=lambda item: item[1], reverse=True) for k, v in lookup_dict.items()}
    return lookup_dict


def lookup_similarity_record(row, lookup_dict: dict, cutoff: float = 0.5):
    lookup = lookup_dict.get(row["_repopath"], None)
    return [] if lookup is None else [x for x in lookup if x[1] >= cutoff]
