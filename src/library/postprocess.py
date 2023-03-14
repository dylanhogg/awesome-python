import json
import itertools
import pandas as pd
from collections import Counter


def write_tags(github_json_filename: str, github_tags_json_filename: str, most_common: int):
    with open(github_json_filename, "r") as fin:
        data = json.load(fin)["data"]
        topic_lists = [list(set(t["_topics"])) for t in data if t["_topics"]]
        topic_list = list(itertools.chain(*topic_lists))

        counts = Counter(topic_list)
        most_common = counts.most_common(most_common)
        # above_count_value = [(k, c) for k, c in counts.items() if c >= 100]

        exclude_topics = ["python", "python2", "python3"]
        # top_topics = dict([(k, f"{k} ({c})") for k, c in most_common if k not in exclude_topics])
        top_topics = dict([(k, f"{k}") for k, c in most_common if k not in exclude_topics])
        sorted_topics = dict(sorted(top_topics.items()))

        # json_topics = {"": "All Tags"}
        # json_topics.update(top_topics)
        with open(github_tags_json_filename, "w") as fout:
            json.dump(sorted_topics, fout, indent=4)


def write_best_in_class_data(github_json_filename: str, sort_col: str):
    top = 3
    group_by = "category"
    cols = [
        "category",
        "_organization",
        "_repopath",
        "_reponame",
        "_stars",
        "_forks",
        "_watches",
        "_language",
        "_homepage",
        "_description",
        "_updated_at",
        "_created_at",
        "_age_weeks",
        "_stars_per_week",
        "_topics",
        "_last_commit_date",
        "_pop_score",
        "_readme_filename",
        "_requirements_filenames",
        "rank",
    ]

    df = pd.read_json("./github_data.json", orient="table")
    df_best = (
        df.assign(
            rank=df.sort_values(["category", sort_col], ascending=[True, False]).groupby([group_by]).cumcount() + 1
        )
        .query(f"rank <= {top}")
        .sort_values([group_by, "rank"], ascending=[True, True])
        .reset_index(drop=True)
    )[cols]

    with open(github_json_filename, "w") as f:
        json_results = df_best.to_json(orient="table", double_precision=2)
        data = json.loads(json_results)
        json.dump(data, f, indent=4)
