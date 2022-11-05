import json
import itertools
from collections import Counter
from datetime import datetime
from loguru import logger


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
