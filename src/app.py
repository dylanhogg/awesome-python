import pandas as pd
from datetime import datetime
import library.env as env
from library.log import get_logger
from library.ghw import GithubWrapper
from urllib.parse import urlparse

logger = get_logger(__name__)


def process(input_file, output_file, token):
    ghw = GithubWrapper(token)

    df = pd.read_csv(input_file)
    df.columns = map(str.lower, df.columns)

    df["_reponame"] = df["githuburl"].apply(lambda x: urlparse(x).path.lstrip("/"))
    df["_repo"] = df["_reponame"].apply(lambda x: ghw.get_repo(x))
    df["_stars"] = df["_reponame"].apply(lambda x: ghw.get_repo(x).stargazers_count)
    df = df.sort_values("_stars", ascending=False)

    def make_line(row):
        repo = row["_repo"]
        url = row["githuburl"]
        name = repo.name
        homepage = (
            f"\n[{repo.homepage}]({repo.homepage})  "
            if repo.homepage is not None and len(repo.homepage) > 0
            else ""
        )
        stars = repo.stargazers_count
        forks = repo.forks_count
        watches = repo.subscribers_count
        updated = repo.updated_at.date()
        created = repo.created_at.date()
        topics = (
            "\n<sub><sup>" + ", ".join(sorted(repo.get_topics())) + "</sup></sub>"
            if len(repo.get_topics()) > 0
            else ""
        )
        description = repo.description
        language = repo.language
        if language is not None and language.lower() != "python":
            logger.info(
                f"Is {name} really a Python library? Main language is {language}."
            )

        return (
            f"[{name}]({url})  "
            f"{homepage}"
            f"\n{description}  "
            f"\n{stars:,} stars, {forks:,} forks, {watches:,} watches  "
            f"\ncreated {created}, updated {updated}, main language {language}  "
            f"{topics}"
            f"\n\n"
        )

    df["_doclines"] = df.apply(lambda x: make_line(x), axis=1)

    lines = [
        "# Crazy Awesome Python",
        "A selection of python libraries and frameworks, "
        "with a bias towards the data/machine learning space. Ordered by stars.  \n\n",
    ]
    lines.extend(list(df["_doclines"]))
    lines.append(
        f"Automatically generated from csv on {datetime.now().date()}.  "
        f"\n\nTo curate your own github list, simply clone and change the input csv file.  "
        f"\n\nInspired by:  "
        f"\n[https://github.com/vinta/awesome-python](https://github.com/vinta/awesome-python)  "
        f"\n[https://github.com/trananhkma/fucking-awesome-python](https://github.com/trananhkma/fucking-awesome-python)  "
    )

    with open(output_file, "w") as out:
        out.write("\n".join(lines))


def main():
    token = env.get_env("GITHUB_ACCESS_TOKEN")
    input_file = "./data/GithubData.csv"
    output_file = "README.md"

    process(input_file, output_file, token)


if __name__ == "__main__":
    main()
