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
    df["_org"] = df["_reponame"].apply(lambda x: ghw.get_repo(x).organization)
    df["_stars"] = df["_reponame"].apply(lambda x: ghw.get_repo(x).stargazers_count)

    df = df.sort_values("_stars", ascending=False)

    def make_line(row):
        repo = row["_repo"]
        url = row["githuburl"]
        name = repo.name
        org = (
            f"[{repo.organization.login}]({repo.organization.blog})"
            if repo.organization is not None
            else "none"
        )
        stars = repo.stargazers_count
        forks = repo.forks_count
        watches = repo.subscribers_count
        updated = repo.updated_at.date()
        topics = ", ".join(sorted(repo.get_topics()))
        description = repo.description

        return (
            f"[{name}]({url})  "
            f"\n{description}  "
            f"\nstars: {stars:,} forks: {forks:,} watches: {watches:,} org: {org}, updated: {updated}  "
            f"\n<sub><sup>topics: {topics}</sup></sub>"
            f"\n\n"
        )

    df["_doclines"] = df.apply(lambda x: make_line(x), axis=1)

    lines = [
        "# Crazy Awesome Python",
        "Some hand curated python libraries and frameworks, "
        "with a focus on the data/machine learning space ordered by stars.\n\n",
    ]
    lines.extend(list(df["_doclines"]))
    lines.append(f"Automatically generated on {datetime.now().date()}\n")

    with open(output_file, "w") as out:
        out.write("\n".join(lines))


def main():
    token = env.get_env("GITHUB_ACCESS_TOKEN")
    input_file = "./data/Crazy Awesome Python - Sheet1.csv"
    output_file = "CRAZY_AWESOME_PYTHON.md"

    process(input_file, output_file, token)


if __name__ == "__main__":
    main()
