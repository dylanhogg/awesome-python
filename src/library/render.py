import sys
import pandas as pd
from datetime import datetime
from typing import List
from library.log import get_logger
from library.ghw import GithubWrapper
from urllib.parse import urlparse

logger = get_logger(__name__)


def get_input_data(csv_location) -> pd.DataFrame:
    df = pd.read_csv(csv_location)
    df.columns = map(str.lower, df.columns)
    assert "githuburl" in df.columns
    assert "category" in df.columns

    duplicated_githuburls = df[df.duplicated(subset=["githuburl"])]
    duplicated_count = len(duplicated_githuburls)
    if duplicated_count > 0:
        logger.warning(
            f"Duplicate githuburl values found in csv: {duplicated_count}\n{duplicated_githuburls}"
        )
        logger.fatal(f"Fix up {duplicated_count} duplicates from {csv_location} and re-run.")
        sys.exit()
    else:
        logger.info("No duplicate githuburl values found in csv :)")

    return df


def make_markdown(row, include_category=False) -> str:
    url = row["githuburl"]
    name = row["_reponame"]
    homepage = row["_homepage"]
    homepage_display = (
        f"\n[{homepage}]({homepage})  "
        if homepage is not None and len(homepage) > 0
        else f"\n[{url}]({url})  "
    )
    category = row["category"]
    category_display = (
        f"category [{category}](categories/{category}.md), "
        if include_category and category is not None and len(category) > 0
        else ""
    )
    stars = row["_stars"]
    forks = row["_forks"]
    watches = row["_watches"]
    updated = row["_updated_at"]
    last_commit_date = row["_last_commit_date"]
    created = row["_created_at"]
    topics = row["_topics"]
    topics_display = (
        "\n<sub><sup>" + ", ".join(sorted(topics)) + "</sup></sub>"
        if len(topics) > 0
        else ""
    )
    description = row["_description"]
    language = row["_language"]
    if language is not None and language.lower() != "python":
        logger.info(f"Is {name} really a Python library? Main language is {language}.")

    return (
        f"### [{name}]({url})  "
        f"{homepage_display}"
        f"\n{description}  "
        f"\n{stars:,} stars, {forks:,} forks, {watches:,} watches  "
        f"\n{category_display}created {created}, last commit {last_commit_date}, main language {language}  "
        f"{topics_display}"
        f"\n\n"
    )


def process(df_input, token) -> pd.DataFrame:
    ghw = GithubWrapper(token)
    df = df_input.copy()
    df["_reponpath"] = df["githuburl"].apply(lambda x: urlparse(x).path.lstrip("/"))
    df["_reponame"] = df["_reponpath"].apply(lambda x: ghw.get_repo(x).name)
    df["_stars"] = df["_reponpath"].apply(lambda x: ghw.get_repo(x).stargazers_count)
    df["_forks"] = df["_reponpath"].apply(lambda x: ghw.get_repo(x).forks_count)
    df["_watches"] = df["_reponpath"].apply(lambda x: ghw.get_repo(x).subscribers_count)
    df["_topics"] = df["_reponpath"].apply(lambda x: ghw.get_repo(x).get_topics())
    df["_language"] = df["_reponpath"].apply(lambda x: ghw.get_repo(x).language)
    df["_homepage"] = df["_reponpath"].apply(lambda x: ghw.get_repo(x).homepage)
    df["_description"] = df["_reponpath"].apply(lambda x: ghw.get_repo(x).description)
    df["_updated_at"] = df["_reponpath"].apply(
        lambda x: ghw.get_repo(x).updated_at.date()
    )
    df["_last_commit_date"] = df["_reponpath"].apply(
        # E.g. Sat, 18 Jul 2020 17:14:09 GMT
        lambda x: datetime.strptime(
            ghw.get_repo(x).get_commits().get_page(0)[0].last_modified,
            "%a, %d %b %Y %H:%M:%S %Z",
        ).date()
    )
    df["_created_at"] = df["_reponpath"].apply(
        lambda x: ghw.get_repo(x).created_at.date()
    )
    return df.sort_values("_stars", ascending=False)


def lines_header(count, category="") -> List[str]:
    return [
        "# Crazy Awesome Python",
        f"A selection of {count} {category} Python libraries and frameworks ordered by stars.  \n\n",
    ]


def add_markdown(df) -> pd.DataFrame:
    df["_doclines_main"] = df.apply(
        lambda x: make_markdown(x, include_category=True), axis=1
    )
    df["_doclines_child"] = df.apply(
        lambda x: make_markdown(x, include_category=False), axis=1
    )
    return df
