import sys
import pandas as pd
from datetime import datetime
from typing import List
from loguru import logger
from urllib.parse import urlparse
from library.ghw import GithubWrapper


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
    organization = row["_organization"]
    homepage = row["_homepage"]
    homepage_display = (
        f"\n[{homepage}]({homepage})  "
        if homepage is not None and len(homepage) > 0
        else f"\n[{url}]({url})  "
    )
    category = row["category"]
    category_display = (
        f"[{category}](categories/{category}.md) category, "
        if include_category and category is not None and len(category) > 0
        else ""
    )
    stars = row["_stars"]
    stars_per_week = row["_stars_per_week"]
    stars_per_week = round(stars_per_week, 2) if stars_per_week < 10 else int(stars_per_week)
    age_weeks = row["_age_weeks"]
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

    header = f"[{name}]({url})" \
        if name == organization \
        else f"[{name}]({url}) by [{organization}](https://github.com/{organization})"

    return (
        f"### {header}  "
        f"{homepage_display}"
        f"\n{description}  "
        f"\n{stars_per_week} stars per week over {age_weeks} weeks  "
        f"\n{stars} stars, {forks} forks, {watches} watches  "
        f"\n{category_display}created {created}, last commit {last_commit_date}, main language {language}  "
        f"{topics_display}"
        f"\n\n"
    )


def process(df_input, token) -> pd.DataFrame:
    ghw = GithubWrapper(token)
    df = df_input.copy()
    df["_repopath"] = df["githuburl"].apply(lambda x: urlparse(x).path.lstrip("/"))
    df["_reponame"] = df["_repopath"].apply(lambda x: ghw.get_repo(x).name)
    df["_stars"] = df["_repopath"].apply(lambda x: ghw.get_repo(x).stargazers_count)
    df["_forks"] = df["_repopath"].apply(lambda x: ghw.get_repo(x).forks_count)
    df["_watches"] = df["_repopath"].apply(lambda x: ghw.get_repo(x).subscribers_count)
    df["_topics"] = df["_repopath"].apply(lambda x: ghw.get_repo(x).get_topics())
    df["_language"] = df["_repopath"].apply(lambda x: ghw.get_repo(x).language)
    df["_homepage"] = df["_repopath"].apply(lambda x: ghw.get_repo(x).homepage)
    df["_description"] = df["_repopath"].apply(lambda x: ghw.get_repo(x).description)
    df["_organization"] = df["_repopath"].apply(
        lambda x: x.split("/")[0]
    )
    df["_updated_at"] = df["_repopath"].apply(
        lambda x: ghw.get_repo(x).updated_at.date()
    )
    df["_last_commit_date"] = df["_repopath"].apply(
        # E.g. Sat, 18 Jul 2020 17:14:09 GMT
        lambda x: datetime.strptime(
            ghw.get_repo(x).get_commits().get_page(0)[0].last_modified,
            "%a, %d %b %Y %H:%M:%S %Z",
        ).date()
    )
    df["_created_at"] = df["_repopath"].apply(
        lambda x: ghw.get_repo(x).created_at.date()
    )
    df["_age_weeks"] = df["_repopath"].apply(
        lambda x: (datetime.now().date() - ghw.get_repo(x).created_at.date()).days // 7
    )
    df["_stars_per_week"] = df["_repopath"].apply(
        lambda x: ghw.get_repo(x).stargazers_count * 7 / (datetime.now().date() - ghw.get_repo(x).created_at.date()).days
    )

    return df.sort_values("_stars", ascending=False)


def lines_header(count, category="") -> List[str]:
    return [
        f"# Crazy Awesome Python",
        f"A selection of {count} {category} Python libraries and frameworks ordered by stars.  \n",
        f"Checkout the interactive version that you can filter and sort: ",
        f"[http://awesome-python.infocruncher.com/](http://awesome-python.infocruncher.com/)  \n\n",
    ]


def add_markdown(df) -> pd.DataFrame:
    df["_doclines_main"] = df.apply(
        lambda x: make_markdown(x, include_category=True), axis=1
    )
    df["_doclines_child"] = df.apply(
        lambda x: make_markdown(x, include_category=False), axis=1
    )
    return df
