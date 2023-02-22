import sys
import pandas as pd
from datetime import datetime
from typing import List, Callable
from loguru import logger
from urllib.parse import urlparse
from library.ghw import GithubWrapper
from library.metrics import PopularityMetrics, StandardMetrics
from library.scorer import PopularityScorer


def get_input_data(csv_location: str) -> pd.DataFrame:
    df = pd.read_csv(csv_location)
    df.columns = map(str.lower, df.columns)
    assert "githuburl" in df.columns
    assert "category" in df.columns

    df["githuburl"] = df["githuburl"].apply(lambda x: x.lower())
    duplicated_githuburls = df[df.duplicated(subset=["githuburl"])]
    duplicated_count = len(duplicated_githuburls)
    if duplicated_count > 0:
        logger.warning(f"Duplicate githuburl values found in csv: {duplicated_count}\n{duplicated_githuburls}")
        logger.error(f"Fix up {duplicated_count} duplicates from {csv_location} and re-run.")
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
        f"[{homepage}]({homepage})  \n[{url}]({url})"
        if homepage is not None and len(homepage) > 0
        else f"[{url}]({url})"
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
    topics_display = "\n<sub><sup>" + ", ".join(sorted(topics)) + "</sup></sub>" if len(topics) > 0 else ""
    description = row["_description"]
    language = row["_language"]
    if language is not None and language.lower() != "python":
        logger.info(f"Is {name} really a Python library? Main language is {language}.")

    header = (
        f"[{name}]({url})"
        if name == organization
        else f"[{name}]({url}) by [{organization}](https://github.com/{organization})"
    )

    return (
        f"### {header}  "
        f"\n{description}  "
        f"\n{homepage_display}  "
        f"\n{stars_per_week} stars per week over {age_weeks} weeks  "
        f"\n{stars:,} stars, {forks:,} forks, {watches:,} watches  "
        f"\n{category_display}created {created}, last commit {last_commit_date}, main language {language}  "
        f"{topics_display}"
        f"\n\n"
    )


def _display_description(ghw, name) -> str:
    repo = ghw.get_repo(name)
    if repo.description is None:
        return f"{name}"
    else:
        assert repo.name is not None
        if (
            repo.description.lower().startswith(repo.name.lower())
            or f"{repo.name.lower()}:" in repo.description.lower()
        ):
            return f"{repo.description}"
        else:
            return f"{repo.name}: {repo.description}"


def _column_apply(df: pd.DataFrame, target: str, source: str, fn: Callable):
    logger.info(f"apply: {source} -> {target}")
    df[target] = df[source].apply(fn)


def process(df_input: pd.DataFrame, token_list: List[str]) -> pd.DataFrame:
    ghw = GithubWrapper(token_list)
    df = df_input.copy()

    # Add repo lookup columns ------------------------------------------------
    logger.info(f"Add repo lookup columns...")
    t0 = datetime.now()
    _column_apply(df, "_repopath", "githuburl", lambda x: urlparse(x).path.lstrip("/"))
    _column_apply(df, "_reponame", "_repopath", lambda x: ghw.get_repo(x).name)
    _column_apply(df, "_stars", "_repopath", lambda x: ghw.get_repo(x).stargazers_count)
    _column_apply(df, "_forks", "_repopath", lambda x: ghw.get_repo(x).forks_count)
    _column_apply(df, "_watches", "_repopath", lambda x: ghw.get_repo(x).subscribers_count)
    _column_apply(df, "_language", "_repopath", lambda x: ghw.get_repo(x).language)
    _column_apply(df, "_homepage", "_repopath", lambda x: ghw.get_repo(x).homepage)
    _column_apply(df, "_description", "_repopath", lambda x: _display_description(ghw, x))
    _column_apply(df, "_organization", "_repopath", lambda x: x.split("/")[0])
    _column_apply(df, "_updated_at", "_repopath", lambda x: ghw.get_repo(x).updated_at.date())
    _column_apply(df, "_created_at", "_repopath", lambda x: ghw.get_repo(x).created_at.date())
    _column_apply(df, "_age_weeks", "_repopath",
                  lambda x: (datetime.now().date() - ghw.get_repo(x).created_at.date()).days // 7
                  )
    _column_apply(df, "_stars_per_week", "_repopath",
                  lambda x: ghw.get_repo(x).stargazers_count
                  * 7
                  / (datetime.now().date() - ghw.get_repo(x).created_at.date()).days
                  )
    timing_lookup = datetime.now() - t0
    logger.info(f"Timing: {timing_lookup.total_seconds()=}")

    # Add standard metric columns ------------------------------------------------
    logger.info(f"Add popularity metric columns...")
    std_metrics = StandardMetrics()
    t0 = datetime.now()
    _column_apply(df, "_topics", "_repopath", lambda x: std_metrics.get_repo_topics(ghw, x))
    _column_apply(df, "_last_commit_date", "_repopath", lambda x: std_metrics.last_commit_date(ghw, x))
    timing_std = datetime.now() - t0
    logger.info(f"Timing: {timing_std.total_seconds()=}")

    # Add popularity metric columns ------------------------------------------------
    logger.info(f"Add popularity metric columns...")
    pop_metrics = PopularityMetrics()
    ghw = GithubWrapper(token_list)
    t0 = datetime.now()

    df["_pop_contributor_count"] = df["_repopath"].apply(
        lambda x: pop_metrics.contributor_count(ghw, x)
    )

    # TODO: optimise contributor_orgs_dict
    # Long, many interations thru repo contibutors for company info
    # Can rate limit, esp if sleep < 2sec
    df_pop_contributor_orgs_dict = df.apply(lambda row: pop_metrics.contributor_orgs_dict(ghw, row["_repopath"]),
                                            axis="columns", result_type="expand")
    df = pd.concat([df, df_pop_contributor_orgs_dict], axis="columns")

    df["_pop_commit_frequency"] = df["_repopath"].apply(
        lambda x: pop_metrics.commit_frequency(ghw, x)
    )

    df["_pop_updated_issues_count"] = df["_repopath"].apply(
        lambda x: pop_metrics.updated_issues_count(ghw, x)
    )

    df["_pop_closed_issues_count"] = df["_repopath"].apply(
        lambda x: pop_metrics.closed_issues_count(ghw, x)
    )

    df["_pop_created_since_days"] = df["_repopath"].apply(
        lambda x: pop_metrics.created_since_days(ghw, x)
    )

    df["_pop_updated_since_days"] = df["_repopath"].apply(
        lambda x: pop_metrics.updated_since_days(ghw, x)
    )

    df_pop_recent_releases_count = df.apply(lambda row: pop_metrics.recent_releases_count_dict(ghw, row["_repopath"]),
                                            axis="columns", result_type="expand")
    df = pd.concat([df, df_pop_recent_releases_count], axis="columns")

    df_pop_comment_frequency = df.apply(lambda row: pop_metrics.comment_frequency(ghw, row["_repopath"]),
                                        axis="columns", result_type="expand")
    df = pd.concat([df, df_pop_comment_frequency], axis="columns")

    timing_pop = datetime.now() - t0
    logger.info(f"Timing: {timing_lookup.total_seconds()=}")
    logger.info(f"Timing: {timing_std.total_seconds()=}")
    logger.info(f"Timing: {timing_pop.total_seconds()=}")

    # TEMP:
    logger.info(f"Write temp process results to csv...")
    df.to_csv("_temp_df_process.csv")
    import json
    with open("_temp_df_process.json", "w") as f:
        json_results = df.to_json(orient="table", double_precision=2)
        data = json.loads(json_results)
        json.dump(data, f, indent=4)
    # /TEMP:

    scorer = PopularityScorer()
    df["_pop_score"] = df.apply(
        lambda row: scorer.score(row), axis="columns"
    )

    return df.sort_values("_pop_score", ascending=False)


def lines_header(count: int, category: str = "") -> List[str]:
    category_line = f"A selection of {count} curated Python libraries and frameworks ordered by stars.  \n"
    if len(category) > 0:
        category_line = (
            f"A selection of {count} curated {category} Python libraries and frameworks ordered by stars.  \n"
        )

    return [
        f"# Crazy Awesome Python",
        category_line,
        f"Checkout the interactive version that you can filter and sort: ",
        f"[https://www.awesomepython.org/](https://www.awesomepython.org/)  \n\n",
    ]


def add_markdown(df: pd.DataFrame) -> pd.DataFrame:
    df["_doclines_main"] = df.apply(lambda x: make_markdown(x, include_category=True), axis=1)
    df["_doclines_child"] = df.apply(lambda x: make_markdown(x, include_category=False), axis=1)
    return df
