from loguru import logger
from library import render, env


def test_render_process():
    token_delim = env.get("GITHUB_ACCESS_TOKEN")
    token_list = token_delim.split("|")

    csv_location = "tests/test_repos_source.csv"

    df_input = render.get_input_data(csv_location)
    test_repos = [
        "https://github.com/dask/dask",
        "https://github.com/dylanhogg/crazy-awesome-python",
        "https://github.com/klen/muffin",
    ]
    # test_repos = ["https://github.com/dylanhogg/crazy-awesome-python"]
    df_input = df_input[df_input["githuburl"].isin(test_repos)]
    assert len(df_input) == len(test_repos)
    df = render.process(df_input, token_list)

    logger.info(f"{df.shape=}")
    logger.info(f"{df.columns=}")

    for i in range(len(df)):
        logger.info(df.iloc[i])

    assert len(df) == len(df_input), "Unexpected number of rows"
    # logger.warning(f"{df.columns=}")
    assert list(df.columns) == [
        "category",
        "githuburl",
        "featured",
        "links",
        "description",
        "_repopath",
        "_reponame",
        "_stars",
        "_forks",
        "_watches",
        "_language",
        "_homepage",
        "_description",
        "_organization",
        "_updated_at",
        "_created_at",
        "_age_weeks",
        "_stars_per_week",
        "_topics",
        "_last_commit_date",
        "_pop_contributor_count",
        "_pop_contributor_orgs_len",
        "_pop_commit_frequency",
        "_pop_updated_issues_count",
        "_pop_closed_issues_count",
        "_pop_created_since_days",
        "_pop_updated_since_days",
        "_pop_recent_releases_count",
        "_pop_recent_releases_estimated_tags",
        "_pop_recent_releases_adjusted_count",
        "_pop_issue_count",
        "_pop_comment_count",
        "_pop_comment_count_lookback_days",
        "_pop_comment_frequency",
        "_pop_score",
    ]
