from loguru import logger
from library import render, env


def test_render_process():
    token = env.get("GITHUB_ACCESS_TOKEN")
    csv_location = "tests/test_repos_source.csv"

    df_input = render.get_input_data(csv_location)
    df_input = df_input.head(1)
    df = render.process(df_input, token)

    logger.info(f"{df.shape=}")
    logger.info(f"{df.columns=}")

    assert df.shape[0] == 1, "Unexpected number of rows"
    assert df.shape[1] == 20, "Unexpected number of columns"
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
        "_topics",
        "_language",
        "_homepage",
        "_description",
        "_organization",
        "_updated_at",
        "_last_commit_date",
        "_created_at",
        "_age_weeks",
        "_stars_per_week",
    ]
