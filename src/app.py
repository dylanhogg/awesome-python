import pandas as pd
from datetime import datetime
import library.env as env
from library.log import get_logger
from library.ghw import GithubWrapper
from urllib.parse import urlparse

logger = get_logger(__name__)


def get_input_data(csv_location) -> pd.DataFrame:
    df = pd.read_csv(csv_location)
    df.columns = map(str.lower, df.columns)
    assert "githuburl" in df.columns
    return df


def process(df_input, token):
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
    df["_created_at"] = df["_reponpath"].apply(
        lambda x: ghw.get_repo(x).created_at.date()
    )
    df = df.sort_values("_stars", ascending=False)

    def make_line(row):
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
            f"[{category}](categories/{category}.md) / "
            if category is not None and len(category) > 0
            else f"none / "
        )
        stars = row["_stars"]
        forks = row["_forks"]
        watches = row["_watches"]
        updated = row["_updated_at"]
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
            logger.info(
                f"Is {name} really a Python library? Main language is {language}."
            )

        return (
            f"### {category_display}[{name}]({url})  "
            f"{homepage_display}"
            f"\n{description}  "
            f"\n{stars:,} stars, {forks:,} forks, {watches:,} watches  "
            f"\ncreated {created}, updated {updated}, main language {language}  "
            f"{topics_display}"
            f"\n\n"
        )

    df["_doclines"] = df.apply(lambda x: make_line(x), axis=1)

    return df


def lines_header(count, category=""):
    return [
        "# Crazy Awesome Python",
        f"A selection of {count} {category} Python libraries and frameworks ordered by stars.  \n\n"
    ]


def write_files(csv_location, token):
    start = datetime.now()
    df_input = get_input_data(csv_location)

    logger.info(f"Processing {len(df_input)} records from {csv_location}")
    df_results = process(df_input, token)

    # Write raw results to csv
    df_results.drop("_doclines", axis=1).to_csv("github_data.csv")

    # Write all results to README.md
    lines_footer = [
        f"This file was automatically generated on {datetime.now().date()}.  "
        f"\n\nTo curate your own github list, simply clone and change the input csv file.  "
        f"\n\nInspired by:  "
        f"\n[https://github.com/vinta/awesome-python](https://github.com/vinta/awesome-python)  "
        f"\n[https://github.com/trananhkma/fucking-awesome-python](https://github.com/trananhkma/fucking-awesome-python)  "
    ]
    lines = []
    lines.extend(lines_header(len(df_results)))
    lines.extend(list(df_results["_doclines"]))
    lines.extend(lines_footer)

    logger.info(f"Writing {len(df_results)} entries to README.md...")
    with open("README.md", "w") as out:
        out.write("\n".join(lines))

    # Write to categories
    categories = df_results["category"].unique()
    for category in categories:
        df_category = df_results[df_results["category"] == category]
        lines = []
        lines.extend(lines_header(len(df_category), category))
        lines.extend(list(df_category["_doclines"]))
        lines.extend(lines_footer)
        filename = f"categories/{category}.md"
        logger.info(f"Writing {len(df_category)} entries to {filename}...")
        with open(filename, "w") as out:
            out.write("\n".join(lines))

    logger.info(f"Finished writing in {datetime.now() - start}")


def main():
    # NOTE: csv location can be local file or google spreadsheet, for example:
    #       https://docs.google.com/spreadsheets/d/<your_doc_id>/export?gid=0&format=csv
    csv_location = env.get_env("CSV_LOCATION")
    token = env.get_env("GITHUB_ACCESS_TOKEN")

    write_files(csv_location, token)


if __name__ == "__main__":
    main()
