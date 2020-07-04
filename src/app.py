import pandas as pd
from datetime import datetime
import library.env as env
from library.log import get_logger
from library.ghw import GithubWrapper
from urllib.parse import urlparse

logger = get_logger(__name__)


def get_input_data(csv_location):
    df = pd.read_csv(csv_location)
    df.columns = map(str.lower, df.columns)
    return df


def process(csv_location, output_file, token):
    ghw = GithubWrapper(token)

    df = get_input_data(csv_location)

    logger.info(f"Processing {len(df)} records to {output_file} from {csv_location}")

    df["_reponpath"] = df["githuburl"].apply(lambda x: urlparse(x).path.lstrip("/"))
    df["_repo"] = df["_reponpath"].apply(lambda x: ghw.get_repo(x))
    df["_repo_name"] = df["_reponpath"].apply(lambda x: ghw.get_repo(x).name)
    df["_stars"] = df["_reponpath"].apply(lambda x: ghw.get_repo(x).stargazers_count)
    df["_topics"] = df["_reponpath"].apply(lambda x: ghw.get_repo(x).get_topics())
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
            f"### [{name}]({url})  "
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
        f"This file was automatically generated on {datetime.now().date()}.  "
        f"\n\nTo curate your own github list, simply clone and change the input csv file.  "
        f"\n\nInspired by:  "
        f"\n[https://github.com/vinta/awesome-python](https://github.com/vinta/awesome-python)  "
        f"\n[https://github.com/trananhkma/fucking-awesome-python](https://github.com/trananhkma/fucking-awesome-python)  "
    )

    with open(output_file, "w") as out:
        out.write("\n".join(lines))

    logger.info(f"Finished writing to {output_file}")


def main():
    # NOTE: csv location can be local file or google spreadsheet, for example:
    #       https://docs.google.com/spreadsheets/d/<your_doc_id>/export?gid=0&format=csv
    csv_location = env.get_env("CSV_LOCATION")
    token = env.get_env("GITHUB_ACCESS_TOKEN")
    output_file = "README.md"
    process(csv_location, output_file, token)


if __name__ == "__main__":
    main()
