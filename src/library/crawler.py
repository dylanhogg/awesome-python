import json
import pandas as pd
from datetime import datetime
from typing import List
from loguru import logger
from library import render, readme, requirements


def _crawl_external_files(df_input: pd.DataFrame):
    df = df_input.copy()

    logger.info("Crawling readme files...")
    df["_readme_filename"] = df["_repopath"].apply(lambda x: readme.get_readme(x))

    # TODO: handle 'main' master branches also:
    df["_readme_giturl"] = df.apply(
        lambda row: f"https://raw.githubusercontent.com/{row['_repopath']}/master/{row['_readme_filename']}"
        if len(row["_readme_filename"]) > 0
        else "",
        axis=1,
    )

    # TODO: get from readme.get_readme above as tuple and zip as per
    #       https://stackoverflow.com/questions/16236684/apply-pandas-function-to-column-to-create-multiple-new-columns
    df["_readme_localurl"] = df.apply(
        lambda row: f"{row['_repopath'].replace('/', '~')}~{row['_readme_filename']}"
        if len(row["_readme_filename"]) > 0
        else "",
        axis=1,
    )

    logger.info("Crawling requirements files...")
    df["_requirements_filenames"] = df["_repopath"].apply(lambda x: requirements.get_requirements(x))

    # TODO: handle 'main' master branches also:
    df["_requirements_giturls"] = df.apply(
        lambda row: list(
            map(
                lambda x: f"https://raw.githubusercontent.com/{row['_repopath']}/master/{x}",
                row["_requirements_filenames"],
            )
        ),
        axis=1,
    )

    # TODO: get from readme.get_readme above as tuple and zip as per
    #       https://stackoverflow.com/questions/16236684/apply-pandas-function-to-column-to-create-multiple-new-columns
    df["_requirements_localurls"] = df.apply(
        lambda row: list(map(lambda x: f"{row['_repopath'].replace('/', '~')}~{x}", row["_requirements_filenames"])),
        axis=1,
    )

    # TODO: parse crawled df["_readme_localurl"] files and extract: pypi links & arxiv links
    # TEMP: move this to a new function:
    import re
    from pathlib import Path

    def _get_arxiv_links(readme_localurl: str) -> list[str]:
        arxiv_links = []
        file = Path(f"./data/{readme_localurl}.html")
        if file.is_file():
            with open(file, "r") as f:
                text = f.read()
                arxiv_links = re.findall(r'https?://arxiv\.org/abs/(\d{4}\.\d{4,5})', text)  # TODO: review regex & compile it!
                arxiv_links = list(dict.fromkeys(arxiv_links))  # Remove duplicates from a list, while preserving order
        return arxiv_links
    # TODO: Remove https://arxiv.org/abs/ prefix from arxiv_links
    df["_arxiv_links"] = df.apply(lambda row: _get_arxiv_links(row["_readme_localurl"]), axis=1)

    def _get_pypi_links(readme_localurl: str) -> list[str]:
        pypi_links = []
        file = Path(f"./data/{readme_localurl}.html")
        if file.is_file():
            with open(file, "r") as f:
                text = f.read()
                pypi_links = re.findall(r'https?://pypi\.org/project/\w+/', text)  # TODO: review regex, esp trailing / & compile it!
                pypi_links = list(dict.fromkeys(pypi_links))  # Remove duplicates from a list, while preserving order
        return pypi_links
    df["_pypi_links"] = df.apply(lambda row: _get_pypi_links(row["_readme_localurl"]), axis=1)

    # TODO: https://huggingface.co/spaces/* e.g. https://huggingface.co/spaces/OFA-Sys/OFA-Visual_Question_Answering
    # TODO: https://wandb.ai/* e.g. https://wandb.ai/eleutherai/neox

    return df


def _save_json_data_files(df: pd.DataFrame,
                          output_json_filename: str,
                          max_ui_topics: int = 4,
                          max_ui_sim: int = 3):

    # Write raw results to json table format (i.e. github_data.json)
    with open(output_json_filename, "w") as f:
        json_results = df.to_json(orient="table", double_precision=2)
        data = json.loads(json_results)
        json.dump(data, f, indent=4)

    # Write raw results to minimised json (i.e. github_data.min.json)
    output_minjson_filename = (
        output_json_filename.replace(".json", ".min.json")
        if ".json" in output_json_filename
        else output_json_filename + ".min.json"
    )
    with open(output_minjson_filename, "w") as f:
        json_results = df.to_json(orient="table", double_precision=2)
        data = json.loads(json_results)
        json.dump(data, f, separators=(",", ":"))

    # Write UI results to minimised json (i.e. github_data.ui.min.json)
    output_ui_minjson_filename = (
        output_json_filename.replace(".json", ".ui.min.json")
        if ".json" in output_json_filename
        else output_json_filename + ".ui.min.json"
    )
    with open(output_ui_minjson_filename, "w") as f:
        # NOTE: this cols list must be synced with app.js DataTable columns for display
        cols = [
            "githuburl",
            "_reponame",
            "_organization",
            "_homepage",
            "_pop_score",
            "_stars",
            "_stars_per_week",
            "_description",
            "_age_weeks",
            "category",
            "_topics",
            "sim",
            "_readme_localurl",
            # TODO: review and trim:
            "_arxiv_links",
            "_pypi_links",
        ]

        df_min_ui = df[cols].copy()
        # NOTE: max_ui_topics & max_ui_sim values impact capability on Javascript UI side!
        df_min_ui["_topics"] = df_min_ui["_topics"].apply(lambda x: x[0:max_ui_topics])
        df_min_ui["sim"] = df_min_ui["sim"].apply(lambda x: x[0:max_ui_sim])
        json_results = df_min_ui[cols].to_json(orient="table", double_precision=2, index=False)
        data = json.loads(json_results)
        json.dump(data, f, separators=(",", ":"))

    # Write raw results to pickle
    output_pickle_filename = (
        output_json_filename.replace(".json", ".pkl")
        if ".json" in output_json_filename
        else output_json_filename + ".pkl"
    )
    df.to_pickle(output_pickle_filename)


def _write_local_markdown_files(df: pd.DataFrame):
    # Add markdown columns for local README.md and categories/*.md file lists.
    logger.info(f"Add markdown columns...")
    df = render.add_markdown(df)

    # Write all results to README.md
    lines_footer = [
        f"This file was automatically generated on {datetime.now().date()}.  "
        f"\n\nTo curate your own github list, simply clone and change the input csv file.  "
        f"\n\nInspired by:  "
        f"\n[https://github.com/vinta/awesome-python](https://github.com/vinta/awesome-python)  "
        f"\n[https://github.com/trananhkma/fucking-awesome-python](https://github.com/trananhkma/fucking-awesome-python)  "
    ]
    lines = []
    lines.extend(render.lines_header(len(df)))
    lines.extend(list(df["_doclines_main"]))
    lines.extend(lines_footer)

    logger.info(f"Writing {len(df)} entries to README.md...")
    with open("README.md", "w") as out:
        out.write("\n".join(lines))

    # Write to categories file
    categories = df["category"].unique()
    for category in categories:
        df_category = df[df["category"] == category]
        lines = []
        lines.extend(render.lines_header(len(df_category), category))
        lines.extend(list(df_category["_doclines_child"]))
        lines.extend(lines_footer)
        filename = f"categories/{category}.md"
        logger.info(f"Writing {len(df_category)} entries to {filename}...")
        with open(filename, "w") as out:
            out.write("\n".join(lines))


def run(
    csv_location: str,
    token_list: List[str],
    output_csv_filename: str,
    output_json_filename: str
):
    start = datetime.now()

    # Read GitHub urls from google docs
    df_input = render.get_input_data(csv_location)
    # df_input = df_input.head(3)  # Testing
    # df_input = df_input.iloc[9:11]  # Testing

    # Augment repo name with metadata from GitHub
    logger.info(f"Processing {len(df_input)} records from {csv_location} with {len(token_list)=}...")
    df = render.process(df_input, token_list)

    # Write raw results to csv (without external file info)
    logger.info(f"Write raw results to csv...")
    df.to_csv(output_csv_filename)

    # Crawl and save external files (e.g. readme.md and requirements.txt files)
    df = _crawl_external_files(df)

    # Write results to various json files
    _save_json_data_files(df, output_json_filename)

    # Write local markdown files for GitHub viewing
    _write_local_markdown_files(df)

    logger.info(f"Finished writing in {datetime.now() - start}")
