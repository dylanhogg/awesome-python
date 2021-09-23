import json
from datetime import datetime
from loguru import logger
from library import log, env, render


def write_files(csv_location, token, output_csv_filename, output_json_filename):
    start = datetime.now()

    # Read github urls from google docs
    df_input = render.get_input_data(csv_location)
    df_input = df_input.head(2)  # Testing

    # Augment repo name with metadata from Github
    logger.info(f"Processing {len(df_input)} records from {csv_location}")
    df_results = render.process(df_input, token)

    # Write raw results to csv
    logger.info(f"Write raw results to csv...")
    df_results.to_csv(output_csv_filename)

    # Write raw results to json table format
    with open(output_json_filename, "w") as f:
        json_results = df_results.to_json(orient="table")
        data = json.loads(json_results)
        json.dump(data, f, indent=4)

    # Add markdown columns
    logger.info(f"Add markdown columns...")
    df_results = render.add_markdown(df_results)

    # Write all results to README.md
    lines_footer = [
        f"This file was automatically generated on {datetime.now().date()}.  "
        f"\n\nTo curate your own github list, simply clone and change the input csv file.  "
        f"\n\nInspired by:  "
        f"\n[https://github.com/vinta/awesome-python](https://github.com/vinta/awesome-python)  "
        f"\n[https://github.com/trananhkma/fucking-awesome-python](https://github.com/trananhkma/fucking-awesome-python)  "
    ]
    lines = []
    lines.extend(render.lines_header(len(df_results)))
    lines.extend(list(df_results["_doclines_main"]))
    lines.extend(lines_footer)

    logger.info(f"Writing {len(df_results)} entries to README.md...")
    with open("README.md", "w") as out:
        out.write("\n".join(lines))

    # Write to categories
    categories = df_results["category"].unique()
    for category in categories:
        df_category = df_results[df_results["category"] == category]
        lines = []
        lines.extend(render.lines_header(len(df_category), category))
        lines.extend(list(df_category["_doclines_child"]))
        lines.extend(lines_footer)
        filename = f"categories/{category}.md"
        logger.info(f"Writing {len(df_category)} entries to {filename}...")
        with open(filename, "w") as out:
            out.write("\n".join(lines))

    logger.info(f"Finished writing in {datetime.now() - start}")


def main():
    log.configure()

    # NOTE: csv location can be local file or google spreadsheet, for example:
    #       https://docs.google.com/spreadsheets/d/<your_doc_id>/export?gid=0&format=csv
    csv_location = env.get("CSV_LOCATION")
    token = env.get("GITHUB_ACCESS_TOKEN")
    output_csv_filename = "github_data.csv"
    output_json_filename = "github_data.json"

    write_files(csv_location, token, output_csv_filename, output_json_filename)


if __name__ == "__main__":
    main()
