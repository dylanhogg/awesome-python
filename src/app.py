from loguru import logger
from pathlib import Path

from library import log, env, crawler, postprocess


def main():
    joblib_cache_exists = Path(".joblib_cache").exists()
    log.configure()
    if joblib_cache_exists:
        logger.warning(".joblib_cache folder exists, not all data will be fresh!")

    # NOTE: csv location can be local file or google spreadsheet, for example:
    #       https://docs.google.com/spreadsheets/d/<your_doc_id>/export?gid=0&format=csv
    csv_location = env.get("CSV_LOCATION")
    token_delim = env.get("GITHUB_ACCESS_TOKEN")
    token_list = token_delim.split("|")

    # Crawl and write files
    github_json_filename = "github_data.json"
    crawler.run(csv_location, token_list, "github_data.csv", github_json_filename)

    # Post-processing
    postprocess.write_tags(github_json_filename, "github_tags_data.json", most_common=200)
    postprocess.write_best_in_class_data(github_json_filename="github_top.json", sort_col="_pop_score")
    postprocess.write_best_in_class_data(github_json_filename="github_hot.json", sort_col="_stars_per_week")

    if joblib_cache_exists:
        logger.warning(".joblib_cache folder exists, not all data will be fresh!")


# def write_tags():
#     # TEMP: remove once tag filtering implemented
#     github_json_filename = "github_data.json"
#     github_tags_json_filename = "github_tags_data.json"
#     postprocess.write_tags(github_json_filename, github_tags_json_filename, most_common=200)


def write_best_in_class_data(github_json_filename: str, sort_col: str):
    # TEMP: remove once tag filtering implemented
    postprocess.write_best_in_class_data(github_json_filename, sort_col)


if __name__ == "__main__":
    # write_tags()
    # write_best_in_class_data(github_json_filename="github_top.json", sort_col="_pop_score")
    # write_best_in_class_data(github_json_filename="github_hot.json", sort_col="_stars_per_week")
    main()
