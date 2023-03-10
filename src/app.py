from library import log, env, crawler, postprocess


def main():
    log.configure()

    # NOTE: csv location can be local file or google spreadsheet, for example:
    #       https://docs.google.com/spreadsheets/d/<your_doc_id>/export?gid=0&format=csv
    csv_location = env.get("CSV_LOCATION")
    token_delim = env.get("GITHUB_ACCESS_TOKEN")
    token_list = token_delim.split("|")
    github_csv_filename = "github_data.csv"
    github_json_filename = "github_data.json"
    crawler.write_files(csv_location, token_list, github_csv_filename, github_json_filename)

    github_tags_json_filename = "github_tags_data.json"
    postprocess.write_tags(github_json_filename, github_tags_json_filename, most_common=200)
    postprocess.write_best_in_class_data(github_json_filename="github_top.json", sort_col="_pop_score")
    postprocess.write_best_in_class_data(github_json_filename="github_hot.json", sort_col="_stars_per_week")


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
