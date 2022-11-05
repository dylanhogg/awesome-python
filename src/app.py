from library import log, env, crawler, postprocess


def main():
    log.configure()

    # NOTE: csv location can be local file or google spreadsheet, for example:
    #       https://docs.google.com/spreadsheets/d/<your_doc_id>/export?gid=0&format=csv
    csv_location = env.get("CSV_LOCATION")
    token = env.get("GITHUB_ACCESS_TOKEN")
    github_csv_filename = "github_data.csv"
    github_json_filename = "github_data.json"
    crawler.write_files(csv_location, token, github_csv_filename, github_json_filename)

    github_tags_json_filename = "github_tags_data.json"
    postprocess.write_tags(github_json_filename, github_tags_json_filename, most_common=200)


def write_tags():
    github_json_filename = "github_data.json"
    github_tags_json_filename = "github_tags_data.json"
    postprocess.write_tags(github_json_filename, github_tags_json_filename, 200)


if __name__ == "__main__":
    # write_tags()
    main()
