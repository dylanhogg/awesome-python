from library import log, env, crawler


def main():
    log.configure()

    # NOTE: csv location can be local file or google spreadsheet, for example:
    #       https://docs.google.com/spreadsheets/d/<your_doc_id>/export?gid=0&format=csv
    csv_location = env.get("CSV_LOCATION")
    token = env.get("GITHUB_ACCESS_TOKEN")
    output_csv_filename = "github_data.csv"
    output_json_filename = "github_data.json"

    crawler.write_files(csv_location, token, output_csv_filename, output_json_filename)


if __name__ == "__main__":
    main()
