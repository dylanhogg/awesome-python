import arxiv
import re
from loguru import logger
from pathlib import Path
from joblib import Memory

memory = Memory(".joblib_cache")


@memory.cache()
def _search_arxiv_ids(ids: list[str]) -> list[arxiv.arxiv.Result]:
    return list(arxiv.Search(id_list=ids).results())


def get_arxiv_links(row) -> list[list[str, str, str]]:
    def _get_entry_id(arxiv_paper) -> str:
        short_id = arxiv_paper.get_short_id()
        find = re.findall(r"(.*)v\d", short_id)  # Cut off vN suffix
        return short_id if len(find) == 0 else find[0]

    def _get_safe_title(arxiv_paper):
        safe_title = " ".join(str.strip(x) for x in arxiv_paper.title.split("\n"))
        return safe_title.replace("  ", " ").replace("\"", "").replace("'", "")

    def _get_author_et_al(arxiv_paper):
        authors = arxiv_paper.authors
        return authors[0] if len(authors) == 1 else f"{authors[0]} et al"

    # Get Arxiv paper_ids as a list, from customarxiv override column, or readme html
    paper_ids = []
    if row["customarxiv"]:
        # TODO: consider breaking up customarxiv from readmearxiv, include both and render _arxiv_links from these?
        paper_ids = row["customarxiv"]
        logger.info(f"Searching arxiv for {len(paper_ids)} paper_ids from 'customarxiv' field")
    else:
        readme_localurl = row["_readme_localurl"]
        file = Path(f"./data/{readme_localurl}.html")
        logger.info(f"Searching arxiv for {len(paper_ids)} paper_ids from {readme_localurl}")
        if file.is_file():
            with open(file, "r") as f:
                text = f.read()
                paper_ids_abs = re.findall(r'https?://arxiv\.org/abs/(\d{4}\.\d{4,5})', text)
                paper_ids_pdf = re.findall(r'https?://arxiv\.org/pdf/(\d{4}\.\d{4,5})v?\d?.pdf', text)
                paper_ids = list(dict.fromkeys(paper_ids_abs + paper_ids_pdf))  # Remove duplicates from a list, while preserving order

    # Process paper_ids and get metadata from arxiv.org search
    results = []
    if paper_ids:
        arxiv_papers = _search_arxiv_ids(paper_ids)
        results = [[_get_entry_id(x), _get_safe_title(x), _get_author_et_al(x)] for x in arxiv_papers]
    return results


def get_pypi_links(row) -> list[str]:
    readme_localurl = row["_readme_localurl"]
    pypi_links = []
    file = Path(f"./data/{readme_localurl}.html")
    if file.is_file():
        with open(file, "r") as f:
            text = f.read()
            pypi_links = re.findall(r'https?://pypi\.org/project/[A-Za-z0-9_-]+/', text)  # TODO: review regex, esp trailing / & compile it!
            pypi_links = list(dict.fromkeys(pypi_links))  # Remove duplicates from a list, while preserving order
    return pypi_links
