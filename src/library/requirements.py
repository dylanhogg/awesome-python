import urllib.request
import urllib.error
from joblib import Memory
from pathlib import Path
from loguru import logger

memory = Memory(".joblib_cache")


def _save_content(repopath, branch, filename, content):
    folder = "data/"
    Path(folder).mkdir(parents=True, exist_ok=True)

    out_filename = folder + repopath.replace("/", "~") + f"~{filename}"
    with open(out_filename, "w") as f:
        f.write(content)

    logger.info(f"Saved file {out_filename}")


def _safe_get_url(repopath, branch, filename):
    try:
        url = f"https://raw.githubusercontent.com/{repopath}/{branch}/{filename}"
        resource = urllib.request.urlopen(url)
        # try:
        #     resource = urllib.request.urlopen(url)
        # except Exception as ex:
        #     # NOTE: this can be expected for many missing files (404)
        #     logger.warning(f"Exception for safe_get_url with repopath (will re-try once): {repopath}")
        #     try:
        #         time.sleep(30)
        #         resource = urllib.request.urlopen(url)
        #     except Exception as ex:
        #         logger.error(f"Exception for safe_get_url after re-try with repopath: {repopath}")
        #         raise ex
        charset = resource.headers.get_content_charset()
        return resource.read().decode(charset).strip()
    except urllib.error.HTTPError as ex:
        return ""


@memory.cache()
def get_requirements(repopath):
    filenames = [
        "requirements.txt",
        "setup.py",  # TODO: needs postprocessing for install_requires etc.
        "pyproject.toml",  # TODO: postprocessing for [build-system] / requires | [tool.poetry] / packages etc
        "Pipfile",  # TODO: postprocessing
    ]

    saved_filenames = []
    for branch in ["master"]:  # NOTE: master redirects to main as at Sep 2021.
        for filename in filenames:
            content = _safe_get_url(repopath, branch, filename)
            if len(content) > 0:
                _save_content(repopath, branch, filename, content)
                saved_filenames.append(filename)

    return saved_filenames  # TODO:  return list of tuples (repopath, branch, filename, local_filename)
