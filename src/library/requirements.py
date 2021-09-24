import urllib.request
import urllib.error
from pathlib import Path
from loguru import logger


def save_content(repopath, branch, filename, content):
    folder = "data/"
    Path(folder).mkdir(parents=True, exist_ok=True)

    out_filename = folder + repopath.replace("/", "-") + f"-{filename}"
    with open(out_filename, "w") as f:
        f.write(content)

    logger.info(f"Saved file {out_filename}")


def safe_get_url(repopath, branch, filename):
    try:
        url = f"https://raw.githubusercontent.com/{repopath}/{branch}/{filename}"
        resource = urllib.request.urlopen(url)
        charset = resource.headers.get_content_charset()
        return resource.read().decode(charset).strip()
    except urllib.error.HTTPError as ex:
        return ""


def get_requirements(repopath):
    filenames = [
        "requirements.txt",
        # "setup.py"  # TODO: needs postprocessing for install_requires etc.
        # toml?
    ]

    for branch in ["master", "main"]:
        for filename in filenames:
            content = safe_get_url(repopath, branch, filename)
            if len(content) > 0:
                save_content(repopath, branch, filename, content)
                return filename  # TODO:  return tuple (repopath, branch, filename, local_filename)

    # Did not locate any readme files in repo
    return ""
