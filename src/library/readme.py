import re
import markdown
import urllib.request
import urllib.error
import docutils.io
import docutils.core
from pathlib import Path
from loguru import logger


# https://stackoverflow.com/questions/47337009/rst2html-on-full-python-project
def rst2html_old(content):
    pub = docutils.core.Publisher(
        source_class=docutils.io.StringInput,
        destination_class=docutils.io.StringOutput)
    pub.set_components('standalone', 'restructuredtext', 'html')
    pub.process_programmatic_settings(None, None, None)
    pub.set_source(source=content)
    pub.publish()
    html = pub.writer.parts['whole']
    return html


# https://www.kite.com/python/docs/docutils.core.publish_parts
def rst2html(content):
    parts = docutils.core.publish_parts(content, writer_name="html")
    return parts["html_body"]


def save_content(repopath, branch, filename, content):
    folder = "data/"
    Path(folder).mkdir(parents=True, exist_ok=True)

    # Original file contents
    out_filename = folder + repopath.replace("/", "~") + f"~{filename}"
    with open(out_filename, "w") as f:
        f.write(content)

    # Link to readme file
    readme_url = f"https://github.com/{repopath}/blob/{branch}/{filename}"

    # Find pip installs
    pip_pattern = re.compile(r'(pip[3]*[ ]+install[ ]+[a-zA-Z0-9 \-_<>=\"\'.,+:/\[\]]+)')  # TODO: review
    pips = []
    for m in re.findall(pip_pattern, content):
        pips.append(m.strip())

    # TODO: analyse requrements.txt, setup.py etc

    # Get HTML from content
    if filename.lower().endswith(".rst"):
        html_content = rst2html(content)
    else:
        # TODO: how to handle non markdown/non rst?
        html_content = markdown.markdown(content)

    # TODO: fix relative images/links
    #  e.g. <img src="docs/img/logo.svg">
    #  to <img src="https://raw.githubusercontent.com/<repopath>/<branch>/docs/img/logo.svg"

    html = "<pre><code>"
    html = html + f"README: <a href='{readme_url}'>{readme_url}</a><br />"
    if len(pips) > 0:
        html = html + "<br />pip install(s):<br />" + "<br />".join(pips)
    html = html + "</code></pre><hr />" + html_content

    with open(f"{out_filename}.html", "w") as f:
        f.write(html)

    logger.info(f"Saved file {out_filename}")


def safe_get_url(repopath, branch, filename):
    try:
        url = f"https://raw.githubusercontent.com/{repopath}/{branch}/{filename}"
        resource = urllib.request.urlopen(url)
        charset = resource.headers.get_content_charset()
        return resource.read().decode(charset).strip()
    except urllib.error.HTTPError as ex:
        return ""


def get_readme(repopath):
    filenames = [
        "README.md",
        "README.rst",
        "README.txt",
        "readme.md",
        "readme.rst",
        "readme.txt"
    ]

    for branch in ["master", "main"]:
        for filename in filenames:
            content = safe_get_url(repopath, branch, filename)
            if len(content) > 0:
                save_content(repopath, branch, filename, content)
                return filename  # TODO:  return tuple (repopath, branch, filename, local_filename)

    # Did not locate any readme files in repo
    return ""
