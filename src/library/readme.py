import re
import markdown
import html2text
import urllib.request
import urllib.error
from bs4 import BeautifulSoup
from pathlib import Path
from loguru import logger


def safe_get_url(repopath, filename):
    Path("data/").mkdir(parents=True, exist_ok=True)

    try:
        url = f"https://raw.githubusercontent.com/{repopath}/master/{filename}"
        resource = urllib.request.urlopen(url)
        charset = resource.headers.get_content_charset()
        content = resource.read().decode(charset).strip()
        # content_no_nl = content.replace("\n", "\\n").replace("\r", "\\r")

        out_filename = "data/" + repopath.replace("/", "-") + f"-{filename}"
        with open(out_filename, "w") as f:
            f.write(content)

        html = markdown.markdown(content)
        # html = re.sub(r'<pre>(.*?)</pre>', ' ', html)
        # html = re.sub(r'<code>(.*?)</code >', ' ', html)
        # html = re.sub("(<!--.*?-->)", "", html, flags=re.DOTALL)
        out_html_filename = "data/" + repopath.replace("/", "-") + f"-{filename}.html"
        with open(out_html_filename, "w") as f:
            f.write(html)

        # soup = BeautifulSoup(html, features="html.parser")
        soup = BeautifulSoup(content, features="html.parser")   # Best? MD content, strip any html
        # for element in soup:
        #     element.extract()
        # txt = "".join(BeautifulSoup(html_content, features="html.parser").findAll(text=True))
        txt = soup.get_text()
        txt = re.sub(r'\n{2,}', '\n\n', txt)
        txt = txt.strip()
        out_txt_filename = "data/" + repopath.replace("/", "-") + f"-{filename}.txt"
        with open(out_txt_filename, "w") as f:
            f.write(txt)

        h = html2text.HTML2Text()
        h.ignore_links = True
        txt_output2 = h.handle(html)
        txt_output2 = txt_output2.strip()
        out_txt_filename2 = "data/" + repopath.replace("/", "-") + f"-{filename}.2.txt"
        with open(out_txt_filename2, "w") as f:
            f.write(txt_output2)

        # return url + ": " + content
        logger.info(f"Saved file {out_filename}")
        return filename
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
    for f in filenames:
        content = safe_get_url(repopath, f)
        if len(content) > 0:
            return content

    return ""
