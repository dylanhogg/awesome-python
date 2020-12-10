import github
from loguru import logger
from typing import List


class GithubWrapper:
    def __init__(self, token):
        self.gh = github.Github(token)
        self.cache = {}

    def get_repo(self, name, use_cache=True) -> github.Repository:
        if name.endswith("/"):
            logger.warning(f"Repo needs to be fixed by removing trailing slash in source csv: {name}")
        key = f"repo_{name}"
        cached = self.cache.get(key, None)
        if cached is None or not use_cache:
            logger.info(f"get_repo: [{name}]")
            self.cache[key] = self.gh.get_repo(name)
            return self.cache[key]
        else:
            logger.info(f"get_repo: [{name}] (cached)")
            return cached

    def get_org_repos(self, name) -> List[github.Repository.Repository]:
        logger.debug(f"get_org_repos: {name}")
        org = self.gh.get_organization(name)
        repos = []
        for repo in org.get_repos():
            repos.append(repo)
        return repos

    def get_organization(self, name) -> github.Organization.Organization:
        logger.debug(f"get_organization: {name}")
        return self.gh.get_organization(name)

    def search_github(self, keywords):
        query = "+".join(keywords) + "+in:readme+in:description"
        result = self.gh.search_repositories(query, "stars", "desc")

        print(f"Found {result.totalCount} repo(s)")
        for repo in result:
            print(repo.clone_url)
