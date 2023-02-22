import time
import github
from loguru import logger
from typing import List
from joblib import Memory

memory = Memory(".joblib_cache")


class GithubWrapper:
    def __init__(self, token_list: List[str]):
        self.token_list = token_list
        self.token_index = 0

    def token(self) -> str:
        current_token = self.token_list[self.token_index]
        logger.trace(f"Using token index {self.token_index}")
        self.token_index += 1
        if self.token_index >= len(self.token_list):
            self.token_index = 0
        return current_token

    @staticmethod
    @memory.cache(ignore=["token"])
    def _get_repo_cached(token, name) -> github.Repository:
        gh = github.Github(token)
        if name.endswith("/"):
            logger.info(f"Removed trailing slash for: {name}")
            name = name.rstrip("/")

        logger.info(f"get_repo (not cached): [{name}]")
        try:
            repo = gh.get_repo(name)
        except Exception as ex:
            logger.warning(f"Exception for get_repo with name (will re-try once): {name} - {ex}")
            try:
                time.sleep(30)
                repo = gh.get_repo(name)
            except Exception as ex:
                logger.error(f"Exception for get_repo after re-try with name: {name} - {ex}")
                raise ex
        return repo

    def get_repo(self, name) -> github.Repository:
        return self._get_repo_cached(self.token(), name)

    # def get_org_repos(self, name) -> List[github.Repository.Repository]:
    #     logger.debug(f"get_org_repos: {name}")
    #     org = self.gh.get_organization(name)
    #     repos = []
    #     for repo in org.get_repos():
    #         repos.append(repo)
    #     return repos
    #
    # def get_organization(self, name) -> github.Organization.Organization:
    #     logger.debug(f"get_organization: {name}")
    #     return self.gh.get_organization(name)
    #
    # def search_github(self, keywords):
    #     query = "+".join(keywords) + "+in:readme+in:description"
    #     result = self.gh.search_repositories(query, "stars", "desc")
    #
    #     print(f"Found {result.totalCount} repo(s)")
    #     for repo in result:
    #         print(repo.clone_url)
