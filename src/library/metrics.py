import requests
import time
import re
from datetime import datetime, timedelta
from loguru import logger
from library.ghw import GithubWrapper
from joblib import Memory

memory = Memory(".joblib_cache")


class PopularityMetrics:
    """
    Inspired by https://github.com/ossf/criticality_score/tree/v1.0.7 (Latest Python version before Go port)
    and https://github.com/ossf/criticality_score/blob/main/Quantifying_criticality_algorithm.pdf
    """

    @staticmethod
    @memory.cache
    def contributor_count(ghw: GithubWrapper, name: str) -> int:
        repo = ghw.get_repo(name)
        try:
            return repo.get_contributors(anon='true').totalCount
        except Exception as ex:
            # Typically a large number of contributors
            logger.warning(f"get_contributors exception: {ex}")
            return 5000

    @staticmethod
    @memory.cache
    def contributor_orgs(ghw: GithubWrapper, name: str) -> dict:
        MAX_CONTRIBUTOR_COUNT = 20  # 10
        repo = ghw.get_repo(name)

        def _filter_name(org_name):
            return org_name.lower().replace(', inc.', '').replace('inc.', '')\
                .replace('llc', '').replace('@', '').replace(' ', '').rstrip(',')

        orgs = set()
        contributors = repo.get_contributors()[:MAX_CONTRIBUTOR_COUNT]
        try:
            # NOTE: Can be expensive if not capped due to `contributor.company` being an API call
            for contributor in contributors:
                # TODO: cache contributor -> company ??
                contributor_company = contributor.company
                if contributor_company:
                    filtered_contributor_company = _filter_name(contributor_company)
                    logger.info(f"{name=}, {contributor.login=}, {filtered_contributor_company=}")
                    orgs.add(filtered_contributor_company)
        except Exception as ex:
            # Typically a large number of contributors
            logger.warning(f"get_contributors exception: {ex}")
            return {
                "_pop_contributor_orgs_len": -1,
                "_pop_contributor_orgs_max": MAX_CONTRIBUTOR_COUNT,
                "_pop_contributor_orgs": None,
                "_pop_contributor_orgs_error": str(ex)
            }
        return {
            "_pop_contributor_orgs_len": len(orgs),
            "_pop_contributor_orgs_max": MAX_CONTRIBUTOR_COUNT,
            "_pop_contributor_orgs": sorted(orgs),
            "_pop_contributor_orgs_error": None
        }

    @staticmethod
    @memory.cache
    def commit_frequency(ghw: GithubWrapper, name: str) -> float:
        repo = ghw.get_repo(name)
        # NOTE: get_stats_commit_activity Returns the last year of commit activity grouped by week
        stats_commit_activity = repo.get_stats_commit_activity()
        assert len(stats_commit_activity) == 52
        total = 0
        for week_stat in stats_commit_activity:
            total += week_stat.total
        return round(total / len(stats_commit_activity), 2)

    @staticmethod
    @memory.cache
    def updated_issues_count(ghw: GithubWrapper, name: str) -> int:
        ISSUE_LOOKBACK_DAYS = 90
        repo = ghw.get_repo(name)
        issues_since_time = datetime.utcnow() - timedelta(days=ISSUE_LOOKBACK_DAYS)
        # NOTE: get_issues includes PR's
        return repo.get_issues(state='all', since=issues_since_time).totalCount

    @staticmethod
    @memory.cache
    def closed_issues_count(ghw: GithubWrapper, name: str) -> int:
        ISSUE_LOOKBACK_DAYS = 90
        # TODO: make generic with updated_issues_count?
        repo = ghw.get_repo(name)
        issues_since_time = datetime.utcnow() - timedelta(days=ISSUE_LOOKBACK_DAYS)
        # NOTE: get_issues includes PR's
        return repo.get_issues(state='closed', since=issues_since_time).totalCount

    @staticmethod
    @memory.cache
    def created_since_days(ghw: GithubWrapper, name: str) -> int:
        repo = ghw.get_repo(name)
        creation_time = repo.created_at

        # See if there exist any commits before this repository creation
        # time on GitHub. If yes, then the repository creation time is not
        # correct, and it was residing somewhere else before. So, use the first
        # commit date.
        prior_creation_commit_count = repo.get_commits(until=creation_time).totalCount
        if prior_creation_commit_count:
            logger.warning(f"{name} has {prior_creation_commit_count=}, repository creation time is not correct, "
                           f"and it was residing somewhere else before")
            # TODO: see how often this happens
            # first_commit_time = self.get_first_commit_time()
            # if first_commit_time:
            #     creation_time = min(creation_time, first_commit_time)

        difference = datetime.utcnow() - creation_time
        return round(difference.days / 30)

    @staticmethod
    @memory.cache
    def updated_since_days(ghw: GithubWrapper, name: str) -> int:
        repo = ghw.get_repo(name)
        last_commit = repo.get_commits()[0]
        last_commit_time = last_commit.commit.author.date
        difference = datetime.utcnow() - last_commit_time
        return round(difference.days / 30)

    @staticmethod
    @memory.cache
    def recent_releases_count(ghw: GithubWrapper, name: str) -> dict:
        RELEASE_LOOKBACK_DAYS = 365
        repo = ghw.get_repo(name)
        recent_releases_count = 0
        for release in repo.get_releases():
            if (datetime.utcnow() -
                    release.created_at).days > RELEASE_LOOKBACK_DAYS:
                continue
            recent_releases_count += 1

        estimated_tags = 0
        # Make rough estimation of tags used in last year from overall
        # project history. This query is extremely expensive, so instead
        # do the rough calculation.
        days_since_creation = PopularityMetrics.created_since_days(ghw, name) * 30
        if days_since_creation:
            total_tags = repo.get_tags().totalCount
            estimated_tags = round(
                (total_tags / days_since_creation) * RELEASE_LOOKBACK_DAYS)

        recent_releases_adjusted_count = recent_releases_count
        if not recent_releases_count:
            recent_releases_adjusted_count = estimated_tags

        return {
            "_pop_recent_releases_count": recent_releases_count,
            "_pop_recent_releases_estimated_tags": estimated_tags,
            "_pop_recent_releases_adjusted_count": recent_releases_adjusted_count  # TODO: review need and name
        }

    @staticmethod
    @memory.cache
    def comment_frequency(ghw: GithubWrapper, name: str) -> dict:
        ISSUE_LOOKBACK_DAYS = 90
        repo = ghw.get_repo(name)
        issues_since_time = datetime.utcnow() - timedelta(days=ISSUE_LOOKBACK_DAYS)
        issue_count = repo.get_issues(state='all', since=issues_since_time).totalCount
        comment_count = repo.get_issues_comments(since=issues_since_time).totalCount
        comment_frequency = round(comment_count / issue_count, 1) if issue_count else 0

        return {
            "_pop_issue_count": issue_count,
            "_pop_comment_count": comment_count,
            "_pop_comment_frequency": comment_frequency
        }

    @staticmethod
    @memory.cache
    def dependents_count(ghw: GithubWrapper, name: str) -> int:
        # return 0  # TODO: this is expensive and can have many fails
        DEPENDENTS_REGEX = re.compile(b'.*[^0-9,]([0-9,]+).*commit results', re.DOTALL)
        FAIL_RETRIES = 12
        # TODO: Take package manager dependency trees into account. If we decide
        # to replace this, then find a solution for C/C++ as well.
        # parsed_url = urllib.parse.urlparse(self.url)
        # repo_name = parsed_url.path.strip('/')
        dependents_url = f'https://github.com/search?q="{name}"&type=commits'
        logger.trace(f"Call dependents_count: {dependents_url=}, {name=}")
        content = b''
        time.sleep(5)
        success = False
        for i in range(FAIL_RETRIES):
            result = requests.get(dependents_url)
            if result.status_code == 200:
                content = result.content
                success = True
                break
            sleep_secs = min(2 ** (i + 2), 15 * 60)
            logger.warning(f"Retry dependents_count #{i}/{FAIL_RETRIES}, sleeping for {sleep_secs} secs...")
            time.sleep(sleep_secs)

        if not success:
            raise Exception(f"dependents_count failed, too many retries ({FAIL_RETRIES})")

        match = DEPENDENTS_REGEX.match(content)
        if not match:
            return 0
        return int(match.group(1).replace(b',', b''))
