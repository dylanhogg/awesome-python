import math
from loguru import logger


class PopularityScorer:
    """
    Inspired by https://github.com/ossf/criticality_score/tree/v1.0.7 (Latest Python version before Go port)
    and https://github.com/ossf/criticality_score/blob/main/Quantifying_criticality_algorithm.pdf
    """

    @staticmethod
    def score(row):
        # Weights for various parameters.
        CREATED_SINCE_WEIGHT = 1
        UPDATED_SINCE_WEIGHT = -1
        CONTRIBUTOR_COUNT_WEIGHT = 1
        ORG_COUNT_WEIGHT = 1
        COMMIT_FREQUENCY_WEIGHT = 1
        RECENT_RELEASES_WEIGHT = 0.5
        CLOSED_ISSUES_WEIGHT = 0.5
        UPDATED_ISSUES_WEIGHT = 0.5
        COMMENT_FREQUENCY_WEIGHT = 1.5
        STARS_PER_WEEK_WEIGHT = 6

        # Max thresholds for various parameters.
        CREATED_SINCE_THRESHOLD = 120
        UPDATED_SINCE_THRESHOLD = 120
        CONTRIBUTOR_COUNT_THRESHOLD = 2000
        ORG_COUNT_THRESHOLD = 10
        COMMIT_FREQUENCY_THRESHOLD = 500
        RECENT_RELEASES_THRESHOLD = 10
        CLOSED_ISSUES_THRESHOLD = 5000
        UPDATED_ISSUES_THRESHOLD = 5000
        COMMENT_FREQUENCY_THRESHOLD = 15
        STARS_PER_WEEK_THRESHOLD = 400

        def _get_param_score(param, max_value, weight=1.0):
            """Return paramater score given its current value, max value and
            parameter weight."""
            if param < 0:
                # Seen for twitter/the-algorithm-ml
                logger.warning(f"Negative value for param in _get_param_score: {param=}, {row['_repopath']=}")
                param = 0  # Handle negative values for log below

            try:
                score = (math.log(1 + param) / math.log(1 + max(param, max_value))) * weight
            except Exception as e:
                logger.error(f"Error in _get_param_score {param=}, {max_value=}, {weight=}, {row['_repopath']=}: {e}")
                raise e

            return score

        total_weight = (
            CREATED_SINCE_WEIGHT
            + UPDATED_SINCE_WEIGHT
            + CONTRIBUTOR_COUNT_WEIGHT
            + ORG_COUNT_WEIGHT
            + COMMIT_FREQUENCY_WEIGHT
            + RECENT_RELEASES_WEIGHT
            + CLOSED_ISSUES_WEIGHT
            + UPDATED_ISSUES_WEIGHT
            + COMMENT_FREQUENCY_WEIGHT
            + STARS_PER_WEEK_WEIGHT
        )

        score_sum = (
            (_get_param_score(row["_pop_created_since_days"], CREATED_SINCE_THRESHOLD, CREATED_SINCE_WEIGHT))
            + (_get_param_score(row["_pop_updated_since_days"], UPDATED_SINCE_THRESHOLD, UPDATED_SINCE_WEIGHT))
            + (_get_param_score(row["_pop_contributor_count"], CONTRIBUTOR_COUNT_THRESHOLD, CONTRIBUTOR_COUNT_WEIGHT))
            + (_get_param_score(row["_pop_contributor_orgs_len"], ORG_COUNT_THRESHOLD, ORG_COUNT_WEIGHT))
            + (_get_param_score(row["_pop_commit_frequency"], COMMIT_FREQUENCY_THRESHOLD, COMMIT_FREQUENCY_WEIGHT))
            + (
                _get_param_score(
                    row["_pop_recent_releases_adjusted_count"], RECENT_RELEASES_THRESHOLD, RECENT_RELEASES_WEIGHT
                )
            )
            + (_get_param_score(row["_pop_closed_issues_count"], CLOSED_ISSUES_THRESHOLD, CLOSED_ISSUES_WEIGHT))
            + (_get_param_score(row["_pop_updated_issues_count"], UPDATED_ISSUES_THRESHOLD, UPDATED_ISSUES_WEIGHT))
            + (_get_param_score(row["_pop_comment_frequency"], COMMENT_FREQUENCY_THRESHOLD, COMMENT_FREQUENCY_WEIGHT))
            + (_get_param_score(row["_stars_per_week"], STARS_PER_WEEK_THRESHOLD, STARS_PER_WEEK_WEIGHT))
        )

        criticality_score = round(100 * score_sum / total_weight, 5)
        logger.trace(f"Calculated {criticality_score=} for {row['_repopath']}")
        return criticality_score
