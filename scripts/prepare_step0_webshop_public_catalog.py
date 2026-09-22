"""Withdrawn offline WebShop candidate-ranking and voting entry point.

The old implementation is retained in Git history, not as an executable legacy
option. Use the native environment's public search/index and let the evaluated
owner choose its own actions; do not precompute an answer catalog for a skill.
"""

from typing import NoReturn

from skillev.evaluation.public_validation_guard import CandidateSelectionForbidden


def main(args: object = None) -> NoReturn:
    del args
    raise CandidateSelectionForbidden(
        "Offline WebShop candidate ranking, voting and fallback are withdrawn. "
        "Evaluate the owner's own native actions against the public product index."
    )


if __name__ == "__main__":
    main()
