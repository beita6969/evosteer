"""Withdrawn order-shuffled WebShop voting and baseline-fallback entry point.

Historical artifacts remain historical. No command-line option or legacy mode
reenables catalog answer selection; the evaluated owner decides its own actions.
"""

from typing import NoReturn

from skillev.evaluation.public_validation_guard import CandidateSelectionForbidden


def main(args: object = None) -> NoReturn:
    del args
    raise CandidateSelectionForbidden(
        "WebShop consensus refinement and conservative candidate fallback are withdrawn. "
        "Evaluate the owner's own native actions without an offline selector."
    )


if __name__ == "__main__":
    main()
