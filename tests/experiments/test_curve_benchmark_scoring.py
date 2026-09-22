"""Answer extraction for the real training-curve benchmarks.

The math grader reads model output, where ``####`` is a markdown heading far
more often than a final-answer line; the gold side keeps GSM8K's ``####``
convention, which is where it actually holds.
"""

from skillev.experiments.curve_benchmarks import _math_answer, _math_prediction, _math_score


def test_math_prediction_prefers_boxed_over_a_markdown_heading():
    output = (
        "#### Case 3: $n$ is a 3-digit number\n"
        "Checking every candidate leaves three survivors.\n"
        "The sum of all such $n$ is \\boxed{279}.\n"
    )
    assert _math_prediction(output) == 279
    assert _math_score(output, "279") == 1.0


def test_math_prediction_still_reads_a_gsm8k_style_answer_line():
    assert _math_prediction("Work it out.\n#### 42") == 42
    # A heading is used only when nothing is boxed anywhere in the output.
    assert _math_prediction("#### 42\nand later \\boxed{7}") == 7


def test_math_answer_keeps_the_gold_hash_convention():
    # Gold parsing is unchanged: GSM8K solutions do end with "#### <answer>".
    assert _math_answer("Natalia sold 48/2 = 24 clips.\n#### 72") == "72"
    assert _math_answer("Tom pays $1,234 in total.\n#### 1,234") == "1234"
    assert _math_answer("no final line") == ""
