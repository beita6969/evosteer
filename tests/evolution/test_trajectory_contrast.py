import json
from types import SimpleNamespace as N

from skillev.diagnostics.skillflow_view import skill_flow_diagnostic
from skillev.evolution.trajectory_contrast import same_source_contrasts


def artifact(identity, *, source="q1", policy="f@1", reset="reset-seed-0", action="left"):
    return N(
        record=N(
            trajectory_id=identity,
            reward=N(
                native_payload={
                    "training_evidence_source": {
                        "benchmark_id": "synthetic",
                        "population_id": "training",
                        "source_question_id": source,
                        "occurrence_id": identity,
                    },
                    "answer_key": "PRIVATE_SENTINEL",
                }
            ),
            initial_context=N(
                query="same prefix " * 40, meta={"environment_reset_identity": reset}
            ),
            steps=(N(action_text=action, observation_text="public feedback"),),
        ),
        manifest=N(
            policy_snapshot=N(snapshot_id=policy),
            library_version="library@1",
            condition_id="same-wire",
            decoding_snapshot_id="same-decoder",
        ),
    )


def test_source_identity_not_question_prefix_and_private_payload_not_exported():
    left = artifact("left")
    assert not same_source_contrasts((left, artifact("right", source="other", action="different")))
    assert not same_source_contrasts((left, artifact("right", policy="f@2", action="different")))
    assert not same_source_contrasts(
        (left, artifact("right", reset="reset-seed-1", action="different"))
    )
    contrasts = same_source_contrasts((left, artifact("right", action="different")))
    assert len(contrasts) == 1
    public = contrasts[0].authoring_value(maximum_characters_per_field=16)
    assert "PRIVATE_SENTINEL" not in json.dumps(public)
    assert public["first_divergent_step"] == 1
    assert public["causal_claim"] is False
    assert public == contrasts[0].authoring_value(maximum_characters_per_field=16)


def test_missing_reset_identity_and_identical_paths_are_not_invented_contrasts():
    assert not same_source_contrasts((artifact("a"), artifact("b")))
    assert not same_source_contrasts(
        (artifact("a", reset=None), artifact("b", reset=None, action="other"))
    )


def test_read_only_cgf_missing_and_extreme_values_are_not_clipped():
    missing = skill_flow_diagnostic("new-skill", ())
    assert missing.g_at_one is None
    assert missing.to_value()["diagnostic_only"] is True
    observed = skill_flow_diagnostic("observed", (1000.0, 1002.0))
    assert observed.g_at_one > 1000
    assert observed.jensen_gap > 0


def test_public_contrast_persistence_roundtrip():
    from skillev.contracts import canonical_json
    from skillev.evolution.public_execution import PublicExecutionSnippet

    contrast = same_source_contrasts((artifact("left"), artifact("right", action="other")))[0]
    snippet = PublicExecutionSnippet(
        "left", "public", (canonical_json(contrast.authoring_value()),)
    )
    assert PublicExecutionSnippet.from_value(snippet.to_value()) == snippet
    assert "PRIVATE_SENTINEL" not in canonical_json(snippet.to_value())
