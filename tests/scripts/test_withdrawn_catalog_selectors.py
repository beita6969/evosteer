"""Old catalog entry points cannot create voted answers, even in legacy mode."""

import importlib
import runpy
import sys
from types import SimpleNamespace

import pytest

from skillev.evaluation.public_validation_guard import CandidateSelectionForbidden


@pytest.mark.parametrize(
    "name", ["prepare_step0_webshop_public_catalog", "refine_step0_webshop_public_catalog"]
)
@pytest.mark.parametrize("entrypoint", ["import", "cli"])
def test_withdrawn_catalog_selector_has_no_generation_or_output(
    tmp_path, monkeypatch, name, entrypoint
):
    output = tmp_path / "must-not-exist"
    if entrypoint == "import":
        module = importlib.import_module("scripts." + name)
        with pytest.raises(CandidateSelectionForbidden):
            module.main(SimpleNamespace(output=output, ballots=1, legacy=True))
    else:
        monkeypatch.delitem(sys.modules, "scripts." + name, raising=False)
        monkeypatch.setattr(
            "sys.argv", [name, "--output", str(output), "--ballots", "1", "--legacy"]
        )
        with pytest.raises(CandidateSelectionForbidden):
            runpy.run_module("scripts." + name, run_name="__main__")
    assert not output.exists()
