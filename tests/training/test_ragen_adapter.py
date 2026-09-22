from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from src import ragen_adapter


def test_webshop_instances_share_the_immutable_product_server(monkeypatch) -> None:
    constructed_servers = []

    class WebAgentTextEnv:
        def __init__(self, *, server=None, **kwargs) -> None:
            del kwargs
            self.server = server or SimpleNamespace(goals=[{}])
            constructed_servers.append(self.server)

    monkeypatch.setitem(
        sys.modules,
        "web_agent_site.envs.web_agent_text_env",
        SimpleNamespace(WebAgentTextEnv=WebAgentTextEnv),
    )
    engine_module = SimpleNamespace(
        DEFAULT_ATTR_PATH="upstream-small-attributes",
        HUMAN_ATTR_PATH="upstream-small-human-attributes",
    )
    monkeypatch.setitem(sys.modules, "web_agent_site.engine.engine", engine_module)
    ragen_adapter._WEBSHOP_SERVER_CACHE.clear()

    first = ragen_adapter.WebShopEnv(
        file_path="items", attr_path="attributes", human_attr_path="human-attributes"
    )
    second = ragen_adapter.WebShopEnv(
        file_path="items", attr_path="attributes", human_attr_path="human-attributes"
    )

    assert first.env.server is second.env.server
    assert constructed_servers == [first.env.server, first.env.server]
    assert engine_module.DEFAULT_ATTR_PATH == "attributes"
    assert engine_module.HUMAN_ATTR_PATH == "human-attributes"


def test_webshop_uses_the_configured_private_search_index(monkeypatch, tmp_path) -> None:
    module = SimpleNamespace(WebAgentTextEnv=object, init_search_engine=lambda _: None)
    constructed = []

    class LuceneSearcher:
        def __init__(self, path) -> None:
            constructed.append(path)

    monkeypatch.setitem(sys.modules, "web_agent_site.envs.web_agent_text_env", module)
    monkeypatch.setitem(sys.modules, "pyserini", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "pyserini.search", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "pyserini.search.lucene",
        SimpleNamespace(LuceneSearcher=LuceneSearcher),
    )
    monkeypatch.setenv("WEBSHOP_SEARCH_INDEX_PATH", str(tmp_path))

    assert ragen_adapter._webshop_environment_class() is object
    module.init_search_engine()
    assert constructed == [str(tmp_path)]


def test_webshop_reset_does_not_substitute_another_goal_in_formal_runtime(
    monkeypatch,
) -> None:
    class BrokenEnv:
        server = SimpleNamespace(goals=[{}])

        def reset(self, *, session):
            raise RuntimeError(f"failed session {session}")

    env = object.__new__(ragen_adapter.WebShopEnv)
    env.env = BrokenEnv()
    env._n_goals = 1
    env.goal_indices = [0]
    monkeypatch.setenv("SKILLEV_FORMAL_RUNTIME", "1")

    with pytest.raises(RuntimeError):
        env.reset(seed=0)


def test_webshop_initialization_failure_is_fatal_in_formal_runtime(monkeypatch) -> None:
    class BrokenWebShopEnv:
        def __init__(self, **kwargs) -> None:
            del kwargs
            raise FileNotFoundError("official resource missing")

    monkeypatch.setattr(ragen_adapter, "_check_webshop", lambda: True)
    monkeypatch.setattr(ragen_adapter, "WebShopEnv", BrokenWebShopEnv)
    monkeypatch.setenv("SKILLEV_FORMAL_RUNTIME", "1")

    with pytest.raises(RuntimeError):
        ragen_adapter.RAGENAdapter().reset("webshop", {})


def test_alfworld_initialization_failure_is_fatal_in_formal_runtime(monkeypatch) -> None:
    class BrokenALFWorldEnv:
        def __init__(self, **kwargs) -> None:
            del kwargs
            raise FileNotFoundError("official resource missing")

    monkeypatch.setattr(ragen_adapter, "_check_alfworld", lambda: True)
    monkeypatch.setattr(ragen_adapter, "ALFWorldEnv", BrokenALFWorldEnv)
    monkeypatch.setenv("SKILLEV_FORMAL_RUNTIME", "1")

    with pytest.raises(RuntimeError):
        ragen_adapter.RAGENAdapter().reset("alfworld", {})


@pytest.mark.parametrize("environment", ["alfworld", "webshop"])
def test_missing_interactive_dependency_is_fatal_in_formal_runtime(
    monkeypatch, environment: str
) -> None:
    checker = f"_check_{environment}"
    monkeypatch.setattr(ragen_adapter, checker, lambda: False)
    monkeypatch.setenv("SKILLEV_FORMAL_RUNTIME", "1")

    with pytest.raises(RuntimeError):
        ragen_adapter.RAGENAdapter().reset(environment, {})


def test_missing_environment_is_fatal_during_formal_step(monkeypatch) -> None:
    monkeypatch.setenv("SKILLEV_FORMAL_RUNTIME", "1")

    with pytest.raises(RuntimeError):
        ragen_adapter.RAGENAdapter().step("act")


def test_textworld_eval_symbol_receives_grammar_variables(monkeypatch) -> None:
    class TerminalSymbol:
        def __init__(self, value) -> None:
            self.value = value

    class EvalSymbol:
        pass

    textgen = SimpleNamespace(EvalSymbol=EvalSymbol, TerminalSymbol=TerminalSymbol)
    monkeypatch.setitem(sys.modules, "textworld", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "textworld.envs", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "textworld.envs.pddl", SimpleNamespace(textgen=textgen))

    ragen_adapter._patch_textworld_eval_symbol()
    symbol = SimpleNamespace(
        expression="item + suffix", context={"variables": {"item": "a", "suffix": "b"}}
    )

    derived = EvalSymbol.derive(symbol)

    assert derived[0].value == "ab"
