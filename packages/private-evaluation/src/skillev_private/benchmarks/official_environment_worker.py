#!/usr/bin/env python3
# mypy: ignore-errors
# ruff: noqa: UP004, UP015, UP032, UP038
"""Standalone official-environment worker (kept compatible with Python 3.8).

This file deliberately imports no SKILLEV module.  Each official benchmark is
loaded from one explicit source checkout in its own pinned interpreter.
Unexpected simulator exceptions are not translated here: the process exits and
the parent reports an infrastructure failure without exposing private stderr.
Manifest mode exposes only model-visible task identity/context and never reads
terminal rewards, goal predicates, or gold action sequences.
"""

import argparse
import copy
import importlib
import json
import os
import random
import sys

if __package__:
    from .alfworld_public_goal import LEGACY_GOAL_BINDING, bound_instruction
    from .scienceworld_native_diagnostics import ScienceWorldNativeDiagnostics
else:
    from alfworld_public_goal import LEGACY_GOAL_BINDING, bound_instruction
    from scienceworld_native_diagnostics import ScienceWorldNativeDiagnostics

PROTOCOL_VERSION = "skillev-official-environment-worker@1"
MAX_MESSAGE_BYTES = 2 * 1024 * 1024


def require_object(value, fields, label):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ValueError("{} has an incompatible field set".format(label))
    return value


def require_text(value, label, allow_empty=False):
    if not isinstance(value, str) or "\x00" in value:
        raise ValueError("{} must be text without NUL".format(label))
    if not allow_empty and not value.strip():
        raise ValueError("{} cannot be empty".format(label))
    return value


def require_int(value, label, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError("{} has an invalid integer".format(label))
    return value


def require_bool(value, label):
    if not isinstance(value, bool):
        raise TypeError("{} must be boolean".format(label))
    return value


def require_number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("{} must be numeric".format(label))
    return float(value)


def require_sequence(value, label):
    if not isinstance(value, (list, tuple)):
        raise TypeError("{} must be a sequence".format(label))
    return value


def require_singleton(value, label):
    sequence = require_sequence(value, label)
    if len(sequence) != 1:
        raise ValueError("{} must contain exactly one item".format(label))
    return sequence[0]


def require_module_in_source(module, source_root):
    module_file = getattr(module, "__file__", None)
    require_text(module_file, "official module file")
    resolved_module = os.path.realpath(module_file)
    resolved_source = os.path.realpath(source_root)
    if os.path.commonpath([resolved_module, resolved_source]) != resolved_source:
        raise ValueError("official module was not imported from the pinned source root")


def single_alfworld_game(data_directory):
    games = []
    trajectories = []
    for root, _directories, files in os.walk(data_directory):
        if "game.tw-pddl" in files:
            games.append(os.path.join(root, "game.tw-pddl"))
        if "traj_data.json" in files:
            trajectories.append(os.path.join(root, "traj_data.json"))
    if (
        len(games) != 1
        or len(trajectories) != 1
        or os.path.dirname(games[0]) != os.path.dirname(trajectories[0])
    ):
        raise ValueError("ALFWorld data directory must contain exactly one complete game")
    return os.path.realpath(games[0])


def available_webshop_actions(value):
    actions = require_object(
        value,
        {"clickables", "has_search_bar"},
        "official WebShop available actions",
    )
    require_bool(actions["has_search_bar"], "has_search_bar")
    clickables = require_sequence(actions["clickables"], "WebShop clickables")
    # WebAgentTextEnv.step routes every nonempty search argument to browser.search,
    # irrespective of the HTML search bar. Expose the native API, not just widgets.
    projected = ["search"]
    for clickable in clickables:
        label = require_text(clickable, "WebShop clickable")
        # The pinned native step explicitly rejects click[search]. The form's
        # real operation is search[query], already exposed above, not a no-op.
        if label.lower() != "search":
            projected.append("click[{}]".format(label))
    if len(set(projected)) != len(projected):
        raise ValueError("official WebShop actions are not unique")
    return projected


def public_webshop_instruction(value):
    """Project the pinned WebShop HTML heading back to its raw goal text.

    The official text environment exposes the rendered ``<h4>`` content, whose
    stable pinned form is ``"Instruction: " + goal_text``.  Task enumeration
    reads ``goal_text`` directly from the official goal inventory, so the
    execution worker must remove that presentation-only heading before the two
    public projections are compared.
    """

    rendered = require_text(value, "WebShop rendered instruction")
    prefix = "Instruction: "
    if not rendered.startswith(prefix):
        raise ValueError("WebShop rendered instruction lacks its official heading")
    return require_text(rendered[len(prefix) :], "WebShop instruction")


def create_webshop_env(source_root, deployment):
    if not isinstance(deployment, dict):
        raise ValueError("WebShop deployment must be an object")
    if "kind" not in deployment:
        raise ValueError("WebShop deployment has no kind")
    kind = require_text(deployment["kind"], "kind")
    if kind == "json-array":
        deployment = require_object(
            deployment,
            {
                "index_path",
                "inventory_size",
                "kind",
                "products_path",
                "seed",
            },
            "WebShop deployment",
        )
        products_path = os.path.realpath(require_text(deployment["products_path"], "products_path"))
        if not os.path.isfile(products_path):
            raise ValueError("WebShop products file does not exist")
    elif kind == "sqlite":
        deployment = require_object(
            deployment,
            {
                "goals_path",
                "index_path",
                "kind",
                "seed",
                "store_path",
            },
            "WebShop deployment",
        )
        product_store_path = os.path.realpath(require_text(deployment["store_path"], "store_path"))
        goals_path = os.path.realpath(require_text(deployment["goals_path"], "goals_path"))
        if not os.path.isfile(product_store_path):
            raise ValueError("WebShop product store does not exist")
        if not os.path.isfile(goals_path):
            raise ValueError("WebShop goal stream does not exist")
    else:
        raise ValueError("WebShop deployment kind is unsupported")
    search_index_path = os.path.realpath(require_text(deployment["index_path"], "index_path"))
    if not os.path.isdir(search_index_path):
        raise ValueError("WebShop search index does not exist")
    seed = require_int(deployment["seed"], "WebShop seed")
    random.seed(seed)
    module = importlib.import_module("web_agent_site.envs.web_agent_text_env")
    require_module_in_source(module, source_root)
    if kind == "sqlite":
        search_module = importlib.import_module("pyserini.search.lucene")
        runtime_module = importlib.import_module("webshop_disk_runtime")
        server = runtime_module.create_disk_backed_server(
            web_agent_module=module,
            product_store_path=product_store_path,
            goals_path=goals_path,
            search_index_path=search_index_path,
            searcher_factory=search_module.LuceneSearcher,
        )
        environment = module.WebAgentTextEnv(
            observation_mode="text",
            file_path=product_store_path,
            server=server,
        )
        environment._skillev_deployment_kind = kind
        return environment
    num_products = require_int(deployment["inventory_size"], "inventory_size", 1)
    if num_products not in (100, 1000, 100000):
        raise ValueError("WebShop inventory_size is unsupported")
    environment = module.WebAgentTextEnv(
        observation_mode="text",
        file_path=products_path,
        num_products=num_products,
        human_goals=True,
    )
    environment._skillev_deployment_kind = kind
    return environment


class WebShopWorker(object):
    def __init__(self, source_root, deployment, task):
        fields = {"goal_index"}
        if isinstance(task, dict) and "observation_mode" in task:
            fields.add("observation_mode")
        task = require_object(task, fields, "WebShop task")
        self.goal_index = require_int(task["goal_index"], "WebShop goal index")
        mode = task.get("observation_mode", "text")
        if mode not in ("text", "text_rich"):
            raise ValueError("unsupported native WebShop observation mode")
        self.env = create_webshop_env(source_root, deployment)
        # Use the released renderer, which preserves button labels and selected
        # options. Do not derive hints or choices from goals/reward components.
        self.env.observation_mode = mode

    def reset(self):
        result = self.env.reset(session=self.goal_index)
        reset = require_sequence(result, "WebShop reset result")
        if len(reset) != 2:
            raise ValueError("WebShop reset result has an incompatible shape")
        observation = require_text(reset[0], "WebShop reset observation")
        instruction = public_webshop_instruction(self.env.get_instruction_text())
        return {
            "available_actions": available_webshop_actions(self.env.get_available_actions()),
            "instruction_text": instruction,
            "observation_text": observation,
        }

    def step(self, action):
        result = require_sequence(self.env.step(action), "WebShop step result")
        if len(result) != 4:
            raise ValueError("WebShop step result has an incompatible shape")
        return {
            "available_actions": available_webshop_actions(self.env.get_available_actions()),
            "observation_text": require_text(result[0], "WebShop observation"),
            "reward": require_number(result[1], "WebShop reward"),
            "terminal": require_bool(result[2], "WebShop terminal"),
        }

    def close(self):
        if self.env._skillev_deployment_kind == "sqlite":
            runtime_module = importlib.import_module("webshop_disk_runtime")
            runtime_module.close_disk_backed_server(self.env.server)
        self.env.close()


class WebShopManifestWorker(object):
    def __init__(self, source_root, deployment, task):
        require_object(task, set(), "WebShop manifest task")
        self.env = create_webshop_env(source_root, deployment)
        server = getattr(self.env, "server", None)
        goals = getattr(server, "goals", None)
        self.goals = require_sequence(goals, "WebShop official goals")
        if not self.goals:
            raise ValueError("WebShop official goal inventory is empty")

    def manifest_count(self):
        return len(self.goals)

    def manifest_batch(self, offset, limit):
        offset = require_int(offset, "WebShop manifest offset")
        limit = require_int(limit, "WebShop manifest limit", 1)
        if offset > len(self.goals) or offset + limit > len(self.goals):
            raise ValueError("WebShop manifest range is outside the official inventory")
        records = []
        for goal_index in range(offset, offset + limit):
            goal = self.goals[goal_index]
            if not isinstance(goal, dict):
                raise TypeError("WebShop official goal must be an object")
            records.append(
                {
                    "category": require_text(goal["category"], "WebShop goal category"),
                    "goal_index": goal_index,
                    "instruction_text": require_text(
                        goal["instruction_text"],
                        "WebShop goal instruction",
                    ),
                }
            )
        return records

    def manifest_materialize(self, index):
        raise ValueError("WebShop manifest records do not require simulator reset")

    def close(self):
        if self.env._skillev_deployment_kind == "sqlite":
            runtime_module = importlib.import_module("webshop_disk_runtime")
            runtime_module.close_disk_backed_server(self.env.server)
        self.env.close()


class ALFWorldWorker(object):
    def __init__(self, source_root, deployment, task):
        deployment = dict(deployment)
        self.goal_binding = deployment.pop("goal_binding", LEGACY_GOAL_BINDING)
        deployment = require_object(
            deployment,
            {
                "config_path",
                "data_directory",
                "instruction_text",
                "seed",
                "train_eval",
            },
            "ALFWorld deployment",
        )
        task = require_object(
            task,
            {"game_id", "max_steps", "simulator_max_steps"},
            "ALFWorld task",
        )
        self.instruction_text = require_text(deployment["instruction_text"], "ALFWorld instruction")
        self.game_id = require_text(task["game_id"], "ALFWorld game_id")
        self.outer_max_steps = require_int(task["max_steps"], "ALFWorld outer max_steps", 1)
        self.simulator_max_steps = require_int(
            task["simulator_max_steps"],
            "ALFWorld simulator max_steps",
            self.outer_max_steps,
        )
        config = load_alfworld_config(deployment["config_path"])
        self.env, self.expected_game = create_alfworld_env(
            source_root,
            config=config,
            data_directory=deployment["data_directory"],
            max_steps=self.simulator_max_steps,
            seed=deployment["seed"],
            train_eval=deployment["train_eval"],
        )

    def reset(self):
        return reset_alfworld_env(
            self.env,
            expected_game=self.expected_game,
            instruction_text=self.instruction_text,
            goal_binding=self.goal_binding,
        )

    def step(self, action):
        result = require_sequence(self.env.step([action]), "ALFWorld step result")
        if len(result) != 4:
            raise ValueError("ALFWorld step result has an incompatible shape")
        observation = require_text(
            require_singleton(result[0], "ALFWorld observations"),
            "ALFWorld observation",
        )
        terminal = require_bool(
            require_singleton(result[2], "ALFWorld terminal flags"),
            "ALFWorld terminal",
        )
        info = result[3]
        if not isinstance(info, dict):
            raise TypeError("ALFWorld step info must be an object")
        commands = require_singleton(info["admissible_commands"], "ALFWorld command batches")
        won = require_bool(
            require_singleton(info["won"], "ALFWorld won flags"),
            "ALFWorld won flag",
        )
        return {
            "admissible_commands": [
                require_text(command, "ALFWorld admissible command")
                for command in require_sequence(commands, "ALFWorld commands")
            ],
            "observation_text": observation,
            "success": won if terminal else None,
            "terminal": terminal,
        }

    def close(self):
        self.env.close()


def load_alfworld_config(config_path):
    config_path = os.path.realpath(require_text(config_path, "ALFWorld config_path"))
    if not os.path.isfile(config_path):
        raise ValueError("ALFWorld config path does not exist")
    yaml_module = importlib.import_module("yaml")
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml_module.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError("ALFWorld config must be an object")
    return config


def create_alfworld_env(
    source_root,
    *,
    config,
    data_directory,
    max_steps,
    seed,
    train_eval,
):
    builder, expected_game = create_alfworld_builder(
        source_root,
        config=config,
        data_directory=data_directory,
        max_steps=max_steps,
        seed=seed,
        train_eval=train_eval,
    )
    return builder.init_env(batch_size=1), expected_game


def create_alfworld_builder(
    source_root,
    *,
    config,
    data_directory,
    max_steps,
    seed,
    train_eval,
):
    data_directory = os.path.realpath(require_text(data_directory, "ALFWorld data_directory"))
    if not os.path.isdir(data_directory):
        raise ValueError("ALFWorld data directory does not exist")
    expected_game = single_alfworld_game(data_directory)
    train_eval = require_text(train_eval, "ALFWorld train_eval")
    path_keys = {
        "train": "data_path",
        "eval_in_distribution": "eval_id_data_path",
        "eval_out_of_distribution": "eval_ood_data_path",
    }
    if train_eval not in path_keys:
        raise ValueError("ALFWorld train_eval is unsupported")
    max_steps = require_int(max_steps, "ALFWorld max_steps", 1)
    seed = require_int(seed, "ALFWorld seed")

    config = copy.deepcopy(config)
    dataset = config["dataset"]
    dataset[path_keys[train_eval]] = data_directory
    dataset["num_train_games"] = -1
    dataset["num_eval_games"] = -1
    config["general"]["random_seed"] = seed
    config["rl"]["training"]["max_nb_steps_per_episode"] = max_steps
    config["dagger"]["training"]["max_nb_steps_per_episode"] = max_steps

    module = importlib.import_module("alfworld.agents.environment")
    require_module_in_source(module, source_root)
    environment_type = module.get_environment("AlfredTWEnv")
    builder = environment_type(config, train_eval=train_eval)
    game_files = require_sequence(builder.game_files, "ALFWorld collected games")
    if len(game_files) != 1 or os.path.realpath(game_files[0]) != expected_game:
        raise ValueError("ALFWorld did not bind the pinned single game")
    return builder, expected_game


def reset_alfworld_env(env, *, expected_game, instruction_text, goal_binding=LEGACY_GOAL_BINDING):
    result = require_sequence(env.reset(), "ALFWorld reset result")
    if len(result) != 2:
        raise ValueError("ALFWorld reset result has an incompatible shape")
    observation = require_text(
        require_singleton(result[0], "ALFWorld reset observations"),
        "ALFWorld reset observation",
    )
    info = result[1]
    if not isinstance(info, dict):
        raise TypeError("ALFWorld reset info must be an object")
    commands = require_singleton(info["admissible_commands"], "ALFWorld reset command batches")
    game_file = require_text(
        require_singleton(info["extra.gamefile"], "ALFWorld reset game files"),
        "ALFWorld reset game file",
    )
    if os.path.realpath(game_file) != expected_game:
        raise ValueError("ALFWorld reset selected another game")
    return {
        "admissible_commands": [
            require_text(command, "ALFWorld admissible command")
            for command in require_sequence(commands, "ALFWorld commands")
        ],
        "instruction_text": bound_instruction(
            observation, require_text(instruction_text, "ALFWorld instruction"), goal_binding
        ),
        "observation_text": observation,
    }


class ALFWorldManifestWorker(object):
    """Reset explicit games through one loaded official TextWorld environment."""

    def __init__(self, source_root, deployment, task):
        deployment = require_object(
            deployment,
            {"config_path", "games", "seed"},
            "ALFWorld manifest deployment",
        )
        task = require_object(
            task,
            {"batch_size", "max_steps"},
            "ALFWorld manifest task",
        )
        self.source_root = source_root
        self.config = load_alfworld_config(deployment["config_path"])
        self.seed = require_int(deployment["seed"], "ALFWorld seed")
        self.max_steps = require_int(
            task["max_steps"],
            "ALFWorld manifest max_steps",
            1,
        )
        self.batch_size = require_int(
            task["batch_size"],
            "ALFWorld manifest batch_size",
            1,
        )
        raw_games = require_sequence(deployment["games"], "ALFWorld manifest games")
        if not raw_games:
            raise ValueError("ALFWorld manifest game inventory is empty")
        games = []
        game_ids = []
        for value in raw_games:
            item = require_object(
                value,
                {
                    "data_directory",
                    "game_id",
                    "instruction_text",
                    "train_eval",
                },
                "ALFWorld manifest game",
            )
            game_id = require_text(item["game_id"], "ALFWorld game_id")
            data_directory = os.path.realpath(
                require_text(item["data_directory"], "ALFWorld data directory")
            )
            games.append(
                {
                    "expected_game": single_alfworld_game(data_directory),
                    "game_id": game_id,
                    "instruction_text": require_text(
                        item["instruction_text"],
                        "ALFWorld instruction",
                    ),
                    "train_eval": require_text(
                        item["train_eval"],
                        "ALFWorld train_eval",
                    ),
                }
            )
            game_ids.append(game_id)
        if game_ids != sorted(game_ids) or len(set(game_ids)) != len(game_ids):
            raise ValueError("ALFWorld manifest games must have sorted unique IDs")
        train_eval_values = {item["train_eval"] for item in games}
        if len(train_eval_values) != 1 or not train_eval_values <= {
            "train",
            "eval_in_distribution",
            "eval_out_of_distribution",
        }:
            raise ValueError("ALFWorld manifest must contain one supported split")
        if self.batch_size > len(games):
            raise ValueError("ALFWorld manifest batch_size exceeds its game count")
        self.games = games

        first = games[0]
        builder, _expected_game = create_alfworld_builder(
            source_root,
            config=self.config,
            data_directory=os.path.dirname(first["expected_game"]),
            max_steps=self.max_steps,
            seed=self.seed,
            train_eval=first["train_eval"],
        )
        builder.game_files = [item["expected_game"] for item in games]
        builder.num_games = len(games)
        self.env = builder.init_env(batch_size=self.batch_size)

    def manifest_count(self):
        return len(self.games)

    def manifest_batch(self, offset, limit):
        offset = require_int(offset, "ALFWorld manifest offset")
        limit = require_int(limit, "ALFWorld manifest limit", 1)
        if offset > len(self.games) or offset + limit > len(self.games):
            raise ValueError("ALFWorld manifest range is outside the explicit inventory")
        return [{"game_id": item["game_id"]} for item in self.games[offset : offset + limit]]

    def manifest_materialize(self, index):
        index = require_int(index, "ALFWorld manifest index")
        if index >= len(self.games):
            raise ValueError("ALFWorld manifest index is outside the explicit inventory")
        return self.manifest_materialize_batch([index] * self.batch_size)[0]

    def manifest_materialize_batch(self, indices):
        indices = require_sequence(indices, "ALFWorld manifest indices")
        if len(indices) != self.batch_size:
            raise ValueError("ALFWorld materialization batch has an incompatible length")
        items = []
        for raw_index in indices:
            index = require_int(raw_index, "ALFWorld manifest index")
            if index >= len(self.games):
                raise ValueError("ALFWorld manifest index is outside the explicit inventory")
            items.append(self.games[index])
        self.env.gamefiles = [item["expected_game"] for item in items]
        self.env.seed(self.seed)
        result = require_sequence(self.env.reset(), "ALFWorld reset result")
        if len(result) != 2:
            raise ValueError("ALFWorld reset result has an incompatible shape")
        observations = require_sequence(result[0], "ALFWorld reset observations")
        info = result[1]
        if not isinstance(info, dict):
            raise TypeError("ALFWorld reset info must be an object")
        commands = require_sequence(
            info["admissible_commands"],
            "ALFWorld reset command batches",
        )
        game_files = require_sequence(
            info["extra.gamefile"],
            "ALFWorld reset game files",
        )
        if not (len(observations) == len(commands) == len(game_files) == self.batch_size):
            raise ValueError("ALFWorld reset batch has an incompatible length")

        remaining = {}
        for item in items:
            remaining.setdefault(item["expected_game"], []).append(item)
        materialized = []
        for position in range(self.batch_size):
            observation = observations[position]
            command_batch = commands[position]
            game_file = game_files[position]
            expected_game = os.path.realpath(require_text(game_file, "ALFWorld reset game file"))
            candidates = remaining.get(expected_game)
            if not candidates:
                raise ValueError("ALFWorld reset selected another game")
            item = candidates.pop()
            materialized.append(
                {
                    "admissible_commands": [
                        require_text(command, "ALFWorld admissible command")
                        for command in require_sequence(
                            command_batch,
                            "ALFWorld commands",
                        )
                    ],
                    "game_id": item["game_id"],
                    "instruction_text": item["instruction_text"],
                    "observation_text": require_text(
                        observation,
                        "ALFWorld reset observation",
                    ),
                }
            )
        return materialized

    def close(self):
        self.env.close()


def create_scienceworld_env(source_root, deployment, max_steps):
    deployment = require_object(
        deployment,
        {"jar_path", "seed", "simplification"},
        "ScienceWorld deployment",
    )
    jar_path = os.path.realpath(require_text(deployment["jar_path"], "ScienceWorld jar_path"))
    if not os.path.isfile(jar_path):
        raise ValueError("ScienceWorld jar does not exist")
    require_int(deployment["seed"], "ScienceWorld seed")
    simplification = require_text(
        deployment["simplification"],
        "ScienceWorld simplification",
        allow_empty=True,
    )
    module = importlib.import_module("scienceworld")
    require_module_in_source(module, source_root)
    return (
        module.ScienceWorldEnv(
            serverPath=jar_path,
            envStepLimit=max_steps,
        ),
        simplification,
    )


def scienceworld_public_state(info):
    # Only observations already returned by this native response. Never query
    # valid action combinations, goal progress, object trees or gold plans here.
    names = {"look": "current_look", "inv": "inventory", "taskDesc": "task_description"}
    return {
        target: info[source]
        for source, target in names.items()
        if source in info and isinstance(info[source], str)
    }


class ScienceWorldWorker(object):
    def __init__(self, source_root, deployment, task):
        diagnostics = deployment.get("private_diagnostics", False)
        deployment = {
            key: value for key, value in deployment.items() if key != "private_diagnostics"
        }
        task = require_object(
            task,
            {"max_steps", "task_name", "variation_index"},
            "ScienceWorld task",
        )
        self.task_name = require_text(task["task_name"], "ScienceWorld task_name")
        self.variation_index = require_int(task["variation_index"], "ScienceWorld variation_index")
        self.max_steps = require_int(task["max_steps"], "ScienceWorld max_steps", 1)
        self.env, simplification = create_scienceworld_env(
            source_root,
            deployment,
            self.max_steps,
        )
        self.env.load(
            self.task_name,
            self.variation_index,
            simplification,
            False,
        )
        self.diagnostics = ScienceWorldNativeDiagnostics(self.env) if diagnostics else None

    def reset(self):
        result = require_sequence(self.env.reset(), "ScienceWorld reset result")
        if len(result) != 2:
            raise ValueError("ScienceWorld reset result has an incompatible shape")
        if self.diagnostics is not None:
            self.diagnostics.reset(result[1])
        return {
            "observation_text": require_text(result[0], "ScienceWorld reset observation"),
            "task_description": require_text(
                self.env.get_task_description(), "ScienceWorld task description"
            ),
            "public_state": scienceworld_public_state(result[1]),
            "native_score": result[1].get("score"),
            "native_moves": result[1].get("moves"),
        }

    def step(self, action):
        result = require_sequence(self.env.step(action), "ScienceWorld step result")
        if len(result) != 4:
            raise ValueError("ScienceWorld step result has an incompatible shape")
        info = result[3]
        if not isinstance(info, dict):
            raise TypeError("ScienceWorld step info must be an object")
        return {
            "observation_text": require_text(result[0], "ScienceWorld observation"),
            "score": require_number(info["score"], "ScienceWorld score"),
            "terminal": require_bool(result[2], "ScienceWorld terminal"),
            "public_state": scienceworld_public_state(info),
            "native_moves": info.get("moves"),
            "private_diagnostics": (
                self.diagnostics.transition(action, info)
                if getattr(self, "diagnostics", None) is not None
                else {}
            ),
        }

    def close(self):
        self.env.close()


class ScienceWorldManifestWorker(object):
    def __init__(self, source_root, deployment, task):
        task = require_object(
            task,
            {"max_steps", "split"},
            "ScienceWorld manifest task",
        )
        self.max_steps = require_int(
            task["max_steps"],
            "ScienceWorld manifest max_steps",
            1,
        )
        split = require_text(task["split"], "ScienceWorld manifest split")
        if split != "test":
            raise ValueError("ScienceWorld production manifest must use the test split")
        self.env, self.simplification = create_scienceworld_env(
            source_root,
            deployment,
            self.max_steps,
        )
        pairs = []
        task_names = require_sequence(
            self.env.get_task_names(),
            "ScienceWorld task names",
        )
        for raw_task_name in task_names:
            task_name = require_text(raw_task_name, "ScienceWorld task name")
            self.env.load(task_name, 0, self.simplification, False)
            variations = require_sequence(
                self.env.get_variations_test(),
                "ScienceWorld test variations",
            )
            for raw_variation in variations:
                pairs.append(
                    (
                        task_name,
                        require_int(
                            raw_variation,
                            "ScienceWorld test variation",
                        ),
                    )
                )
        if not pairs or len(set(pairs)) != len(pairs):
            raise ValueError("ScienceWorld official test inventory is empty or duplicated")
        self.pairs = pairs

    def manifest_count(self):
        return len(self.pairs)

    def manifest_batch(self, offset, limit):
        offset = require_int(offset, "ScienceWorld manifest offset")
        limit = require_int(limit, "ScienceWorld manifest limit", 1)
        if offset > len(self.pairs) or offset + limit > len(self.pairs):
            raise ValueError("ScienceWorld manifest range is outside the official inventory")
        return [
            {"task_name": task_name, "variation_index": variation_index}
            for task_name, variation_index in self.pairs[offset : offset + limit]
        ]

    def manifest_materialize(self, index):
        index = require_int(index, "ScienceWorld manifest index")
        if index >= len(self.pairs):
            raise ValueError("ScienceWorld manifest index is outside the official inventory")
        task_name, variation_index = self.pairs[index]
        self.env.load(
            task_name,
            variation_index,
            self.simplification,
            False,
        )
        reset = require_sequence(self.env.reset(), "ScienceWorld reset result")
        if len(reset) != 2:
            raise ValueError("ScienceWorld reset result has an incompatible shape")
        return {
            "initial_observation": require_text(
                reset[0],
                "ScienceWorld reset observation",
            ),
            "query": require_text(
                self.env.get_task_description(),
                "ScienceWorld task description",
            ),
        }

    def close(self):
        self.env.close()


def initialize(payload):
    data = require_object(
        payload,
        {"benchmark", "deployment", "source_revision", "source_root", "task"},
        "worker initialization",
    )
    source_root = os.path.realpath(require_text(data["source_root"], "source_root"))
    if source_root != os.path.realpath(os.getcwd()):
        raise ValueError("worker cwd differs from its pinned source root")
    require_text(data["source_revision"], "source_revision")
    sys.path.insert(0, source_root)
    benchmark = require_text(data["benchmark"], "benchmark")
    if benchmark == "webshop":
        return WebShopWorker(source_root, data["deployment"], data["task"])
    if benchmark == "alfworld":
        return ALFWorldWorker(source_root, data["deployment"], data["task"])
    if benchmark == "scienceworld":
        return ScienceWorldWorker(source_root, data["deployment"], data["task"])
    raise ValueError("worker benchmark is unsupported")


def initialize_manifest(payload):
    data = require_object(
        payload,
        {"benchmark", "deployment", "source_revision", "source_root", "task"},
        "manifest worker initialization",
    )
    source_root = os.path.realpath(require_text(data["source_root"], "source_root"))
    if source_root != os.path.realpath(os.getcwd()):
        raise ValueError("worker cwd differs from its pinned source root")
    require_text(data["source_revision"], "source_revision")
    sys.path.insert(0, source_root)
    benchmark = require_text(data["benchmark"], "benchmark")
    if benchmark == "webshop":
        return WebShopManifestWorker(source_root, data["deployment"], data["task"])
    if benchmark == "alfworld":
        return ALFWorldManifestWorker(source_root, data["deployment"], data["task"])
    if benchmark == "scienceworld":
        return ScienceWorldManifestWorker(source_root, data["deployment"], data["task"])
    raise ValueError("manifest worker benchmark is unsupported")


def response(request_id, result):
    return {
        "protocol_version": PROTOCOL_VERSION,
        "request_id": request_id,
        "result": result,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--response-fd", required=True, type=int)
    args = parser.parse_args()
    response_stream = os.fdopen(args.response_fd, "w", buffering=1, encoding="utf-8")
    worker = None
    manifest_worker = None
    for raw_line in sys.stdin.buffer:
        if len(raw_line) > MAX_MESSAGE_BYTES:
            raise ValueError("worker request is too large")
        request = require_object(
            json.loads(raw_line.decode("utf-8")),
            {"operation", "payload", "protocol_version", "request_id"},
            "worker request",
        )
        if request["protocol_version"] != PROTOCOL_VERSION:
            raise ValueError("worker protocol version differs")
        request_id = require_int(request["request_id"], "request_id", 1)
        operation = require_text(request["operation"], "operation")
        payload = request["payload"]
        if not isinstance(payload, dict):
            raise TypeError("worker request payload must be an object")
        if operation == "initialize":
            if worker is not None or manifest_worker is not None:
                raise ValueError("worker is already initialized")
            worker = initialize(payload)
            result = {"ready": True}
        elif operation == "initialize_manifest":
            if worker is not None or manifest_worker is not None:
                raise ValueError("worker is already initialized")
            manifest_worker = initialize_manifest(payload)
            result = {"ready": True}
        elif operation == "manifest_count":
            require_object(payload, set(), "manifest count payload")
            if manifest_worker is None:
                raise ValueError("manifest worker is not initialized")
            result = {"count": manifest_worker.manifest_count()}
        elif operation == "manifest_batch":
            batch = require_object(
                payload,
                {"limit", "offset"},
                "manifest batch payload",
            )
            if manifest_worker is None:
                raise ValueError("manifest worker is not initialized")
            result = {
                "records": manifest_worker.manifest_batch(
                    batch["offset"],
                    batch["limit"],
                )
            }
        elif operation == "manifest_materialize":
            item = require_object(
                payload,
                {"index"},
                "manifest materialize payload",
            )
            if manifest_worker is None:
                raise ValueError("manifest worker is not initialized")
            result = manifest_worker.manifest_materialize(item["index"])
        elif operation == "manifest_materialize_batch":
            batch = require_object(
                payload,
                {"indices"},
                "manifest materialize batch payload",
            )
            if manifest_worker is None:
                raise ValueError("manifest worker is not initialized")
            result = {"records": manifest_worker.manifest_materialize_batch(batch["indices"])}
        elif operation == "reset":
            require_object(payload, set(), "reset payload")
            if worker is None:
                raise ValueError("worker is not initialized")
            result = worker.reset()
        elif operation == "step":
            step = require_object(payload, {"action"}, "step payload")
            if worker is None:
                raise ValueError("worker is not initialized")
            result = worker.step(require_text(step["action"], "action"))
        elif operation == "close":
            require_object(payload, set(), "close payload")
            active_worker = worker if worker is not None else manifest_worker
            if active_worker is None:
                raise ValueError("worker is not initialized")
            active_worker.close()
            result = {"closed": True}
        else:
            raise ValueError("worker operation is unsupported")
        response_stream.write(
            json.dumps(
                response(request_id, result),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        )
        if operation == "close":
            return


if __name__ == "__main__":
    main()
