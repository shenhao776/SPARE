from copy import deepcopy
import os

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT, "configs", "paper_tables.yaml")


def _load_yaml(path):
    with open(path, "r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"paper table config must be a mapping: {path}")
    return value


def _model_registry():
    registry = {}
    directory = os.path.join(ROOT, "configs", "models")
    for name in sorted(os.listdir(directory)):
        if name.endswith(".yaml"):
            value = _load_yaml(os.path.join(directory, name))
            if value.get("name") and value.get("model_id"):
                registry[str(value["name"])] = str(value["model_id"])
    return registry


def _dataset_registry():
    registry = {}
    directory = os.path.join(ROOT, "configs", "datasets")
    for name in sorted(os.listdir(directory)):
        if name.endswith(".yaml"):
            stem = name[:-5]
            value = _load_yaml(os.path.join(directory, name))
            scene = value.get("name")
            if scene:
                if scene in registry:
                    raise ValueError(f"duplicate dataset scene name: {scene!r}")
                registry[str(scene)] = {"stem": stem, "config": value}
    return registry


def _env_value(value):
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def load_paper_config(full_variant=None):

    config = deepcopy(_load_yaml(CONFIG_PATH))
    datasets = config.get("datasets") or {}
    backbones = list(config.get("backbones") or [])
    if not datasets or not backbones:
        raise ValueError("paper config requires non-empty datasets and backbones")

    models = _model_registry()
    missing_models = [name for name in backbones if name not in models]
    if missing_models:
        raise ValueError("unknown paper backbones: " + ", ".join(missing_models))
    config["model_ids"] = {name: models[name] for name in backbones}

    dataset_registry = _dataset_registry()
    scene_thresholds = {}
    scene_stems = {}
    for dataset, spec in datasets.items():
        scenes = list((spec or {}).get("scenes") or [])
        if not scenes:
            raise ValueError(f"paper dataset {dataset!r} has no scenes")
        for scene in scenes:
            if scene not in dataset_registry:
                raise ValueError(f"paper scene {scene!r} has no dataset YAML")
            thresholds = dataset_registry[scene]["config"].get("thresholds") or {}
            if "test_m" not in thresholds:
                raise ValueError(f"dataset {scene!r} has no thresholds.test_m")
            scene_thresholds[scene] = float(thresholds["test_m"])
            scene_stems[scene] = dataset_registry[scene]["stem"]
    config["scene_thresholds"] = scene_thresholds
    config["scene_stems"] = scene_stems
    config["all_scene_stems"] = {
        scene: item["stem"] for scene, item in dataset_registry.items()
    }

    if full_variant is not None:

        def expand(rows):
            expanded = []
            for row in rows or []:
                item = deepcopy(row)
                if item.get("kind") != "handcrafted":
                    overrides = {
                        key: _env_value(value)
                        for key, value in (item.pop("overrides", {}) or {}).items()
                    }
                    item["variant"] = {**full_variant, **overrides}
                expanded.append(item)
            return expanded

        config["ablation_rows"] = expand(config.get("ablation_rows"))
        config["prose_variants"] = expand(config.get("prose_variants"))
    return config


def format_path(config, key, **values):

    paths = config["paths"]
    return str(paths[key]).format(**{**paths, **values})
