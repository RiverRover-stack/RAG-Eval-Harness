import pytest
import yaml

from rag_eval.config.run_config import config_hash, load_run_config, split_hashes


@pytest.fixture
def base_yaml(tmp_path):
    path = tmp_path / "_base.yaml"
    path.write_text(
        yaml.dump(
            {
                "name": "_base",
                "retrieval": {"top_k": 5, "dense": {"enabled": True}},
                "generation": {"enabled": False},
                "eval": {"datasets": ["docs_synth_v1"]},
            }
        )
    )
    return path


def test_rejects_typo_at_top_level(tmp_path, base_yaml):
    path = tmp_path / "typo.yaml"
    path.write_text(yaml.dump({"name": "x", "retrieval_typo": {}}))
    with pytest.raises(Exception, match="retrieval_typo"):
        load_run_config(path)


def test_rejects_typo_in_nested_section(tmp_path, base_yaml):
    path = tmp_path / "typo.yaml"
    path.write_text(yaml.dump({"name": "x", "retrieval": {"top_kk": 5}}))
    with pytest.raises(Exception, match="top_kk"):
        load_run_config(path)


def test_extends_merges_and_child_overrides_win(tmp_path, base_yaml):
    child = tmp_path / "child.yaml"
    child.write_text(
        yaml.dump({"name": "child", "extends": "_base.yaml", "retrieval": {"top_k": 10}})
    )
    cfg = load_run_config(child)
    assert cfg.name == "child"
    assert cfg.retrieval.top_k == 10
    # inherited from base, untouched by the child's override
    assert cfg.retrieval.dense.enabled is True
    assert cfg.generation.enabled is False
    assert cfg.eval.datasets == ["docs_synth_v1"]


def test_extends_chain_two_levels(tmp_path, base_yaml):
    mid = tmp_path / "mid.yaml"
    mid.write_text(yaml.dump({"name": "mid", "extends": "_base.yaml", "retrieval": {"top_k": 7}}))
    leaf = tmp_path / "leaf.yaml"
    leaf.write_text(yaml.dump({"name": "leaf", "extends": "mid.yaml"}))
    cfg = load_run_config(leaf)
    assert cfg.retrieval.top_k == 7
    assert cfg.generation.enabled is False


def test_circular_extends_raises(tmp_path):
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text(yaml.dump({"name": "a", "extends": "b.yaml"}))
    b.write_text(yaml.dump({"name": "b", "extends": "a.yaml"}))
    with pytest.raises(ValueError, match="circular"):
        load_run_config(a)


@pytest.mark.parametrize(
    ("field", "raw", "expected"),
    [
        ("retrieval.top_k", "10", 10),
        ("retrieval.bm25.k1", "0.5", 0.5),
        ("retrieval.dense.enabled", "false", False),
        ("retrieval.fusion.method", "rrf_custom", "rrf_custom"),
    ],
)
def test_set_override_coerces_yaml_scalars(tmp_path, base_yaml, field, raw, expected):
    child = tmp_path / "child.yaml"
    child.write_text(yaml.dump({"name": "child", "extends": "_base.yaml"}))
    cfg = load_run_config(child, overrides=[f"{field}={raw}"])
    value = cfg
    for part in field.split("."):
        value = getattr(value, part)
    assert value == expected


def test_set_override_sets_nested_bool(tmp_path, base_yaml):
    child = tmp_path / "child.yaml"
    child.write_text(yaml.dump({"name": "child", "extends": "_base.yaml"}))
    cfg = load_run_config(child, overrides=["retrieval.bm25.enabled=true"])
    assert cfg.retrieval.bm25.enabled is True


def test_set_override_malformed_raises(tmp_path, base_yaml):
    child = tmp_path / "child.yaml"
    child.write_text(yaml.dump({"name": "child", "extends": "_base.yaml"}))
    with pytest.raises(ValueError, match="path.to.field=value"):
        load_run_config(child, overrides=["retrieval.top_k10"])


def test_config_hash_stable_under_key_reordering(tmp_path):
    a = tmp_path / "a.yaml"
    a.write_text(yaml.dump({"name": "x", "retrieval": {"top_k": 5}, "generation": {"enabled": False}}))
    b = tmp_path / "b.yaml"
    b.write_text(yaml.dump({"generation": {"enabled": False}, "name": "x", "retrieval": {"top_k": 5}}))
    assert config_hash(load_run_config(a)) == config_hash(load_run_config(b))


def test_config_hash_changes_with_value(tmp_path, base_yaml):
    child = tmp_path / "child.yaml"
    child.write_text(yaml.dump({"name": "child", "extends": "_base.yaml"}))
    cfg1 = load_run_config(child)
    cfg2 = load_run_config(child, overrides=["retrieval.top_k=99"])
    assert config_hash(cfg1) != config_hash(cfg2)


def test_split_hashes_isolates_retrieval_from_generation(tmp_path, base_yaml):
    child = tmp_path / "child.yaml"
    child.write_text(yaml.dump({"name": "child", "extends": "_base.yaml"}))
    base_cfg = load_run_config(child)
    retrieval_changed = load_run_config(child, overrides=["retrieval.top_k=99"])
    generation_changed = load_run_config(
        child, overrides=["generation.llm.model=some-other-model"]
    )

    base_r, base_g = split_hashes(base_cfg)
    r_r, r_g = split_hashes(retrieval_changed)
    g_r, g_g = split_hashes(generation_changed)

    assert r_r != base_r
    assert r_g == base_g
    assert g_g != base_g
    assert g_r == base_r
