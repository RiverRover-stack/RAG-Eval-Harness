from datetime import UTC, datetime, timedelta

import pytest

from rag_eval.config.run_config import RunConfig, config_hash, split_hashes
from rag_eval.runs.manifest import list_runs, load_run, new_run, pin_run, write_manifest


@pytest.fixture
def cfg() -> RunConfig:
    return RunConfig(name="baseline-dense")


def test_new_run_creates_a_directory_named_with_timestamp_name_and_hash(tmp_path, cfg):
    run_dir = new_run(cfg, "configs/baseline.yaml", runs_root=tmp_path)
    assert run_dir.exists()
    assert cfg.name in run_dir.name
    assert config_hash(cfg)[:6] in run_dir.name


def test_write_manifest_round_trips_through_load_run(tmp_path, cfg):
    run_dir = new_run(cfg, "configs/baseline.yaml", runs_root=tmp_path)
    metrics = {"docs_synth_v1": {"n": 10, "mrr": 0.75}}
    written = write_manifest(
        run_dir,
        cfg,
        "configs/baseline.yaml",
        metrics=metrics,
        corpus_sha="abc123",
    )
    loaded = load_run(run_dir)
    assert loaded.run_id == written.run_id == run_dir.name
    assert loaded.metrics == metrics
    assert loaded.corpus_sha == "abc123"
    assert loaded.config_hash == config_hash(cfg)
    retrieval_hash, generation_hash = split_hashes(cfg)
    assert loaded.retrieval_hash == retrieval_hash
    assert loaded.generation_hash == generation_hash
    assert (run_dir / "config.resolved.yaml").exists()


def test_load_run_by_bare_id_resolves_against_runs_root(tmp_path, cfg):
    run_dir = new_run(cfg, "configs/baseline.yaml", runs_root=tmp_path)
    write_manifest(run_dir, cfg, "configs/baseline.yaml", metrics={})
    loaded = load_run(run_dir.name, runs_root=tmp_path)
    assert loaded.run_id == run_dir.name


def test_load_run_missing_manifest_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_run("does-not-exist", runs_root=tmp_path)


def test_list_runs_sorted_oldest_first_and_skips_pinned_dir(tmp_path, cfg):
    (tmp_path / "_pinned").mkdir()
    now = datetime.now(UTC)
    later_cfg = RunConfig(name="other")
    d1 = new_run(cfg, "c.yaml", runs_root=tmp_path, timestamp=now)
    write_manifest(d1, cfg, "c.yaml", metrics={}, created_at=now)
    later = now + timedelta(minutes=5)
    d2 = new_run(later_cfg, "c.yaml", runs_root=tmp_path, timestamp=later)
    write_manifest(d2, later_cfg, "c.yaml", metrics={}, created_at=later)

    runs = list_runs(runs_root=tmp_path)
    assert [r.run_id for r in runs] == [d1.name, d2.name]


def test_pin_run_copies_into_pinned_dir(tmp_path, cfg):
    run_dir = new_run(cfg, "c.yaml", runs_root=tmp_path)
    write_manifest(run_dir, cfg, "c.yaml", metrics={"d": {"n": 1}})
    pinned = pin_run(run_dir.name, runs_root=tmp_path, name="0001-first-baseline")
    assert pinned.exists()
    assert (pinned / "manifest.json").exists()
    loaded = load_run(pinned)
    assert loaded.metrics == {"d": {"n": 1}}


def test_pin_run_refuses_to_overwrite_existing_pin(tmp_path, cfg):
    run_dir = new_run(cfg, "c.yaml", runs_root=tmp_path)
    write_manifest(run_dir, cfg, "c.yaml", metrics={})
    pin_run(run_dir.name, runs_root=tmp_path, name="dup")
    with pytest.raises(FileExistsError):
        pin_run(run_dir.name, runs_root=tmp_path, name="dup")


def test_pin_run_missing_source_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        pin_run("nope", runs_root=tmp_path)
