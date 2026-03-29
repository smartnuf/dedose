import importlib
import json
import sys
from datetime import timedelta
from pathlib import Path

import pytest


@pytest.fixture
def dedose(tmp_path):
    """Provide a reloaded dedose module using an isolated state file."""
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    added_path = False
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
        added_path = True

    module = importlib.import_module("dedose")
    module = importlib.reload(module)
    isolated_state = tmp_path / "state.json"
    module.set_state_file(str(isolated_state))

    yield module

    module.set_state_file(None)
    if added_path:
        sys.path.remove(str(src_path))


def build_plan(dedose_module, **overrides):
    """Construct a Plan with sensible defaults for tests."""
    params = dict(
        name="test-plan",
        start_date="2025-01-01",
        T=2.0,
        D=60.0,
        F=0.5,
        epsilon=0.1,
        max_daily_tablets=None,
        last_processed_date=None,
        residual=0.0,
        current_ideal=2.0,
        total_tablets_given=0,
        total_ideal_sum=0.0,
    )
    params.update(overrides)
    return dedose_module.Plan(**params)


def test_ideal_curve_matches_parameters(dedose):
    plan = build_plan(dedose)

    today_ideal = plan.ideal_for_date(plan.start())
    assert today_ideal == pytest.approx(plan.T)

    day_D = plan.start() + timedelta(days=int(plan.D))
    ideal_at_D = plan.ideal_for_date(day_D)
    assert ideal_at_D == pytest.approx(plan.F + plan.epsilon, rel=1e-6)


def test_quantized_schedule_tracks_ideal_sum(dedose):
    plan = build_plan(dedose, T=1.5, F=0.25, epsilon=0.05, D=45.0)

    target_date = plan.start() + timedelta(days=89)
    dose = plan.advance_through(target_date)

    assert dose >= 0
    diff = abs(plan.total_tablets_given - plan.total_ideal_sum)
    assert diff < 1.0  # bounded integral error
    assert plan.residual == pytest.approx(plan.total_ideal_sum - plan.total_tablets_given, abs=1e-6)


def test_state_file_override_isolated_storage(dedose, tmp_path):
    store = dedose.load_store()
    assert store["plans"] == {}

    plan = build_plan(dedose)
    dedose.save_plan(store, plan)

    assert dedose.STATE_FILE == tmp_path / "state.json"
    assert dedose.STATE_FILE.exists()

    with dedose.STATE_FILE.open("r", encoding="utf-8") as fh:
        persisted = json.load(fh)
    assert "plans" in persisted
    assert plan.name in persisted["plans"]
