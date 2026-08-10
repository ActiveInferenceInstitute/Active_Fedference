"""Matched Four-Rooms and Key-Door hierarchy pilot controls."""

from __future__ import annotations

import pytest

from fedference.hierarchy_tasks import (
    KEY_DOOR,
    METHODS,
    HierarchyTask,
    run_hierarchy_task_pilot,
    simulate_hierarchy_task,
)


def test_hierarchy_pilot_keeps_task_and_control_units_explicit() -> None:
    report = run_hierarchy_task_pilot(seeds=(0, 1))
    assert report["tasks"] == ["four-rooms", "key-door"]
    assert report["methods"] == list(METHODS)
    key_door = {
        row["method"]: row["success_rate"]
        for row in report["summaries"]
        if row["task_id"] == "key-door"
    }
    assert key_door["learned"] == 1.0
    assert key_door["oracle"] == 1.0
    assert key_door["flat"] == 0.0
    assert key_door["shuffled"] == 0.0
    assert "episode success" in report["primary_estimand"]


def test_key_door_shuffled_control_fails_closed_at_gate() -> None:
    row = simulate_hierarchy_task(KEY_DOOR, method="shuffled", seed=0)
    assert row["success"] is False
    assert row["blocked"] is True


def test_hierarchy_task_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="unknown hierarchy method"):
        simulate_hierarchy_task(KEY_DOOR, method="invalid")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"task_id": "other"}, "task_id"),
        ({"grid": ()}, "grid"),
        ({"grid": ("..", ".")}, "equal width"),
        ({"horizon": 0}, "horizon"),
        ({"start": (2, 0)}, "start"),
        ({"task_id": "key-door"}, "key-door"),
    ],
)
def test_hierarchy_task_constructor_rejects_invalid_contracts(kwargs, message) -> None:
    base = {
        "task_id": "four-rooms",
        "grid": ("..", ".."),
        "start": (0, 0),
        "goal": (1, 1),
        "horizon": 2,
    }
    base.update(kwargs)
    with pytest.raises(ValueError, match=message):
        HierarchyTask(**base)


def test_hierarchy_pilot_rejects_empty_and_duplicate_seed_sets() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        run_hierarchy_task_pilot(seeds=())
    with pytest.raises(ValueError, match="unique"):
        run_hierarchy_task_pilot(seeds=(0, 0))
    with pytest.raises(ValueError, match="HierarchyTask"):
        simulate_hierarchy_task("not-a-task", method="flat")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-negative"):
        simulate_hierarchy_task(KEY_DOOR, method="flat", seed=-1)
