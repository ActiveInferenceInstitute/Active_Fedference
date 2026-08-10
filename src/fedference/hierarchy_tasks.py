"""Small deterministic Four-Rooms and Key-Door hierarchy pilots (MAJ-5).

These environments are deliberately compact validation tasks, not a claim that
the generic POMDP spine has been shown to improve every hierarchical problem.
The runner keeps task, seed, horizon, and control identity explicit so a null
or reversed pilot remains usable evidence.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Literal

import numpy as np

TaskMethod = Literal["flat", "oracle", "learned", "shuffled", "non-gating"]
Coordinate = tuple[int, int]


def _in_bounds(task: HierarchyTask, coordinate: Coordinate) -> bool:
    return 0 <= coordinate[0] < len(task.grid) and 0 <= coordinate[1] < len(task.grid[0])


@dataclass(frozen=True)
class HierarchyTask:
    """Grid task with an optional key-gated door."""

    task_id: str
    grid: tuple[str, ...]
    start: Coordinate
    goal: Coordinate
    horizon: int
    key: Coordinate | None = None
    door: Coordinate | None = None

    def __post_init__(self) -> None:
        if self.task_id not in ("four-rooms", "key-door"):
            raise ValueError("task_id must be 'four-rooms' or 'key-door'")
        if not self.grid or any(not isinstance(row, str) or not row for row in self.grid):
            raise ValueError("grid must contain non-empty rows")
        width = len(self.grid[0])
        if any(len(row) != width for row in self.grid):
            raise ValueError("grid rows must have equal width")
        if self.horizon < 1:
            raise ValueError("horizon must be positive")
        for name, coordinate in (("start", self.start), ("goal", self.goal)):
            if not _in_bounds(self, coordinate) or self.grid[coordinate[0]][coordinate[1]] == "#":
                raise ValueError(f"{name} must be an open grid cell")
        if self.task_id == "key-door" and (self.key is None or self.door is None):
            raise ValueError("key-door must declare key and door coordinates")


FOUR_ROOMS = HierarchyTask(
    task_id="four-rooms",
    grid=(
        ".........",
        "....#....",
        "....#....",
        ".........",
        "####.####",
        ".........",
        "....#....",
        "....#....",
        ".........",
    ),
    start=(1, 1),
    goal=(7, 7),
    horizon=32,
)

KEY_DOOR = HierarchyTask(
    task_id="key-door",
    grid=(
        "#########",
        "#S..#..G#",
        "#...#...#",
        "#..K.D..#",
        "#...#...#",
        "#########",
    ),
    start=(1, 1),
    goal=(1, 7),
    horizon=28,
    key=(3, 3),
    door=(3, 5),
)

TASKS: tuple[HierarchyTask, ...] = (FOUR_ROOMS, KEY_DOOR)
METHODS: tuple[TaskMethod, ...] = ("flat", "oracle", "learned", "shuffled", "non-gating")


def _neighbors(coordinate: Coordinate) -> tuple[Coordinate, ...]:
    row, column = coordinate
    return ((row - 1, column), (row, column + 1), (row + 1, column), (row, column - 1))


def _walkable(task: HierarchyTask, coordinate: Coordinate, *, has_key: bool) -> bool:
    if not _in_bounds(task, coordinate):
        return False
    cell = task.grid[coordinate[0]][coordinate[1]]
    if cell == "#":
        return False
    return coordinate != task.door or has_key


def _shortest_path(
    task: HierarchyTask,
    start: Coordinate,
    target: Coordinate,
    *,
    has_key: bool,
) -> tuple[Coordinate, ...] | None:
    """Return a deterministic shortest path, excluding ``start``."""
    queue: deque[Coordinate] = deque([start])
    previous: dict[Coordinate, Coordinate | None] = {start: None}
    while queue:
        current = queue.popleft()
        if current == target:
            path: list[Coordinate] = []
            cursor: Coordinate | None = current
            while cursor is not None and cursor != start:
                path.append(cursor)
                cursor = previous[cursor]
            path.reverse()
            return tuple(path)
        for candidate in _neighbors(current):
            if candidate in previous or not _walkable(task, candidate, has_key=has_key):
                continue
            previous[candidate] = current
            queue.append(candidate)
    return None


def _waypoints(task: HierarchyTask, method: TaskMethod) -> tuple[Coordinate, ...]:
    if task.task_id == "key-door":
        if method in ("oracle", "learned"):
            assert task.key is not None
            return (task.key, task.goal)
        if method == "shuffled":
            assert task.key is not None
            return (task.goal, task.key)  # deliberately violates the key gate
    return (task.goal,)


def simulate_hierarchy_task(
    task: HierarchyTask,
    *,
    method: TaskMethod,
    seed: int = 0,
) -> dict[str, object]:
    """Simulate one seeded episode under one declared hierarchy control."""
    if not isinstance(task, HierarchyTask):
        raise ValueError("task must be a HierarchyTask")
    if method not in METHODS:
        raise ValueError(f"unknown hierarchy method {method!r}")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    position = task.start
    has_key = False
    path: list[Coordinate] = []
    waypoints = _waypoints(task, method)
    target_index = 0
    blocked = False
    for _step in range(task.horizon):
        if position == task.goal:
            break
        if target_index >= len(waypoints):
            break
        target = waypoints[target_index]
        planned = _shortest_path(task, position, target, has_key=has_key)
        if planned is None or not planned:
            blocked = True
            break
        next_position = planned[0]
        # The flat and non-gating controls do not receive a room/key-state
        # update.  They therefore fail closed at a gated door rather than
        # silently acquiring oracle memory.
        if method in ("flat", "non-gating") and next_position == task.door and not has_key:
            blocked = True
            break
        if task.key is not None and next_position == task.key and method not in ("flat", "non-gating"):
            has_key = True
        position = next_position
        path.append(position)
        if position == target:
            target_index += 1
    success = position == task.goal and not blocked
    return {
        "task_id": task.task_id,
        "method": method,
        "seed": int(seed),
        "horizon": task.horizon,
        "success": bool(success),
        "steps": len(path),
        "has_key": has_key,
        "blocked": blocked,
        "path": [list(coordinate) for coordinate in path],
        "primary_estimand": "episode success within a fixed horizon",
        "independent_unit": "task; seed-level episodes are nested",
    }


def run_hierarchy_task_pilot(
    *,
    seeds: tuple[int, ...] = (0, 1, 2),
) -> dict[str, object]:
    """Run both task units and all matched hierarchy controls."""
    if not seeds or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError("seeds must contain non-negative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    rows = [
        simulate_hierarchy_task(task, method=method, seed=seed)
        for task in TASKS
        for method in METHODS
        for seed in seeds
    ]
    summaries: list[dict[str, object]] = []
    for task in TASKS:
        for method in METHODS:
            subset = [row for row in rows if row["task_id"] == task.task_id and row["method"] == method]
            summaries.append(
                {
                    "task_id": task.task_id,
                    "method": method,
                    "successes": int(sum(bool(row["success"]) for row in subset)),
                    "n_episodes": len(subset),
                    "success_rate": float(np.mean([bool(row["success"]) for row in subset])),
                    "horizon": task.horizon,
                    "independent_unit": "task; seed-level episodes are nested",
                }
            )
    return {
        "status": "pilot",
        "tasks": [task.task_id for task in TASKS],
        "methods": list(METHODS),
        "seeds": list(seeds),
        "rows": rows,
        "summaries": summaries,
        "primary_estimand": "episode success within a fixed horizon",
        "independent_unit": "task; seeds and episodes are nested",
        "falsifier": (
            "learned hierarchy must beat matched flat, shuffled, and non-gating "
            "controls across both tasks"
        ),
        "no_claim": (
            "task-family pilot does not establish a general hierarchy advantage "
            "or a server robustness theorem"
        ),
    }


__all__ = [
    "FOUR_ROOMS",
    "KEY_DOOR",
    "METHODS",
    "TASKS",
    "HierarchyTask",
    "TaskMethod",
    "run_hierarchy_task_pilot",
    "simulate_hierarchy_task",
]
