from __future__ import annotations
import heapq
from collections import deque
from itertools import permutations
import numpy as np
from .env import ACTION_DELTAS
CARDINAL_ACTIONS = tuple(ACTION_DELTAS)

def _cell(value) -> tuple[int, int]:
    arr = np.asarray(value, dtype=np.int64).reshape(-1)
    return (int(arr[0]), int(arr[1]))

def _boxes(values) -> tuple[tuple[int, int], ...]:
    arr = np.asarray(values, dtype=np.int64).reshape(-1, 2)
    return tuple(sorted(((int(row), int(col)) for row, col in arr)))

def _floor_cells(grid_n: int, obstacles) -> set[tuple[int, int]]:
    blocked = {_cell(value) for value in np.asarray(obstacles).reshape(-1, 2)}
    return {(row, col) for row in range(grid_n) for col in range(grid_n) if (row, col) not in blocked}

def _reachable_paths(agent, boxes, floor):
    occupied = set(boxes)
    paths = {agent: ()}
    queue = deque([agent])
    while queue:
        cell = queue.popleft()
        for action in CARDINAL_ACTIONS:
            dr, dc = ACTION_DELTAS[action]
            nxt = (cell[0] + dr, cell[1] + dc)
            if nxt in floor and nxt not in occupied and (nxt not in paths):
                paths[nxt] = paths[cell] + (action,)
                queue.append(nxt)
    return paths

def interaction_plan(agent, boxes, *, grid_n=10, obstacles=(), rng=None):
    rng = np.random.default_rng() if rng is None else rng
    agent = _cell(agent)
    boxes = _boxes(boxes)
    floor = _floor_cells(grid_n, obstacles)
    walking_paths = _reachable_paths(agent, boxes, floor)
    occupied = set(boxes)
    candidates = []
    for box in boxes:
        for action in CARDINAL_ACTIONS:
            dr, dc = ACTION_DELTAS[action]
            stand = (box[0] - dr, box[1] - dc)
            destination = (box[0] + dr, box[1] + dc)
            if stand in walking_paths and destination in floor and (destination not in occupied):
                candidates.append(walking_paths[stand] + (action,))
    if not candidates:
        return None
    return list(candidates[int(rng.integers(len(candidates)))])

def _assignment_distance(boxes, targets) -> int:
    return min((sum((abs(box[0] - target[0]) + abs(box[1] - target[1]) for box, target in zip(boxes, order))) for order in permutations(targets)))

def _static_corner(cell, targets, floor) -> bool:
    if cell in targets:
        return False
    row, col = cell
    up = (row - 1, col) not in floor
    down = (row + 1, col) not in floor
    left = (row, col - 1) not in floor
    right = (row, col + 1) not in floor
    return (up or down) and (left or right)

def multi_box_solution(agent, boxes, targets, *, grid_n: int=10, obstacles=(), max_expansions: int=200000):
    agent = _cell(agent)
    boxes = _boxes(boxes)
    targets = _boxes(targets)
    floor = _floor_cells(grid_n, obstacles)
    if agent not in floor or any((box not in floor for box in boxes)):
        return None
    if any((_static_corner(box, set(targets), floor) for box in boxes)):
        return None
    if set(boxes) == set(targets):
        return []
    start = (agent, boxes)
    best_cost = {start: 0}
    parent = {}
    counter = 0
    frontier = [(_assignment_distance(boxes, targets), 0, counter, start)]
    expansions = 0
    while frontier and expansions < max_expansions:
        _, cost, _, state = heapq.heappop(frontier)
        if cost != best_cost.get(state):
            continue
        expansions += 1
        cur_agent, cur_boxes = state
        if set(cur_boxes) == set(targets):
            segments = []
            while state != start:
                previous, segment = parent[state]
                segments.append(segment)
                state = previous
            plan = []
            for segment in reversed(segments):
                plan.extend(segment)
            return plan
        walking_paths = _reachable_paths(cur_agent, cur_boxes, floor)
        occupied = set(cur_boxes)
        for box in cur_boxes:
            for action in CARDINAL_ACTIONS:
                dr, dc = ACTION_DELTAS[action]
                stand = (box[0] - dr, box[1] - dc)
                destination = (box[0] + dr, box[1] + dc)
                if stand not in walking_paths:
                    continue
                if destination not in floor or destination in occupied:
                    continue
                next_boxes = tuple(sorted((destination if value == box else value for value in cur_boxes)))
                if any((_static_corner(value, set(targets), floor) for value in next_boxes)):
                    continue
                segment = walking_paths[stand] + (action,)
                next_state = (box, next_boxes)
                next_cost = cost + len(segment)
                if next_cost >= best_cost.get(next_state, 2 ** 63):
                    continue
                best_cost[next_state] = next_cost
                parent[next_state] = (state, segment)
                counter += 1
                priority = next_cost + _assignment_distance(next_boxes, targets)
                heapq.heappush(frontier, (priority, next_cost, counter, next_state))
    return None
