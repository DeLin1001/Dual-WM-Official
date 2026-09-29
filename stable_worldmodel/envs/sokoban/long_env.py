from __future__ import annotations
from itertools import permutations
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_worldmodel import spaces as swm_spaces
from .env import ACTION_DELTAS, NOOP, SokobanEnv
DEFAULT_OBSTACLES = np.array([[3, 3], [3, 4], [3, 5], [6, 4], [6, 5], [6, 6]], dtype=np.int64)

class SokobanLongEnv(SokobanEnv):

    def __init__(self, grid_n: int=10, render_mode: str='rgb_array', terminate_on_success: bool=True, img_size: int=224, render_target: bool=False, obstacles=None, success_gap_radius: int=0, require_agent_at_goal: bool=False, agent_gap_radius: int=0, success_total_gap: int | None=None):
        if grid_n < 8:
            raise ValueError('SokobanLongEnv requires grid_n >= 8')
        super().__init__(grid_n=grid_n, render_mode=render_mode, terminate_on_success=terminate_on_success, img_size=img_size, render_target=render_target)
        self.env_name = 'SokobanLong'
        self.success_gap_radius = int(success_gap_radius)
        self.require_agent_at_goal = bool(require_agent_at_goal)
        self.agent_gap_radius = int(agent_gap_radius)
        self.agent_goal_cell = None
        self.success_total_gap = None if success_total_gap is None else int(success_total_gap)
        self.obstacles = np.asarray(DEFAULT_OBSTACLES if obstacles is None else obstacles, dtype=np.int64).reshape(-1, 2)
        self._obstacle_set = {(int(cell[0]), int(cell[1])) for cell in self.obstacles}
        if any((not self._inside_board(cell) for cell in self.obstacles)):
            raise ValueError('All obstacles must lie inside the board')
        self.observation_space = spaces.Box(low=0.0, high=float(self.grid_n), shape=(6,), dtype=np.float32)
        self.box_cells = np.zeros((2, 2), dtype=np.int64)
        self.target_cells = np.zeros((2, 2), dtype=np.int64)
        self._deadlock_pending = False
        self._initial_success_pending = False

    def reset(self, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        options = options or {}
        swm_spaces.reset_variation_space(self.variation_space, seed, options, ())
        state = options.get('state')
        goal_state = options.get('goal_state', options.get('target_state'))
        if state is None:
            self._sample_interactive_configuration()
            if goal_state is not None:
                self._set_goal_state(goal_state)
        else:
            self._set_state(state)
            if goal_state is None:
                self.target_cells = np.array([[2, 8], [7, 1]], dtype=np.int64)
            else:
                self._set_goal_state(goal_state)
        self._deadlock_pending = False
        self._validate_configuration()
        self._initial_success_pending = bool(self.terminate_on_success and self._is_success())
        info = self._get_info()
        info['success'] = float(self._is_success())
        return (self._get_obs(), info)

    def step(self, action):
        action_idx = self._parse_action(action)
        old_agent = self.agent_cell.copy()
        old_boxes = self.box_cells.copy()
        if self._initial_success_pending and self.terminate_on_success and self._is_success():
            self._initial_success_pending = False
            info = self._get_info()
            info['success'] = 1.0
            return (self._get_obs(), 0.0, True, False, info)
        self._initial_success_pending = False
        if self._deadlock_pending:
            self._deadlock_pending = False
            info = self._get_info()
            info['deadlock'] = 1.0
            info['blocked'] = float(action_idx != NOOP)
            return (self._get_obs(), 0.0, False, True, info)
        delta = ACTION_DELTAS.get(action_idx)
        if delta is not None:
            delta = np.asarray(delta, dtype=np.int64)
            agent_next = self.agent_cell + delta
            if self._is_floor(agent_next):
                matches = np.flatnonzero(np.all(self.box_cells == agent_next[None], axis=1))
                if matches.size:
                    box_idx = int(matches[0])
                    box_next = self.box_cells[box_idx] + delta
                    occupied = np.any(np.all(self.box_cells == box_next[None], axis=1))
                    if self._is_floor(box_next) and (not occupied):
                        self.box_cells[box_idx] = box_next
                        self.agent_cell = agent_next
                else:
                    self.agent_cell = agent_next
        success = self._is_success()
        info = self._get_info()
        info['success'] = float(success)
        info['is_push'] = float(not np.array_equal(old_boxes, self.box_cells))
        info['blocked'] = float(action_idx != NOOP and np.array_equal(old_agent, self.agent_cell))
        terminated = bool(success and self.terminate_on_success)
        return (self._get_obs(), 0.0, terminated, False, info)

    def _candidate_cells(self, *, interior=False):
        margin = 1 if interior else 0
        return [(row, col) for row in range(margin, self.grid_n - margin) for col in range(margin, self.grid_n - margin) if (row, col) not in self._obstacle_set]

    def _sample_interactive_configuration(self, max_attempts=500):
        from .level_utils import interaction_plan
        floor = self._candidate_cells()
        targets_pool = self._candidate_cells(interior=True)
        for _ in range(max_attempts):
            targets = np.asarray([targets_pool[i] for i in self.np_random.choice(len(targets_pool), size=2, replace=False)], dtype=np.int64)
            physical_indices = self.np_random.choice(len(floor), size=3, replace=False)
            agent = np.asarray(floor[int(physical_indices[0])])
            boxes = np.asarray([floor[int(i)] for i in physical_indices[1:]], dtype=np.int64)
            if set(map(tuple, boxes)) == set(map(tuple, targets)):
                continue
            plan = interaction_plan(agent, boxes, grid_n=self.grid_n, obstacles=self.obstacles, rng=self.np_random)
            if plan:
                self.agent_cell = agent.astype(np.int64)
                self.box_cells = boxes.astype(np.int64)
                self.target_cells = targets.astype(np.int64)
                return
        raise RuntimeError('Could not sample an interactive SokobanLong level')

    def sample_reachable_targets(self, max_attempts=200, max_expansions=200000):
        from .level_utils import multi_box_solution
        pool = self._candidate_cells(interior=True)
        for _ in range(max_attempts):
            target_indices = self.np_random.choice(len(pool), size=2, replace=False)
            targets = np.asarray([pool[int(index)] for index in target_indices], dtype=np.int64)
            if set(map(tuple, targets)) == set(map(tuple, self.box_cells)):
                continue
            plan = multi_box_solution(self.agent_cell, self.box_cells, targets, grid_n=self.grid_n, obstacles=self.obstacles, max_expansions=max_expansions)
            if plan:
                self.target_cells = targets
                return True
        return False

    def request_deadlock_reset(self):
        self._deadlock_pending = True

    def _inside_board(self, cell) -> bool:
        cell = np.asarray(cell).reshape(-1)
        return bool(0 <= cell[0] < self.grid_n and 0 <= cell[1] < self.grid_n)

    def _is_floor(self, cell) -> bool:
        cell = np.asarray(cell, dtype=np.int64).reshape(-1)
        return self._inside_board(cell) and tuple(cell[:2]) not in self._obstacle_set

    def _validate_configuration(self):
        occupied = [self.agent_cell, *self.box_cells, *self.target_cells]
        if not all((self._is_floor(cell) for cell in occupied)):
            raise ValueError('Agent, boxes, and targets must be on floor cells')
        physical = [tuple(self.agent_cell), *map(tuple, self.box_cells)]
        if len(set(physical)) != len(physical):
            raise ValueError('Agent and boxes may not overlap')
        if len(set(map(tuple, self.box_cells))) != 2:
            raise ValueError('The two boxes may not overlap')
        if len(set(map(tuple, self.target_cells))) != 2:
            raise ValueError('The two targets may not overlap')

    def _is_success(self) -> bool:
        if self.success_total_gap is not None and self.agent_goal_cell is not None:
            agent_gap = int(np.abs(self.agent_cell - self.agent_goal_cell).sum())
            return self._box_target_distance() + agent_gap <= self.success_total_gap
        box_ok = self._box_target_distance() <= self.success_gap_radius if self.success_gap_radius > 0 else set(map(tuple, self.box_cells)) == set(map(tuple, self.target_cells))
        if not box_ok:
            return False
        if self.require_agent_at_goal and self.agent_goal_cell is not None:
            agent_gap = int(np.abs(self.agent_cell - self.agent_goal_cell).sum())
            return agent_gap <= self.agent_gap_radius
        return True

    def _box_target_distance(self) -> float:
        return float(min((sum((np.abs(self.box_cells[i] - target).sum() for (i, target) in enumerate(order))) for order in permutations(self.target_cells))))

    def _get_obs(self):
        return np.concatenate([self.agent_cell, self.box_cells.reshape(-1)]).astype(np.float32)

    def _get_info(self):
        return {'env_name': self.env_name, 'proprio': self.agent_cell.astype(np.float32).copy(), 'state': self._get_obs(), 'pos_agent': self.agent_cell.astype(np.float32).copy(), 'pos_box': self.box_cells.astype(np.float32).copy(), 'pos_target': self.target_cells.astype(np.float32).copy(), 'agent_target_cell': self.agent_goal_cell.astype(np.float32).copy() if self.agent_goal_cell is not None else np.full(2, -1.0, dtype=np.float32), 'distance_to_target': self._box_target_distance(), 'is_push': 0.0, 'blocked': 0.0, 'deadlock': 0.0, 'action_source': int(self._last_action_source)}

    def _cell_rect(self, cell):
        (row, col) = (int(v) for v in np.asarray(cell).reshape(-1)[:2])
        x0 = col * self.cell
        y0 = row * self.cell
        return (x0, y0, x0 + self.cell, y0 + self.cell)

    def _draw_brackets(self, draw, cell, color):
        (x0, y0, x1, y1) = self._cell_rect(cell)
        (margin, segment, width) = (16, 16, 5)
        for (sx, sy, dx, dy) in [(x0 + margin, y0 + margin, 1, 1), (x1 - margin, y0 + margin, -1, 1), (x0 + margin, y1 - margin, 1, -1), (x1 - margin, y1 - margin, -1, -1)]:
            draw.line([sx, sy, sx + dx * segment, sy], fill=color, width=width)
            draw.line([sx, sy, sx, sy + dy * segment], fill=color, width=width)

    def _draw_wall_panel(self, draw, cell, wall, edge, brace):
        (x0, y0, x1, y1) = self._cell_rect(cell)
        (x0, y0, x1, y1) = (x0 + 3, y0 + 3, x1 - 3, y1 - 3)
        draw.rectangle([x0, y0, x1, y1], fill=wall, outline=edge, width=4)
        draw.line([x0 + 4, y0 + 4, x1 - 4, y1 - 4], fill=brace, width=4)
        draw.line([x1 - 4, y0 + 4, x0 + 4, y1 - 4], fill=brace, width=4)

    def _draw_crate(self, draw, cell, crate, edge, brace):
        (x0, y0, x1, y1) = self._cell_rect(cell)
        inset = 12
        draw.rounded_rectangle([x0 + inset, y0 + inset, x1 - inset, y1 - inset], radius=12, fill=crate, outline=edge, width=4)
        draw.line([x0 + inset + 12, y0 + inset + 10, x1 - inset - 12, y1 - inset - 10], fill=brace, width=9)

    def _render_frame(self):
        from PIL import Image, ImageDraw
        vs = self.variation_space
        floor = tuple((int(v) for v in vs['background']['color'].value))
        wall = tuple((int(v) for v in vs['wall']['color'].value))
        crate = tuple((int(v) for v in vs['box']['color'].value))
        target = tuple((int(v) for v in vs['target']['color'].value))
        agent_color = tuple((int(v) for v in vs['agent']['color'].value))
        floor_motif = self._shade(floor, 1.12)
        wall_edge = self._shade(wall, 0.68)
        wall_brace = self._shade(wall, 1.38)
        crate_edge = self._shade(crate, 1.28)
        crate_brace = self._shade(crate, 0.82)
        size = self.grid_n * self.cell
        img = Image.new('RGB', (size, size), floor)
        draw = ImageDraw.Draw(img)
        for row in range(self.grid_n):
            for col in range(self.grid_n):
                mx = col * self.cell + self.cell // 2
                my = row * self.cell + self.cell // 2
                k = 10
                draw.polygon([(mx, my - k), (mx + k, my), (mx, my + k), (mx - k, my)], outline=floor_motif, width=3)
        border_width = 10
        draw.rectangle([1, 1, size - 2, size - 2], outline=wall_edge, width=border_width)
        for obstacle in self.obstacles:
            self._draw_wall_panel(draw, obstacle, wall, wall_edge, wall_brace)
        show_targets = bool(self.render_target) or bool(int(vs['rendering']['render_target'].value))
        if show_targets:
            for target_cell in self.target_cells:
                self._draw_brackets(draw, target_cell, target)
        for box in self.box_cells:
            self._draw_crate(draw, box, crate, crate_edge, crate_brace)
        (x0, y0, x1, y1) = self._cell_rect(self.agent_cell)
        (cx, cy) = ((x0 + x1) // 2, (y0 + y1) // 2)
        px = 4
        sx0 = cx - len(self.SPRITE[0]) * px // 2
        sy0 = cy - len(self.SPRITE) * px // 2
        palette = dict(self.SPRITE_PAL)
        palette['A'] = agent_color
        for (row_idx, row) in enumerate(self.SPRITE):
            for (col_idx, char) in enumerate(row):
                if char == '.':
                    continue
                draw.rectangle([sx0 + col_idx * px, sy0 + row_idx * px, sx0 + (col_idx + 1) * px - 1, sy0 + (row_idx + 1) * px - 1], fill=palette[char])
        return np.asarray(img.resize((self.img_size, self.img_size), Image.Resampling.LANCZOS), dtype=np.uint8)

    def _set_state(self, state):
        state = np.asarray(state, dtype=np.int64).reshape(-1)
        if state.size != 6:
            raise ValueError('SokobanLong state must contain 6 values')
        self.agent_cell = state[:2].copy()
        self.box_cells = state[2:].reshape(2, 2).copy()

    def _set_goal_state(self, goal_state):
        goal_state = np.asarray(goal_state, dtype=np.int64).reshape(-1)
        if goal_state.size not in (4, 6):
            raise ValueError('SokobanLong goal state must contain 4 or 6 values')
        self.agent_goal_cell = goal_state[:2].astype(np.int64).copy() if goal_state.size == 6 else None
        box_values = goal_state if goal_state.size == 4 else goal_state[2:]
        self.target_cells = box_values.reshape(2, 2).copy()
        self._initial_success_pending = self._is_success()
