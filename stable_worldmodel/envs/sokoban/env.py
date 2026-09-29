import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_worldmodel import spaces as swm_spaces
DEFAULT_VARIATIONS = ()
(NOOP, UP, DOWN, LEFT, RIGHT) = (0, 1, 2, 3, 4)
ACTION_DELTAS = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}
IMG_SIZE = 224

class SokobanEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 10}

    def __init__(self, grid_n: int=5, render_mode: str='rgb_array', terminate_on_success: bool=True, img_size: int=IMG_SIZE, render_target: bool=False):
        assert render_mode in self.metadata['render_modes']
        self.render_mode = render_mode
        self.grid_n = int(grid_n)
        self.img_size = int(img_size)
        self.cell = 96
        self.terminate_on_success = bool(terminate_on_success)
        self.render_target = bool(render_target)
        self.observation_space = spaces.Box(low=0.0, high=float(self.grid_n), shape=(4,), dtype=np.float32)
        self.action_space = spaces.Discrete(5)
        self.env_name = 'Sokoban'
        self.variation_space = self._build_variation_space()
        self.agent_cell = np.zeros(2, dtype=np.int64)
        self.box_cell = np.zeros(2, dtype=np.int64)
        self.target_cell = np.zeros(2, dtype=np.int64)
        self._last_action_source = 0

    def _build_variation_space(self):
        return swm_spaces.Dict({'agent': swm_spaces.Dict({'color': swm_spaces.RGBBox(init_value=np.array([255, 0, 0], dtype=np.uint8))}), 'box': swm_spaces.Dict({'color': swm_spaces.RGBBox(init_value=np.array([196, 116, 44], dtype=np.uint8))}), 'target': swm_spaces.Dict({'color': swm_spaces.RGBBox(init_value=np.array([0, 200, 80], dtype=np.uint8))}), 'wall': swm_spaces.Dict({'color': swm_spaces.RGBBox(init_value=np.array([94, 110, 118], dtype=np.uint8))}), 'background': swm_spaces.Dict({'color': swm_spaces.RGBBox(init_value=np.array([112, 130, 138], dtype=np.uint8))}), 'rendering': swm_spaces.Dict({'render_target': swm_spaces.Discrete(2, init_value=int(self.render_target))})})

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        swm_spaces.reset_variation_space(self.variation_space, seed, options, DEFAULT_VARIATIONS)
        state = options.get('state')
        if state is not None:
            self._set_state(state)
            target = options.get('target_state')
            if target is not None:
                self.target_cell = self._to_cell(target)
            else:
                self.target_cell = self._sample_free_cell(exclude=[self.agent_cell, self.box_cell])
        else:
            self._sample_solvable_level()
        obs = self._get_obs()
        info = self._get_info()
        info['distance_to_target'] = float(np.abs(self.box_cell - self.target_cell).sum())
        info['success'] = 0.0
        return (obs, info)

    def _sample_solvable_level(self):
        from stable_worldmodel.envs.sokoban.expert import bfs_solution
        for _ in range(1000):
            self.target_cell = self._sample_interior_cell(exclude=[])
            self.box_cell = self._sample_free_cell(exclude=[self.target_cell])
            self.agent_cell = self._sample_free_cell(exclude=[self.target_cell, self.box_cell])
            if bfs_solution(self.agent_cell, self.box_cell, self.target_cell, grid_n=self.grid_n) is not None:
                return
        raise RuntimeError('SokobanEnv.reset: failed to sample a solvable level in 1000 tries')

    def step(self, action):
        action_idx = self._parse_action(action)
        delta = ACTION_DELTAS.get(action_idx)
        old_agent = self.agent_cell.copy()
        old_box = self.box_cell.copy()
        if delta is not None:
            agent_next = self.agent_cell + delta
            if self._in_bounds(agent_next):
                if np.array_equal(agent_next, self.box_cell):
                    box_next = self.box_cell + delta
                    if self._in_bounds(box_next):
                        self.box_cell = box_next
                        self.agent_cell = agent_next
                else:
                    self.agent_cell = agent_next
        dist = float(np.abs(self.box_cell - self.target_cell).sum())
        terminated = bool(self.box_cell.tolist() == self.target_cell.tolist())
        truncated = False
        reward = 0.0
        obs = self._get_obs()
        info = self._get_info()
        info['distance_to_target'] = dist
        info['success'] = float(terminated)
        info['is_push'] = float(not np.array_equal(old_box, self.box_cell))
        info['blocked'] = float(action_idx != NOOP and np.array_equal(old_agent, self.agent_cell))
        info['action_source'] = int(self._last_action_source)
        terminated = terminated and self.terminate_on_success
        return (obs, reward, terminated, truncated, info)

    def render(self):
        return self._render_frame()

    def _parse_action(self, action):
        if isinstance(action, np.ndarray) or (isinstance(action, (list, tuple)) and len(action) > 1):
            arr = np.asarray(action).reshape(-1)
            idx = int(np.argmax(arr))
        else:
            idx = int(np.asarray(action).reshape(-1)[0])
        if not self.action_space.contains(idx):
            raise ValueError(f'Invalid Sokoban action {idx}; expected 0..4')
        return idx

    def sample_reachable_target(self):
        from stable_worldmodel.envs.sokoban.expert import bfs_solution
        for _ in range(1000):
            target = self._sample_interior_cell(exclude=[self.box_cell])
            if bfs_solution(self.agent_cell, self.box_cell, target, grid_n=self.grid_n):
                self.target_cell = target
                return target.copy()
        raise RuntimeError('Could not sample a reachable Sokoban target')

    def _to_cell(self, cell):
        cell = np.asarray(cell, dtype=np.int64).reshape(-1)
        return np.clip(cell, 0, self.grid_n - 1)

    def _in_bounds(self, cell):
        return bool(0 <= cell[0] < self.grid_n and 0 <= cell[1] < self.grid_n)

    def _sample_free_cell(self, exclude):
        while True:
            cell = self.np_random.integers(0, self.grid_n, size=2)
            if not any((np.array_equal(cell, e) for e in exclude)):
                return cell

    def _sample_interior_cell(self, exclude):
        if self.grid_n < 3:
            raise ValueError('Sokoban grid_n must be at least 3')
        while True:
            cell = self.np_random.integers(1, self.grid_n - 1, size=2)
            if not any((np.array_equal(cell, e) for e in exclude)):
                return cell

    def _get_obs(self):
        return np.array([float(self.agent_cell[0]), float(self.agent_cell[1]), float(self.box_cell[0]), float(self.box_cell[1])], dtype=np.float32)

    def _get_info(self):
        agent = self.agent_cell.astype(np.float32)
        box = self.box_cell.astype(np.float32)
        target = self.target_cell.astype(np.float32)
        return {'env_name': self.env_name, 'proprio': agent.copy(), 'state': self._get_obs(), 'pos_agent': agent.copy(), 'pos_box': box.copy(), 'pos_target': target.copy(), 'is_push': 0.0, 'blocked': 0.0, 'action_source': int(self._last_action_source)}
    SPRITE = ['....GGGGGG....', '...GGGGGGGG...', '..GGGGGGGGGG..', '.GGGGGGGGGGGG.', '...SSSSSSSS...', '..SSSBSSBSSS..', '..SSSBSSBSSS..', '..SSSSSSSSSS..', '...SMMMMMMS...', '....MMMMMM....', '..AAAAAAAAAA..', '.AAAOOOOOAAA.', '.AAAOOOOOAAA.', '..AAAAAAAAAA..', '..HHH....HHH..', '.HHHH....HHHH.']
    SPRITE_PAL = {'G': (76, 165, 82), 'S': (244, 208, 172), 'B': (40, 40, 44), 'M': (120, 72, 40), 'O': (250, 210, 120), 'H': (120, 72, 40)}
    OUTLINE = (40, 40, 44)

    def _cell_rect(self, cell):
        (r, c) = (int(cell[0]) + 1, int(cell[1]) + 1)
        x0 = c * self.cell
        y0 = r * self.cell
        return (x0, y0, x0 + self.cell, y0 + self.cell)

    @staticmethod
    def _shade(color, factor):
        arr = np.asarray(color, dtype=np.float32) * factor
        return tuple((int(min(255, max(0, v))) for v in arr))

    def _render_frame(self):
        from PIL import Image, ImageDraw
        vs = self.variation_space
        floor = tuple((int(v) for v in vs['background']['color'].value))
        wall = tuple((int(v) for v in vs['wall']['color'].value))
        crate = tuple((int(v) for v in vs['box']['color'].value))
        target = tuple((int(v) for v in vs['target']['color'].value))
        agent_c = tuple((int(v) for v in vs['agent']['color'].value))
        floor_motif = self._shade(floor, 1.12)
        wall_edge = self._shade(wall, 0.78)
        wall_brace = self._shade(wall, 1.38)
        crate_edge = self._shade(crate, 1.28)
        crate_brace = self._shade(crate, 0.82)
        body_dark = self._shade(agent_c, 0.78)
        cell = self.cell
        size = (self.grid_n + 2) * cell
        img = Image.new('RGB', (size, size), floor)
        d = ImageDraw.Draw(img)
        for r in range(self.grid_n + 2):
            for c in range(self.grid_n + 2):
                mx = c * cell + cell // 2
                my = r * cell + cell // 2
                k = 10
                d.polygon([(mx, my - k), (mx + k, my), (mx, my + k), (mx - k, my)], outline=floor_motif, width=3)
        for r in range(self.grid_n + 2):
            for c in range(self.grid_n + 2):
                if r not in (0, self.grid_n + 1) and c not in (0, self.grid_n + 1):
                    continue
                (x0, y0, x1, y1) = (c * cell + 3, r * cell + 3, (c + 1) * cell - 3, (r + 1) * cell - 3)
                d.rectangle([x0, y0, x1, y1], fill=wall, outline=wall_edge, width=4)
                d.line([x0 + 4, y0 + 4, x1 - 4, y1 - 4], fill=wall_brace, width=4)
                d.line([x1 - 4, y0 + 4, x0 + 4, y1 - 4], fill=wall_brace, width=4)
        render_target = bool(self.render_target) or bool(int(vs['rendering']['render_target'].value))
        if render_target:
            (x0, y0, x1, y1) = self._cell_rect(self.target_cell)
            (m, seg, w) = (16, 16, 5)
            for (sx0, sy0, dx, dy) in [(x0 + m, y0 + m, 1, 1), (x1 - m, y0 + m, -1, 1), (x0 + m, y1 - m, 1, -1), (x1 - m, y1 - m, -1, -1)]:
                d.line([sx0, sy0, sx0 + dx * seg, sy0], fill=target, width=w)
                d.line([sx0, sy0, sx0, sy0 + dy * seg], fill=target, width=w)
        (x0, y0, x1, y1) = self._cell_rect(self.box_cell)
        inset = 12
        d.rounded_rectangle([x0 + inset, y0 + inset, x1 - inset, y1 - inset], radius=12, fill=crate, outline=crate_edge, width=4)
        d.line([x0 + inset + 12, y0 + inset + 10, x1 - inset - 12, y1 - inset - 10], fill=crate_brace, width=9)
        d.rounded_rectangle([x0 + inset, y0 + inset, x1 - inset, y1 - inset], radius=12, outline=crate_edge, width=3)
        (x0, y0, x1, y1) = self._cell_rect(self.agent_cell)
        (cx, cy) = ((x0 + x1) // 2, (y0 + y1) // 2)
        px = 4
        sprite_w = len(self.SPRITE[0]) * px
        sprite_h = len(self.SPRITE) * px
        (sx0, sy0) = (cx - sprite_w // 2, cy - sprite_h // 2)
        pal = dict(self.SPRITE_PAL)
        pal['A'] = agent_c
        for (j, row) in enumerate(self.SPRITE):
            for (i, ch) in enumerate(row):
                if ch == '.':
                    continue
                color = body_dark if ch == 'a' else pal[ch]
                d.rectangle([sx0 + i * px, sy0 + j * px, sx0 + (i + 1) * px - 1, sy0 + (j + 1) * px - 1], fill=color)
        img = img.resize((self.img_size, self.img_size), Image.LANCZOS)
        return np.asarray(img, dtype=np.uint8)

    def _set_state(self, state):
        state = np.asarray(state, dtype=np.float64).reshape(-1)
        self.agent_cell = self._to_cell(state[:2])
        self.box_cell = self._to_cell(state[2:4])

    def _set_goal_state(self, goal_state):
        goal_state = np.asarray(goal_state, dtype=np.float64).reshape(-1)
        self.target_cell = self._to_cell(goal_state[2:4])
