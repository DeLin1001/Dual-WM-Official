import mujoco
import numpy as np
from dm_control import mjcf
from ogbench.manipspace import lie
from ogbench.manipspace.envs.manipspace_env import ManipSpaceEnv
from stable_worldmodel import spaces as swm_spaces
from stable_worldmodel.envs.utils import perturb_camera_angle
DEFAULT_VARIATIONS = ('cube.start_position', 'cube.start_yaw', 'cube.goal_position', 'cube.goal_yaw')

class CubeEnv(ManipSpaceEnv):

    def __init__(self, env_type='single', ob_type='states', permute_blocks=True, multiview=False, height=224, width=224, *args, **kwargs):
        self._env_type = env_type
        self._permute_blocks = permute_blocks
        self._multiview = multiview
        self.env_name = 'Cube'
        if self._env_type == 'single':
            self._num_cubes = 1
        elif self._env_type == 'double':
            self._num_cubes = 2
        elif self._env_type == 'triple':
            self._num_cubes = 3
        elif self._env_type == 'quadruple':
            self._num_cubes = 4
        elif self._env_type == 'octuple':
            self._num_cubes = 8
        else:
            raise ValueError(f'Invalid env_type: {env_type}')
        super().__init__(*args, height=height, width=width, **kwargs)
        self._ob_type = ob_type
        self._cube_colors = np.array([self._colors['red'], self._colors['blue'], self._colors['orange'], self._colors['green'], self._colors['purple'], self._colors['yellow'], self._colors['magenta'], self._colors['gray']])
        self._cube_success_colors = np.array([self._colors['lightred'], self._colors['lightblue'], self._colors['lightorange'], self._colors['lightgreen'], self._colors['lightpurple'], self._colors['lightyellow'], self._colors['lightmagenta'], self._colors['white']])
        self._target_task = 'cube'
        self._target_block = 0
        default_start_position = np.array([0.3, 0.0], dtype=np.float64)
        self.variation_space = swm_spaces.Dict({'cube': swm_spaces.Dict({'color': swm_spaces.Box(low=0.0, high=1.0, shape=(self._num_cubes, 3), dtype=np.float64, init_value=self._cube_colors[:self._num_cubes, :3].copy()), 'size': swm_spaces.Box(low=0.01, high=0.03, shape=(self._num_cubes,), dtype=np.float64, init_value=0.02 * np.ones((self._num_cubes,), dtype=np.float32)), 'start_position': swm_spaces.Box(low=np.tile(self._object_sampling_bounds[0], (self._num_cubes, 1)), high=np.tile(self._object_sampling_bounds[1], (self._num_cubes, 1)), shape=(self._num_cubes, 2), dtype=np.float64, init_value=np.tile(default_start_position, (self._num_cubes, 1))), 'start_yaw': swm_spaces.Box(low=0.0, high=2 * np.pi, shape=(self._num_cubes,), dtype=np.float64, init_value=np.zeros((self._num_cubes,), dtype=np.float32)), 'goal_position': swm_spaces.Box(low=np.tile(self._target_sampling_bounds[0], (self._num_cubes, 1)), high=np.tile(self._target_sampling_bounds[1], (self._num_cubes, 1)), shape=(self._num_cubes, 2), dtype=np.float64, init_value=np.tile(default_start_position, (self._num_cubes, 1))), 'goal_yaw': swm_spaces.Box(low=0.0, high=2 * np.pi, shape=(self._num_cubes,), dtype=np.float64, init_value=np.zeros((self._num_cubes,), dtype=np.float32))}), 'agent': swm_spaces.Dict({'color': swm_spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float64, init_value=self._colors['purple'][:3]), 'ee_start_position': swm_spaces.Box(low=self._arm_sampling_bounds[0], high=self._arm_sampling_bounds[1], shape=(3,), dtype=np.float64, init_value=np.mean(self._arm_sampling_bounds, axis=0))}), 'floor': swm_spaces.Dict({'color': swm_spaces.Box(low=0.0, high=1.0, shape=(2, 3), dtype=np.float64, init_value=np.array([[0.08, 0.11, 0.16], [0.15, 0.18, 0.25]]))}), 'camera': swm_spaces.Dict({'angle_delta': swm_spaces.Box(low=-10.0, high=10.0, shape=(2, 2) if self._multiview else (1, 2), dtype=np.float64, init_value=np.zeros([2, 2]) if self._multiview else np.zeros([1, 2]))}), 'light': swm_spaces.Dict({'intensity': swm_spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float64, init_value=[0.7])})})

    def set_tasks(self):
        if self._env_type == 'single':
            self.task_infos = [{'task_name': 'task1_horizontal', 'init_xyzs': np.array([[0.425, 0.1, 0.02]]), 'goal_xyzs': np.array([[0.425, -0.1, 0.02]])}, {'task_name': 'task2_vertical1', 'init_xyzs': np.array([[0.35, 0.0, 0.02]]), 'goal_xyzs': np.array([[0.5, 0.0, 0.02]])}, {'task_name': 'task3_vertical2', 'init_xyzs': np.array([[0.5, 0.0, 0.02]]), 'goal_xyzs': np.array([[0.35, 0.0, 0.02]])}, {'task_name': 'task4_diagonal1', 'init_xyzs': np.array([[0.35, -0.2, 0.02]]), 'goal_xyzs': np.array([[0.5, 0.2, 0.02]])}, {'task_name': 'task5_diagonal2', 'init_xyzs': np.array([[0.35, 0.2, 0.02]]), 'goal_xyzs': np.array([[0.5, -0.2, 0.02]])}]
        elif self._env_type == 'double':
            self.task_infos = [{'task_name': 'task1_single_pnp', 'init_xyzs': np.array([[0.425, 0.0, 0.02], [0.425, -0.1, 0.02]]), 'goal_xyzs': np.array([[0.425, 0.0, 0.02], [0.425, 0.1, 0.02]])}, {'task_name': 'task2_double_pnp1', 'init_xyzs': np.array([[0.35, -0.1, 0.02], [0.5, -0.1, 0.02]]), 'goal_xyzs': np.array([[0.35, 0.1, 0.02], [0.5, 0.1, 0.02]])}, {'task_name': 'task3_double_pnp2', 'init_xyzs': np.array([[0.35, 0.0, 0.02], [0.5, 0.0, 0.02]]), 'goal_xyzs': np.array([[0.425, -0.2, 0.02], [0.425, 0.2, 0.02]])}, {'task_name': 'task4_swap', 'init_xyzs': np.array([[0.425, -0.1, 0.02], [0.425, 0.1, 0.02]]), 'goal_xyzs': np.array([[0.425, 0.1, 0.02], [0.425, -0.1, 0.02]])}, {'task_name': 'task5_stack', 'init_xyzs': np.array([[0.425, -0.2, 0.02], [0.425, 0.2, 0.02]]), 'goal_xyzs': np.array([[0.425, 0.0, 0.02], [0.425, 0.0, 0.06]])}]
        elif self._env_type == 'triple':
            self.task_infos = [{'task_name': 'task1_single_pnp', 'init_xyzs': np.array([[0.35, -0.1, 0.02], [0.35, 0.1, 0.02], [0.5, -0.1, 0.02]]), 'goal_xyzs': np.array([[0.35, -0.1, 0.02], [0.35, 0.1, 0.02], [0.5, 0.1, 0.02]])}, {'task_name': 'task2_triple_pnp', 'init_xyzs': np.array([[0.35, -0.2, 0.02], [0.35, 0.0, 0.02], [0.35, 0.2, 0.02]]), 'goal_xyzs': np.array([[0.5, 0.0, 0.02], [0.5, 0.2, 0.02], [0.5, -0.2, 0.02]])}, {'task_name': 'task3_pnp_from_stack', 'init_xyzs': np.array([[0.425, 0.2, 0.02], [0.425, 0.2, 0.06], [0.425, 0.2, 0.1]]), 'goal_xyzs': np.array([[0.35, -0.1, 0.02], [0.5, -0.2, 0.02], [0.5, 0.0, 0.02]])}, {'task_name': 'task4_cycle', 'init_xyzs': np.array([[0.35, 0.0, 0.02], [0.5, -0.1, 0.02], [0.5, 0.1, 0.02]]), 'goal_xyzs': np.array([[0.5, -0.1, 0.02], [0.5, 0.1, 0.02], [0.35, 0.0, 0.02]])}, {'task_name': 'task5_stack', 'init_xyzs': np.array([[0.35, -0.1, 0.02], [0.5, -0.2, 0.02], [0.5, 0.0, 0.02]]), 'goal_xyzs': np.array([[0.425, 0.2, 0.02], [0.425, 0.2, 0.06], [0.425, 0.2, 0.1]])}]
        elif self._env_type == 'quadruple':
            self.task_infos = [{'task_name': 'task1_double_pnp', 'init_xyzs': np.array([[0.35, -0.1, 0.02], [0.35, 0.1, 0.02], [0.5, -0.1, 0.02], [0.5, 0.1, 0.02]]), 'goal_xyzs': np.array([[0.35, -0.25, 0.02], [0.35, 0.1, 0.02], [0.5, -0.1, 0.02], [0.5, 0.25, 0.02]])}, {'task_name': 'task2_quadruple_pnp', 'init_xyzs': np.array([[0.325, -0.2, 0.02], [0.325, 0.2, 0.02], [0.525, -0.2, 0.02], [0.525, 0.2, 0.02]]), 'goal_xyzs': np.array([[0.375, 0.1, 0.02], [0.475, 0.1, 0.02], [0.375, -0.1, 0.02], [0.475, -0.1, 0.02]])}, {'task_name': 'task3_pnp_from_square', 'init_xyzs': np.array([[0.425, -0.02, 0.02], [0.425, 0.02, 0.02], [0.425, -0.02, 0.06], [0.425, 0.02, 0.06]]), 'goal_xyzs': np.array([[0.525, -0.2, 0.02], [0.325, 0.2, 0.02], [0.325, -0.2, 0.02], [0.525, 0.2, 0.02]])}, {'task_name': 'task4_cycle', 'init_xyzs': np.array([[0.525, -0.1, 0.02], [0.525, 0.1, 0.02], [0.325, 0.1, 0.02], [0.325, -0.1, 0.02]]), 'goal_xyzs': np.array([[0.525, 0.1, 0.02], [0.325, 0.1, 0.02], [0.325, -0.1, 0.02], [0.525, -0.1, 0.02]])}, {'task_name': 'task5_stack', 'init_xyzs': np.array([[0.5, -0.05, 0.02], [0.5, -0.2, 0.02], [0.35, -0.2, 0.02], [0.35, -0.05, 0.02]]), 'goal_xyzs': np.array([[0.425, 0.2, 0.02], [0.425, 0.2, 0.06], [0.425, 0.2, 0.1], [0.425, 0.2, 0.14]])}]
        elif self._env_type == 'octuple':
            self.task_infos = [{'task_name': 'task1_quadruple_pnp', 'init_xyzs': np.array([[0.325, -0.15, 0.02], [0.425, -0.15, 0.02], [0.425, -0.05, 0.02], [0.525, -0.05, 0.02], [0.325, 0.05, 0.02], [0.425, 0.05, 0.02], [0.425, 0.15, 0.02], [0.525, 0.15, 0.02]]), 'goal_xyzs': np.array([[0.525, 0.05, 0.02], [0.425, -0.15, 0.02], [0.425, -0.05, 0.02], [0.325, 0.15, 0.02], [0.525, -0.15, 0.02], [0.425, 0.05, 0.02], [0.425, 0.15, 0.02], [0.325, -0.05, 0.02]])}, {'task_name': 'task2_octuple_pnp1', 'init_xyzs': np.array([[0.4, -0.15, 0.02], [0.4, -0.05, 0.02], [0.4, 0.05, 0.02], [0.4, 0.15, 0.02], [0.45, -0.15, 0.02], [0.45, -0.05, 0.02], [0.45, 0.05, 0.02], [0.45, 0.15, 0.02]]), 'goal_xyzs': np.array([[0.525, 0.2, 0.02], [0.525, 0.0, 0.02], [0.525, -0.2, 0.02], [0.425, -0.2, 0.02], [0.325, -0.2, 0.02], [0.325, 0.0, 0.02], [0.325, 0.2, 0.02], [0.425, 0.2, 0.02]])}, {'task_name': 'task3_octuple_pnp2', 'init_xyzs': np.array([[0.32, -0.15, 0.02], [0.32, 0.05, 0.02], [0.39, -0.05, 0.02], [0.39, 0.15, 0.02], [0.46, -0.15, 0.02], [0.46, 0.05, 0.02], [0.53, -0.05, 0.02], [0.53, 0.15, 0.02]]), 'goal_xyzs': np.array([[0.32, 0.15, 0.02], [0.46, 0.15, 0.02], [0.39, 0.05, 0.02], [0.53, 0.05, 0.02], [0.32, -0.05, 0.02], [0.46, -0.05, 0.02], [0.39, -0.15, 0.02], [0.53, -0.15, 0.02]])}, {'task_name': 'task4_stack1', 'init_xyzs': np.array([[0.4, -0.025, 0.02], [0.4, -0.025, 0.06], [0.4, 0.025, 0.02], [0.4, 0.025, 0.06], [0.45, -0.025, 0.02], [0.45, -0.025, 0.06], [0.45, 0.025, 0.02], [0.45, 0.025, 0.06]]), 'goal_xyzs': np.array([[0.525, 0.05, 0.02], [0.525, -0.05, 0.02], [0.425, -0.15, 0.06], [0.425, -0.15, 0.02], [0.425, 0.15, 0.06], [0.425, 0.15, 0.02], [0.325, -0.05, 0.02], [0.325, 0.05, 0.02]])}, {'task_name': 'task5_stack2', 'init_xyzs': np.array([[0.5, 0.2, 0.06], [0.5, 0.2, 0.02], [0.5, -0.2, 0.06], [0.5, -0.2, 0.02], [0.35, -0.2, 0.06], [0.35, -0.2, 0.02], [0.35, 0.2, 0.06], [0.35, 0.2, 0.02]]), 'goal_xyzs': np.array([[0.325, 0.0, 0.02], [0.325, 0.0, 0.06], [0.425, 0.2, 0.02], [0.425, 0.2, 0.06], [0.525, 0.0, 0.02], [0.525, 0.0, 0.06], [0.425, -0.2, 0.02], [0.425, -0.2, 0.06]])}]
        if self._reward_task_id == 0:
            self._reward_task_id = 2

    def reset(self, seed=None, options=None, *args, **kwargs):
        options = options or {}
        swm_spaces.reset_variation_space(self.variation_space, seed=None, options=options, default_variations=DEFAULT_VARIATIONS)
        (ob, info) = super().reset(*args, options=options, **kwargs)
        if 'state' in options and options['state'] is not None:
            state = options['state']
            assert isinstance(state, np.ndarray), 'State option must be a numpy ndarray!'
            assert state.ndim == 1, 'State option must be a 1D array!'
            assert state.shape[0] == self._model.nq + self._model.nv, f'State option must have shape ({self._model.nq + self._model.nv},)!'
            qpos = state[:self._model.nq]
            qvel = state[self._model.nq:]
            self.set_state(qpos, qvel)
            self.pre_step()
            self.post_step()
            ob = self.compute_observation()
            info = self.get_reset_info()
        return (ob, info)

    def add_objects(self, arena_mjcf):
        cube_outer_mjcf = mjcf.from_path((self._desc_dir / 'cube_outer.xml').as_posix())
        arena_mjcf.include_copy(cube_outer_mjcf)
        distance = 0.05
        for i in range(self._num_cubes):
            cube_mjcf = mjcf.from_path((self._desc_dir / 'cube_inner.xml').as_posix())
            pos = -distance * (self._num_cubes - 1) + 2 * distance * i
            cube_mjcf.find('body', 'object_0').pos[1] = pos
            cube_mjcf.find('body', 'object_target_0').pos[1] = pos
            for tag in ['body', 'joint', 'geom', 'site']:
                for item in cube_mjcf.find_all(tag):
                    if hasattr(item, 'name') and item.name is not None and item.name.endswith('_0'):
                        item.name = item.name[:-2] + f'_{i}'
            arena_mjcf.include_copy(cube_mjcf)
        self._cube_geoms_list = []
        for i in range(self._num_cubes):
            self._cube_geoms_list.append(arena_mjcf.find('body', f'object_{i}').find_all('geom'))
        self._cube_target_geoms_list = []
        for i in range(self._num_cubes):
            self._cube_target_geoms_list.append(arena_mjcf.find('body', f'object_target_{i}').find_all('geom'))
        self.cameras = {'front': {'pos': (1.287, 0.0, 0.509), 'xyaxes': (0.0, 1.0, 0.0, -0.342, 0.0, 0.94)}, 'front_pixels': {'pos': (1.053, -0.014, 0.639), 'xyaxes': (0.0, 1.0, 0.0, -0.628, 0.001, 0.778)}, 'side_pixels': {'pos': (0.414, -0.753, 0.639), 'xyaxes': (1.0, 0.0, 0.0, -0.001, 0.628, 0.778)}, 'side': {'pos': (1.287, 0.0, 0.509), 'xyaxes': (1.0, 0.0, 0.0, -0.001, 0.628, 0.778)}}
        for (camera_name, camera_kwargs) in self.cameras.items():
            arena_mjcf.worldbody.add('camera', name=camera_name, **camera_kwargs)

    def post_compilation_objects(self):
        self._cube_geom_ids_list = [[self._model.geom(geom.full_identifier).id for geom in cube_geoms] for cube_geoms in self._cube_geoms_list]
        self._cube_target_mocap_ids = [self._model.body(f'object_target_{i}').mocapid[0] for i in range(self._num_cubes)]
        self._cube_target_geom_ids_list = [[self._model.geom(geom.full_identifier).id for geom in cube_target_geoms] for cube_target_geoms in self._cube_target_geoms_list]

    def modify_mjcf_model(self, mjcf_model):
        grid_texture = mjcf_model.find('texture', 'grid')
        texture_changed = grid_texture.rgb1 is None or not np.allclose(grid_texture.rgb1, self.variation_space['floor']['color'].value[0])
        texture_changed = texture_changed or (grid_texture.rgb2 is None or not np.allclose(grid_texture.rgb2, self.variation_space['floor']['color'].value[1]))
        grid_texture.rgb1 = self.variation_space['floor']['color'].value[0]
        grid_texture.rgb2 = self.variation_space['floor']['color'].value[1]
        agent_color_changed = np.allclose(mjcf_model.find('material', 'ur5e/robotiq/black').rgba[:3], self.variation_space['agent']['color'].value)
        agent_color_changed = agent_color_changed or np.allclose(mjcf_model.find('material', 'ur5e/robotiq/pad_gray').rgba[:3], self.variation_space['agent']['color'].value)
        mjcf_model.find('material', 'ur5e/robotiq/black').rgba[:3] = self.variation_space['agent']['color'].value
        mjcf_model.find('material', 'ur5e/robotiq/pad_gray').rgba[:3] = self.variation_space['agent']['color'].value
        size_changed = False
        for i in range(self._num_cubes):
            body = mjcf_model.find('body', f'object_{i}')
            if body:
                for geom in body.find_all('geom'):
                    desired_size = self.variation_space['cube']['size'].value[i] * np.ones(3, dtype=np.float32)
                    if geom.size is None or not np.allclose(geom.size, desired_size):
                        size_changed = True
                    geom.size = desired_size
            target_body = mjcf_model.find('body', f'object_target_{i}')
            if target_body:
                for geom in target_body.find_all('geom'):
                    desired_size = self.variation_space['cube']['size'].value[i] * np.ones(3, dtype=np.float32)
                    if geom.size is None or not np.allclose(geom.size, desired_size):
                        size_changed = True
                    geom.size = desired_size
        camera_angle_changed = False
        cameras_to_vary = ['front_pixels', 'side_pixels'] if self._multiview else ['front_pixels']
        for (i, cam_name) in enumerate(cameras_to_vary):
            cam = mjcf_model.find('camera', cam_name)
            cam.xyaxes = perturb_camera_angle(self.cameras[cam_name]['xyaxes'], self.variation_space['camera']['angle_delta'].value[i])
            camera_angle_changed = camera_angle_changed or not np.allclose(cam.xyaxes, self.cameras[cam_name]['xyaxes'])
        light = mjcf_model.find('light', 'global')
        desired_diffuse = self.variation_space['light']['intensity'].value[0] * np.ones(3, dtype=np.float32)
        light_changed = light.diffuse is None or not np.allclose(light.diffuse, desired_diffuse)
        light.diffuse = desired_diffuse
        if size_changed or light_changed or texture_changed or camera_angle_changed or agent_color_changed:
            self.mark_dirty()
        return mjcf_model

    def initialize_episode(self):
        if not hasattr(self, '_prev_qpos'):
            self._prev_qpos = self._data.qpos.copy()
            self._prev_qvel = self._data.qvel.copy()
        for i in range(self._num_cubes):
            for gid in self._cube_geom_ids_list[i]:
                self._model.geom(gid).rgba[:3] = self.variation_space['cube']['color'].value[i]
                self._model.geom(gid).rgba[3] = 1.0
            for gid in self._cube_target_geom_ids_list[i]:
                self._model.geom(gid).rgba[:3] = self.variation_space['cube']['color'].value[i]
        self._data.qpos[self._arm_joint_ids] = self._home_qpos
        mujoco.mj_kinematics(self._model, self._data)
        if self._mode == 'data_collection':
            self.initialize_arm()
            for i in range(self._num_cubes):
                xy = self.variation_space['cube']['start_position'].value[i]
                obj_pos = (*xy, 0.02)
                yaw = self.variation_space['cube']['start_yaw'].value[i]
                obj_ori = lie.SO3.from_z_radians(yaw).wxyz.tolist()
                self._data.joint(f'object_joint_{i}').qpos[:3] = obj_pos
                self._data.joint(f'object_joint_{i}').qpos[3:] = obj_ori
            self.set_new_target(return_info=False)
            self._cur_goal_ob = np.zeros_like(self.compute_observation())
        else:
            if self._permute_blocks:
                permutation = self.np_random.permutation(self._num_cubes)
            else:
                permutation = np.arange(self._num_cubes)
            init_xyzs = self.cur_task_info['init_xyzs'].copy()[permutation]
            goal_xyzs = self.cur_task_info['goal_xyzs'].copy()[permutation]
            saved_qpos = self._data.qpos.copy()
            saved_qvel = self._data.qvel.copy()
            self.initialize_arm()
            for i in range(self._num_cubes):
                self._data.joint(f'object_joint_{i}').qpos[:3] = goal_xyzs[i]
                self._data.joint(f'object_joint_{i}').qpos[3:] = lie.SO3.identity().wxyz.tolist()
                self._data.mocap_pos[self._cube_target_mocap_ids[i]] = goal_xyzs[i]
                self._data.mocap_quat[self._cube_target_mocap_ids[i]] = lie.SO3.identity().wxyz.tolist()
            mujoco.mj_forward(self._model, self._data)
            for _ in range(2):
                self.step(self.action_space.sample())
            self._cur_goal_ob = self.compute_oracle_observation() if self._use_oracle_rep else self.compute_observation()
            if self._render_goal:
                self._cur_goal_rendered = self.render()
            else:
                self._cur_goal_rendered = None
            self._data.qpos[:] = saved_qpos
            self._data.qvel[:] = saved_qvel
            self.initialize_arm()
            for i in range(self._num_cubes):
                obj_pos = init_xyzs[i].copy()
                obj_pos[:2] += self.np_random.uniform(-0.01, 0.01, size=2)
                self._data.joint(f'object_joint_{i}').qpos[:3] = obj_pos
                yaw = self.np_random.uniform(0, 2 * np.pi)
                obj_ori = lie.SO3.from_z_radians(yaw).wxyz.tolist()
                self._data.joint(f'object_joint_{i}').qpos[3:] = obj_ori
                self._data.mocap_pos[self._cube_target_mocap_ids[i]] = goal_xyzs[i]
                self._data.mocap_quat[self._cube_target_mocap_ids[i]] = lie.SO3.identity().wxyz.tolist()
        self.pre_step()
        mujoco.mj_forward(self._model, self._data)
        self.post_step()
        self._success = False

    def initialize_arm(self):
        eff_pos = self.variation_space['agent']['ee_start_position'].value
        cur_ori = self._effector_down_rotation
        yaw = self.np_random.uniform(-np.pi, np.pi)
        rotz = lie.SO3.from_z_radians(yaw)
        eff_ori = rotz @ cur_ori
        T_wp = lie.SE3.from_rotation_and_translation(eff_ori, eff_pos)
        T_wa = T_wp @ self._T_pa
        qpos_init = self._ik.solve(pos=T_wa.translation(), quat=T_wa.rotation().wxyz, curr_qpos=self._home_qpos)
        self._data.qpos[self._arm_joint_ids] = qpos_init
        mujoco.mj_forward(self._model, self._data)

    def set_new_target(self, return_info=True, p_stack=0.5):
        assert self._mode == 'data_collection'
        block_xyzs = np.array([self._data.joint(f'object_joint_{i}').qpos[:3] for i in range(self._num_cubes)])
        top_blocks = []
        for i in range(self._num_cubes):
            for j in range(self._num_cubes):
                if i == j:
                    continue
                if block_xyzs[j][2] > block_xyzs[i][2] and np.linalg.norm(block_xyzs[i][:2] - block_xyzs[j][:2]) < 0.02:
                    break
            else:
                top_blocks.append(i)
        self._target_block = self.np_random.choice(top_blocks)
        stack = len(top_blocks) >= 2 and self.np_random.uniform() < p_stack
        if stack:
            block_idx = self.np_random.choice(list(set(top_blocks) - {self._target_block}))
            block_pos = self._data.joint(f'object_joint_{block_idx}').qpos[:3]
            tar_pos = np.array([block_pos[0], block_pos[1], block_pos[2] + 0.04])
        else:
            xy = self.variation_space['cube']['goal_position'].value[0]
            tar_pos = (*xy, 0.02)
        yaw = self.variation_space['cube']['goal_yaw'].value[0]
        tar_ori = lie.SO3.from_z_radians(yaw).wxyz.tolist()
        for i in range(self._num_cubes):
            if i == self._target_block:
                self._data.mocap_pos[self._cube_target_mocap_ids[i]] = tar_pos
                self._data.mocap_quat[self._cube_target_mocap_ids[i]] = tar_ori
            else:
                self._data.mocap_pos[self._cube_target_mocap_ids[i]] = (0, 0, -0.3)
                self._data.mocap_quat[self._cube_target_mocap_ids[i]] = lie.SO3.identity().wxyz.tolist()
        for i in range(self._num_cubes):
            if self._visualize_info and i == self._target_block:
                for gid in self._cube_target_geom_ids_list[i]:
                    self._model.geom(gid).rgba[3] = 0.2
            else:
                for gid in self._cube_target_geom_ids_list[i]:
                    self._model.geom(gid).rgba[3] = 0.0
        if return_info:
            return (self.compute_observation(), self.get_reset_info())

    def _compute_successes(self):
        cube_successes = []
        for i in range(self._num_cubes):
            obj_pos = self._data.joint(f'object_joint_{i}').qpos[:3]
            tar_pos = self._data.mocap_pos[self._cube_target_mocap_ids[i]]
            if np.linalg.norm(obj_pos - tar_pos) <= 0.04:
                cube_successes.append(True)
            else:
                cube_successes.append(False)
        return cube_successes

    def set_target_pos(self, cube_id, target_pos, target_quat=None):
        num_target_pos = len(self._cube_target_mocap_ids)
        if cube_id < 0 or cube_id >= num_target_pos:
            raise ValueError(f'cube_id out of range (maximum {num_target_pos - 1})')
        mocap_id = self._cube_target_mocap_ids[cube_id]
        self._data.mocap_pos[mocap_id] = np.asarray(target_pos, dtype=np.float64)
        if target_quat is not None:
            self._data.mocap_quat[mocap_id] = target_quat

    def post_step(self):
        cube_successes = self._compute_successes()
        if self._mode == 'data_collection':
            self._success = cube_successes[self._target_block]
        else:
            self._success = all(cube_successes)
        for i in range(self._num_cubes):
            if self._visualize_info and (self._mode == 'task' or i == self._target_block):
                for gid in self._cube_target_geom_ids_list[i]:
                    self._model.geom(gid).rgba[3] = 0.2
            else:
                for gid in self._cube_target_geom_ids_list[i]:
                    self._model.geom(gid).rgba[3] = 0.0

    def get_reset_info(self):
        reset_info = self.compute_ob_info()
        reset_info['env_name'] = self.env_name
        reset_info['target'] = self._cur_goal_ob
        reset_info['success'] = self._success
        return reset_info

    def get_step_info(self):
        ob_info = self.compute_ob_info()
        ob_info['env_name'] = self.env_name
        ob_info['target'] = self._cur_goal_ob
        ob_info['success'] = self._success
        return ob_info

    def add_object_info(self, ob_info):
        for i in range(self._num_cubes):
            ob_info[f'privileged/block_{i}_pos'] = self._data.joint(f'object_joint_{i}').qpos[:3].copy()
            ob_info[f'privileged/block_{i}_quat'] = self._data.joint(f'object_joint_{i}').qpos[3:].copy()
            ob_info[f'privileged/block_{i}_yaw'] = np.array([lie.SO3(wxyz=self._data.joint(f'object_joint_{i}').qpos[3:]).compute_yaw_radians()])
        if self._mode == 'data_collection':
            ob_info['privileged/target_task'] = self._target_task
            target_mocap_id = self._cube_target_mocap_ids[self._target_block]
            ob_info['privileged/target_block'] = self._target_block
            ob_info['privileged/target_block_pos'] = self._data.mocap_pos[target_mocap_id].copy()
            ob_info['privileged/target_block_yaw'] = np.array([lie.SO3(wxyz=self._data.mocap_quat[target_mocap_id]).compute_yaw_radians()])

    def compute_observation(self):
        if self._ob_type == 'pixels':
            return self.get_pixel_observation()
        else:
            xyz_center = np.array([0.425, 0.0, 0.0])
            xyz_scaler = 10.0
            gripper_scaler = 3.0
            ob_info = self.compute_ob_info()
            ob = [ob_info['proprio/joint_pos'], ob_info['proprio/joint_vel'], (ob_info['proprio/effector_pos'] - xyz_center) * xyz_scaler, np.cos(ob_info['proprio/effector_yaw']), np.sin(ob_info['proprio/effector_yaw']), ob_info['proprio/gripper_opening'] * gripper_scaler, ob_info['proprio/gripper_contact']]
            for i in range(self._num_cubes):
                ob.extend([(ob_info[f'privileged/block_{i}_pos'] - xyz_center) * xyz_scaler, ob_info[f'privileged/block_{i}_quat'], np.cos(ob_info[f'privileged/block_{i}_yaw']), np.sin(ob_info[f'privileged/block_{i}_yaw'])])
            return np.concatenate(ob)

    def compute_oracle_observation(self):
        xyz_center = np.array([0.425, 0.0, 0.0])
        xyz_scaler = 10.0
        ob_info = self.compute_ob_info()
        ob = []
        for i in range(self._num_cubes):
            ob.append((ob_info[f'privileged/block_{i}_pos'] - xyz_center) * xyz_scaler)
        return np.concatenate(ob)

    def compute_reward(self):
        if self._reward_task_id is None:
            return super().compute_reward()
        successes = self._compute_successes()
        reward = float(sum(successes) - len(successes))
        return reward

    def render(self, camera='front_pixels', *args, **kwargs):
        return super().render(*args, camera=camera, **kwargs)

    def render_multiview(self, camera='front_pixels', *args, **kwargs):
        if not self._multiview:
            return self.render(*args, camera=camera, **kwargs)
        cam_names = list(self.cameras.keys())
        multi_view = {cam: self.render(*args, camera=cam, **kwargs) for cam in cam_names}
        return multi_view
