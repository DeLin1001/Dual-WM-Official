from gymnasium.envs import registration
WORLDS = set()
DISCRETE_WORLDS = set()

def register(id, entry_point, discrete=False):
    registration.register(id=id, entry_point=entry_point)
    WORLDS.add(id)
    if discrete:
        DISCRETE_WORLDS.add(id)
register('swm/TwoRoom-v1', 'stable_worldmodel.envs.two_room.env:TwoRoomEnv', discrete=False)
register('swm/ReacherDMControl-v0', 'stable_worldmodel.envs.dmcontrol.reacher:ReacherDMControlWrapper', discrete=False)
register('swm/PushT-v1', 'stable_worldmodel.envs.pusht.env:PushT', discrete=False)
register('swm/OGBCube-v0', 'stable_worldmodel.envs.ogbench.cube_env:CubeEnv', discrete=False)
register('swm/SokobanLong-v0', 'stable_worldmodel.envs.sokoban.long_env:SokobanLongEnv', discrete=True)
