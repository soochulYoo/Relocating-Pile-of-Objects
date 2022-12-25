from gym.envs.registration import register

register(
    id='relocate-v0',
    entry_point='gym_relocate.envs:RelocateEnv',
    max_episode_steps=200,
)