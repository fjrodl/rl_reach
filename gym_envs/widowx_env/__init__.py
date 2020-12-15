from gym.envs.registration import register


# Pybullet environment + fixed goal + gym environment
register(
    id='widowx_reacher-v1',
    entry_point='widowx_env.envs.1_widowx_pybullet_fixed_gymEnv:WidowxEnv',
    max_episode_steps=100)

# Pybullet environment + fixed goal + goal environment
register(
    id='widowx_reacher-v2',
    entry_point='widowx_env.envs.2_widowx_pybullet_fixed_goalEnv:WidowxEnv',
    max_episode_steps=100)

# Pybullet environment + random goal + gym environment
register(
    id='widowx_reacher-v3',
    entry_point='widowx_env.envs.3_widowx_pybullet_random_gymEnv:WidowxEnv',
    max_episode_steps=100)

# Pybullet environment + random goal + goal environment
register(
    id='widowx_reacher-v4',
    entry_point='widowx_env.envs.4_widowx_pybullet_random_goalEnv:WidowxEnv',
    max_episode_steps=100)


# # Pybullet environment + fixed goal + gym environment + obs2
# register(id='widowx_reacher-v2',
#          entry_point='widowx_env.envs.1_widowx_pybullet_fixed_gymEnv_obs2:WidowxEnv',
#          max_episode_steps=100
#          )

# # Pybullet environment + fixed goal + gym environment + obs3
# register(id='widowx_reacher-v3',
#          entry_point='widowx_env.envs.1_widowx_pybullet_fixed_gymEnv_obs3:WidowxEnv',
#          max_episode_steps=100
#          )

# # Pybullet environment + fixed goal + gym environment + obs4
# register(id='widowx_reacher-v4',
#          entry_point='widowx_env.envs.1_widowx_pybullet_fixed_gymEnv_obs4:WidowxEnv',
#          max_episode_steps=100
#          )

# # Pybullet environment + fixed goal + gym environment + obs5
# register(id='widowx_reacher-v5',
#          entry_point='widowx_env.envs.1_widowx_pybullet_fixed_gymEnv_obs5:WidowxEnv',
#          max_episode_steps=100
#          )

# #############

# # Pybullet environment + fixed goal + gym environment + reward 2
# register(id='widowx_reacher-v9',
#          entry_point='widowx_env.envs.5_widowx_pybullet_fixed_gymEnv_reward2:WidowxEnv',
#          max_episode_steps=100
#          )
