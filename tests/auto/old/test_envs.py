import gym
import widowx_env
import pytest
from stable_baselines3.common.env_checker import check_env



@pytest.mark.parametrize("env_id", ['widowx_reacher-v1', 'widowx_reacher-v2', 'widowx_reacher-v3', 'widowx_reacher-v4'])
def test_envs(env_id):
    env = gym.make(env_id)
    # check_env(env)

    env.reset()

    for t in range(100):
        action = env.action_space.sample()
        _, _, done, _ = env.step(action)
    
    assert t == 99
    assert done == True

    env.close()


# doesn't work due to max_episode_steps=100 in __init__.py
# @pytest.mark.parametrize("env_id", ['widowx_reacher-v2', 'widowx_reacher-v4'])
# def test_goalEnvs(env_id):
#     env = gym.make(env_id)
#     assert env == isinstance(env, gym.GoalEnv)


