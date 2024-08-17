import gymnasium as gym
import time
import glfw
import torch
from gymnasium import ObservationWrapper
from matplotlib import pyplot as plt
from stable_baselines3 import PPO, TD3, SAC
from stable_baselines3 import DDPG
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
from stable_baselines3.common.noise import NormalActionNoise

import mimoEnv


def main():
    env = gym.make('MIMoReach-v0')

    max_steps = 20000

    # A2C

    # model = A2C(policy="MultiInputPolicy", env=env, verbose=1)

    # DDPG
    #
    # n_actions = env.action_space.shape[-1]
    # action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    # model = DDPG("MultiInputPolicy", env, action_noise=action_noise, verbose=1)

    # # PPO

    model = PPO("MultiInputPolicy", env=env, verbose=1)

    # TD3
    # n_actions = env.action_space.shape[-1]
    # action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    # model = TD3("MultiInputPolicy", env=env, action_noise=action_noise, verbose=1)

    # SAC

    # model = SAC("MultiInputPolicy", env=env, verbose=1)

    model.learn(total_timesteps=10000)
    env = model.get_env()
    obs = env.reset()

    # for i in range (0, 20):
    #     action, _state = model.predict(obs, deterministic=True)
    #     env.reset()
    #     print("The action is:", action)

    # Running
    #
    episode_count = 0
    action_step = 0
    average_step = 0

    for step in range(max_steps):
        action_step += 1
        action, _state = model.predict(obs, deterministic = False)
        obs, reward, done, info = env.step(action)
        if done:
            env.reset()
            episode_count += 1
            print("The step of action is:", action_step)
            average_step = average_step + action_step
            action_step = 0
            print(episode_count, "to success")
    if episode_count > 0:
        current_average_step = average_step / episode_count
        print("The average step is:", current_average_step)

    # env = gym.make('MIMoReach-v0', render_mode = 'human')
    #
    # max_steps = 2000
    #
    # _ = env.reset()
    #
    # start = time.time()
    # for step in range(max_steps):
    #     action = env.action_space.sample()
    #     obs, reward, done, trunc, info = env.step(action)
    #     env.render()
    #     if done or trunc:
    #         env.reset()
    #
    # print("Elapsed time: ", time.time() - start, "Simulation time:", max_steps * env.dt)
    # env.close()


if __name__ == "__main__":
    main()