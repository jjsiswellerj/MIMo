import gymnasium as gym
import os
import torch
from matplotlib import pyplot as plt
from stable_baselines3 import PPO, TD3, SAC, DDPG, A2C
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.results_plotter import ts2xy, load_results

import mimoEnv


def train_and_plot(model_class, env_id, policy, log_dir, total_timesteps):
    # 确保日志目录存在
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    env = gym.make(env_id)
    env = Monitor(env, log_dir)

    if model_class in [DDPG, TD3]:
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        model = model_class(policy, env, action_noise=action_noise, verbose=1)
    else:
        model = model_class(policy, env, verbose=1)

    model.learn(total_timesteps=total_timesteps)
    env = model.get_env()

    x, y = ts2xy(load_results(log_dir), x_axis='timesteps')
    y_avg = [sum(y[:i + 1]) / (i + 1) for i in range(len(y))]

    return x, y_avg


def main():
    env_id = 'MIMoReach-v0'
    total_timesteps = 1000000
    log_dirs = {
        "PPO": "./logs/ppo/",
        "A2C": "./logs/a2c/",
        "SAC": "./logs/sac/",
        "DDPG": "./logs/ddpg/",
        "TD3": "./logs/td3/"
    }

    if not os.path.exists("./logs/"):
        os.makedirs("./logs/")

    results = {}

    results["PPO"] = train_and_plot(PPO, env_id, "MultiInputPolicy", log_dirs["PPO"], total_timesteps)
    results["A2C"] = train_and_plot(A2C, env_id, "MultiInputPolicy", log_dirs["A2C"], total_timesteps)
    results["SAC"] = train_and_plot(SAC, env_id, "MultiInputPolicy", log_dirs["SAC"], total_timesteps)
    results["DDPG"] = train_and_plot(DDPG, env_id, "MultiInputPolicy", log_dirs["DDPG"], total_timesteps)
    results["TD3"] = train_and_plot(TD3, env_id, "MultiInputPolicy", log_dirs["TD3"], total_timesteps)

    plt.figure(figsize=(12, 8))

    for algo, (x, y_avg) in results.items():
        plt.plot(x, y_avg, label=algo)

    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.title('Average reward over time for different algorithms')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
