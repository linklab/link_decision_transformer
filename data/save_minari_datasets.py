import os
import gymnasium as gym
import numpy as np

import collections
import pickle

from minari import DataCollector


def get_dataset(env_name, dataset_id):
    env = gym.make(env_name)
    env = DataCollector(env, record_infos=True, max_buffer_steps=100000)

    total_episodes = 1000

    for episode in range(total_episodes):
        env.reset(seed=123)
        while True:
            # random action policy
            action = env.action_space.sample()
            obs, rew, terminated, truncated, info = env.step(action)

            if terminated:
                print(episode, terminated, truncated, rew, "!!!!!")

            if terminated or truncated:
                break

    dataset = env.create_dataset(
        dataset_id=dataset_id,
        algorithm_name="Random-Policy",
        code_permalink="https://github.com/Farama-Foundation/Minari",
        author="Farama",
        author_email="contact@farama.org"
    )
    episodes = []

    for episode_data in dataset:
        episode_dict = collections.defaultdict(list)
        episode_dict['observations'] = np.array(episode_data.observations)[:-1]
        episode_dict['actions'] = np.array(episode_data.actions)
        episode_dict['rewards'] = np.array(episode_data.rewards)
        episode_dict['terminations'] = np.array(episode_data.terminations)
        episode_dict['truncations'] = np.array(episode_data.truncations)
        episode_dict['infos'] = np.array(episode_data.infos)

        episodes.append(episode_dict)

    return episodes


def download_minari_data(env_name="MountainCarContinuous-v0", dataset_id="MountainCarContinuous-v0-medium-v0"):
    # datasets = []

    data_dir = 'save/'

    print(data_dir)

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    pkl_file_path = os.path.join(data_dir, env_name)

    print("processing: ", env_name)

    episodes = get_dataset(env_name, dataset_id=dataset_id)

    returns = np.array([np.sum(p['rewards']) for p in episodes])
    num_samples = np.sum([p['rewards'].shape[0] for p in episodes])
    print(f'Number of samples collected: {num_samples}')
    print(f'Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}')

    with open(f'{pkl_file_path}.pkl', 'wb') as f:
        pickle.dump(episodes, f)


if __name__ == "__main__":
    download_minari_data()
