# pip install minari
# pip install chardet
# https://minari.farama.org/main/content/basic_usage/
# /Users/yhhan/.minari
import minari
from minari import DataCollector
import gymnasium as gym

env = gym.make('MountainCarContinuous-v0')
env = DataCollector(env, record_infos=True, max_buffer_steps=100000)

total_episodes = 100

for _ in range(total_episodes):
    env.reset(seed=123)
    while True:
        # random action policy
        action = env.action_space.sample()
        obs, rew, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break

dataset = env.create_dataset(
    dataset_id="MountainCarContinuous-v0-medium-v0",
    algorithm_name="Random-Policy",
    code_permalink="https://github.com/Farama-Foundation/Minari",
    author="Farama",
    author_email="contact@farama.org"
)

print(minari.list_local_datasets())
print()

dataset2 = minari.load_dataset("MountainCarContinuous-v0-medium-v0")

for episode_data in dataset2:
    print(episode_data)
    print(episode_data.id)
    print(episode_data.seed)
    print(episode_data.total_timesteps)
    print(episode_data.observations.shape)
    print(episode_data.actions.shape)
    print(episode_data.rewards.shape)
    print(episode_data.terminations.shape)
    print(episode_data.truncations.shape)
    print(episode_data.infos)
    print()