import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('metrics/lunarlander.csv')

plt.style.use('ggplot')

fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(15, 15))

# Total reward per episode
axs[0, 0].plot(data['episode'], data['total_reward'], marker='o', linestyle='-', color='b')
axs[0, 0].set_title('Total Reward per Episode')
axs[0, 0].set_xlabel('Episode')
axs[0, 0].set_ylabel('Total Reward')

# Average rolling reward per episode
axs[0, 1].plot(data['episode'], data['average_rolling_reward'], marker='o', linestyle='-', color='r')
axs[0, 1].set_title('Average Rolling Reward per Episode')
axs[0, 1].set_xlabel('Episode')
axs[0, 1].set_ylabel('Average Rolling Reward')

# Loss per episode (if available)
if 'loss' in data.columns and data['loss'].notna().any():
    axs[1, 0].plot(data['episode'], data['loss'], marker='o', linestyle='-', color='g')
    axs[1, 0].set_title('Loss per Episode')
    axs[1, 0].set_xlabel('Episode')
    axs[1, 0].set_ylabel('Loss')

# Duration per episode
axs[1, 1].plot(data['episode'], data['duration'], marker='o', linestyle='-', color='m')
axs[1, 1].set_title('Duration per Episode')
axs[1, 1].set_xlabel('Episode')
axs[1, 1].set_ylabel('Duration (s)')

# Epsilon decay per episode
axs[2, 0].plot(data['episode'], data['epsilon'], marker='o', linestyle='-', color='c')
axs[2, 0].set_title('Epsilon Decay Over Episodes')
axs[2, 0].set_xlabel('Episode')
axs[2, 0].set_ylabel('Epsilon')

# Steps per episode
axs[2, 1].plot(data['episode'], data['steps'], marker='o', linestyle='-', color='y')
axs[2, 1].set_title('Steps per Episode')
axs[2, 1].set_xlabel('Episode')
axs[2, 1].set_ylabel('Steps')

# Adjust layout to prevent overlap
plt.tight_layout(pad=3.0)

plt.show()

