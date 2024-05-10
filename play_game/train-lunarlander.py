import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.regularizers import l2
import numpy as np
import gymnasium as gym
from datetime import datetime
from collections import deque
import random
import time
import csv


env = gym.make("LunarLander-v2", render_mode="human")
env.reset()

model = Sequential([
    tf.keras.layers.Input(shape=(env.observation_space.shape[0],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(env.action_space.n, activation='linear')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')

model.summary()


episodes = 500
memory = deque(maxlen=100000)
batch_size = 512
train_data = {'states': [], 'targets': []}

start_epsilon = 1.0
epsilon_min = 0.3
epsilon = start_epsilon
epsilon_decay = 0.995
gamma = 0.99

rolling_rewards = deque(maxlen=100)

metrics_filepath = 'metrics/lunarlander.csv'
metrics_key = datetime.now().strftime('%Y%m%d%H%M%S')


def write_to_csv(metric):
    with open(metrics_filepath, 'a', newline='') as file:
        fieldnames = list(metric.keys())

        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if file.tell() == 0:
            writer.writeheader()

        writer.writerow(metric)


def choose_action(state):
    if np.random.rand() < epsilon:
        return env.action_space.sample()

    q_values = model.predict(state)

    return np.argmax(q_values[0])


def replay_experience(replay_memory):
    if len(replay_memory) <= batch_size:
        return # Ensure there is enough data in memory to sample a batch

    # Sample a random minibatch of transitions (state, action, reward, next_state, done) from the memory
    minibatch = random.sample(replay_memory, batch_size)

    # Extract elements of each experience
    states, actions, rewards, next_states, dones = map(np.array, zip(*minibatch))

    # Reshape states and next_states for prediction
    states = np.vstack(states)  # Stack states vertically to create a proper batch
    next_states = np.vstack(next_states)  # Stack next states vertically

    # Predict Q-values for next_states and current states in one go
    next_q_values = model.predict(next_states)
    current_q_values = model.predict(states)

    # Calculate the maximum Q-value for each next state
    max_next_q_values = np.max(next_q_values, axis=1)

    # Compute target Q-values using the Bellman equation
    target_q_values = rewards + gamma * max_next_q_values * (1 - dones)

    # Update the Q-values for the actions taken
    targets = current_q_values.copy()  # Start with current predictions
    for idx, action in enumerate(actions):
        targets[idx, action] = target_q_values[idx]  # Update only the actions that were taken

    return states, targets


def train_model(train_data):
    if train_data['states']:
        # Train the model on the states and their updated Q-values
        history = model.fit(np.array(train_data['states']), np.array(train_data['targets']), epochs=1, verbose=1)
        train_data['states'].clear()
        train_data['targets'].clear()
        return history.history['loss'][0]


for e in range(episodes):
    start_time = time.time()

    state, _ = env.reset()
    state = np.reshape(state, [1, env.observation_space.shape[0]])
    done = False
    total_reward = 0
    step_count = 0

    while True:
        action = choose_action(state)

        observation, reward, terminated, truncated, info = env.step(action)
        next_state = np.reshape(observation, [1, env.observation_space.shape[0]])
        done = terminated or truncated

        memory.append((state, action, reward, next_state, done))

        state = next_state
        total_reward += reward
        step_count += 1

        if done:
            end_time = time.time()
            episode_duration = end_time - start_time
            rolling_rewards.append(total_reward)
            average_rolling_reward = sum(rolling_rewards) / len(rolling_rewards)
            training_data = replay_experience(memory)
            loss = None
            if training_data:
                states, targets = training_data
                train_data['states'].extend(states)
                train_data['targets'].extend(targets)
                if len(train_data['states']) >= batch_size:
                    loss = train_model(train_data)

            write_to_csv({
                'metrics_key': metrics_key,
                'episode': e+1,
                'total_reward': total_reward,
                'average_rolling_reward': average_rolling_reward,
                'loss': loss,
                'duration': episode_duration,
                'epsilon': epsilon,
                'steps': step_count
            })

            print(f"episode: {e+1} - reward: {total_reward:.2f} - duration: {episode_duration:.2f}s - epsilon: {epsilon:.3f} - steps: {step_count} - reward_avg: {average_rolling_reward:.2f} - loss: {loss if loss is not None else 'N/A'}")
            break

    epsilon = max(epsilon_min, epsilon * epsilon_decay)


model.save('models/lunarlander.keras')
env.close()

