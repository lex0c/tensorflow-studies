import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import gymnasium as gym
import datetime
from collections import deque
import random


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


env = gym.make("LunarLander-v2", render_mode="human")

env.reset()

model = Sequential([
    Dense(24, input_shape=(env.observation_space.shape[0],), activation='relu'),
    Dense(24, activation='relu'),
    Dense(env.action_space.n, activation='linear')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['accuracy'])

model.summary()


episodes = 1000
memory = deque(maxlen=2048)

start_epsilon = 1.0
epsilon_min = 0.1
epsilon = start_epsilon
epsilon_decay = start_epsilon / (episodes / 2)


def choose_action(state):
    if np.random.rand() <= epsilon:
        return env.action_space.sample()

    q_values = model.predict(state)

    return np.argmax(q_values[0])


def replay_experience(replay_memory, batch_size=64, gamma=0.99):
    if len(replay_memory) < batch_size:
        return

    minibatch = random.sample(replay_memory, batch_size)

    states, actions, rewards, next_states, dones = map(np.array, zip(*minibatch))

    next_states = np.array([x.reshape((env.observation_space.shape[0],)) for x in next_states])

    target_q_values = rewards + gamma * np.max(model.predict(next_states), axis=1) * (1 - dones)

    states = np.array([x.reshape((env.observation_space.shape[0],)) for x in states])

    target_f = model.predict(states)

    for i, action in enumerate(actions):
        target_f[i][action] = target_q_values[i]

    model.fit(states, target_f, epochs=1, verbose=0)


for e in range(episodes):
    state, _ = env.reset()
    state = np.reshape(state, [1, env.observation_space.shape[0]])
    done = False

    while True:
        action = choose_action(state)

        observation, reward, terminated, truncated, info = env.step(action)
        next_state = np.reshape(observation, [1, env.observation_space.shape[0]])
        done = terminated or truncated

        memory.append((state, action, reward, next_state, done))

        state = next_state
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if done:
            break

        replay_experience(memory)


model.save('lunarlander.keras')
env.close()

