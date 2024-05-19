import tensorflow as tf
import gymnasium as gym
from tensorflow.keras.models import load_model
import time
import numpy as np


model = load_model('models/cartpole.keras')
model.summary()

env = gym.make("CartPole-v1", render_mode="human")
env.reset()

episodes = 10

for e in range(episodes):
    start_time = time.time()

    state, _ = env.reset()
    state = np.reshape(state, [1, env.observation_space.shape[0]])
    done = False
    total_reward = 0
    step_count = 0

    while True:
        action = np.argmax(model.predict(state, verbose=0))

        observation, reward, terminated, truncated, info = env.step(action)
        next_state = np.reshape(observation, [1, env.observation_space.shape[0]])
        done = terminated or truncated

        state = next_state
        total_reward += reward
        step_count += 1

        if done:
            end_time = time.time()
            episode_duration = end_time - start_time
            print(f"episode: {e+1} - reward: {total_reward:.2f} - duration: {episode_duration:.2f}s - steps: {step_count}")
            break


env.close()

