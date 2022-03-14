import os
import json
import numpy as np
from datetime import datetime

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from collections import deque, namedtuple
import random
from matplotlib import pyplot as plt
from itertools import product

Experience = namedtuple('Experience',
                        ['state', 'action', 'reward', 'next_state', 'done'])  # namedtuple for storing experience


class DQNAgent:
    def __init__(self, env, **kwargs):
        # define the environment
        self.env = env
        self.winning_score = 200
        self.state_size = env.observation_space.shape[0]

        # quantize the action space
        self.bins_per_coord = kwargs.get('bins_per_coord', 4)
        self.action_space = self._get_action_space(self.env, self.bins_per_coord)  # quantized action space
        self.action_size = len(self.action_space)
        self.inverse_action_space = dict(zip(self.action_space.values(), self.action_space.keys()))

        # experience replay memory stored as deque
        self.memory = deque(maxlen=kwargs.get('memory_size', 10_000))  # experience replay memory

        # hyper-parameters
        self.gamma = kwargs.get('gamma', 0.95)  # discount rate
        self.epsilon_max = kwargs.get('epsilon_max', 1.0)  # max exploration rate
        self.epsilon = self.epsilon_max
        self.epsilon_min = kwargs.get('epsilon_min', 0.01)  # min exploration rate
        self.epsilon_decay = kwargs.get('epsilon_decay', 0.995)  # exploration rate decay
        self.learning_rate = kwargs.get('learning_rate', 0.0001)  # learning rate
        self.batch_size = kwargs.get('batch_size', 64)  # size of minibatch
        self.frame_skipping = kwargs.get('frame_skipping', 4)  # number of frames to skip
        self.target_update_freq = kwargs.get('target_update_freq', 10)  # how often to update the target network
        self.max_episode_length = kwargs.get('max_episode_length', 300)  # maximum episode length
        self.n_episodes = kwargs.get('n_episodes', 200)  # number of episodes to train for
        self.n_great_episodes = kwargs.get('n_great_episodes',
                                           5)  # number of required successful episodes to stop training

        # build the networks
        self.model = self._build_model(first_layer_size=kwargs.get('first_layer_size', 24),
                                       second_layer_size=kwargs.get('second_layer_size', 30))

        self.target_model = self._build_model(first_layer_size=kwargs.get('first_layer_size', 24),
                                              second_layer_size=kwargs.get('second_layer_size', 30))

        self.episodes_avg_rewards = []
        self.episodes_scores = []

        self.id = self._get_id()  # unique id for the agent
        self.declaration_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        self.trained_episodes = 0
        self.is_trained = False

    def _build_model(self, first_layer_size=24, second_layer_size=30):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(first_layer_size, input_dim=self.state_size, activation='relu'))
        model.add(Dense(second_layer_size, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=tf.losses.Huber(delta=1.0), optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def _get_id(self):
        obj_dir = os.path.join(os.getcwd(), 'DQN')
        obj_lst = os.listdir(obj_dir)

        if len(obj_lst) == 0:
            return 0

        return max([int(obj.split('_')[0]) for obj in obj_lst]) + 1

    def _get_action_space(self, env, bins_per_coord):
        lower_bounds = env.action_space.low
        upper_bounds = env.action_space.high

        action_ids = {}
        bins = [np.linspace(lower_bounds[i], upper_bounds[i], num=bins_per_coord)
                for i in range(len(lower_bounds))]

        for i, a in enumerate(product(*bins)):
            action_ids[i] = a

        return action_ids

    def remember(self, state, action, reward, next_state, done):
        self.memory.append(Experience(state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            a_num = np.random.choice(list(self.action_space.keys()), size=1)[0]
            return self.action_space[a_num]

        act_values = self.model.predict(state)
        return self.action_space.get(np.argmax(act_values[0]))

    def replay(self):
        # check if memory is enough
        if len(self.memory) < self.batch_size:
            return

        # sample batch_size from memory
        minibatch = np.array(random.sample(self.memory, self.batch_size))
        rewards = minibatch[:, 2]

        next_states = minibatch[:, 3]
        next_states = np.array([s for s in next_states]).reshape(self.batch_size, -1)

        states = minibatch[:, 0]
        states = np.array([s for s in states]).reshape(self.batch_size, -1)

        actions = minibatch[:, 1]
        actions = np.array([self.inverse_action_space[a] for a in actions])

        dones = np.array(minibatch[:, 4], dtype=int)

        target = rewards + self.gamma * np.amax(self.target_model.predict(next_states), axis=1) * (1 - dones)
        target_f = self.model.predict(states)
        target_f[np.arange(self.batch_size), actions] = target
        self.model.fit(states, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i]
        self.target_model.set_weights(target_weights)

    def train(self):
        for e in range(1, self.n_episodes + 1):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            time_step = 0
            episode_rewards = deque(maxlen=100)  # record the episode rewards in the last 100 episodes
            status = 'failed'
            episode_score = 0
            successful_episodes_counter = 0

            self.trained_episodes += 1

            while not done and time_step < self.max_episode_length:
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])

                if time_step % self.frame_skipping == 0:
                    self.remember(state, action, reward, next_state, done)
                    self.replay()
                    episode_rewards.append(reward)

                if time_step % self.target_update_freq == 0:
                    self.target_train()

                state = next_state
                time_step += 1
                episode_score += reward

                if episode_score > self.winning_score:
                    status = 'success'
                    break

            episode_reward = np.mean(episode_rewards)
            print(
                f'Episode: {e}/{self.n_episodes}, episode score: {episode_score}, episode average reward:'
                f' {episode_reward}, finished after {time_step} steps with status {status}.')

            self.episodes_avg_rewards.append(episode_reward)
            self.episodes_scores.append(episode_score)

            successful_episodes_counter = successful_episodes_counter + 1 if status == 'succeed' else 0

            if successful_episodes_counter == self.n_great_episodes:
                print(f'Training finished after {e} episodes, after {self.n_great_episodes} successful episodes.')
                self.is_trained = True
                break

        self.save_results()
        return self.trained_episodes, self.is_trained

    def test(self, n_test_episodes=100):
        scores = []
        for e in range(1, n_test_episodes + 1):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            time_step = 0
            episode_score = 0
            status = 'failed'

            while not done and time_step < self.max_episode_length:
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])

                state = next_state
                time_step += 1
                episode_score += reward

                if episode_score > self.winning_score:
                    status = 'success'
                    break

            print(
                f'Test episode: {e}/{n_test_episodes}, episode score: {episode_score}, finished after {time_step}'
                f' steps with status {status}.')

            scores.append(episode_score)

        return np.mean(scores)

    def plot_rewards(self, results_dir=None, show=True):
        plt.plot(range(1, len(self.episodes_avg_rewards) + 1), self.episodes_avg_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Reward per episode')

        if results_dir is not None:
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            plt.savefig(f'{results_dir}/rewards.png')

        if show:
            plt.show()
        else:
            plt.clf()

    def plot_scores(self, results_dir=None, show=True):
        plt.plot(range(1, len(self.episodes_scores) + 1), self.episodes_scores)
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.title('Score per episode')

        if results_dir is not None:
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            plt.savefig(f'{results_dir}/scores.png')

        if show:
            plt.show()
        else:
            plt.clf()

    def save_parameters(self, results_dir=None):
        params_to_save = {
            'epsilon_max': self.epsilon_max,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'gamma': self.gamma,
            'batch_size': self.batch_size,
            'target_update_freq': self.target_update_freq,
            'frame_skipping': self.frame_skipping,
            'max_episode_length': self.max_episode_length,
            'n_episodes': self.n_episodes,
            'bins_per_coord': self.bins_per_coord,
            'buffer_size': self.memory.maxlen,
        }

        with open(f'{results_dir}/parameters.json', 'w') as f:
            json.dump(params_to_save, f, indent=4)

    def save_results(self):
        dir_path = f'DQN/{self.id}_{self.declaration_time}'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        self.save(f'{dir_path}/dqn.h5')
        self.plot_rewards(results_dir=dir_path, show=False)
        self.plot_scores(results_dir=dir_path, show=False)
        self.save_parameters(results_dir=dir_path)

    def get_rewards(self):
        return self.episodes_avg_rewards

    def get_scores(self):
        return self.episodes_scores


class DDQNAgent:
    def __init__(self, env, **kwargs):
        # define the environment
        self.env = env
        self.winning_score = 200
        self.state_size = env.observation_space.shape[0]

        # quantize the action space
        self.bins_per_coord = kwargs.get('bins_per_coord', 4)
        self.action_space = self._get_action_space(env, self.bins_per_coord)  # quantized action space
        self.action_size = len(self.action_space)
        self.inverse_action_space = dict(zip(self.action_space.values(), self.action_space.keys()))

        # experience replay memory stored as deque
        self.memory = deque(maxlen=kwargs.get('buffer_size', 10_000))  # experience replay memory

        # hyper-parameters
        self.gamma = kwargs.get('gamma', 0.95)  # discount rate
        self.epsilon_max = kwargs.get('epsilon_max', 1.0)  # max exploration rate
        self.epsilon = self.epsilon_max
        self.epsilon_min = kwargs.get('epsilon_min', 0.01)  # min exploration rate
        self.epsilon_decay = kwargs.get('epsilon_decay', 0.995)  # exploration rate decay
        self.learning_rate = kwargs.get('learning_rate', 0.0001)  # learning rate
        self.batch_size = kwargs.get('batch_size', 64)  # size of minibatch
        self.frame_skipping = kwargs.get('frame_skipping', 4)  # number of frames to skip
        self.target_update_freq = kwargs.get('target_update_freq', 10)  # how often to update the target network
        self.max_episode_length = kwargs.get('max_episode_length', 300)  # maximum episode length
        self.n_episodes = kwargs.get('n_episodes', 200)  # number of episodes to train for
        self.n_great_episodes = kwargs.get('n_great_episodes',
                                           5)  # number of required successful episodes to stop training

        # build the networks
        self.model = self._build_model(first_layer_size=kwargs.get('first_layer_size', 24),
                                       second_layer_size=kwargs.get('second_layer_size', 30))

        self.target_model = self._build_model(first_layer_size=kwargs.get('first_layer_size', 24),
                                              second_layer_size=kwargs.get('second_layer_size', 30))

        self.episodes_avg_rewards = []
        self.episodes_scores = []

        self.id = self._get_id()
        self.declaration_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        self.trained_episodes = 0
        self.is_trained = False

    def _build_model(self, first_layer_size=24, second_layer_size=30):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(first_layer_size, input_dim=self.state_size, activation='relu'))
        model.add(Dense(second_layer_size, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=tf.losses.Huber(delta=1.0), optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def _get_id(self):
        obj_dir = os.path.join(os.getcwd(), 'DDQN')
        if not os.path.exists(obj_dir):
            os.makedirs(obj_dir)
        obj_lst = os.listdir(obj_dir)

        if len(obj_lst) == 0:
            return 0

        return max([int(obj.split('_')[0]) for obj in obj_lst]) + 1

    def _get_action_space(self, env, bins_per_coord):
        lower_bounds = env.action_space.low
        upper_bounds = env.action_space.high

        action_ids = {}
        bins = [np.linspace(lower_bounds[i], upper_bounds[i], num=bins_per_coord)
                for i in range(len(lower_bounds))]

        for i, a in enumerate(product(*bins)):
            action_ids[i] = a

        return action_ids

    def remember(self, state, action, reward, next_state, done):
        self.memory.append(Experience(state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            a_num = np.random.choice(list(self.action_space.keys()), size=1)[0]
            return self.action_space[a_num]

        act_values = self.model.predict(state)
        return self.action_space.get(np.argmax(act_values[0]))

    def replay(self):
        # check if memory is enough
        if len(self.memory) < self.batch_size:
            return

        # sample batch_size from memory
        minibatch = np.array(random.sample(self.memory, self.batch_size))
        rewards = minibatch[:, 2]

        next_states = minibatch[:, 3]
        next_states = np.array([s for s in next_states]).reshape(self.batch_size, -1)

        states = minibatch[:, 0]
        states = np.array([s for s in states]).reshape(self.batch_size, -1)

        actions = minibatch[:, 1]
        actions = np.array([self.inverse_action_space[a] for a in actions])

        dones = np.array(minibatch[:, 4], dtype=int)

        # target = rewards + self.gamma * self.target_model.predict(
        #     np.argmax(self.model.predict(next_states), axis=1)) * (1 - dones)
        target = rewards + self.gamma * self.target_model.predict(next_states)[
            np.arange(self.batch_size), np.argmax(self.model.predict(next_states), axis=1)] * (1 - dones)

        # target = rewards + self.gamma * np.amax(self.target_model.predict(next_states), axis=1) * (1 - dones)
        target_f = self.model.predict(states)
        target_f[np.arange(self.batch_size), actions] = target
        self.model.fit(states, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i]
        self.target_model.set_weights(target_weights)

    def train(self):
        for e in range(1, self.n_episodes + 1):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            time_step = 0
            episode_rewards = deque(maxlen=100)  # record the episode rewards in the last 100 episodes
            status = 'failed'
            episode_score = 0
            successful_episodes_counter = 0

            while not done and time_step < self.max_episode_length:
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])

                if time_step % self.frame_skipping == 0:
                    self.remember(state, action, reward, next_state, done)
                    self.replay()
                    episode_rewards.append(reward)

                if time_step % self.target_update_freq == 0:
                    self.target_train()

                state = next_state
                episode_score += reward
                time_step += 1

                if episode_score > self.winning_score:
                    status = 'success'
                    break

            episode_reward = np.mean(episode_rewards)
            print(
                "Episode: {}/{}, episode score: {}, episode average reward: {}, finished after {} steps with status {}.".
                    format(e, self.n_episodes, episode_score, episode_reward, time_step, status))

            self.episodes_avg_rewards.append(episode_reward)
            self.episodes_scores.append(episode_score)

            successful_episodes_counter = successful_episodes_counter + 1 if status == 'succeed' else 0

            if successful_episodes_counter == self.n_great_episodes:
                print(f'Training finished after {e} episodes, after {self.n_great_episodes} successful episodes.')
                self.is_trained = True
                break

        self.save_results()
        return self.trained_episodes, self.is_trained

    def test(self, n_test_episodes=100):
        scores = []
        for e in range(1, n_test_episodes + 1):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            time_step = 0
            episode_score = 0
            status = 'failed'

            while not done and time_step < self.max_episode_length:
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])

                state = next_state
                time_step += 1
                episode_score += reward

                if episode_score > self.winning_score:
                    status = 'success'
                    break

            print(
                f'Test episode: {e}/{n_test_episodes}, episode score: {episode_score}, finished after {time_step}'
                f' steps with status {status}.')

            scores.append(episode_score)

        return np.mean(scores)

    def plot_rewards(self, results_dir=None, show=True):
        plt.plot(range(1, len(self.episodes_avg_rewards) + 1), self.episodes_avg_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Reward per episode')

        if results_dir is not None:
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            plt.savefig(f'{results_dir}/rewards.png')

        if show:
            plt.show()

    def plot_scores(self, results_dir=None, show=True):
        plt.plot(range(1, len(self.episodes_scores) + 1), self.episodes_scores)
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.title('Score per episode')

        if results_dir is not None:
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            plt.savefig(f'{results_dir}/scores.png')

        if show:
            plt.show()

    def save_parameters(self, results_dir=None):
        params_to_save = {
            'epsilon_max': self.epsilon_max,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'gamma': self.gamma,
            'batch_size': self.batch_size,
            'target_update_freq': self.target_update_freq,
            'frame_skipping': self.frame_skipping,
            'max_episode_length': self.max_episode_length,
            'n_episodes': self.n_episodes,
            'bins_per_coord': self.bins_per_coord,
            'buffer_size': self.memory.maxlen,
        }

        with open(f'{results_dir}/parameters.json', 'w') as f:
            json.dump(params_to_save, f, indent=4)

    def save_results(self):
        dir_path = f'DDQN/{self.id}_{self.declaration_time}'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        self.save(f'{dir_path}/ddqn.h5')
        self.plot_rewards(results_dir=dir_path, show=False)
        self.plot_scores(results_dir=dir_path, show=False)
        self.save_parameters(results_dir=dir_path)

    def get_rewards(self):
        return self.episodes_avg_rewards

    def get_scores(self):
        return self.episodes_scores
