import random

import matplotlib.pyplot as plt
import numpy as np


class WindyGridworldEnv:
    def __init__(self, width=10, high=7, wind=None, mode='human', actions=4, kind="grid", episode_length=100):
        self.width = width
        self.high = high
        self.wind = wind
        self.mode = mode
        self.kind = kind
        self.actions = actions
        self.state = [0, 0]
        self.goal = [7, 3]
        self.counter = 0
        self.last_counter = 0
        self.episode_length = episode_length
        self.map = np.zeros((self.width, self.high))
        if self.wind is None:
            self.wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
        if self.kind == 'grid':
            self.actions = 4
        if self.kind == 'king':
            self.actions = 8
        if self.mode == 'human':
            self.path_x = []
            self.path_y = []

    def reset(self):
        self.last_counter = self.counter
        self.counter = 0
        for i in range(self.width):
            wind_strength = self.wind[i]
            self.map[i, :] = wind_strength
        self.state = [0, 3]

        if self.mode == 'human':
            self.path_x = []
            self.path_y = []

            self.path_x.append(self.state[0])
            self.path_y.append(self.state[1])

        return self.state

    def grid_action(self, action):
        curr_x, curr_y = self.state
        if action == 0:
            # left
            curr_x = max(0, curr_x - 1)
        elif action == 1:
            # Up
            curr_y = min(self.high-1, curr_y + 1)
        elif action == 2:
            # right
            curr_x = min(self.width-1, curr_x + 1)
        elif action == 3:
            # down
            curr_y = max(0, curr_y - 1)
        else:
            raise RuntimeError('action is incorrect')

        self.state = [curr_x, curr_y]

    def king_action(self, action):
        curr_x, curr_y = self.state
        if action == 0:
            # Up
            curr_y = min(self.high - 1, curr_y + 1)
        elif action == 1:
            # Up + right
            curr_y = min(self.high-1, curr_y + 1)
            curr_x = min(self.width - 1, curr_x + 1)
        elif action == 2:
            # right
            curr_x = min(self.width-1, curr_x + 1)
        elif action == 3:
            # down + right
            curr_y = max(0, curr_y - 1)
            curr_x = min(self.width-1, curr_x + 1)
        elif action == 4:
            # down
            curr_y = max(0, curr_y - 1)
        elif action == 5:
            # down + left
            curr_y = max(0, curr_y - 1)
            curr_x = max(0, curr_x - 1)
        elif action == 6:
            # left
            curr_x = max(0, curr_x - 1)
        elif action == 7:
            # up + left
            curr_y = min(self.high - 1, curr_y + 1)
            curr_x = max(0, curr_x - 1)
        else:
            raise RuntimeError('action is incorrect')

        self.state = [curr_x, curr_y]

    def wind_correction(self):
        curr_x, curr_y = self.state
        wind_strength = self.wind[curr_x]
        # wind_strength = random.choice([wind_strength, wind_strength - 1, wind_strength + 1])
        curr_y = min(self.high - 1, curr_y + wind_strength)
        curr_y = max(0, curr_y)

        self.state = [curr_x, curr_y]

    def step(self, action):

        # change state
        if self.kind == 'grid':
            self.grid_action(action)
        elif self.kind == 'king':
            self.king_action(action)
        else:
            raise RuntimeError('no kind')

        self.wind_correction()

        # change reward and check if done
        self.counter += 1

        if self.state == self.goal:
            reward = 1
            done = True
        else:
            done = self.counter == self.episode_length
            reward = -1

        # check for error
        if self.counter > self.episode_length:
            raise RuntimeError('too long')

        # for render
        if self.mode == 'human':
            self.path_x.append(self.state[0])
            self.path_y.append(self.state[1])

        return self.state, reward, done

    def action_values(self):
        return np.arange(self.actions)

    def render(self):
        if self.mode == 'human':
            plt.cla()
            plt.imshow(self.map.T, origin='lower')
            plt.plot(self.path_x, self.path_y, c='red', alpha=0.7, linewidth=5.5)
            plt.scatter(self.state[0], self.state[1], s=100, c='red')
            plt.scatter(self.goal[0], self.goal[1], s=100, c='blue')
            plt.pause(0.05)


def run_env():
    env = WindyGridworldEnv()
    times = 100000
    state = env.reset()
    for i in range(times):
        action = random.choice(env.action_values())
        next_state, reward, done = env.step(action)

        if done:
            state = env.reset()
        else:
            state = next_state
        if i % 100 == 0:
            env.render()
        print(f'\r[iter {i}] step: {env.counter}, state: {state}, reward: {reward}, done: {done}', end='')
        if done:
            print()

    plt.show()


if __name__ == '__main__':
    run_env()
