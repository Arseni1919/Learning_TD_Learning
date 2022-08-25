import numpy as np
import random
import matplotlib.pyplot as plt
from impl_env_windy_gridworld import WindyGridworldEnv


def epsilon_greedy(q_func, state, actions):
    if random.random() > EPSILON:
        actions_dict = {action: q_func[state[0], state[1], action] for action in actions}
        action = max(actions_dict, key=actions_dict.get)
        return action

    action = random.choice(actions)
    return action


def run_q_learning(q_func, env):
    times = 100000
    actions = env.action_values()
    episodes = 1
    total_return = 0
    state = env.reset()
    for i in range(times):
        action = epsilon_greedy(q_func, state, actions)
        next_state, reward, done = env.step(action)

        # Learning
        if not done:
            actions_dict = {action: q_func[next_state[0], next_state[1], action] for action in actions}
            action = max(actions_dict, key=actions_dict.get)
            max_q = q_func[next_state[0], next_state[1], action]
            new_q = q_func[state[0], state[1], action] + ALPHA * (reward + GAMMA * max_q - q_func[state[0], state[1], action])
            q_func[state[0], state[1], action] = new_q
        else:
            new_q = q_func[state[0], state[1], action] + ALPHA * (reward - q_func[state[0], state[1], action])
            q_func[state[0], state[1], action] = new_q

        # render + metrics
        if i % 20 == 0 or episodes > 250:
            env.render()

        total_return += reward
        print(f'\r[iter {i}, episode {episodes} step {env.last_counter}] state: {state}, reward: {total_return}, done: {done}',
              end='')

        if not done:
            state = next_state
        else:
            # End of episode
            episodes += 1
            total_return = 0
            state = env.reset()
            print()

    plt.show()


def main():
    # env = WindyGridworldEnv()
    # env = WindyGridworldEnv(kind='king', episode_length=200)
    env = WindyGridworldEnv(episode_length=200)
    q_func = np.zeros((env.width, env.high, env.actions))
    run_q_learning(q_func, env)


if __name__ == '__main__':
    EPSILON = 0.2
    ALPHA = 0.2
    GAMMA = 0.9
    main()
