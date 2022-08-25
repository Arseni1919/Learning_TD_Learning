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


def run_sarsa(q_func, env):
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
            next_action = epsilon_greedy(q_func, next_state, actions)
            new_q = q_func[state[0], state[1], action] + ALPHA * (reward + GAMMA * q_func[next_state[0], next_state[1], next_action] - q_func[state[0], state[1], action])
            q_func[state[0], state[1], action] = new_q
        else:
            new_q = q_func[state[0], state[1], action] + ALPHA * (reward - q_func[state[0], state[1], action])
            q_func[state[0], state[1], action] = new_q

        # render + metrics
        if i % 20 == 0 or episodes > 450:
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
    env = WindyGridworldEnv(kind='king')
    q_func = np.zeros((env.width, env.high, env.actions))
    run_sarsa(q_func, env)


if __name__ == '__main__':
    EPSILON = 0.1
    ALPHA = 0.5
    GAMMA = 0.9
    main()
