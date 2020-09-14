import random
import numpy as np

q_table = np.zeros((1, 9))


class Env:
    def __init__(self):
        self.rewards = {
                'correct': 1,
                'incorrect': -1
                }
        self.actions = range(0, 9)

    def step(self, action):
        correct = input(f"Is {action} correct?")
        if correct.lower().startswith('y'):
            return self.rewards['correct']
        else:
            return self.rewards['incorrect']


def main(learning_rate, env):
    alpha = 0.1
    gamma = 0.6
    epsilon = learning_rate
    state = 0

    while True:
        if random.uniform(0, 1) < epsilon:
            action = random.choice(env.actions)
        else:
            action = np.argmax(q_table[state])

        reward = env.step(action)
        old_value = q_table[state, action]
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * 0)

        q_table[state, action] = new_value

        return action


if __name__ == '__main__':
    env = Env()
    print("Training")
    for i in range(0, 20):
        main(1, env)
    print("Done")
    print(q_table)
