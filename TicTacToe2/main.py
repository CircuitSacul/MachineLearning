# Import Libraries
import os
import pickle
import random
from time import sleep

import numpy as np
from progress.bar import Bar

# Set Globals
# Q-Tables:
C1_TABLE = "c1_q_table.pickle"
C2_TABLE = "c2_q_table.pickle"

# Action/Observation Space
ACTION_SPACE = 9
OBSERVATION_SPACE = 3 ** 9
ACTIONS = {
    0: (0, 0),
    1: (0, 1),
    2: (0, 2),
    3: (1, 0),
    4: (1, 1),
    5: (1, 2),
    6: (2, 0),
    7: (2, 1),
    8: (2, 2),
}

# Rewards/Penalties
WIN_REWARD = 10
LOSE_REWARD = -10
DRAW_REWARD = 0
ACTION_REWARD = -1
INVALID_REWARD = -20

# Board Square States
EMPTY = 0
X = 1
O = 2

# Game States
GOING = "going"
WIN = "win"
DRAW = "draw"

# Action Types
INVALID = "invalidAction"


def color(r, g, b, text):
    return "\033[38;2;{};{};{}m{}\033[38;2;255;255;255m".format(r, g, b, text)


def get_name(in_type):
    name = str(in_type).split(" ")[1].replace(">", "")
    return name


class DefaultCheck:
    def __init__(self, include=[], exclude=[]):
        self.include = include
        self.exclude = exclude

    def check(self, user_input):
        if self.include != [] and user_input not in self.include:
            print(color(255, 0, 0, f"Input must be in {self.include}"))
            return False
        if self.exclude != [] and user_input in self.exclude:
            print(color(255, 0, 0, f"Input cannot be in {self.exclude}"))
            return False
        return True


def safe_input(prompt=None, in_type=str, check=None):
    user_input = input(prompt if prompt is not None else "")

    try:
        converted = in_type(user_input)
    except ValueError:
        print(color(255, 0, 0, f"Expected type {get_name(in_type)}"))
        return safe_input(prompt, in_type, check)

    valid = check(converted) if check is not None else True
    if not valid:
        return safe_input(prompt, in_type, check)

    return converted


class Game:
    def __init__(self):
        self.board = np.zeros((3, 3))
        self.status = GOING

    def step(self, action, number):
        if not self.is_valid(action):
            return INVALID

        self.board[ACTIONS[action]] = number

        if self.is_winner(number):
            self.status = WIN
            return WIN
        elif self.is_draw():
            self.status = DRAW
            return DRAW

    def get_state(self):
        max_num = 3
        arr2 = self.board.reshape(-1)
        degrees = max_num ** np.arange(len(arr2))
        state = arr2 @ degrees
        return state

    def is_valid(self, action):
        if self.board[ACTIONS[action]] != EMPTY:
            return False
        return True

    def win_indexes(self, n):
        for r in range(n):
            yield [(r, c) for c in range(n)]
        for c in range(n):
            yield [(r, c) for r in range(n)]
        yield [(i, i) for i in range(n)]
        yield [(i, n - 1 - i) for i in range(n)]

    def is_winner(self, decorator):
        n = len(self.board)
        for indexes in self.win_indexes(n):
            if all(self.board[r][c] == decorator for r, c in indexes):
                return True
        return False

    def is_draw(self):
        for row in self.board:
            if EMPTY in row:
                return False
        return True

    def print_board(self):
        os.system("clear")
        for row in self.board:
            for item in row:
                if item == X:
                    print("X", end=" ")
                elif item == O:
                    print("O", end=" ")
                else:
                    print("_", end=" ")
            print()


class Computer:
    def __init__(self, number, q_table_path, alpha=0, gamma=0, epsilon=0):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.q_table_path = q_table_path
        self.q_table = self.load_q_table(q_table_path)

        self.last_action = None
        self.last_state = None

        self.going = True
        self.number = number

    def load_q_table(self, path):
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        else:
            return np.zeros((OBSERVATION_SPACE, ACTION_SPACE))

    def save_q_table(self, path=None):
        if path is None:
            path = self.q_table_path
        with open(path, "wb+") as f:
            pickle.dump(self.q_table, f)

    def reward(self, old_state, action, reward, new_state):
        old_value = self.q_table[int(old_state), int(action)]
        next_max = np.max(self.q_table[int(new_state)])

        new_value = (1 - self.alpha) * old_value + self.alpha * (
            reward + self.gamma * next_max
        )
        self.q_table[int(old_state), int(action)] = new_value

    def get_action(self, game, is_random=None):
        if is_random is None:
            is_random = random.uniform(0, 1) < self.epsilon

        state = game.get_state()

        if is_random:
            action = random.randrange(0, ACTION_SPACE)
        else:
            action = np.argmax(self.q_table[int(state)])

        if not game.is_valid(action):
            self.q_table[int(state), int(action)] = INVALID_REWARD
            return self.get_action(game, is_random=is_random)
        return action

    def next_move(self, game):
        state = game.get_state()
        if self.last_state is not None:
            if game.status != GOING:
                if game.status == WIN:
                    if game.is_winner(self.number):
                        self.reward(
                            self.last_state,
                            self.last_action,
                            WIN_REWARD,
                            state,
                        )
                    else:
                        self.reward(
                            self.last_state,
                            self.last_action,
                            LOSE_REWARD,
                            state,
                        )
                elif game.status == DRAW:
                    self.reward(
                        self.last_state, self.last_action, DRAW_REWARD, state
                    )
                self.going = False
                return
            else:
                self.reward(
                    self.last_state, self.last_action, ACTION_REWARD, state
                )

        action = self.get_action(game)
        game.step(action, self.number)

        self.last_state = state
        self.last_action = action


class Player:
    def __init__(self, number):
        self.number = number
        self.going = True

    def next_move(self, game):
        game.print_board()
        if game.status != GOING:
            if game.status == DRAW:
                print("Draw")
            elif game.status == WIN and game.is_winner(self.number):
                print(color(0, 255, 0, "You Win"))
            else:
                print(color(255, 0, 0, "You Lose"))
            self.going = False
            return

        col = safe_input(prompt="Col: ", in_type=int) - 1
        row = safe_input(prompt="Row: ", in_type=int) - 1
        action = None
        for item in ACTIONS:
            if ACTIONS[item] == (row, col):
                action = item
        if action is None or not game.is_valid(action):
            print(color(255, 0, 0, "Invalid Move"))
            sleep(1)
            return self.next_move(game)
        game.step(action, self.number)


def train(p1, p2, loops):
    game = Game()
    p_list = [p1, p2]

    with Bar("Training", max=loops/1000) as bar:
        for i in range(0, loops):
            if i % 1000 == 0:
                bar.next()
            game.__init__()
            p1.going = True
            p2.going = True
            while p1.going or p2.going:
                for ai in p_list:
                    if ai.going:
                        ai.next_move(game)

    for ai in p_list:
        ai.save_q_table()


def play(p1, p2):
    game = Game()
    p_list = [p1, p2]

    while p1.going or p2.going:
        for p in p_list:
            if p.going:
                p.next_move(game)


if __name__ == "__main__":
    print(
        color(
            255, 255, 255, "Q-Learning TicTacToe AI Created by Lucas Daniels"
        )
    )
    check = DefaultCheck(include=[0, 1, 2])
    running = True
    while running:
        choice = safe_input(
            prompt="0: Quit\n1: Play\n2: Train\n>",
            in_type=int,
            check=check.check,
        )
        if choice == 2:
            c1 = Computer(1, C1_TABLE, alpha=0.1, gamma=0.8, epsilon=0.1)
            c2 = Computer(2, C2_TABLE, alpha=0.1, gamma=0.8, epsilon=0.1)
            train(
                c1,
                c2,
                safe_input(
                    prompt="How many games? (Recommended is 10,000 or 100,000)\n>",
                    in_type=int,
                ),
            )
        elif choice == 1:
            is_p_first = (
                safe_input(prompt="Do you want to play first? (y/n)\n>")
                .lower()
                .startswith("y")
            )
            if is_p_first:
                p1 = Player(1)
                p2 = Computer(2, C2_TABLE)
            else:
                p2 = Player(2)
                p1 = Computer(1, C1_TABLE)
            play(p1, p2)
        elif choice == 0:
            print("Exitting")
            running = False
