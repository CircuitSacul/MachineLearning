import numpy as np
import pickle
import os
import random
from time import sleep

Q_TABLE_PATH = 'q_table.pickle'


class Player:
    def __init__(self):
        pass

    def get_action(self, game):
        col = int(input('Col: '))-1
        row = int(input('Row: '))-1
        for m in game.actions:
            if game.actions[m] == (row, col):
                move = m
        if not game.is_valid(move):
            print("Invalid Move")
            return self.get_action(game)
        return move

    def reward(self, reward, game):
        game.print_board()
        if reward == -10:
            print("You Lost")
        elif reward == 10:
            print("You Win")


class Computer:
    def __init__(self, game, learn_rate=0, explore_rate=0):
        self.action_space = game.action_space
        self.actions = [x for x in range(0, self.action_space)]
        self.observation_space = game.observation_space

        if os.path.exists(Q_TABLE_PATH):
            with open(Q_TABLE_PATH, 'rb') as f:
                self.q_table = pickle.load(f)
        else:
            self.q_table = np.zeros((self.observation_space, self.action_space))

        self.alpha = learn_rate
        self.gamma = 0.8
        self.epsilon = explore_rate

    def get_action(self, game):
        if random.uniform(0, 1) < self.epsilon:
            action = random.choice(self.actions)
        else:
            action = np.argmax(self.q_table[game.get_state()])

        if not game.is_valid(action):
            self.q_table[game.get_state(), action] = -10
            return self.get_action(game)

        return action

    def reward(self, old_state, action, reward, new_state):
        old_value = self.q_table[old_state, action]
        next_max = np.max(self.q_table[new_state])

        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_table[old_state, action] = new_value

    def save_data(self):
        with open(Q_TABLE_PATH, 'wb+') as f:
            pickle.dump(self.q_table, f)


class Game:
    def __init__(self):
        self.observation_space = 19683
        self.action_space = 9
        self.board = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]
        self.actions = {
            0: (0, 0),
            1: (0, 1),
            2: (0, 2),
            3: (1, 0),
            4: (1, 1),
            5: (1, 2),
            6: (2, 0),
            7: (2, 1),
            8: (2, 2)
        }
        self.rewards = {
            'win': 10,
            'lose': -10,
            'draw': 0,
            'invalid': -10
        }

    def step(self, action, player_num):
        if not self.is_valid(action):
            return self.rewards['invalid'], False
        index = self.actions[action]
        self.board[index[0]][index[1]] = player_num
        if self.is_winner(player_num):
            return self.rewards['win'], True
        found = False
        for sub_list in self.board:
            for item in sub_list:
                if item == 0:
                    found = True
        if found == False:
            return 0, True
        return -1, False

    def get_state(self):
        max_num = 3
        arr2 = sum(self.board, [])
        degrees = max_num ** np.arange(len(arr2))
        state = arr2 @ degrees
        return state

    def is_valid(self, action):
        index = self.actions[action]
        if self.board[index[0]][index[1]] != 0:
            return False
        return True

    def print_board(self):
        chars = ['_', 'X', 'O']
        os.system('clear')
        for row in self.board:
            for item in row:
                print(chars[item], end = ' ')
            print()

    def win_indexes(self, n):
        # Rows
        for r in range(n):
            yield [(r, c) for c in range(n)]
        # Columns
        for c in range(n):
            yield [(r, c) for r in range(n)]
        # Diagonal top left to bottom right
        yield [(i, i) for i in range(n)]
        # Diagonal top right to bottom left
        yield [(i, n - 1 - i) for i in range(n)]

    def is_winner(self, decorator):
        n = len(self.board)
        for indexes in self.win_indexes(n):
            if all(self.board[r][c] == decorator for r, c in indexes):
                return True
        return False


def train():
    game = Game()
    computer = Computer(game, learn_rate=0.1, explore_rate=0.1)

    for _ in range(0, 100000):
        c1_states = []
        c2_states = []
        running = True
        game.__init__()
        while True:
            state = game.get_state()
            c1_states.append(state)

            if len(c1_states) == 4:
                computer.reward(c1_states[0], c1_states[1], c1_states[2], c1_states[3])
                c1_states = []
                c1_states.append(state)
            if running == False:
                break

            move = computer.get_action(game)
            c1_states.append(move)
            reward, done = game.step(move, 1)
            c1_states.append(reward)
            if done:
                c2_states[2] = -1*reward
                running = False

            #sleep(1)
            #for row in game.board:
            #    print(row)
            #print()

            state = game.get_state()
            c2_states.append(state)

            if len(c2_states) == 4:
                computer.reward(c2_states[0], c2_states[1], c2_states[2], c2_states[3])
                c2_states = []
                c2_states.append(state)
            if running == False:
                break

            move = computer.get_action(game)
            c2_states.append(move)
            reward, done = game.step(move, 2)
            c2_states.append(reward)

            if done:
                c1_states[2] = -1*reward
                running = False

            #sleep(1)
            #for row in game.board:
            #    print(row)
            #print()

    computer.save_data()


def game(is_p_first):
    game = Game()
    player = Player()
    computer = Computer(game)

    if is_p_first:
        player_dict = {'p': (player, 1), 'c': (computer, 2)}
    else:
        player_dict = {'c': (computer, 1), 'p': (player, 2)}

    going = True
    game.print_board()
    while going:
        for p in player_dict:
            game.print_board()

            move = player_dict[p][0].get_action(game)
            reward, done = game.step(move, player_dict[p][1])

            if p == 'p':
                player_dict[p][0].reward(reward, game)
            elif done == True:
                player_dict['p'][0].reward(-1*reward, game)

            if done == True:
                going = False
                break


def menu():
    print("0: QUIT")
    print("1: PLAY")
    print("2: TRAIN")
    choice = int(input('>'))
    return choice


def main():
    global Q_TABLE_PATH
    while True:
        choice = menu()
        if choice == 0:
            exit()
        elif choice == 1:
            is_p_first = input('Do you want to be first? (y/n)').lower().startswith('y')
            game(is_p_first)
        elif choice == 2:
            train()


main()
