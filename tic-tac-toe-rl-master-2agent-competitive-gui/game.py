import random
import sys
import copy

class Game:
    def __init__(self):
        self.board = [-1.0] * 9
        self.winning_combos = (
        [6, 7, 8], [3, 4, 5], [0, 1, 2], [0, 3, 6], [1, 4, 7], [2, 5, 8],
        [0, 4, 8], [2, 4, 6],)
        self.corners = [0,2,6,8]
        self.sides = [1,3,5,7]
        self.middle = 4
        self.p1_marker, self.p2_marker = self.get_marker()

    def get_marker(self):
        return (1.0,0.0)

    def reset(self):
        self.board = [-1.0] * 9
        return self.board

    def step(self, action, marker):
      over = False
      reward = 0
      
      self.make_move(self.board, action, marker)
      
      if(self.is_winner(self.board, marker)):
        reward = 100
        over = True

      # drawing
      elif self.is_board_full():
        reward = 10
        over = True
      
      return self.board, reward, over

    def is_winner(self, board, marker):
        for combo in self.winning_combos:
            if (board[combo[0]] == board[combo[1]] == board[combo[2]] == marker):
                return True
        return False

    def get_winning_combo(self, board):
        for combo in self.winning_combos:
            if (board[combo[0]] == board[combo[1]] == board[combo[2]]):
                return [combo[0], combo[1], combo[2]]
        return [None, None, None]

    def is_space_free(self, board, index):
        "checks for free space of the board"
        return board[index] == -1.0

    def is_board_full(self):
        "checks if the board is full"
        for i in range(1,9):
            if self.is_space_free(self.board, i):
                return False
        return True

    def make_move(self,board,index,move):
        board[index] =  move

    def choose_random_move(self, move_list):
        possible_winning_moves = []
        for index in move_list:
            if self.is_space_free(self.board, index):
                possible_winning_moves.append(index)
        if len(possible_winning_moves) != 0:
            return random.choice(possible_winning_moves)
        else:
            return None