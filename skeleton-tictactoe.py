# based on code from https://stackabuse.com/minimax-and-alpha-beta-pruning-in-python

import time
import sys
import math as m


class Game:
  MINIMAX = 0
  ALPHABETA = 1
  HUMAN = 2
  AI = 3

  def __init__(self,
               recommend=True,
               n=3,
               blocks=[],
               s=3,
               depth1=0,
               depth2=0,
               e1=1,
               e2=2,
               t=10):
    E = {1: self.e1, 2: self.e2}
    self.n = n
    self.initialize_game()
    self.add_blocks(blocks)
    self.s = s
    self.recommend = recommend
    self.depth1 = depth1
    self.depth2 = depth2
    self.score1 = E[e1]
    self.score2 = E[e2]
    self.time_limit = t

  def initialize_game(self):

    self.current_state = [['.' for x in range(0, self.n)]
                          for x in range(0, self.n)]
    # Player X always plays first
    self.player_turn = 'X'

  def draw_board(self):
    print()
    for x in range(0, self.n):
      for y in range(0, self.n):
        print(F'{self.current_state[x][y]}', end="")
      print()
    print()

  def is_valid(self, px, py):
    if px < 0 or px > self.n - 1 or py < 0 or py > self.n - 1:
      return False
    elif self.current_state[px][py] != '.':
      return False
    else:
      return True

  def is_end(self):
    # Vertical win
    for i in range(0, self.n):
      column = ''
      for j in range(0, self.n):
        column += self.current_state[j][i]
      if 'X' * self.s in column:
        return 'X'
      elif 'O' * self.s in column:
        return 'O'
    # Horizontal win
    for i in range(0, self.n):
      row = ''.join(self.current_state[i])
      if 'X' * self.s in row:
        return 'X'
      elif 'O' * self.s in row:
        return 'O'
    # diagonals:
    for x in range(0, self.n):
      anti_diagonal = ''
      anti_diagonal2 = ''
      diagonal = ''
      diagonal2 = ''
      for i, j in zip(range(x, -1, -1), range(0, x + 1)):
        k = self.n - i - 1
        l = j
        diagonal += self.current_state[k][l]
        diagonal2 += self.current_state[self.n - k - 1][self.n - l - 1]
        anti_diagonal += self.current_state[i][j]
        anti_diagonal2 += self.current_state[self.n - i - 1][self.n - i - 1]
      xpattern = 'X' * self.s
      opattern = 'O' * self.s
      if (xpattern in anti_diagonal or xpattern in anti_diagonal2
          or xpattern in diagonal or xpattern in diagonal2):
        return 'X'
      elif (opattern in anti_diagonal or opattern in anti_diagonal2
            or opattern in diagonal or opattern in diagonal2):
        return 'O'

    # Is whole board full?
    for i in range(0, self.n):
      for j in range(0, self.n):
        # There's an empty field, we continue the game
        if (self.current_state[i][j] == '.'):
          return None
    # It's a tie!
    return '.'

  def check_end(self):
    self.result = self.is_end()
    # Printing the appropriate message if the game has ended
    if self.result != None:
      if self.result == 'X':
        print('The winner is X!')
      elif self.result == 'O':
        print('The winner is O!')
      elif self.result == '.':
        print("It's a tie!")
      self.initialize_game()
    return self.result

  def input_move(self):
    while True:
      print(F'Player {self.player_turn}, enter your move:')
      px = int(input('enter the row number: '))
      py = int(input('enter the column number: '))
      if self.is_valid(px, py):
        return (px, py)
      else:
        print('The move is not valid! Try again.')

  def switch_player(self):
    if self.player_turn == 'X':
      self.player_turn = 'O'
    elif self.player_turn == 'O':
      self.player_turn = 'X'
    return self.player_turn

  def minimax(self, max=False, d=0, start=0, limit=10):
    # Minimizing for 'X' and maximizing for 'O'
    # Possible values are:
    # -maxint - win for 'X'
    # 0  - a tie
    # maxint  - loss for 'X'
    # We're initially setting it to inf or -inf as worse than the worst case:
    t = time.time() - start
    value = float('inf')
    if max:
      value = float('-inf')
    x = None
    y = None
    result = self.is_end()
    if result == 'X':
      return (-sys.maxsize + 1, x, y)
    elif result == 'O':
      return (sys.maxsize - 1, x, y)
    elif result == '.':
      return (0, x, y)
    elif d <= 0 or t == (limit - 0.5):  #max depth reached or time
      for i in range(0, self.n):
        for j in range(0, self.n):
          if self.current_state[i][j] == '.':
            if max:
              self.current_state[i][j] = 'O'
              v = self.score1()
              print('2: ', v)
              if v > value:
                value = v
                x = i
                y = j
            else:
              self.current_state[i][j] = 'X'
              v = self.score2()
              # print('v1: ', v)
              if v < value:
                value = v
                x = i
                y = j
            self.current_state[i][j] = '.'
      return (value, x, y)
    d -= 1
    for i in range(0, self.n):
      for j in range(0, self.n):
        if self.current_state[i][j] == '.':
          if max:
            self.current_state[i][j] = 'O'
            (v, _, _) = self.minimax(max=False, d=d)
            if v > value:
              value = v
              x = i
              y = j
          else:
            self.current_state[i][j] = 'X'
            (v, _, _) = self.minimax(max=True, d=d)
            if v < value:
              value = v
              x = i
              y = j
          self.current_state[i][j] = '.'
    return (value, x, y)

  def alphabeta(self, alpha=-2, beta=2, max=False, d=0, start=0, limit=10):
    # Minimizing for 'X' and maximizing for 'O'
    # Possible values are:
    # -1 - win for 'X'
    # 0  - a tie
    # 1  - loss for 'X'
    # We're initially setting it to 2 or -2 as worse than the worst case:
    t = time.time() - start
    value = float('inf')
    if max:
      value = float('-inf')
    x = None
    y = None
    result = self.is_end()
    # print(start, t, limit)
    if result == 'X':
      return (-sys.maxsize + 1, x, y)
    elif result == 'O':
      return (sys.maxsize - 1, x, y)
    elif result == '.':
      return (0, x, y)
    elif d <= 0 or t >= (limit - 0.5):  #max depth or time reached
      if t >= limit:
        print('time limit reached')
        if max: return (float('-inf'), x, y)
        else: return (float('inf'), x, y)
      for i in range(0, self.n):
        for j in range(0, self.n):
          if self.current_state[i][j] == '.':
            if max:
              self.current_state[i][j] = 'O'
              v = self.score1()
              # print('2: ', v)
              if v > value:
                value = v
                x = i
                y = j
            else:
              self.current_state[i][j] = 'X'
              v = self.score2()
              # print('v1: ', v)
              if v < value:
                value = v
                x = i
                y = j
            self.current_state[i][j] = '.'
      return (value, x, y)
    d -= 1
    for i in range(0, self.n):
      for j in range(0, self.n):
        if self.current_state[i][j] == '.':
          if max:
            self.current_state[i][j] = 'O'
            (v, _, _) = self.alphabeta(alpha,
                                       beta,
                                       max=False,
                                       d=d,
                                       start=start,
                                       limit=limit)
            if v > value:
              value = v
              x = i
              y = j
          else:
            self.current_state[i][j] = 'X'
            (v, _, _) = self.alphabeta(alpha,
                                       beta,
                                       max=True,
                                       d=d,
                                       start=start,
                                       limit=limit)
            if v < value:
              value = v
              x = i
              y = j
          self.current_state[i][j] = '.'
          if max:
            if value >= beta:
              return (value, x, y)
            if value > alpha:
              alpha = value
          else:
            if value <= alpha:
              return (value, x, y)
            if value < beta:
              beta = value
    return (value, x, y)

  def play(self, algo=None, player_x=None, player_o=None):
    d1 = self.depth1
    d2 = self.depth2
    limit = self.time_limit
    if algo == None:
      algo = self.ALPHABETA
    if player_x == None:
      player_x = self.HUMAN
    if player_o == None:
      player_o = self.HUMAN
    while True:
      self.draw_board()
      if self.check_end():
        return
      start = time.time()
      if algo == self.MINIMAX:
        if self.player_turn == 'X':
          (m, x, y) = self.minimax(max=False, d=d1, start=start, limit=limit)
        else:
          (m, x, y) = self.minimax(max=True, d=d2, start=start, limit=limit)
      else:  # algo == self.ALPHABETA
        if self.player_turn == 'X':
          (m, x, y) = self.alphabeta(max=False, d=d1, start=start, limit=limit)
        else:
          (m, x, y) = self.alphabeta(max=True, d=d2, start=start, limit=limit)
      end = time.time()
      if (self.player_turn == 'X'
          and player_x == self.HUMAN) or (self.player_turn == 'O'
                                          and player_o == self.HUMAN):
        if self.recommend:
          print(F'Evaluation time: {round(end - start, 7)}s')
          print(F'Recommended move: x = {x}, y = {y}')
        (x, y) = self.input_move()
      if (self.player_turn == 'X'
          and player_x == self.AI) or (self.player_turn == 'O'
                                       and player_o == self.AI):
        print(F'Evaluation time: {round(end - start, 7)}s')
        print(
            F'Player {self.player_turn} under AI control plays: x = {x}, y = {y}'
        )
      self.current_state[x][y] = self.player_turn
      self.switch_player()

  def e1(self):
    score = 0
    lines = []
    for i in range(0, self.n):
      column = ''
      for j in range(0, self.n):
        column += self.current_state[j][i]
      lines.append(column)
    # Horizontal
    for i in range(0, self.n):
      row = ''.join(self.current_state[i])
      lines.append(row)

    for x in range(0, self.n):
      anti_diagonal = ''
      anti_diagonal2 = ''
      diagonal = ''
      diagonal2 = ''
      for i, j in zip(range(x, -1, -1), range(0, x + 1)):
        k = self.n - i - 1
        l = j
        diagonal += self.current_state[k][l]
        diagonal2 += self.current_state[self.n - k - 1][self.n - l - 1]
        anti_diagonal += self.current_state[i][j]
        anti_diagonal2 += self.current_state[self.n - i - 1][self.n - i - 1]
      lines += [diagonal, diagonal2, anti_diagonal, anti_diagonal2]

    for line in lines:
      if 'X' * self.s in line:
        score = -sys.maxsize + 1
        return score
      elif 'O' * self.s in line:
        score = sys.maxsize - 1
        return score
      elif 'X' not in line and 'O' in line:
        score += line.count('O')
      elif 'X' in line and 'O' not in line:
        score -= line.count('X')
    return score

  def e2(self):
    score = 0
    center = self.n // 2
    for i in range(0, self.n):
      for j in range(0, self.n):
        element = self.current_state[i][j]
        x = i - center
        y = j - center
        distance = m.sqrt(x**2 + y**2)
        if element == 'X':
          score -= center - distance
        elif element == 'O':
          score += center - distance
    return score

  def add_blocks(self, b):
    for block in b:
      self.current_state[block[0]][block[1]] = '*'


def main():
  GAMES = [{"n": 4, "b": 4, "s": 3, "t": 5}]
  g = Game(recommend=True, n=4, depth1=7, depth2=0, s=4, t=5)
  # g.play(algo=Game.MINIMAX, player_x=Game.AI, player_o=Game.AI)
  g.play(
      algo=Game.ALPHABETA,
      player_x=Game.AI,
      player_o=Game.HUMAN,
  )


if __name__ == "__main__":
  main()
