# based on code from https://stackabuse.com/minimax-and-alpha-beta-pruning-in-python

import time
import sys
import math as m
from collections import Counter


class Game:
  MINIMAX = 0
  ALPHABETA = 1
  HUMAN = 2
  AI = 3

  def __init__(self,
               recommend=False,
               n=3,
               blocks=[],
               s=3,
               depth1=0,
               depth2=0,
               e1=1,
               e2=2,
               t=10,
               f=None):
    E = {1: self.e1, 2: self.e2}
    self.n = n
    self.initialize_game()
    if blocks:
      self.add_blocks(blocks)
    self.s = s
    self.recommend = recommend
    self.depth1 = depth1
    self.depth2 = depth2
    self.score1 = E[e1]
    self.score2 = E[e2]
    self.time_limit = t
    self.f = f
    self.moves = 0
    self.e1_invocations = 0
    self.e2_invocations = 0
    self.invocations = 0
    self.total_depth_evals = Counter(
        {i: 0
         for i in range(1,
                        max(depth2, depth1) + 1)})

  def initialize_game(self):

    self.current_state = [['.' for x in range(0, self.n)]
                          for x in range(0, self.n)]
    # Player X always plays first
    self.player_turn = 'X'

  def draw_board(self):
    print()
    for x in range(0, self.n):
      for y in range(0, self.n):
        print(F'{self.current_state[x][y]}', end='')
        if self.f: self.f.write(F'{self.current_state[x][y]}')
      print()
      if self.f: self.f.write('\n')
    print()
    if self.f: self.f.write('\n')

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
        print('It\'s a tie!')
      if self.f:
        if self.result == ',':
          self.f.write('It\'s a tie.\n')
        else:
          self.f.write(f'The winner is {self.result}.\n\n')
        avg_time = sum(self.total_times) / len(self.total_times)
        avg_ev_depth = sum(self.total_depths) / len(self.total_depths)

        self.f.write(f'6(b)i\tAverage evaluation time: {avg_time:.2f}s\n')
        self.f.write(
            f'6(b)ii\tTotal heuristic evaluations: {self.total_evals}\n')
        self.f.write(
            f'6(b)iii\tEvaluations by depth: {dict(self.total_depth_evals)}\n')
        self.f.write(f'6(b)iv\tAverage evaluation depth: {avg_ev_depth:.2f}\n')
        self.f.write(f'6(b)vi Total moves: {self.total_moves}\n')
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
    self.moves += 1
    if self.player_turn == 'X':
      self.player_turn = 'O'
    elif self.player_turn == 'O':
      self.player_turn = 'X'
    return self.player_turn

  def minimax(self, max=False, d=0, start=0, limit=10, current=0):
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
      self.depth_evals[current] += 1
      return (-sys.maxsize + 1, x, y)
    elif result == 'O':
      self.depth_evals[current] += 1
      return (sys.maxsize - 1, x, y)
    elif result == '.':
      self.depth_evals[current] += 1
      return (0, x, y)
    elif current == d or t >= (limit - 0.5):  #max depth reached or time

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
      self.depth_evals[current] += 1
      return (value, x, y)
    current += 1
    for i in range(0, self.n):
      for j in range(0, self.n):
        if self.current_state[i][j] == '.':
          if max:
            self.current_state[i][j] = 'O'
            (v, _, _) = self.minimax(max=False,
                                     d=d,
                                     start=start,
                                     limit=limit,
                                     current=current)
            if v > value:
              value = v
              x = i
              y = j
          else:
            self.current_state[i][j] = 'X'
            (v, _, _) = self.minimax(max=True,
                                     d=d,
                                     start=start,
                                     limit=limit,
                                     current=current)
            if v < value:
              value = v
              x = i
              y = j
          self.current_state[i][j] = '.'
    self.depth_evals[current] += 1
    return (value, x, y)

  def alphabeta(self,
                alpha=-2,
                beta=2,
                max=False,
                d=0,
                start=0,
                limit=10,
                current=0):
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
      self.depth_evals[current] += 1
      return (-sys.maxsize + 1, x, y)
    elif result == 'O':
      self.depth_evals[current] += 1
      return (sys.maxsize - 1, x, y)
    elif result == '.':
      self.depth_evals[current] += 1
      return (0, x, y)
    elif current == d or t >= (limit - 0.5):  #max depth or time reached
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
      self.depth_evals[current] += 1
      return (value, x, y)
    current += 1
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
                                       limit=limit,
                                       current=current)
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
                                       limit=limit,
                                       current=current)
            if v < value:
              value = v
              x = i
              y = j
          self.current_state[i][j] = '.'
          if max:
            if value >= beta:
              self.depth_evals[current] += 1
              return (value, x, y)
            if value > alpha:
              alpha = value
          else:
            if value <= alpha:
              self.depth_evals[current] += 1
              return (value, x, y)
            if value < beta:
              beta = value
    self.depth_evals[current] += 1
    return (value, x, y)

  def play(self, algo=None, player_x=None, player_o=None):
    self.total_evals = 0
    self.total_times = []
    self.total_depths = []
    self.total_moves = 0
    moves = 0
    limit = self.time_limit
    if algo == None:
      algo = self.ALPHABETA
    if player_x == None:
      player_x = self.HUMAN
    if player_o == None:
      player_o = self.HUMAN
    while True:
      moves += 1
      self.depth_evals = Counter(
          {i: 0
           for i in range(1,
                          max(self.depth1, self.depth2) + 1)})
      d1 = self.depth1
      d2 = self.depth2
      self.draw_board()
      r = self.check_end()
      if r:
        ans = (r, self.invocations, self.moves, self.total_depth_evals,
               (time.time() - start),
               (sum(self.total_depths) / len(self.total_depths)))
        return ans

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
        # if self.recommend:
        #   print(F'Evaluation time: {round(end - start, 7)}s')
        #   if self.f:
        #     self.f.write(F'i\t\tEvaluation time: {round(end - start, 7)}s\n')
        #   print(F'Recommended move: x = {x}, y = {y}')
        (x, y) = self.input_move()
      if (self.player_turn == 'X'
          and player_x == self.AI) or (self.player_turn == 'O'
                                       and player_o == self.AI):
        print(F'Evaluation time: {round(end - start, 7)}s')
        print(
            F'Player {self.player_turn} under AI control plays: x = {x}, y = {y}'
        )
        avg = sum(k * v for k, v in self.depth_evals.items()) / sum(
            self.depth_evals.values())
        if self.f:
          self.f.write(
              F'Player {self.player_turn} under AI control plays: row:{x}, column:{y}\n\n'
          )
          self.f.write(F'i\t\tEvaluation time: {round(end - start, 7)}s\n')
          self.f.write(f'ii\tHeuristic evaluations: ')
          if self.player_turn == 'X':
            self.f.write(f'{self.e1_invocations}\n')
          else:
            self.f.write(f'{self.e2_invocations}\n')
          self.f.write(
              f'iii\tEvaluations by depth: {dict(self.depth_evals)}\n')

          self.f.write(f'iv\tAverage evaluation depth: {avg:.2f}\n')
      # self.e1_invocations = 0
      # self.e2_invocations = 0
      self.current_state[x][y] = self.player_turn
      self.switch_player()
      self.total_evals += sum(self.depth_evals.values())

      self.total_times.append(round(end - start, 7))
      self.total_depths.append(avg)
      self.total_moves += moves
      self.total_depth_evals += self.depth_evals

  def e1(self):
    self.invocations += 1
    self.e1_invocations += 1
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
    self.invocations += 1
    self.e2_invocations += 1
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
  #yapf: disable
  GAMES = [
      {'n':4,'b':4, 's':3, 't':5,
          'd1':6, 'd2':6, 'algo': False,'blocks': [(0, 0), (0, 3), (3, 0), (3, 3)]},
      {'n':4,'b':4, 's':3, 't':1,
          'd1':6, 'd2':6, 'algo': True,'blocks': [(1, 0), (2, 2)]},
      {'n': 5, 'b':4 ,'s':4 , 't':1,
          'd1':6, 'd2':6, 'algo':True, 'blocks':[]},
      {'n':5, 'b':4,'s':4, 't':5,
      'd1':6, 'd2':6, 'algo':True, 'blocks':[]},
      {'n':8, 'b':5,'s':5, 't':1,
      'd1':2, 'd2':6, 'algo':True, 'blocks':[]},
      {'n':8, 'b':5,'s':5, 't':5,
      'd1':2, 'd2':6, 'algo':True, 'blocks':[]},
      {'n':8, 'b':6,'s':5, 't':1,
      'd1':6, 'd2':6, 'algo':True, 'blocks':[]},
      {'n':8, 'b':6,'s':5, 't':1,
      'd1':6, 'd2':6, 'algo':True, 'blocks': [(0,0),(1,1),
                                              (2,2),(3,3),
                                              (4,4),(5,5),
                                              (6,6),(7,7)]},
  ]
  # for game in GAMES:
  #   n, b, s, t = (game["n"],game["b"], game["s"],game["t"] )
  #   file = f'gameTrace-{n}{b}{s}{t}.txt'
  #   with open(file, 'w') as f:
  #     f.write(f'n={n} b={b} s={s} t={t}\n')
  #     f.write(f'blocs={game["blocks"]}\n')
  #     f.write(f'\nPlayer 1: AI d={game["d1"]} a={game["algo"]}, e1\n')
  #     f.write(f'Player 2: AI d={game["d1"]} a={game["algo"]}, e2\n\n')

  #     g = Game(recommend=False,
  #            n=game['n'],
  #            s=game['s'],
  #            t=game['t'],
  #            depth1=game['d1'],
  #            depth2=game['d2'],
  #            blocks=game['blocks'], f=f)
  #     g.play(
  #     algo=game['algo'],
  #     player_x=Game.AI,
  #     player_o=Game.AI,
  #     )
  r = 10
  FILE = 'scoreboard.txt'
  with open(FILE, 'w')as f:
    for game in GAMES:

      wins_e1 = 0
      wins_e2 = 0
      i = []
      ii = 0
      iii = Counter({})
      iv = []
      v = []
      vi = []
      f.write(f'n={game["n"]} b={len(game["blocks"])} s={game["s"]} t={game["t"]}\n')
      f.write(f'\nPlayer 1: d={game["d1"]} a={game["algo"]}\n')
      f.write(f'Player 2: d={game["d2"]} a={game["algo"]}\n')
      f.write(f'\n10 games\n')
      for j in range(0, r):
        g = Game(recommend=False,
              n=game['n'],
              s=game['s'],
              t=game['t'],
              depth1=game['d1'],
              depth2=game['d2'],
              blocks=game['blocks'])
        w, evals, moves, evals_depth, t, avg_ev_depth= g.play(
        algo=game['algo'],
        player_x=Game.AI,
        player_o=Game.AI,
        )
        if w == 'X': wins_e1 +=1
        elif w=='O': wins_e2 +=1
        i.append(t/moves)
        ii += evals
        iii += evals_depth
        iv.append(avg_ev_depth)
        vi.append(moves)
      for j in range(0,r):
        g = Game(recommend=False,
              n=game['n'],
              s=game['s'],
              t=game['t'],
              depth1=game['d1'],
              depth2=game['d2'],
              blocks=game['blocks'], e1=2, e2=1)
        w, evals, moves, evals_depth, t, avg_ev_depth= g.play(
        algo=game['algo'],
        player_x=Game.AI,
        player_o=Game.AI,
        )
        if w == 'O': wins_e1 +=1
        elif w == 'X': wins_e2 +=1
        i.append(t/moves)
        ii += evals
        iii += evals_depth
        iv.append(avg_ev_depth)
        vi.append(moves)
      total_wins = wins_e1+wins_e2
      f.write(f'Total wins for heuristic e1: {wins_e1} ({wins_e1/total_wins*100:.1f}%)\n')
      f.write(f'Total wins for heuristic e2: {wins_e2} ({wins_e2/total_wins*100:.1f}%)\n')
      f.write(f'\ni\tAverage evaluation time: {sum(i)/len(i):.2f}\n')
      f.write(f'ii\tTotal heuristic evaluations: {ii}\n')
      f.write(f'iii\tEvaluations by depth: {dict(iii)}\n')
      f.write(f'iv\t Average evaluation depth: {sum(iv)/len(iv):.2f}\n')
      f.write(f'vi\tAverage moves per game: {sum(vi)/len(vi):.2f}\n\n')


if __name__ == '__main__':
  main()
