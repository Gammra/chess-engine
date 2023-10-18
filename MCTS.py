import numpy as np
import math

class MCTS:

    def search(color, visited, game, nnet, Q_values, N, P):
        state = 0
        enemy_color = 1 - color
        enemy_attacked, xrays = game.all_squares_attacked(enemy_color)
        moves = game.generate_legal_moves(color, enemy_attacked, xrays)

        if game.is_mate(0, 1) or game.is_mate(1, 0):
            if game.checkmate == 0:
                return -1
            elif game.checkmate == 1:
                return 1
            elif game.checkmate == 2:
                return 0

        if state not in visited:
            visited[state] = True
            P[state], v = nnet.predict(state)
            return -v

        max_UCB, best_action = -np.Infinity, -1
        for move in moves:
            UCB = Q_values[(state, move)] + P[(state, move)] * \
                  math.sqrt(sum(N[state])/ (1+N[(state, move)]))
            if UCB > max_UCB:
                max_UCB = UCB
                best_action = move

        game.make_move(best_action)
        v = MCTS(enemy_color, visited, game, nnet)
        game.undo_move()

        Q_values[state][best_action] = (N[state][best_action] * Q_values[state][best_action] + v) / \
                                       (N[state][best_action] +1)
        N[state][best_action] += 1
        return -v