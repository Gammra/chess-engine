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

    """
    def execute_episode(nnet):
        numSims = 10
        examples = []
        game = Game(0, 1)

        while True:
            visited = {}
            Q_values = {}
            N = {}
            # for _ in range(numSims):
                # MCTS(0, visited, game, nnet, Q_values, N)
            board3d = 0 # implement
            #examples.append([board3d, pi, None])
            #action = random.choice(len(mcts.pi(s)), p=mcts.pi(s))
            #game.make_move(action)
            if game.is_mate(0,1) or game.is_mate(1,0):
                if game.checkmate == 0:
                    examples[0][2] = -1
                elif game.checkmate == 1:
                    examples[0][2] = 1
                elif game.checkmate == 2:
                    examples[0][2] = 0
                return examples

    def policyIterSP(game, num_iters, num_eps, threshold):
        engine_old = Engine()
        engine_new = Engine()
        nnet = engine_old.build_model()
        new_nnet = engine_new.build_model()
        examples = []
        for i in range(num_iters):
            for e in range(num_eps):
                examples += execute_episode(game, nnet)
            # new_nnet = trainNNet(examples)
            # score = pit(new_nnet, nnet)
            # if score > threshold:
            #    nnet = new_nnet
        return nnet
    """

