import copy
import rl
from game import Game


def self_play(num_iters, num_games, engine):
    winner_model = engine

    for iter_num in range(num_iters):
        # it is the first iteration, so put two copies against each other
        if iter_num == 0:
            contestor_model = copy.deepcopy(engine)

        score = [0, 0]
        for game_num in range(num_games):
            winner = play_game(winner_model, contestor_model, game_num)
            if winner == 0:
                score[game_num % 2] += 1
            elif winner == 1:
                score[1 - (game_num % 2)] += 1

        # the contestor model has won by a point, therefore it becomes the new winner model
        if score[1] >= 1 + score[0]:
            winner_model = contestor_model

        # train the contestor

def play_game(winner_model, contestor_model, game_num):
    moves = []
    while not game.is_mate(0, 1) and not game.is_mate(1, 0):
        game = Game(0, 1)
        # alternate colors every game
        if game_num % 2 == 0:
            move = winner_model.make_CPU_move(0, game, 3)
        else:
            move = contestor_model.make_CPU_move(0, game, 3)
        moves.append(move)
        if game.is_mate(0, 1):
            break

        if game_num % 2 == 0:
            move = contestor_model.make_CPU_move(1, game, 3)
        else:
            move = winner_model.make_CPU_move(1, game, 3)
        if game.is_mate(1, 0):
            break
        moves.append(move)

    if game.checkmate == 0:
        return 0
    elif game.checkmate == 1:
        return 1
    return 2
