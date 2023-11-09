import copy
import rl
from game import Game


def self_play(num_iters, num_games, engine):
    winner_model = engine

    for iter_num in range(num_iters):
        if iter_num != 0:
            contestor_model = 0
        else:
            contestor_model = copy.deepcopy(engine)

        score = [0, 0]
        for game_num in range(num_games):
            winner = play_game(winner_model, contestor_model, game_num)
            if winner == 0:
                score[game_num % 2] += 1
            elif winner == 1:
                score[1 - (game_num % 2)] += 1


def play_game(winner_model, contestor_model, game_num):
    while not game.is_mate(0, 1) and not game.is_mate(1, 0):
        game = Game(0, 1)
        if game_num % 2 == 0:
            winner_model.make_CPU_move(0, game, 3)
        else:
            contestor_model.make_CPU_move(0, game, 3)
        if game.is_mate(0, 1):
            break
        if game_num % 2 == 0:
            contestor_model.make_CPU_move(1, game, 3)
        else:
            winner_model.make_CPU_move(1, game, 3)
        if game.is_mate(1, 0):
            break

    if game.checkmate == 0:
        return 0
    elif game.checkmate == 1:
        return 1
    return 0
