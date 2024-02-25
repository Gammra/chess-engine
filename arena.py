import copy
import tensorflow as tf
import random
from game import Game

def self_play(num_iters, num_games, engine):
    winner_model = copy.deepcopy(engine)
    contestor_model = copy.deepcopy(engine)

    train_player_col = 0
    for iter_num in range(num_iters):
        gamma = 0.99
        winner = 2
        game_record = []

        # pit model against itself until 1 model wins
        while winner == 2:
            winner = play_game(winner_model, contestor_model, train_player_col, game_record)
        update_from_games(contestor_model, winner, game_record, gamma)

        score = [0, 0]
        game_record = []
        for game_num in range(num_games):
            winner = play_game(winner_model, contestor_model, game_num, game_record)
            if winner == 0:
                score[game_num % 2] += 1
            elif winner == 1:
                score[1 - (game_num % 2)] += 1

        # the contestor model has won by a point, therefore it becomes the new winner model
        if score[1] >= 1 + score[0]:
            winner_model = copy.deepcopy(contestor_model)

        train_player_col = 1 - train_player_col

    return winner_model


def play_game(winner_model, contestor_model, game_num, game_record):
    game = Game(0, 1)
    while not game.is_mate(0, 1) and not game.is_mate(1, 0):
        # stalemate - game has gone on far too long
        if len(game_record) > 300:
            return 2

        # alternate colors every game
        if game_num % 2 == 0:
            move = winner_model.make_CPU_move(0, game, 1)
        else:
            move = contestor_model.make_CPU_move(0, game, 1)
        game_record.append(move)
        if game.is_mate(0, 1):
            break

        if game_num % 2 == 0:
            move = contestor_model.make_CPU_move(1, game, 1)
        else:
            move = winner_model.make_CPU_move(1, game, 1)
        if game.is_mate(1, 0):
            break
        game_record.append(move)

    if game.checkmate == 0:
        return 0
    elif game.checkmate == 1:
        return 1
    return 2


def update_from_games(engine, winner, game_record, gamma):
    all_grads = []
    color = 0
    temp_game = Game(0, 1)
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-6)

    for move in game_record:
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            enemy_color = 1 - color
            if color == winner:
                reward = 0.05
            else:
                reward = -0.05
            if color == 1:
                reward = 0 - reward

            curr_Q = engine.predict(temp_game)
            state = temp_game.make_move(move, save=True, promo_code=3)
            next_Q = engine.predict(temp_game)
            temp_game.undo_move(state)

            loss_value = (reward + (gamma * next_Q) - curr_Q)
            grad = tape.gradient(loss_value, engine.model.trainable_variables)
            all_grads.append(grad)

            temp_game.make_move(move, save=True, promo_code=3)
            color = enemy_color

    for grad in all_grads:
        optimizer.apply_gradients(zip(grad, engine.model.trainable_variables))
