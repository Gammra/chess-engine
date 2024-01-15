import random
import pandas as pd
import matplotlib as plt
plt.use("qt5agg")
import seaborn as sns
import numpy as np
import tensorflow as tf
from game import Game
from engine import Engine
from itertools import chain


def get_q_values(engine, game, moves, color, move_count):
    q_values = []
    promo_num = 0
    for move in moves:
        state = game.make_move(move, save=True, promo_code=promo_num % 4)
        if move_count <= 9:
            q_values.append(random.uniform(-1, 1))
        else:
            q_values.append(engine.predict(game))
        game.undo_move(state)
        if game.board.board[move[0]].piece_type == 1 and engine.is_promo(color, move):
            promo_num += 1
    return q_values


def get_reward(game, color, winner, move_selected, move_actual, enemy_attacked, player_attacked, curr_move):
    from training_data import is_promo
    reward = 0
    enemy_color = 1 - color

    # reward early development
    if 2 <= game.board.board[move_selected[0]].piece_type <= 3 and curr_move <= 12:
        if color == 0:
            reward += 0.5
        if color == 1:
            reward -= 0.5

    # reward castling
    if game.is_legal_castle(move_selected, enemy_attacked):
        if color == 0:
            reward += 0.5
        else:
            reward -= 0.5

    # reward promos
    if is_promo(color, move_selected):
        if color == 0:
            reward += 6
        else:
            reward -= 6

    # reward if the move was the actual move selected by a pro
    if move_selected == move_actual:
        if color == 0:
            reward += 1
            if color == winner:
                reward += 0.5
        else:
            reward -= 1
            if color == winner:
                reward -= 0.5

    # reward taking
    if game.board.board[move_selected[1]].color == enemy_color:
        if color == 0:
            reward += 0.2
        else:
            reward -= 0.2

        # avoid unfavorable trades
        if enemy_attacked[move_selected[1]] == 1 and game.board.board[move_selected[0]].piece_type < game.board.board[move_selected[1]].piece_type:
            if color == 0:
                reward -= 4
            if color == 1:
                reward += 4

    # avoid hanging
    if enemy_attacked[move_selected[1]] == 1 and player_attacked[move_selected[1]] == 0:
        if color == 0:
            reward -= 7.5
        if color == 1:
            reward += 7.5

    return reward


def reinforcement_training(engine, games, promos, metadata, epsilon, gamma):
    from training_data import is_promo

    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-6)
    game_num = 0
    all_grads = []
    loss_over_time = []
    for game, promo in zip(games, promos):
        # model created by the previous iteration will serve as the q_value generator
        if game_num % 3 == 0 and game_num != 0:
            engine = Engine("predictor_model.h5")
            all_grads = []

        player_color = 0
        enemy_color = 1
        temp_game = Game(0, 1)

        winner = metadata[game_num][6]
        if winner == "[Result \"1-0\"]":
            winner = 0
        elif winner == "[Result \"0-1\"]":
            winner = 1
        else:
            winner = 2

        move_count = 0
        promo_num = 0
        for move in game:
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(engine.model.trainable_variables)

                curr_Q = engine.predict(temp_game)
                player_attacked, dummy = temp_game.all_squares_attacked(player_color)
                enemy_attacked, xrays = temp_game.all_squares_attacked(enemy_color)
                m = temp_game.generate_legal_moves(player_color, enemy_attacked, xrays)
                moves = list(chain.from_iterable(m))

                # use epsilon method to balance exploration/exploitation
                sample_epsilon = np.random.rand()
                if sample_epsilon > epsilon:
                    # choose the highest rated move out of all moves
                    q_values = get_q_values(engine, temp_game, moves, player_color, move_count)
                    action = np.argmax(q_values)
                    if player_color == 1:
                        action = np.argmin(q_values)
                    next_Q = q_values[action]
                else:
                    # sample 9 random moves, and add the correct move (only if there are more than 9 possible already)
                    if len(moves) >= 9:
                        moves = random.sample(moves, 9)
                        moves.append(move)
                    q_values = get_q_values(engine, temp_game, moves, player_color, move_count)
                    action = np.random.choice(len(moves))
                    next_Q = q_values[action]

                reward = get_reward(temp_game, player_color, winner, moves[action], move, enemy_attacked, player_attacked, move_count)

                loss_value = (reward + (gamma * next_Q) - curr_Q)
                if player_color == 1:
                    loss_value = 0 - loss_value
                loss_over_time.append(loss_value[0].numpy()[0])

            grads = tape.gradient(loss_value, engine.model.trainable_variables)
            all_grads.append(grads)

            temp_game.make_move(move, save=False, promo_code=promo[promo_num])

            if is_promo(player_color, move) and temp_game.board.board[move[0]].piece_type == 1:
                promo_num += 1

            player_color = 1 - player_color
            enemy_color = 1 - player_color
            move_count += 1

        if game_num % 2 == 0 and game_num != 0:
            # apply all gradients from sample games
            for grad in all_grads:
                optimizer.apply_gradients(zip(grad, engine.model.trainable_variables))
            engine.model.save("predictor_model.h5", overwrite=True, save_format="h5")
            loss_chart = pd.DataFrame()
            loss_chart["loss"] = loss_over_time
            loss_chart["iter"] = range(1, len(loss_chart) + 1)
            sns.scatterplot(data=loss_chart, x="iter", y="loss")

        epsilon = epsilon * 0.99
        game_num += 1

        if game_num > 1000:
            break
