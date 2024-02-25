from game import Game
import seaborn as sns
import numpy as np
import tensorflow as tf

import random
from itertools import chain
import pandas as pd
import matplotlib as plt
plt.use("qt5agg")


PAWN_VALUE = 1
KNIGHT_VALUE = 3.1
BISHOP_VALUE = 3.2
ROOK_VALUE = 4.9
QUEEN_VALUE = 9.8

PIECE_VALUES = [0, PAWN_VALUE, KNIGHT_VALUE, BISHOP_VALUE, ROOK_VALUE, QUEEN_VALUE]


def get_q_values(engine, game, moves, color):
    q_values = []
    promo_num = 0
    for move in moves:
        # this is temporary as ep is causing problems
        if game.board.board[move[1]].color == -1 and game.board.board[move[1]].piece_type == 1 and abs(move[1] - move[0]) % 8 != 0:
            q_values.append(engine.predict(game))
            continue
        state = game.make_move(move, save=True, promo_code=promo_num % 4)
        q_values.append(engine.predict(game))
        game.undo_move(state)
        if game.board.board[move[0]].piece_type == 1 and engine.is_promo(color, move):
            promo_num += 1
    return q_values


def get_reward(game, color, winner, move_selected, move_actual, enemy_attacked, player_attacked):
    from training_data import is_promo
    init_pos = move_selected[0]
    final_pos = move_selected[1]
    reward = 0
    enemy_color = 1 - color

    if game.board.board[init_pos].piece_type == 1 and ((init_pos // 8) == 1 or (init_pos // 8) == 6):
        # reward moving d and e pawns off original squares
        if 3 <= move_selected[0] % 8 <= 4:
            reward += 0.001

    # reward moving knights and bishops off original squares
    if game.board.board[init_pos].piece_type == 2 or game.board.board[init_pos].piece_type == 3:
        if (init_pos // 8) == 0 or (init_pos // 8) == 7:
            reward += 0.001

    # keep knights off the rim
    if game.board.board[init_pos].piece_type == 2 and ((final_pos % 8 == 0) or (final_pos % 8 == 7)):
        reward -= 0.001

    # reward castling
    if game.board.board[init_pos].piece_type == 0 and 2 <= abs(init_pos-final_pos) <= 3:
        reward += 0.005

    # reward promos
    if is_promo(color, move_selected):
        reward += 0.005

    # reward if the move was the actual move selected by a pro, and pro won
    if move_selected == move_actual and color == winner:
        reward += 0.005

    # reward taking
    if game.board.board[move_selected[1]].color == enemy_color:
        reward += 0.005

    # reward taking hanging pieces
    if game.board.board[final_pos].color == enemy_color and enemy_attacked[final_pos] != 1:
        reward += PIECE_VALUES[game.board.board[final_pos].piece_type] * 0.05

    # avoid hanging
    if enemy_attacked[final_pos] == 1 and player_attacked[final_pos] == 0:
        reward -= PIECE_VALUES[game.board.board[init_pos].piece_type] * 0.05

    if color == 1:
        reward = 0 - reward
    return reward


def reinforcement_training(engine, model_name, moves, game_nums, game_states, metadata, epsilon, gamma, batch_size, epochs):
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-6)
    all_grads = []
    loss_over_time = []

    original_epsilon = epsilon

    sample = random.sample(range(0, len(game_states)), batch_size*2)
    hold_out = sample[batch_size:]  # TODO implement validation
    sample = sample[:batch_size]

    for epoch in range(epochs):
        np.random.shuffle(sample)

        for i in sample:
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(engine.model.trainable_variables)

                temp_game = Game(0, 1)
                temp_game.init_from_state(game_states[i])
                move = moves[i].tolist()
                player_color = temp_game.board.board[move[0]].color
                enemy_color = 1 - player_color
                winner = metadata[game_nums[i]][6]
                if winner == "[Result \"1-0\"]":
                    winner = 0
                elif winner == "[Result \"0-1\"]":
                    winner = 1
                else:
                    winner = 2

                curr_Q = engine.predict(temp_game)

                player_attacked, dummy = temp_game.all_squares_attacked(player_color)
                enemy_attacked, xrays = temp_game.all_squares_attacked(enemy_color)
                m = temp_game.generate_legal_moves(player_color, enemy_attacked, xrays)
                curr_moves = list(chain.from_iterable(m))
                if len(curr_moves) == 0:
                    continue

                # use epsilon method to balance exploration/exploitation
                sample_epsilon = np.random.rand()
                if sample_epsilon > epsilon:
                    # choose the highest rated move out of all moves
                    q_values = get_q_values(engine, temp_game, curr_moves, player_color)
                    action = np.argmax(q_values)
                    next_Q = q_values[action]
                else:
                    # choose a random move
                    q_values = get_q_values(engine, temp_game, curr_moves, player_color)
                    action = np.random.choice(len(curr_moves))
                    next_Q = q_values[action]

                reward = get_reward(temp_game, player_color, winner, curr_moves[action], move, enemy_attacked, player_attacked)

                loss_value = (reward + (gamma * next_Q) - curr_Q)

                loss_over_time.append(loss_value[0].numpy()[0])

            grads = tape.gradient(loss_value, engine.model.trainable_variables)
            all_grads.append(grads)

        # apply all gradients from sample positions
        for grad in all_grads:
            optimizer.apply_gradients(zip(grad, engine.model.trainable_variables))
        engine.model.save(f"predictor_model{epoch}.h5", overwrite=True, save_format="h5")
        loss_chart = pd.DataFrame()
        loss_chart["loss"] = loss_over_time
        loss_chart["iter"] = range(1, len(loss_chart) + 1)
        sns.scatterplot(data=loss_chart, x="iter", y="loss")

        epsilon = original_epsilon * epsilon

    engine.model.save(model_name, save_format="h5")