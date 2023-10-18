import numpy as np
import tensorflow as tf
import os
from game import Game
from MCTS import MCTS
from engine import Engine
from itertools import chain


def get_q_values(engine, game, moves, color):
    q_values = []
    promo_num = 0
    for move in moves:
        state = game.make_move(move, save=True, promo_code=promo_num % 4)
        q_values.append(engine.predict(game))
        game.undo_move(state)
        if game.board.board[move[0]].piece_type == 1 and engine.is_promo(color, move):
            promo_num += 1
    return q_values


def get_reward(game, color, winner, move_selected, move_actual, enemy_attacked, player_attacked):
    reward = 0
    enemy_color = 1 - color

    # reward if the move was the actual move selected by a pro
    if move_selected == move_actual and color == winner:
        if color == 0:
            reward += 1
        else:
            reward -= 1

    # reward taking
    if game.board.board[move_selected[1]].color == enemy_color:
        if color == 0:
            reward += 0.1
        else:
            reward -= 0.1

    # avoid hanging
    if enemy_attacked[move_selected[1]] == 1 and player_attacked[move_selected[1]] == 0:
        if color == 0:
            reward -= 0.2
        if color == 1:
            reward += 0.2

    return reward


def reinforcement_training(engine, games, metadata, epsilon, gamma):
    optimizer = tf.keras.optimizers.Adam()
    game_num = 0
    for game in games:
        if game_num > 1:
            predictor = Engine("predictor_model_iter_" + (game_num-1))
        else:
            predictor = engine

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

        for move in game:
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(engine.model.trainable_variables)
                curr_Q = predictor.predict(temp_game)
                player_attacked, dummy = temp_game.all_squares_attacked(player_color)
                enemy_attacked, xrays = temp_game.all_squares_attacked(enemy_color)
                m = temp_game.generate_legal_moves(0, enemy_attacked, xrays)
                moves = list(chain.from_iterable(m))
                q_values = get_q_values(predictor, temp_game, moves, player_color)

                sample_epsilon = np.random.rand()
                if sample_epsilon <= epsilon:
                    action = np.random.choice(len(moves))
                    next_Q = q_values[action]
                else:
                    action = np.argmax(q_values)
                    next_Q = q_values[action]

                reward = get_reward(temp_game, player_color, winner, moves[action], move, enemy_attacked,
                                         player_attacked)

                loss_value = (reward + (gamma * next_Q) - curr_Q)

            grads = tape.gradient(loss_value, engine.model.trainable_variables)

            optimizer.apply_gradients(zip(grads, engine.model.trainable_variables))

            temp_game.make_move(move, save=False, promo_code=0)
            player_color = 1 - player_color
            enemy_color = 1 - player_color
            epsilon = epsilon * 0.9

        if game_num != 0:
            engine.model.save("predictor_model_iter_" + game_num)
            if game_num > 1:
                os.remove("predictor_model_iter_" + (game_num-1))
        game_num += 1

        if game_num > 1000:
            break

# MCTS stuff
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

