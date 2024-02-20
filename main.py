import numpy as np

from game import Game
from board import Board
from board import Piece
from engine import Engine
import arena
import json
import rl
import training_data
import os
import time
import test
import pygame
import tensorflow as tf

pygame.init()
pygame.display.set_caption("Jack's Chess")

SCREEN_WIDTH = 480
SCREEN_HEIGHT = 480
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))


def display_menu():
    print("Welcome to Jack's chess program featuring Scooter the chess engine!\n")
    mode = input("1: Play against the engine\n"
                 "2: Load a model\n"
                 "3: Train a model\n"
                 "4: Debug\n"
                 "5: Quit\n"
                 "Please select what you would like to do by inputting one of the numbers above: ")
    return mode


def display_debug_menu():
    print("Please select which type of testing you would like to perform.")
    test_mode = input("1: Engine testing\n"
                      "2: Two-player testing\n"
                      "3: perft testing\n"
                      "4: Speed testing\n"
                      "Please select what you would like to do by inputting one of the numbers above: ")
    return test_mode


def run_game(engine):
    player_color = 0
    CPU_color = 1

    color_selected = False
    while not color_selected:
        player_color = ord(input("Please select a color. Press 0 to play as white, and 1 to play as black: ")) - 48
        CPU_color = 1 - player_color
        if not 0 <= player_color <= 1:
            print("Invalid input. Input must be 1 or 0.\n")
        else:
            color_selected = True
            os.system('cls')

    game = Game(player_color, CPU_color)
    engine.cpu_color = CPU_color

    # run the game loop until a mate is reached
    while game.checkmate == -1:
        if player_color == 0:
            game.draw_board(SCREEN)
            game.draw_pos(SCREEN)
            if player_color == 0:
                game.make_player_move()
            else:
                engine.make_CPU_move(game.CPU_color, game, 3)
            if game.is_mate(0, 1):
                break

            game.draw_board(SCREEN)
            game.draw_pos(SCREEN)
            if player_color == 1:
                game.make_player_move()
            else:
                engine.make_CPU_move(game.CPU_color, game, 3)
            if game.is_mate(1, 0):
                break

    if game.checkmate == 0:
        print("Black wins!")
    elif game.checkmate == 1:
        print("White Wins!")
    if game.checkmate == 2:
        print("The game has ended in stalemate.")


def load_model(engine, model_name):
    engine.model = tf.keras.models.load_model(model_name)


"""
If the user entered a filename, load the chosen PGN file. Convert from PGN format to a list of moves (pos1, pos2), and 
evaluate each position reached in each game.
If the user did not enter a filename, load the default preprocessed data.
Then pretrain the model using this data, and enter the reinforcement learning loop (if the user wants, they can skip
directly to RL).
"""
def train_model(engine, training_filename, model_name):
    pgn_name = training_filename
    training_filename = "pgns/" + training_filename
    conv_size, conv_depth = 64, 3

    # if a training filename was entered
    if training_filename != "pgns/.pgn":
        metadata, games, promos = training_data.import_train_data(training_filename)
        with open(f"training_data/{pgn_name}_metadata.txt", "w") as f:
            json.dump(metadata, f)
        moves, game_nums, game_states = training_data.generate_states(games, promos)
    # load default data
    else:
        moves, game_nums, game_states, metadata = training_data.load_all_train_data()

    engine.build_model(conv_size, conv_depth)
    engine.compile_model()

    epsilon = 0.97
    gamma = 0.97
    batch_size = 512
    epochs = 24
    rl.reinforcement_training(engine, model_name, moves, game_nums, game_states, metadata, epsilon, gamma, batch_size, epochs)
    # arena.self_play(10, 10, engine)


def run_debug():
    test_mode = 1
    done = False
    while not done:
        test_mode = display_debug_menu()
        if not 49 <= ord(test_mode) <= 52:
            print("Invalid input. Input must be a number 1-4.")
        else:
            done = True
            os.system('cls')

    # debug training
    if test_mode == "1":
        test.debug_training()

    # debug with the ability to move both black and white pieces
    elif test_mode == "2":
        test.two_player_testing(SCREEN)

    # debug chess sim using perft results
    elif test_mode == "3":
        test.perft_testing()

    # run speed profiling
    elif test_mode == "4":
        test.profiling()


def main():
    done = False
    while not done:
        mode = display_menu()
        engine = Engine()
        if mode == "1":
            run_game(engine)
        elif mode == "2":
            model_name = input("Input a model name: ")
            load_model(engine, model_name)
        elif mode == "3":
            training_data_name = input("Input a training data filename (leave blank to load default data): ") + ".pgn"
            model_name = input("Input a model name: ") + ".h5"
            train_model(engine, training_data_name, model_name)
        elif mode == "4":
            run_debug()
        elif mode == "5":
            done = True
            exit(0)
        else:
            print("Invalid input. Input must be a number 1-5. \n")
            time.sleep(3)
            os.system('cls')


# returns a game class initialized to the position from a FEN string.
def read_fen(filename):
    fen_game = Game(0, 1)
    fen_game.init_from_fen(0, 1)
    fen_board = Board()
    fen_board.init_blank()

    with open(filename) as f:
        fen = f.read()
        fen = fen.split("/")
        states = fen[7].split(" ")
        fen[7] = fen[7].split(" ")[0]

        for row in range(0, len(fen)):
            col = 0
            for char in fen[row]:
                curr_square = col + (56 - row * 8)
                ascii_val = ord(char)
                # white pieces
                # king
                if ascii_val == 75:
                    curr = Piece(0, 0)
                    fen_board.board[curr_square] = curr
                    fen_game.white_piece_pos[0].append(curr_square)
                # pawn
                elif ascii_val == 80:
                    curr = Piece(0, 1)
                    fen_board.board[curr_square] = curr
                    fen_game.white_piece_pos[1].append(curr_square)
                # knight
                elif ascii_val == 78:
                    curr = Piece(0, 2)
                    fen_board.board[curr_square] = curr
                    fen_game.white_piece_pos[2].append(curr_square)
                # bishop
                elif ascii_val == 66:
                    curr = Piece(0, 3)
                    fen_board.board[curr_square] = curr
                    fen_game.white_piece_pos[3].append(curr_square)
                # rook
                elif ascii_val == 82:
                    curr = Piece(0, 4)
                    fen_board.board[curr_square] = curr
                    fen_game.white_piece_pos[4].append(curr_square)
                # queen
                elif ascii_val == 81:
                    curr = Piece(0, 5)
                    fen_board.board[curr_square] = curr
                    fen_game.white_piece_pos[5].append(curr_square)

                # black pieces
                # king
                elif ascii_val == 107:
                    curr = Piece(1, 0)
                    fen_board.board[curr_square] = curr
                    fen_game.black_piece_pos[0].append(curr_square)
                # pawn
                elif ascii_val == 112:
                    curr = Piece(1, 1)
                    fen_board.board[curr_square] = curr
                    fen_game.black_piece_pos[1].append(curr_square)
                # knight
                elif ascii_val == 110:
                    curr = Piece(1, 2)
                    fen_board.board[curr_square] = curr
                    fen_game.black_piece_pos[2].append(curr_square)
                # bishop
                elif ascii_val == 98:
                    curr = Piece(1, 3)
                    fen_board.board[curr_square] = curr
                    fen_game.black_piece_pos[3].append(curr_square)
                # rook
                elif ascii_val == 114:
                    curr = Piece(1, 4)
                    fen_board.board[curr_square] = curr
                    fen_game.black_piece_pos[4].append(curr_square)
                # queen
                elif ascii_val == 113:
                    curr = Piece(1, 5)
                    fen_board.board[curr_square] = curr
                    fen_game.black_piece_pos[5].append(curr_square)

                # blank space
                if 47 < ascii_val < 58:
                    num_spaces = ascii_val - 48
                    for i in range(0, num_spaces):
                        col += 1
                else:
                    col += 1
        fen_game.board = fen_board

        # determine who's turn it is
        if ord(states[1][0]) == 98:
            fen_game.player_color, fen_game.CPU_color = fen_game.CPU_color, fen_game.player_color

        # determine which castling moves are legal
        for char in states[2]:
            # white kingside
            if ord(char) == 75:
                fen_game.castles_kingside[0] = 1
            # white queenside
            if ord(char) == 81:
                fen_game.castles_queenside[0] = 1
            # black kingside
            if ord(char) == 107:
                fen_game.castles_kingside[1] = 1
            # black queenside
            if ord(char) == 113:
                fen_game.castles_queenside[1] = 1

        # determine if any pawns are capable of being captured en passant
        if len(states[3]) != 0 and ord(states[3][0]) != 45:
            pawn_column = ord(states[2][0]) - 97
            if fen_game.player_color == 0:
                fen_game.en_passant[1] = 48 + pawn_column
            else:
                fen_game.en_passant[0] = 0 + pawn_column

    return fen_game


if __name__ == "__main__":
    main()
