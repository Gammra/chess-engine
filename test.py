from engine import Engine
from game import Game
from main import read_fen
import cProfile
import pstats


@staticmethod
def is_promo(color, move):
    if color == 0:
        if move[1] // 8 == 7:
            return True
    else:
        if move[1] // 8 == 0:
            return True
    return False


def perft(game, depth, color, enemy_color):
    num_moves = 0
    promo_num = 0
    enemy_attacked, xrays = game.all_squares_attacked(enemy_color)
    moves = game.generate_legal_moves(color, enemy_attacked, xrays)
    for piece_type in moves:
        if depth == 1:
            num_moves += len(piece_type)
            continue
        for move in piece_type:
            save_state = game.make_move(move, True, promo_num % 4)
            num_moves += perft(game, depth-1, enemy_color, color)
            game.undo_move(save_state)
            if game.board.board[move[0]].piece_type == 1 and is_promo(color, move):
                promo_num += 1

    return num_moves


def debug_training():
    from main import train_model
    engine = Engine()
    training_filename = input("Input a training data filename(leave blank to load default data): ") + ".pgn"
    model_name = input("Please enter a name for the model: ") + ".h5"
    train_model(engine, training_filename, model_name)
    engine.model.summary()


def two_player_testing(SCREEN):
    from main import read_fen

    game = Game(0, 1)
    is_fen = input("Would you like to load the board from a fen string? (y or n): ")

    if is_fen == 'y':
        filename = input("Input a filename: ")
        game = read_fen(filename)

    while game.checkmate == -1:
        game.draw_board(SCREEN)
        game.draw_pos(SCREEN)
        game.make_player_move()
        if game.is_mate(0, 1):
            break

        game.player_color, game.CPU_color = \
            game.CPU_color, game.player_color

        game.draw_board(SCREEN)
        game.draw_pos(SCREEN)
        game.make_player_move()
        if game.is_mate(1, 0):
            break

        game.player_color, game.CPU_color = \
            game.CPU_color, game.player_color

    if game.checkmate == 0:
        print("Black wins!")
    elif game.checkmate == 1:
        print("White Wins!")
    if game.checkmate == 2:
        print("The game has ended in stalemate.")


def perft_testing():
    pos_2_game = read_fen("perft_positions/pos2.txt")
    print("perft testing from pos2:")
    print("Expected: 48 Actual:", perft(pos_2_game, 1, 0, 1))
    print("Expected 2039 Actual:", perft(pos_2_game, 2, 0, 1))
    print("Expected 97862 Actual:", perft(pos_2_game, 3, 0, 1))
    print("Expected 4085603 Actual:", perft(pos_2_game, 4, 0, 1))

    pos_3_game = read_fen("perft_positions/pos3.txt")
    print("perft testing from pos3:")
    print("Expected: 14 Actual:", perft(pos_3_game, 1, 0, 1))
    print("Expected: 191 Actual:", perft(pos_3_game, 2, 0, 1))
    print("Expected: 2812 Actual:", perft(pos_3_game, 3, 0, 1))
    print("Expected: 43238 Actual:", perft(pos_3_game, 4, 0, 1))

    pos_4_game = read_fen("perft_positions/pos4.txt")
    print("perft testing from pos4:")
    print("Expected: 6 Actual:", perft(pos_4_game, 1, 0, 1))
    print("Expected: 264 Actual:", perft(pos_4_game, 2, 0, 1))
    print("Expected: 9467 Actual:", perft(pos_4_game, 3, 0, 1))
    print("Expected: 422333 Actual:", perft(pos_4_game, 4, 0, 1))

    pos_5_game = read_fen("perft_positions/pos5.txt")
    print("perft testing from pos5:")
    print("Expected: 44 Actual:", perft(pos_5_game, 1, 0, 1))
    print("Expected: 1486 Actual:", perft(pos_5_game, 2, 0, 1))
    print("Expected: 62379 Actual:", perft(pos_5_game, 3, 0, 1))
    print("Expected: 2103487 Actual:", perft(pos_5_game, 4, 0, 1))

    pos_6_game = read_fen("perft_positions/pos6.txt")
    print("perft testing from pos6:")
    print("Expected: 46 Actual:", perft(pos_6_game, 1, 0, 1))
    print("Expected: 2079 Actual:", perft(pos_6_game, 2, 0, 1))
    print("Expected: 89890 Actual:", perft(pos_6_game, 3, 0, 1))
    print("Expected: 3894594 Actual:", perft(pos_6_game, 4, 0, 1))

    init_pos_game = Game(0, 1)
    print("perft testing from starting position:")
    print("Expected: 400 Actual:", perft(init_pos_game, 2, 0, 1))
    print("Expected: 8902 Actual:", perft(init_pos_game, 3, 0, 1))
    print("Expected: 197281 Actual:", perft(init_pos_game, 4, 0, 1))


def profiling():
    # test move generation speed
    runtime_test_game = read_fen("perft_positions/pos2.txt")
    pr = cProfile.Profile()
    pr.enable()
    perft(runtime_test_game, 3, 0, 1)
    pr.disable()
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.dump_stats(filename='perft_pos2_depth3')
