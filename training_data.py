import json
import random
import numpy as np
from game import Game

PAWN_VALUE = 1
KNIGHT_VALUE = 3.1
BISHOP_VALUE = 3.2
ROOK_VALUE = 4.9
QUEEN_VALUE = 9.8


def is_promo(color, move):
    if color == 0:
        if move[1] // 8 == 7:
            return True
    else:
        if move[1] // 8 == 0:
            return True
    return False


def load_all_train_data():
    naka_test_pos = np.load('training_data/nakamurapositions.npy')
    naka_test_evals = np.load('training_data/nakamuraevaluations.npy')
    carlsen_test_pos = np.load('training_data/carlsenpositions.npy')
    carlsen_test_evals = np.load('training_data/carlsenevaluations.npy')

    return np.concatenate((naka_test_pos, carlsen_test_pos)), np.concatenate((naka_test_evals, carlsen_test_evals))


def import_train_data(filename):
    with open(filename) as f:
        data = f.read()
        data = data.split("\n")
        metadata = []
        games = []

        is_meta = True
        last = 0
        # split the metadata from the PGNs
        for i in range(0, len(data)):
            if data[i] == "" and is_meta:
                curr_meta = data[last:i]
                metadata.append(curr_meta)
                is_meta = False
                last = i + 1
            elif data[i] == "" and not is_meta:
                curr_game = data[last:i]
                games.append(curr_game)
                is_meta = True
                last = i + 1

        # now go through the games, converting all moves into (pos1, pos2) format and concatenating them into one list
        converted_games = []
        promos = []
        for i in range(0, len(games)):
            curr_game = []
            curr_promos = []
            temp_game = Game(0, 1)
            for j in range(0, len(games[i])):
                games[i][j] = games[i][j].split(" ")
                t, p = convert_to_numeric(games[i][j], temp_game)
                curr_game += t
                curr_promos += p
            converted_games.append(curr_game)
            if len(curr_promos) == 0:
                curr_promos.append(0)
            promos.append(curr_promos)

        return metadata, converted_games, promos


def convert_to_numeric(game, temp_game):
    numeric_game = []
    promos = []
    color = 0
    for i in range(0, len(game), 1):
        # end is reached
        if game[i] == "":
            break

        move = []
        piece_list = []

        game[i] = game[i].replace('+', '')
        if i % 2 == 0:
            game[i] = game[i].split(".")[1]
            color = 0
            enemy_color = 1
            piece_list = temp_game.white_piece_pos
        else:
            color = 1
            enemy_color = 0
            piece_list = temp_game.black_piece_pos

        enemy_attacked, xrays = temp_game.all_squares_attacked(enemy_color)

        # pawn move
        if len(game[i]) == 2 or (len(game[i]) == 4 and game[i][1] == "x" and 97 <= ord(game[i][0]) <= 104):
            from_col = ord(game[i][0])-97
            col = ord(game[i][0])-97
            row = ord(game[i][1])-49
            moving_to = row*8 + col
            if len(game[i]) == 4:
                col = ord(game[i][2])-97
                row = ord(game[i][3])-49
                moving_to = row * 8 + col
            possible = []
            for pos in piece_list[1]:
                if pos % 8 == from_col:
                    possible.append(pos)
            for pos in possible:
                pawn_moves = temp_game.board.generate_pawn_moves(pos, temp_game.en_passant)
                move = (pos, moving_to)
                if moving_to in pawn_moves and temp_game.is_legal_move(move, pawn_moves, enemy_attacked, xrays):
                    temp_game.make_move(move, save=False)
                    numeric_game.append(move)
                    break
            continue

        # kingside castle
        elif game[i] == "O-O":
            if color == 0:
                move = (4, 6)
            else:
                move = (60, 62)
            temp_game.make_move(move, save=False)
            numeric_game.append(move)
            continue

        piece_type = -1
        generation = temp_game.board.generate_king_moves
        if game[i][0] == "K":
            piece_type = 0
        elif game[i][0] == "Q":
            piece_type = 5
            generation = temp_game.board.generate_queen_moves
        elif game[i][0] == "N":
            piece_type = 2
            generation = temp_game.board.generate_knight_moves
        elif game[i][0] == "B":
            piece_type = 3
            generation = temp_game.board.generate_bishop_moves
        elif game[i][0] == "R":
            piece_type = 4
            generation = temp_game.board.generate_rook_moves

        # unambiguous move
        if len(game[i]) == 3:
            col = ord(game[i][1])-97
            row = ord(game[i][2])-49
            moving_to = row*8 + col
            dummy = []
            for pos in piece_list[piece_type]:
                move = (pos, moving_to)
                moves = []

                if piece_type == 2 or piece_type == 0:
                    moves = generation(pos)
                else:
                    moves = generation(pos, False, dummy)

                if moving_to in moves and temp_game.is_legal_move(move, moves, enemy_attacked, xrays):
                    move = (pos, moving_to)
                    temp_game.make_move(move, save=False)
                    numeric_game.append(move)
                    break
            continue

        # promotion
        elif len(game[i]) == 4 and game[i][2] == "=":
            col = ord(game[i][0]) - 97
            moving_to = 0
            move = []
            if color == 0:
                moving_to = 56 + col
                move = (moving_to - 8, moving_to)
            else:
                moving_to = 0 + col
                move = (moving_to + 8, moving_to)
            promo_piece = 0
            if game[i][3] == "Q":
                promo_piece = 5
            elif game[i][3] == "N":
                promo_piece = 2
            elif game[i][3] == "B":
                promo_piece = 3
            elif game[i][3] == "R":
                promo_piece = 4
            temp_game.make_move(move, save=False, promo_code=promo_piece-2)
            numeric_game.append(move)
            promos.append(promo_piece-2)
            continue

        # unambiguous capture
        elif len(game[i]) == 4 and game[i][1] == "x":
            col = ord(game[i][2])-97
            row = ord(game[i][3])-49
            moving_to = row*8 + col
            dummy = []
            for pos in piece_list[piece_type]:
                move = (pos, moving_to)
                moves = []

                if piece_type == 2 or piece_type == 0:
                    moves = generation(pos)
                else:
                    moves = generation(pos, False, dummy)

                if moving_to in moves and temp_game.is_legal_move(move, moves, enemy_attacked, xrays):
                    move = (pos, moving_to)
                    temp_game.make_move(move, save=False)
                    numeric_game.append(move)
                    break
            continue

        # ambiguous move
        elif len(game[i]) == 4:
            col = ord(game[i][2]) - 97
            row = ord(game[i][3]) - 49
            moving_to = row*8 + col
            # row is ambiguous
            if 49 <= ord(game[i][1]) <= 56:
                from_row = ord(game[i][1])-49
                for pos in piece_list[piece_type]:
                    if pos // 8 == from_row:
                        move = (pos, moving_to)
                        temp_game.make_move(move, save=False)
                        numeric_game.append(move)
                        break
            # col is ambiguous
            elif 97 <= ord(game[i][1]) <= 104:
                from_col = ord(game[i][1]) - 97
                for pos in piece_list[piece_type]:
                    if pos % 8 == from_col:
                        move = (pos, moving_to)
                        temp_game.make_move(move, save=False)
                        numeric_game.append(move)
                        break
            continue

        # queenside castle
        elif game[i] == "O-O-O":
            if color == 0:
                move = (4, 2)
            else:
                move = (60, 58)
            temp_game.make_move(move, save=False)
            numeric_game.append(move)
            continue

        # ambiguous capture
        elif len(game[i]) == 5:
            col = ord(game[i][3]) - 97
            row = ord(game[i][4]) - 49
            moving_to = row*8 + col
            # row is ambiguous
            if 49 <= ord(game[i][1]) <= 56:
                from_row = ord(game[i][1])-49
                for pos in piece_list[piece_type]:
                    if pos // 8 == from_row:
                        move = (pos, moving_to)
                        temp_game.make_move(move, save=False)
                        numeric_game.append(move)
                        break
            # col is ambiguous
            elif 97 <= ord(game[i][1]) <= 104:
                from_col = ord(game[i][1]) - 97
                for pos in piece_list[piece_type]:
                    if pos % 8 == from_col:
                        move = (pos, moving_to)
                        temp_game.make_move(move, save=False)
                        numeric_game.append(move)
                        break
            continue

        # promotion and capture
        elif len(game[i]) == 6:
            from_col = ord(game[i][0]) - 97
            from_pos = 0
            if color == 0:
                from_pos = 48 + from_col
            else:
                from_pos = 8 + from_col

            col = ord(game[i][2]) - 97
            moving_to = 0
            if color == 0:
                moving_to = 56 + col
            else:
                moving_to = 0 + col

            move = (from_pos, moving_to)
            promo_piece = 0
            if game[i][5] == "Q":
                promo_piece = 5
            elif game[i][5] == "N":
                promo_piece = 2
            elif game[i][5] == "B":
                promo_piece = 3
            elif game[i][5] == "R":
                promo_piece = 4
            temp_game.make_move(move, save=False, promo_code=promo_piece - 2)
            numeric_game.append(move)
            promos.append(promo_piece - 2)
            continue

    return numeric_game, promos


def split_and_eval(games, promos):
    positions = []
    evaluations = []

    for game, promo in zip(games, promos):
        temp_game = Game(0, 1)
        move_count = 0

        promo_num = 0
        for move in game:
            piece_type = temp_game.board.board[move[0]].piece_type
            color = temp_game.board.board[move[0]].color
            w_attacked, dummy = temp_game.all_squares_attacked(0)
            b_attacked, dummy = temp_game.all_squares_attacked(1)

            # convert the board state to a numpy array that can be fed into model
            board3d = split_dims(temp_game.board, w_attacked, b_attacked)
            # add to the list of corresponding positions and evaluations
            positions.append(board3d)
            if material_count(temp_game) != 0:
                evaluations.append(material_count(temp_game))
            else:
                evaluations.append(random.uniform(-1, 1))

            temp_game.make_move(move, save=False, promo_code=promo[promo_num])
            move_count += 1

            if is_promo(color, move) and temp_game.board.board[move[0]].piece_type == 1:
                promo_num += 1

    positions = np.array(positions)
    evaluations = np.array(evaluations)

    save = input("Would you like to save the training data? ")

    if save == "y":
        save_training_data(positions, evaluations, promos)

    return positions, evaluations


def material_count(game):
    w_total = 0
    b_total = 0
    for pos in game.white_piece_pos[1]:
        if pos != -1:
            w_total += PAWN_VALUE
    for pos in game.white_piece_pos[2]:
        if pos != -1:
            w_total += KNIGHT_VALUE
    for pos in game.white_piece_pos[3]:
        if pos != -1:
            w_total += BISHOP_VALUE
    for pos in game.white_piece_pos[4]:
        if pos != -1:
            w_total += ROOK_VALUE
    for pos in game.white_piece_pos[5]:
        if pos != -1:
            w_total += QUEEN_VALUE
    for pos in game.black_piece_pos[1]:
        if pos != -1:
            b_total += PAWN_VALUE
    for pos in game.black_piece_pos[2]:
        if pos != -1:
            b_total += KNIGHT_VALUE
    for pos in game.black_piece_pos[3]:
        if pos != -1:
            b_total += BISHOP_VALUE
    for pos in game.black_piece_pos[4]:
        if pos != -1:
            b_total += ROOK_VALUE
    for pos in game.black_piece_pos[5]:
        if pos != -1:
            b_total += QUEEN_VALUE
    return w_total - b_total


def save_training_data(positions, evaluations, promos):
        filename = input("Input a filename: ")
        pos_name = filename + "positions"
        eval_name = filename + "evaluations"
        promo_name = filename + "promos"
        np.save(pos_name, positions)
        np.save(eval_name, evaluations)
        np.save(promo_name, promos)


def split_dims(board, w_attacked, b_attacked):
        board3d = np.zeros((14, 8, 8), dtype=np.int8)
        for i in range(0, 64):
            col = (i % 8)
            row = 7 - (i // 8)
            piece = board.board[i]
            board3d[12][row][col] = w_attacked[i]
            board3d[13][row][col] = b_attacked[i]
            if piece.piece_type == -1:
                continue
            else:
                if piece.color == 0:
                    board3d[piece.piece_type][row][col] = 1
                else:
                    board3d[piece.piece_type+6][row][col] = 1
        return board3d