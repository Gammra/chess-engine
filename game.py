from board import Board
from board import Piece
import pygame
import os

# load piece and board images
board_img = pygame.image.load(os.path.join('assets', 'board.png'))
wp_img = pygame.image.load(os.path.join('assets', 'wp.png'))
bp_img = pygame.image.load(os.path.join('assets', 'bp.png'))
wk_img = pygame.image.load(os.path.join('assets', 'wk.png'))
bk_img = pygame.image.load(os.path.join('assets', 'bk.png'))
wq_img = pygame.image.load(os.path.join('assets', 'wq.png'))
bq_img = pygame.image.load(os.path.join('assets', 'bq.png'))
wr_img = pygame.image.load(os.path.join('assets', 'wr.png'))
br_img = pygame.image.load(os.path.join('assets', 'br.png'))
wb_img = pygame.image.load(os.path.join('assets', 'wb.png'))
bb_img = pygame.image.load(os.path.join('assets', 'bb.png'))
wn_img = pygame.image.load(os.path.join('assets', 'wn.png'))
bn_img = pygame.image.load(os.path.join('assets', 'bn.png'))

empty = Piece(-1, -1)  # fills blank spaces


class SaveState:
    def __init__(self, move, piece_moved, piece_captured, castle_queenside, castle_kingside, en_passant, promo, is_ep):
        self.move = move
        self.piece_moved = piece_moved
        self.piece_captured = piece_captured
        self.castle_queenside = castle_queenside
        self.castle_kingside = castle_kingside
        self.en_passant = en_passant
        self.promo = promo
        self.is_ep = is_ep


class Game:
    def __init__(self, player_color, CPU_color):
        self.player_color = player_color
        self.CPU_color = CPU_color
        self.board = Board()
        self.board.init_board()
        self.checkmate = -1
        self.en_passant = [-1, -1]
        self.castles_queenside = [1, 1]
        self.castles_kingside = [1, 1]
        self.white_piece_pos = self.init_pos(0)
        self.black_piece_pos = self.init_pos(1)

    def init_from_fen(self, player_color, CPU_color):
        self.player_color = player_color
        self.CPU_color = CPU_color
        self.board.init_blank()
        self.checkmate = -1
        self.en_passant = [-1, -1]
        self.castles_queenside = [0, 0]
        self.castles_kingside = [0, 0]
        self.white_piece_pos = [[], [], [], [], [], []]
        self.black_piece_pos = [[], [], [], [], [], []]

    @staticmethod
    def init_pos(color):
        piece_pos = [[], [], [], [], [], []]
        if color == 0:  # white
            piece_pos[0].append(4)
            for i in range(8, 16):
                piece_pos[1].append(i)
            piece_pos[5].append(3)
            piece_pos[4].append(0)
            piece_pos[4].append(7)
            piece_pos[3].append(2)
            piece_pos[3].append(5)
            piece_pos[2].append(1)
            piece_pos[2].append(6)
        else:  # black
            piece_pos[0].append(60)
            for i in range(48, 56):
                piece_pos[1].append(i)
            piece_pos[5].append(59)
            piece_pos[4].append(56)
            piece_pos[4].append(63)
            piece_pos[3].append(58)
            piece_pos[3].append(61)
            piece_pos[2].append(57)
            piece_pos[2].append(62)
        return piece_pos

    @staticmethod
    def draw_board(screen):
        screen.blit(board_img, (0, 0))
        pygame.display.update()

    def draw_pos(self, screen):
        for i in range(64):
            curr_piece = self.board.board[i]

            # white
            if self.player_color == 0:
                x = 60 * (i % 8)
                y = 60 * (7 - (i // 8))
            # black
            else:
                x = 60 * (7 - (i % 8))
                y = 60 * (i // 8)

            # white
            if curr_piece.color == 0:
                if curr_piece.piece_type == 0:
                    screen.blit(wk_img, (x, y))
                elif curr_piece.piece_type == 1:
                    screen.blit(wp_img, (x, y))
                elif curr_piece.piece_type == 2:
                    screen.blit(wn_img, (x, y))
                elif curr_piece.piece_type == 3:
                    screen.blit(wb_img, (x, y))
                elif curr_piece.piece_type == 4:
                    screen.blit(wr_img, (x, y))
                elif curr_piece.piece_type == 5:
                    screen.blit(wq_img, (x, y))

            # black
            elif curr_piece.color == 1:
                if curr_piece.piece_type == 0:
                    screen.blit(bk_img, (x, y))
                elif curr_piece.piece_type == 1:
                    screen.blit(bp_img, (x, y))
                elif curr_piece.piece_type == 2:
                    screen.blit(bn_img, (x, y))
                elif curr_piece.piece_type == 3:
                    screen.blit(bb_img, (x, y))
                elif curr_piece.piece_type == 4:
                    screen.blit(br_img, (x, y))
                elif curr_piece.piece_type == 5:
                    screen.blit(bq_img, (x, y))
            pygame.display.update()

    def square_at_pos(self, mouse_pos):
        x, y = mouse_pos
        if self.player_color == 0:
            x = x // 60
            y = 7 - (y // 60)
        # board view is inverted for black
        else:
            x = 7 - (x // 60)
            y = y // 60
        return (y * 8) + x

    def select_square(self):
        while True:
            for event in pygame.event.get():
                # player clicked the location of the piece they want to move
                if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    # get the position of the mouse upon a click and check what square is at that position
                    mouse_pos = pygame.mouse.get_pos()
                    board_pos = self.square_at_pos(mouse_pos)
                    return board_pos

    # first, wait for the player to select a square containing one of their pieces.
    # then wait until the player selects a square that is legal for the selected piece to move to
    def make_player_move(self):
        turn_complete = False
        while not turn_complete:
            init_pos = self.select_square()
            piece_selected = self.board.board[init_pos]
            # guarantee that the piece selected belongs to the player
            if piece_selected.color == self.player_color:
                target_pos = self.select_square()
                move = (init_pos, target_pos)
                enemy_attacked, xrays = self.all_squares_attacked(self.CPU_color)

                piece_moves = []
                if piece_selected.piece_type == 0:
                    piece_moves = self.board.generate_king_moves(move[0])
                elif piece_selected.piece_type == 1:
                    piece_moves = self.board.generate_pawn_moves(move[0], self.en_passant)
                elif piece_selected.piece_type == 2:
                    piece_moves = self.board.generate_knight_moves(move[0])
                elif piece_selected.piece_type == 3:
                    piece_moves = self.board.generate_bishop_moves(move[0], True, xrays)
                elif piece_selected.piece_type == 4:
                    piece_moves = self.board.generate_rook_moves(move[0], True, xrays)
                elif piece_selected.piece_type == 5:
                    piece_moves = self.board.generate_queen_moves(move[0], True, xrays)

                if self.is_legal_move(move, piece_moves, enemy_attacked, xrays):
                    self.make_move(move, save=False)
                    turn_complete = True
                else:
                    print("This is not a legal move. Please select another move.\n")

    def all_squares_attacked(self, color):
        squares_attacked = [0]*64
        xrays = []
        if color == 0:
            pieces = self.white_piece_pos
        else:
            pieces = self.black_piece_pos

        king_moves = self.board.generate_king_moves(pieces[0][0])
        pawn_moves = []
        for piece_pos in pieces[1]:
            if piece_pos == -1:
                continue
            pawn_moves += self.board.generate_pawn_attacks(piece_pos, self.en_passant)
        knight_moves = []
        for piece_pos in pieces[2]:
            if piece_pos == -1:
                continue
            knight_moves += self.board.generate_knight_moves(piece_pos)
        bishop_moves = []
        for piece_pos in pieces[3]:
            if piece_pos == -1:
                continue
            bishop_moves += self.board.generate_bishop_moves(piece_pos, True, xrays)
        rook_moves = []
        for piece_pos in pieces[4]:
            if piece_pos == -1:
                continue
            rook_moves += self.board.generate_rook_moves(piece_pos, True, xrays)
        queen_moves = []
        for piece_pos in pieces[5]:
            if piece_pos == -1:
                continue
            queen_moves += self.board.generate_queen_moves(piece_pos, True, xrays)

        all_moves = [king_moves, pawn_moves, knight_moves, bishop_moves, rook_moves, queen_moves]
        for i in all_moves:
            for j in i:
                squares_attacked[j] = 1

        return squares_attacked, xrays

    def move_checks_player(self, move, enemy_attacked, xrays):
        piece_color = self.board.board[move[0]].color
        enemy_color = 1 - piece_color
        pieces = self.white_piece_pos
        if piece_color == 1:
            pieces = self.black_piece_pos
        king_pos = pieces[0][0]

        # if you are in check, you must get out of check
        if enemy_attacked[king_pos] == 1:
            save_state = self.make_move(move, save=True, promo_code=0)
            king_pos_after_move = pieces[0][0]
            # dummy is unimportant xray data
            temp_enemy_attacked, dummy = self.all_squares_attacked(enemy_color)
            self.undo_move(save_state)
            if temp_enemy_attacked[king_pos_after_move] == 1:
                return True

        # king can't walk into check
        if self.board.board[move[0]].piece_type == 0 and enemy_attacked[move[1]] == 1:
            return True

        # check if move reveals a discovered check against king
        for xray in xrays:
            # xray[0] is the xrayed piece, xray[1] is the vector of the xray
            if xray[0] == move[0] and abs(move[1]-move[0]) % xray[1] != 0:
                return True
            if xray[0] == move[0] and abs(xray[1]) == 1 and (move[1] // 8) != (move[0] // 8):
                return True

        # check if an en passant reveals a check
        if self.board.board[move[0]].piece_type == 1 and abs(move[1] - move[0]) % 8 != 0 \
                and self.board.board[move[1]].color == -1:
            if (piece_color == 0 and self.en_passant[1] == move[1] - 8) \
                    or (piece_color == 1 and self.en_passant[0] == move[1] + 8):
                save_state = self.make_move(move, save=True)
                king_pos_after_move = pieces[0][0]
                temp_enemy_attacked, dummy = self.all_squares_attacked(enemy_color)
                self.undo_move(save_state)
                if temp_enemy_attacked[king_pos_after_move] == 1:
                    return True

        return False

    """
    You can castle if:
    you are not in check
    the spaces between the rook and king are not obstructed
    you have not already castled or moved the king or rook in question
    the squares the king passes through are not attacked by an enemy piece
    """
    def is_legal_castle(self, move, enemy_attacked):
        #  you can't castle out of check
        if enemy_attacked[move[0]] == 1:
            return False

        # white kingside
        if move[1] == 6:
            if self.board.board[5].color != -1 or self.board.board[6].color != -1:
                return False
            if self.castles_kingside[0] == 0 or enemy_attacked[5] == 1 or enemy_attacked[6] == 1:
                return False
        # white queenside
        elif move[1] == 2:
            if self.board.board[1].color != -1 or self.board.board[2].color != -1 or self.board.board[3].color != -1:
                return False
            if self.castles_queenside[0] == 0 or enemy_attacked[2] == 1 or enemy_attacked[3] == 1:
                return False
        # black kingside
        elif move[1] == 62:
            if self.board.board[61].color != -1 or self.board.board[62].color != -1:
                return False
            if self.castles_kingside[1] == 0 or enemy_attacked[61] == 1 or enemy_attacked[62] == 1:
                return False
        # black queenside
        elif move[1] == 58:
            if self.board.board[57].color != -1 or self.board.board[58].color != -1 or self.board.board[59].color != -1:
                return False
            if self.castles_queenside[1] == 0 or enemy_attacked[58] == 1 or enemy_attacked[59] == 1:
                return False

        return True

    """
    Assumes a player piece is selected.
    A move is legal if:
    the square is not obstructed by a player's piece
    the piece can attack that square
    the move doesn't put the player in check
    """
    def is_legal_move(self, move, piece_moves, enemy_attacked, xrays):
        init_piece = self.board.board[move[0]]
        target_piece = self.board.board[move[1]]

        # one can't move where they already have a piece
        if target_piece.color == init_piece.color:
            return False

        # check if move is a castle, if it is a legal castle return true if not return false
        elif init_piece.piece_type == 0 and abs(move[1] - move[0]) == 2:
            if self.is_legal_castle(move, enemy_attacked):
                return True
            else:
                return False

        # if the piece is capable of moving to a square, return false if moving there puts the player in check
        # else return false
        elif move[1] in piece_moves:
            if not self.move_checks_player(move, enemy_attacked, xrays):
                return True
            else:
                return False

        else:
            return False

    def promote_pawn(self, color, pos_promoted, promo_code):
        promoted_piece_type = -1

        # this perft_code will be something other than -1 when a promotion is made by the cpu or by perft in testing
        if promo_code != -1:
            promoted_piece_type = promo_code + 2
        # prompt the player for what they want to promote the pawn to
        else:
            promoted_piece_type = \
                -47 + ord(input(
                  "1: Knight\n"
                  "2: Bishop\n"
                  "3: Rook\n"
                  "4: Queen\n"
                  "Please select what you would to promote the pawn to: "))

        # get piece list for given color
        piece_positions = []
        if color == 0:
            piece_positions = self.white_piece_pos
        else:
            piece_positions = self.black_piece_pos

        # put the promoted piece on the board, and add its position to the players piece position list
        promoted_piece = Piece(color, promoted_piece_type)
        self.board.board[pos_promoted] = promoted_piece
        piece_positions[promoted_piece_type].append(pos_promoted)

        # find the pawn that was promoted and remove its position from the position list
        piece_positions[1].remove(pos_promoted)

    def capture(self, move, piece_attacked):
        cap_piece_list = []
        pos_captured = move[1]
        if piece_attacked.color == 1:
            cap_piece_list = self.black_piece_pos[piece_attacked.piece_type]
        elif piece_attacked.color == 0:
            cap_piece_list = self.white_piece_pos[piece_attacked.piece_type]

        # if a rook is captured, player can no longer castle on that side
        if piece_attacked.piece_type == 4:
            if pos_captured == 0:
                self.castles_queenside[0] = 0
            elif pos_captured == 7:
                self.castles_kingside[0] = 0
            elif pos_captured == 56:
                self.castles_queenside[1] = 0
            elif pos_captured == 63:
                self.castles_kingside[1] = 0

        cap_piece_list.remove(pos_captured)

    def attack_en_passant(self, save, state, piece_color):
        ep_list = []
        enemy_color = 1 - piece_color
        pos_pawn_killed = self.en_passant[enemy_color]
        if piece_color == 0:
            ep_list = self.black_piece_pos[1]
        else:
            ep_list = self.white_piece_pos[1]

        if save:
            state.piece_captured = self.board.board[pos_pawn_killed]
            state.is_ep = True

        self.board.board[pos_pawn_killed] = empty
        self.en_passant[enemy_color] = -1
        ep_list.remove(pos_pawn_killed)

    def make_move(self, move, save, promo_code=-1):
        piece_moved = self.board.board[move[0]]
        piece_attacked = self.board.board[move[1]]
        piece_color = piece_moved.color
        enemy_color = 1 - piece_color
        state = SaveState(move, piece_moved, empty, self.castles_queenside.copy(), self.castles_kingside.copy(), self.en_passant.copy(), False, False)

        piece_list = []
        if piece_color == 0:
            piece_list = self.white_piece_pos[piece_moved.piece_type]
        else:
            piece_list = self.black_piece_pos[piece_moved.piece_type]

        # move the piece on the board
        self.board.board[move[0]] = empty
        self.board.board[move[1]] = piece_moved

        # adjust the player position list to reflect the move
        moved = piece_list.index(move[0])
        piece_list[moved] = move[1]

        # special pawn moves
        if piece_moved.piece_type == 1:
            # pawn is now en passant-able
            if abs(move[1] - move[0]) == 16:
                self.en_passant[piece_color] = move[1]

            # move is en passant because the pawn is moving diagonally to an empty square
            if piece_attacked.color == -1 and abs(move[1] - move[0]) % 8 != 0:
                self.attack_en_passant(save, state, piece_color)
                if save:
                    return state
                return
            else:
                self.en_passant[enemy_color] = -1

            # promotion
            if move[1] // 8 == 7 or move[1] // 8 == 0:
                self.promote_pawn(piece_moved.color, move[1], promo_code)
                if save:
                    state.promo = True

        # player made a move, so reset the en passant state
        else:
            self.en_passant[enemy_color] = -1

        # if move is a capture, adjust the enemy player position list to reflect the move
        if piece_attacked.color == enemy_color:
            self.capture(move, piece_attacked)
            if save:
                state.piece_captured = piece_attacked

        # if king moves, you can no longer castle
        # if move is a castle, move the rook
        if piece_moved.piece_type == 0:
            self.castles_kingside[piece_moved.color] = 0
            self.castles_queenside[piece_moved.color] = 0
            if abs(move[1] - move[0]) == 2:
                self.move_rook_castle(move[1])

        # if a rook is moved, the player can't castle on that side
        elif piece_moved.piece_type == 4:
            if move[0] == 0 or move[0] == 56:
                self.castles_queenside[piece_moved.color] = 0
            elif move[0] == 7 or move[0] == 63:
                self.castles_kingside[piece_moved.color] = 0

        if save:
            return state

    def undo_move(self, save_state):
        piece_moved = save_state.piece_moved
        piece_captured = save_state.piece_captured

        # reset states
        self.castles_kingside = save_state.castle_kingside
        self.castles_queenside = save_state.castle_queenside
        self.en_passant = save_state.en_passant

        # return the piece that was moved back to its original position
        self.board.board[save_state.move[0]] = piece_moved

        # return the piece_captured to its original position (if there was no capture put an empty piece there)
        # if a pawn was captured en passant, replace the position moved to by the pawn attacking with an empty piece
        if save_state.is_ep:
            pos_pawn_killed = self.en_passant[piece_captured.color]
            self.board.board[pos_pawn_killed] = piece_captured
            self.board.board[save_state.move[1]] = empty
        elif not save_state.promo:
            self.board.board[save_state.move[1]] = piece_captured

        # check if move was a promo. if so, remove the promoted piece from the player's piece list
        # also add the pawn that was promoted back to the player's piece list
        if save_state.promo:
            promo_piece_list = []
            promo_piece = self.board.board[save_state.move[1]]
            self.board.board[save_state.move[1]] = piece_captured
            if piece_moved.color == 0:
                promo_piece_list = self.white_piece_pos
            else:
                promo_piece_list = self.black_piece_pos
            promo_piece_list[promo_piece.piece_type].remove(save_state.move[1])
            promo_piece_list[1].insert(0, save_state.move[0])
        # adjust player piece pos to original state
        else:
            piece_list = []
            if piece_moved.color == 0:
                piece_list = self.white_piece_pos[piece_moved.piece_type]
            else:
                piece_list = self.black_piece_pos[piece_moved.piece_type]
            moved = piece_list.index(save_state.move[1])
            piece_list[moved] = save_state.move[0]

        # if move is castle, move the rook back
        if save_state.piece_moved.piece_type == 0 and abs(save_state.move[1] - save_state.move[0]) == 2:
            self.undo_move_rook_castle(save_state.move[1])
            return

        # check if move was a capture
        if piece_captured.color != -1:
            cap_piece_list = []
            if piece_captured.color == 0:
                cap_piece_list = self.white_piece_pos[piece_captured.piece_type]
            elif piece_captured.color == 1:
                cap_piece_list = self.black_piece_pos[piece_captured.piece_type]
            if not save_state.is_ep:
                cap_piece_list.insert(0, save_state.move[1])
            else:
                cap_piece_list.insert(0, self.en_passant[piece_captured.color])

    def move_rook_castle(self, pos):
        if pos == 2:
            self.board.board[3] = self.board.board[0]
            self.board.board[0] = empty
            rook = self.white_piece_pos[4].index(0)
            self.white_piece_pos[4][rook] = 3
        elif pos == 6:
            self.board.board[5] = self.board.board[7]
            self.board.board[7] = empty
            rook = self.white_piece_pos[4].index(7)
            self.white_piece_pos[4][rook] = 5
        elif pos == 58:
            self.board.board[59] = self.board.board[56]
            self.board.board[56] = empty
            rook = self.black_piece_pos[4].index(56)
            self.black_piece_pos[4][rook] = 59
        elif pos == 62:
            self.board.board[61] = self.board.board[63]
            self.board.board[63] = empty
            rook = self.black_piece_pos[4].index(63)
            self.black_piece_pos[4][rook] = 61

    def undo_move_rook_castle(self, pos):
        if pos == 2:
            self.board.board[0] = self.board.board[3]
            self.board.board[3] = empty
            rook = self.white_piece_pos[4].index(3)
            self.white_piece_pos[4][rook] = 0
        elif pos == 6:
            self.board.board[7] = self.board.board[5]
            self.board.board[5] = empty
            rook = self.white_piece_pos[4].index(5)
            self.white_piece_pos[4][rook] = 7
        elif pos == 58:
            self.board.board[56] = self.board.board[59]
            self.board.board[59] = empty
            rook = self.black_piece_pos[4].index(59)
            self.black_piece_pos[4][rook] = 56
        elif pos == 62:
            self.board.board[63] = self.board.board[61]
            self.board.board[61] = empty
            rook = self.black_piece_pos[4].index(61)
            self.black_piece_pos[4][rook] = 63

    def generate_legal_moves(self, color, enemy_attacked, xrays):
        all_moves = [[], [], [], [], [], []]

        if color == 0:
            pieces = self.white_piece_pos
        else:
            pieces = self.black_piece_pos

        king_moves = self.board.generate_king_moves(pieces[0][0])
        for i in king_moves:
            move = (pieces[0][0], i)
            if self.is_legal_move(move, king_moves, enemy_attacked, xrays):
                all_moves[5].append(move)
        all_moves[5] += self.generate_legal_castles(color, enemy_attacked)

        for piece_pos in pieces[1]:
            pawn_moves = self.board.generate_pawn_moves(piece_pos, self.en_passant)
            for i in pawn_moves:
                move = (piece_pos, i)
                if self.is_legal_move(move, pawn_moves, enemy_attacked, xrays):
                    all_moves[2].append(move)
                    if (i // 8) == 0 or (i // 8) == 7:
                        all_moves[2].append(move)
                        all_moves[2].append(move)
                        all_moves[2].append(move)

        for piece_pos in pieces[2]:
            knight_moves = self.board.generate_knight_moves(piece_pos)
            for i in knight_moves:
                move = (piece_pos, i)
                if self.is_legal_move(move, knight_moves, enemy_attacked, xrays):
                    all_moves[0].append(move)

        for piece_pos in pieces[3]:
            bishop_moves = self.board.generate_bishop_moves(piece_pos, True, xrays)
            for i in bishop_moves:
                move = (piece_pos, i)
                if self.is_legal_move(move, bishop_moves, enemy_attacked, xrays):
                    all_moves[1].append(move)

        for piece_pos in pieces[4]:
            rook_moves = self.board.generate_rook_moves(piece_pos, True, xrays)
            for i in rook_moves:
                move = (piece_pos, i)
                if self.is_legal_move(move, rook_moves, enemy_attacked, xrays):
                    all_moves[3].append(move)

        for piece_pos in pieces[5]:
            queen_moves = self.board.generate_queen_moves(piece_pos, True, xrays)
            for i in queen_moves:
                move = (piece_pos, i)
                if self.is_legal_move(move, queen_moves, enemy_attacked, xrays):
                    all_moves[4].append(move)

        return all_moves

    def generate_legal_castles(self, color, enemy_attacked):
        legal_castles = []
        if color == 0:
            qs = (self.white_piece_pos[0][0], 2)
            ks = (self.white_piece_pos[0][0], 6)
            if self.is_legal_castle(qs, enemy_attacked):
                legal_castles.append(qs)
            if self.is_legal_castle(ks, enemy_attacked):
                legal_castles.append(ks)
        else:
            qs = (self.black_piece_pos[0][0], 58)
            ks = (self.black_piece_pos[0][0], 62)
            if self.is_legal_castle(qs, enemy_attacked):
                legal_castles.append(qs)
            if self.is_legal_castle(ks, enemy_attacked):
                legal_castles.append(ks)
        return legal_castles

    def is_mate(self, player_color, enemy_color):
        player_attacked, xrays = self.all_squares_attacked(player_color)
        moves = self.generate_legal_moves(enemy_color, player_attacked, xrays)

        no_moves = True
        for movesets in moves:
            if len(movesets) != 0:
                no_moves = False

        if no_moves:
            if player_color == 0 and player_attacked[self.black_piece_pos[0][0]] == 1:
                self.checkmate = 1
                return True
            elif player_color == 1 and player_attacked[self.white_piece_pos[0][0]] == 1:
                self.checkmate = 0
                return True
            elif player_color == 0 and player_attacked[self.black_piece_pos[0][0]] != 1:
                self.checkmate = 2
                return True
            elif player_color == 1 and player_attacked[self.white_piece_pos[0][0]] != 1:
                self.checkmate = 2
                return True
        return False
