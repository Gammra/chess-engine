import json

with open("movesets/king_moves.json", "r") as f:
    KING_MOVES = json.load(f)
with open("movesets/knight_moves.json", "r") as f:
    KNIGHT_MOVES = json.load(f)

"""
PIECE CODES:
0 - white
1 - black
KING - 0
PAWN - 1
KNIGHT - 2
BISHOP - 3
ROOK - 4
QUEEN - 5
e.g. (0, 0, 0) = white king
the number variable differentiates between duplicate pieces
"""


class Piece:
    def __init__(self, color, piece_type):
        self.color = color
        self.piece_type = piece_type


empty = Piece(-1, -1)  # fills blank spaces


class Board:
    def __init__(self):
        self.board = []

    def init_board(self):
        # white pieces
        self.board.append(Piece(0, 4))
        self.board.append(Piece(0, 2))
        self.board.append(Piece(0, 3))
        self.board.append(Piece(0, 5))
        self.board.append(Piece(0, 0))
        self.board.append(Piece(0, 3))
        self.board.append(Piece(0, 2))
        self.board.append(Piece(0, 4))

        # white pawns
        for i in range(8, 16, 1):
            self.board.append(Piece(0, 1))

        # empty spaces
        for i in range(16, 48, 1):
            self.board.append(empty)

        # black pawns
        for i in range(48, 56, 1):
            self.board.append(Piece(1, 1))

        # black pieces
        self.board.append(Piece(1, 4))
        self.board.append(Piece(1, 2))
        self.board.append(Piece(1, 3))
        self.board.append(Piece(1, 5))
        self.board.append(Piece(1, 0))
        self.board.append(Piece(1, 3))
        self.board.append(Piece(1, 2))
        self.board.append(Piece(1, 4))

    def init_blank(self):
        self.board = [empty]*64

    # returns a list of [N,E,S,W] offsets
    def generate_offsets(self, pos):
        offsets = [7 - (pos // 8), 7 - (pos % 8), (pos // 8), pos % 8]
        return offsets

    def generate_king_moves(self, pos):
        return KING_MOVES[pos]

    def generate_pawn_moves(self, pos, en_passant):
        offsets = self.generate_offsets(pos)
        piece_color = self.board[pos].color
        pawn_moves = []

        if piece_color == 0:
            enemy_color = 1
            # move 2
            if offsets[2] == 1:
                if self.board[pos+8].color == -1 and self.board[pos+16].color == -1:
                    pawn_moves.append(pos+16)
            # move 1
            if self.board[pos+8].color == -1:
                pawn_moves.append(pos+8)
            # take up right
            if offsets[1] != 0 and self.board[pos+9].color == enemy_color:
                pawn_moves.append(pos+9)
            # take up left
            if offsets[3] != 0 and self.board[pos+7].color == enemy_color:
                pawn_moves.append(pos+7)
            # en passant
            if en_passant[enemy_color] != -1:
                if en_passant[enemy_color]+1 == pos and offsets[3] != 0:
                    pawn_moves.append(en_passant[enemy_color] + 8)
                if en_passant[enemy_color]-1 == pos and offsets[1] != 0:
                    pawn_moves.append(en_passant[enemy_color] + 8)
        else:
            enemy_color = 0
            # move 2
            if offsets[0] == 1:
                if self.board[pos-8].color == -1 and self.board[pos-16].color == -1:
                    pawn_moves.append(pos-16)
            # move 1
            if self.board[pos-8].color == -1:
                pawn_moves.append(pos-8)
            # take down right
            if offsets[1] != 0 and self.board[pos-7].color == enemy_color:
                pawn_moves.append(pos-7)
            # take down left
            if offsets[3] != 0 and self.board[pos-9].color == enemy_color:
                pawn_moves.append(pos-9)
            # en passant
            if en_passant[enemy_color] != -1:
                if en_passant[enemy_color]+1 == pos and offsets[3] != 0:
                    pawn_moves.append(en_passant[enemy_color] - 8)
                if en_passant[enemy_color]-1 == pos and offsets[1] != 0:
                    pawn_moves.append(en_passant[enemy_color] - 8)
        return pawn_moves

    def generate_knight_moves(self, pos):
        return KNIGHT_MOVES[pos]

    def generate_bishop_moves(self, pos, xray_on, xrays):
        offsets = self.generate_offsets(pos)
        color = self.board[pos].color
        bishop_moves = []
        xray_found = 0

        # up right
        for i in range(1, min(offsets[0], offsets[1]) + 1, 1):
            bishop_moves.append(pos + (9*i))
            if self.board[pos + (9*i)].color != -1:
                if xray_on:
                    xray_found = self.check_xray(9, min(offsets[0], offsets[1]) - i, pos + (9*i), color, xrays)
                break
        # down right
        for i in range(1, min(offsets[1], offsets[2]) + 1, 1):
            bishop_moves.append(pos - (7*i))
            if self.board[pos - (7*i)].color != -1:
                if xray_on and not xray_found:
                    xray_found = self.check_xray(-7, min(offsets[1], offsets[2]) - i, pos - (7*i), color, xrays)
                break
        # down left
        for i in range(1, min(offsets[2], offsets[3]) + 1, 1):
            bishop_moves.append(pos - (9*i))
            if self.board[pos - (9*i)].color != -1:
                if xray_on and not xray_found:
                    xray_found = self.check_xray(-9, min(offsets[2], offsets[3]) - i, pos - (9*i), color, xrays)
                break
        # up left
        for i in range(1, min(offsets[3], offsets[0]) + 1, 1):
            bishop_moves.append(pos + (7*i))
            if self.board[pos + (7*i)].color != -1:
                if xray_on and not xray_found:
                    self.check_xray(7, min(offsets[3], offsets[0]) - i, pos + (7*i), color, xrays)
                break
        return bishop_moves

    def generate_rook_moves(self, pos, xray_on, xrays):
        offsets = self.generate_offsets(pos)
        color = self.board[pos].color
        rook_moves = []
        xray_found = 0

        # up
        for i in range(1, offsets[0] + 1, 1):
            rook_moves.append(pos + (8 * i))
            if self.board[pos + (8 * i)].color != -1:
                if xray_on:
                    xray_found = self.check_xray(8, offsets[0] - i, pos + (8 * i), color, xrays)
                break
        # right
        for i in range(1, offsets[1] + 1, 1):
            rook_moves.append(pos + (1 * i))
            if self.board[pos + (1 * i)].color != -1:
                if xray_on and not xray_found:
                    xray_found = self.check_xray(1, offsets[1] - i, pos + (1 * i), color, xrays)
                break
        # down
        for i in range(1, offsets[2] + 1, 1):
            rook_moves.append(pos - (8 * i))
            if self.board[pos - (8 * i)].color != -1:
                if xray_on and not xray_found:
                    xray_found = self.check_xray(-8, offsets[2] - i, pos - (8 * i), color, xrays)
                break
        # left
        for i in range(1, offsets[3] + 1, 1):
            rook_moves.append(pos - (1 * i))
            if self.board[pos - (1 * i)].color != -1:
                if xray_on and not xray_found:
                    xray_found = self.check_xray(-1, offsets[3] - i, pos - (1 * i), color, xrays)
                break
        return rook_moves

    def generate_queen_moves(self, pos, xray_on, xrays):
        queen_moves = self.generate_bishop_moves(pos, xray_on, xrays) + self.generate_rook_moves(pos, xray_on, xrays)
        return queen_moves

    def check_xray(self, vector, offset, pos, color, xrays):
        is_xray = 0
        piece_xrayed = pos
        for i in range(0, offset+1, 1):
            # only look for xrays through opponents pieces
            if self.board[pos].color == color:
                return False
            # king was found so return if it is being xrayed, and add the piece xrayed and through what vector
            elif self.board[pos].piece_type == 0 and self.board[pos].color != color and is_xray:
                xrays.append([piece_xrayed, vector])
                return True
            # opponents piece was found. if it is the first to and only to be found, then there is a xray
            elif self.board[pos].color != color and self.board[pos].color != -1:
                if is_xray == 0:
                    is_xray = 1
                else:
                    return False
            pos = pos + vector
        return False

    def generate_pawn_attacks(self, pos, en_passant):
        offsets = self.generate_offsets(pos)
        piece_color = self.board[pos].color
        pawn_moves = []
        enemy_color = 1

        if piece_color == 0:
            # take up right
            if offsets[1] != 0:
                pawn_moves.append(pos+9)
            # take up left
            if offsets[3] != 0:
                pawn_moves.append(pos+7)
            # en passant
            if en_passant[enemy_color] != -1:
                if en_passant[enemy_color] + 1 == pos and offsets[1] != 0:
                    pawn_moves.append(en_passant[enemy_color] + 8)
                if en_passant[enemy_color] - 1 == pos and offsets[3] != 0:
                    pawn_moves.append(en_passant[enemy_color] + 8)
        else:
            enemy_color = 0
            # take down right
            if offsets[1] != 0:
                pawn_moves.append(pos-7)
            # take down left
            if offsets[3] != 0:
                pawn_moves.append(pos-9)
            # en passant
            if en_passant[enemy_color] != -1:
                if en_passant[enemy_color] + 1 == pos and offsets[1] != 0:
                    pawn_moves.append(en_passant[enemy_color] - 8)
                if en_passant[enemy_color] - 1 == pos and offsets[3] != 0:
                    pawn_moves.append(en_passant[enemy_color] - 8)
        return pawn_moves
