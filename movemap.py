import json


def generate_offsets(pos):
    offsets = [7 - (pos // 8), 7 - (pos % 8), (pos // 8), pos % 8]
    return offsets


def king_moves(pos):
    offsets = generate_offsets(pos)
    km = []

    # up
    if offsets[0] != 0:
        km.append(pos + 8)
    # up right
    if offsets[0] != 0 and offsets[1] != 0:
        km.append(pos + 9)
    # right
    if offsets[1] != 0:
        km.append(pos + 1)
    # down right
    if offsets[1] != 0 and offsets[2] != 0:
        km.append(pos - 7)
    # down
    if offsets[2] != 0:
        km.append(pos - 8)
    # down left
    if offsets[2] != 0 and offsets[3] != 0:
        km.append(pos - 9)
    # left
    if offsets[3] != 0:
        km.append(pos - 1)
    # up left
    if offsets[0] != 0 and offsets[3] != 0:
        km.append(pos + 7)

    return km


def knight_moves(pos):
    offsets = generate_offsets(pos)
    knm = []

    # up left
    if offsets[0] > 1 and offsets[3] > 0:
        knm.append(pos + 15)
    # up right
    if offsets[0] > 1 and offsets[1] > 0:
        knm.append(pos + 17)
    # right up
    if offsets[1] > 1 and offsets[0] > 0:
        knm.append(pos + 10)
    # right down
    if offsets[1] > 1 and offsets[2] > 0:
        knm.append(pos - 6)
    # down right
    if offsets[2] > 1 and offsets[1] > 0:
        knm.append(pos - 15)
    # down left
    if offsets[2] > 1 and offsets[3] > 0:
        knm.append(pos - 17)
    # left down
    if offsets[3] > 1 and offsets[2] > 0:
        knm.append(pos - 10)
    # left up
    if offsets[3] > 1 and offsets[0] > 0:
        knm.append(pos + 6)

    return knm


def generate_king_moveset():
    moves = []
    for i in range(0, 64):
        curr = king_moves(i)
        moves.append(curr)
    with open("movesets/king_moves.json", "w") as f:
        json.dump(moves, f)


def generate_knight_moveset():
    moves = []
    for i in range(0, 64):
        curr = knight_moves(i)
        moves.append(curr)
    with open("movesets/knight_moves.json", "w") as f:
        json.dump(moves, f)


def dump_all():
    generate_king_moveset()
    generate_knight_moveset()
