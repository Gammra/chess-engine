from training_data import split_dims
from tensorflow import keras
import numpy as np



class Engine:
    def __init__(self, model_name="regression_model.h5"):
        self.model = keras.models.load_model(model_name)
        self.move_num = 0
        self.cpu_color = 1

    def is_promo(self, color, move):
        if color == 0:
            if move[1] // 8 == 7:
                return True
        else:
            if move[1] // 8 == 0:
                return True
        return False

    def build_model(self, conv_size, conv_depth):
        board3d = keras.layers.Input(shape=(14, 8, 8))
        x = keras.layers.Conv2D(filters=conv_size, kernel_size=3, padding='same')(board3d)
        for _ in range(conv_depth):
            previous = x
            x = keras.layers.Conv2D(filters=conv_size, kernel_size=3, padding='same')(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Activation('relu')(x)
            x = keras.layers.Conv2D(filters=conv_size, kernel_size=3, padding='same')(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Add()([x, previous])
            x = keras.layers.Activation('relu')(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(1, 'linear')(x)
        return keras.models.Model(inputs=board3d, outputs=x)

    def train(self, train_pos, train_eval, model_name, num_epochs):
        self.model.compile(optimizer=keras.optimizers.Adam(1e-5), loss='mean_squared_error')
        self.model.summary()
        self.model.fit(train_pos, train_eval, batch_size=512, validation_split=0.1, epochs=num_epochs)
        self.model.save(model_name)

    def predict(self, game):
        w_attacked, dummy = game.all_squares_attacked(0)
        b_attacked, dummy = game.all_squares_attacked(1)
        board3d = split_dims(game.board, w_attacked, b_attacked)
        board3d = np.expand_dims(board3d, 0)
        return self.model(board3d)

    def heuristic_elim(self, game, move, color, enemy_attacked):
        piece_type = game.board.board[move[0]].piece_type
        if piece_type == 0:
            # don't move king off back rank in the beginning
            if self.move_num < 6 and enemy_attacked[move[0]] != 1 and 56 > move[1] > 7:
                return True


        return False

    # Utilize search to find the move with the highest evaluation in the position; make that move.
    def make_CPU_move(self, color, game, depth):
        if color == 0:
            enemy_is_maxing = False
        else:
            enemy_is_maxing = True


        enemy_color = 1-color
        enemy_attacked, xray = game.all_squares_attacked(1-color)
        moves = game.generate_legal_moves(color, enemy_attacked, xray)

        best_move = ()
        if color == 0:
            best_eval = -np.Infinity
        else:
            best_eval = np.Infinity

        promo_num = 0
        # if the move is a promotion, need to know which piece is best to promote to
        best_promo = -1
        nodes_searched = 0
        for moveset in moves:
            for move in moveset:
                # eliminate faulty moves
                if self.heuristic_elim(game, move, color, enemy_attacked):
                    continue

                initial_eval = self.predict(game)

                save_state = game.make_move(move, save=True, promo_code=promo_num % 4)
                if color == 0:
                    n_nodes = 1
                    curr_eval, n_nodes = self.search(game, depth-1, best_eval, np.Infinity, enemy_is_maxing, enemy_color, n_nodes, initial_eval)
                else:
                    n_nodes = 1
                    curr_eval, n_nodes = self.search(game, depth-1, -np.Infinity, best_eval, enemy_is_maxing, enemy_color, n_nodes, initial_eval)

                if color == 0 and curr_eval > best_eval:
                        best_eval = curr_eval
                        best_move = move
                        best_promo = promo_num
                elif curr_eval < best_eval:
                        best_eval = curr_eval
                        best_move = move
                        best_promo = promo_num

                game.undo_move(save_state)

                if game.board.board[move[0]].piece_type == 1 and self.is_promo(color, move):
                    promo_num += 1

                nodes_searched += n_nodes

        print(f"nodes seached = {nodes_searched}")
        game.make_move(best_move, save=False, promo_code=best_promo % 4)
        self.move_num += 1
        return best_move

    # performs a minimax search with alpha-beta pruning and returns the best possible evaluation from the position
    def search(self, game, depth, alpha, beta, maxing_player, color, n_nodes, init_eval):
        if depth == 0:
            pred = self.predict(game)

            # if the move has improved the evaluation, continue down the tree
            if self.cpu_color == 0 and pred-init_eval >= 0.1:
                return self.search(game, depth+3, alpha, beta, maxing_player, color, n_nodes, init_eval)
            elif self.cpu_color == 0 and init_eval-pred >= 0.1:
                return self.search(game, depth+3, alpha, beta, maxing_player, color, n_nodes, init_eval)

            # else return the prediction
            return pred, (n_nodes + 1)

        promo_num = 0
        enemy_attacked, xray = game.all_squares_attacked(1-color)
        moves = game.generate_legal_moves(color, enemy_attacked, xray)
        if maxing_player:
            max_eval = -np.Infinity
            for piece_type in moves:
                for move in piece_type:
                    save_state = game.make_move(move, save=True, promo_code=promo_num % 4)
                    evaluation, n_nodes = self.search(game, depth-1, alpha, beta, False, 1, n_nodes, init_eval)
                    n_nodes += 1
                    game.undo_move(save_state)
                    max_eval = max(evaluation, max_eval)
                    alpha = max(alpha, evaluation)
                    if beta <= alpha:
                        break

                    if game.board.board[move[0]].piece_type == 1 and self.is_promo(color, move):
                        promo_num += 1
            return max_eval, n_nodes
        elif not maxing_player:
            min_eval = np.Infinity
            for piece_type in moves:
                for move in piece_type:
                    save_state = game.make_move(move, save=True, promo_code=promo_num % 4)
                    evaluation, n_nodes = self.search(game, depth-1, alpha, beta, True, 0, n_nodes, init_eval)
                    n_nodes += 1
                    game.undo_move(save_state)
                    min_eval = min(min_eval, evaluation)
                    beta = min(beta, evaluation)
                    if beta <= alpha:
                        break

                    if game.board.board[move[0]].piece_type == 1 and self.is_promo(color, move):
                        promo_num += 1
            return min_eval, n_nodes
