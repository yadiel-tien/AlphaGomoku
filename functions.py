import numpy as np


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


def apply_move(state: np.ndarray, move: tuple[int, int, int]) -> np.ndarray:
    new_state = state.copy()
    row, col, _ = move
    new_state[row, col, 0] = 1
    new_state[:, :, [0, 1]] = new_state[:, :, [1, 0]]
    return new_state


def is_onboard(i, j, h, w) -> bool:
    return 0 <= i < h and 0 <= j < w


def is_win(state: np.ndarray, move: tuple[int, int, int] = None) -> bool:
    if move is None:
        return False
    h0, w0, _ = move
    h, w, _ = state.shape
    for dh, dw in [(1, 0), (1, 1), (0, 1), (-1, 1)]:
        count = 1
        for direction in (-1, 1):
            for step in range(1, 5):
                i, j = h0 + step * dh * direction, w0 + step * dw * direction
                if is_onboard(i, j, h, w) and state[i, j, 1]:
                    count += 1
                    if count > 4:
                        return True
                else:
                    break
    return False


def available_moves(state: np.ndarray) -> list:
    board = state[:, :, 0] + state[:, :, 1]
    return list(map(tuple, np.argwhere(board == 0)))



