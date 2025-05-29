import numpy as np


def apply_action(state: np.ndarray, action: int) -> np.ndarray:
    """执行落子，返回new state。用于模拟"""
    new_state = state.copy()
    row, col = divmod(action, state.shape[1])
    new_state[row, col, 0] = 1
    new_state[:, :, [0, 1]] = new_state[:, :, [1, 0]]
    return new_state


def is_onboard(i, j, h, w) -> bool:
    return 0 <= i < h and 0 <= j < w


def is_win(state: np.ndarray, action) -> bool:
    """检查落子后是否已经获胜，state是执行落子后的，检查[:,:,1]平面"""
    if action is None:
        return False
    h, w, _ = state.shape
    h0, w0 = divmod(action, w)
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
