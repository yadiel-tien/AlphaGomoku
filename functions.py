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
                if 0 <= i < h and 0 <= j < w and state[i, j, 1]:
                    count += 1
                    if count == 5:
                        return True
                else:
                    break
    return False


def apply_symmetry(nparray, idx, shape):
    """可对state或prob进行对称变换"""
    if shape[0] != shape[1]:
        raise ValueError("Symmetry functions require a square board (rows == cols).")

    result = None
    if nparray.ndim == 1:  # prob
        reshaped = nparray.reshape(shape)
        if idx < 4:  # 旋转 0°, 90°, 180°, 270°
            result = np.rot90(reshaped, k=idx, axes=(0, 1)).flatten()
        elif idx == 4:  # 水平翻转
            result = np.flip(reshaped, axis=1).flatten()
        elif idx == 5:  # 垂直翻转
            result = np.flip(reshaped, axis=0).flatten()
        elif idx == 6:  # 主对角线翻转
            result = np.transpose(reshaped, axes=(1, 0)).flatten()
        elif idx == 7:  # 副对角线翻转
            result = np.transpose(reshaped, axes=(1, 0)).flatten()
            result = np.flip(result.reshape(shape), axis=0).flatten()
    elif nparray.ndim == 3:  # state
        if idx < 4:  # 旋转 0°, 90°, 180°, 270°
            result = np.rot90(nparray, k=idx, axes=(0, 1))
        elif idx == 4:  # 水平翻转
            result = np.flip(nparray, axis=1)
        elif idx == 5:  # 垂直翻转
            result = np.flip(nparray, axis=0)
        elif idx == 6:  # 主对角线翻转
            result = np.transpose(nparray, axes=(1, 0, 2))
        elif idx == 7:  # 副对角线翻转
            result = np.transpose(nparray, axes=(1, 0, 2))
            result = np.flip(result, axis=0)
    if result is None:
        raise ValueError(f'Invalid symmetry index: {idx}')
    return result


def reverse_symmetry_prob(nparray, idx, shape):
    """反转概率分布"""
    if shape[0] != shape[1]:
        raise ValueError("Symmetry functions require a square board (rows == cols).")

    result = None
    reshaped = nparray.reshape(shape)
    if idx < 4:  # 逆向旋转
        result = np.rot90(reshaped, k=-idx, axes=(0, 1)).flatten()
    elif idx == 4:  # 水平翻转
        result = np.flip(reshaped, axis=1).flatten()
    elif idx == 5:  # 垂直翻转
        result = np.flip(reshaped, axis=0).flatten()
    elif idx == 6:  # 主对角线翻转
        result = np.transpose(reshaped, axes=(1, 0)).flatten()
    elif idx == 7:  # 副对角线翻转
        result = np.flip(reshaped, axis=0).flatten()
        result = np.transpose(result.reshape(shape), axes=(1, 0)).flatten()

    if result is None:
        raise ValueError(f'Invalid symmetry index: {idx}')

    return result
