import numpy as np


from env import BaseEnv



def get_class(name: str) -> type[BaseEnv]:
    from chess import ChineseChess
    from gomoku import Gomoku
    if name == 'Gomoku':
        return Gomoku
    elif name == 'ChineseChess':
        return ChineseChess
    else:
        raise ValueError('Unknown game class')


def apply_symmetry(nparray, idx, shape):
    """可对state或prob进行对称变换"""
    if shape[0] != shape[1] and idx in (1, 3, 6, 7):
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
    if shape[0] != shape[1] and idx in (1, 3, 6, 7):
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
