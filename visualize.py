import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_policy_heatmap(pi: np.ndarray, shape=(9, 9), title='Policy Heatmap', save_path=None):
    """
    可视化 policy 分布的热力图。 save_path: 如果设置路径，则保存到该路径
    """
    H, W = shape
    assert pi.shape[0] == H * W, f"pi 应该是一维长度为 {H * W}，但现在是 {pi.shape}"

    policy_matrix = pi.reshape(H, W)

    plt.figure(figsize=(6, 6))
    sns.heatmap(policy_matrix, cmap='YlOrRd', linewidths=0.5, square=True, cbar=True, annot=False)

    plt.title(title)
    plt.xlabel('Column')
    plt.ylabel('Row')
    plt.gca().invert_yaxis()  # 让(0,0) 在左上角像棋盘
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"热力图保存到 {save_path}")
    else:
        plt.show()
