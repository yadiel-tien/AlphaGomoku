from PIL import Image
import numpy as np
import os


def find_and_crop_centered_pieces(image_path, output_dir, box_size=130, min_opacity=50, step=10):
    """
    在图像上滑动130x130方框，找到中心非透明且四周距离相等的区域进行裁剪保存

    参数:
        image_path: 输入图片路径(PNG透明背景)
        output_dir: 输出目录
        box_size: 方框大小(默认130x130)
        min_opacity: 中心区域最小不透明度阈值(0-255)
        step: 滑动步长(默认10像素)
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载图像
    img = Image.open(image_path)
    if img.mode != 'RGBA':
        raise ValueError("图片必须包含Alpha通道(RGBA模式)")

    img_array = np.array(img)
    alpha = img_array[:, :, 3]  # Alpha通道

    # 2. 计算滑动范围
    height, width = img.height, img.width
    half_box = box_size // 2
    count = 0
    last_x = 0
    last_y = 0

    # 3. 滑动方框遍历图像
    for y in range(half_box, height - half_box, step):
        for x in range(half_box, width - half_box, step):
            # 3.1 检查中心区域是否非透明
            center_alpha = alpha[y - 10:y + 10, x - 10:x + 10]  # 检查20x20中心区域
            if np.mean(center_alpha) < min_opacity:
                continue

            # 3.2 获取当前方框区域
            box = img_array[y - half_box:y + half_box, x - half_box:x + half_box]

            # 3.3 检查是否有效(防止边缘越界)
            if box.shape[0] != box_size or box.shape[1] != box_size:
                continue

            # 3.4 检查棋子是否居中(计算非透明区域到四边的距离)
            non_transparent = np.where(box[:, :, 3] > 0)
            if len(non_transparent[0]) == 0:
                continue

            min_y, max_y = np.min(non_transparent[0]), np.max(non_transparent[0])
            min_x, max_x = np.min(non_transparent[1]), np.max(non_transparent[1])

            # 计算到四边的距离是否近似相等(允许±5像素误差)
            left_dist = min_x
            right_dist = box_size - max_x - 1
            top_dist = min_y
            bottom_dist = box_size - max_y - 1
            # 不可离边界太近
            if left_dist < 2 or right_dist < 2 or top_dist < 2 or bottom_dist < 2:
                continue
            # 居中
            if (abs(left_dist - right_dist) > 1 or
                    abs(top_dist - bottom_dist) > 1):
                continue
            # 避免重复截图
            if abs(last_x - x) < 20 and abs(last_y - y) < 20:
                continue

            # 4. 保存符合条件的区域
            piece = Image.fromarray(box, 'RGBA')
            piece.save(f"{output_dir}/piece_{count}.png")
            count += 1
            last_x = x
            last_y = y

    print(f"共找到并保存 {count} 个居中棋子")


if __name__ == "__main__":
    find_and_crop_centered_pieces(
        image_path="graphics/chess/pieces.png",
        output_dir="graphics/",
        box_size=150,
        min_opacity=50,  # 中心区域至少100/255不透明度
        step=1  # 增大步长加快处理速度
    )
