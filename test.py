import cv2
import numpy as np
import matplotlib.pyplot as plt

# 手动设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 导入灰度图像
img_path = input("input your img path:")
img = cv2.imread(img_path)

# Check if the image was loaded successfully
if img is None:
    print(f"Error: Unable to load image at {img_path}")
else:
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 灰度区间划分：定义分段区间
    a, b = 50, 200  # 假设划分为 [0, a), [a, b), [b, 255]
    M = 255
    s = M / (b - a)
    # 分段线性变换函数
    def piecewise_linear_transform(img, a, b):
        output = np.zeros_like(img, dtype=np.float32)

        # 对于 [0, a) 区间，进行线性变换
        output[img < a] = a

        # 对于 [a, b) 区间，进行线性变换
        output[(img >= a) & (img < b)] = img[(img >= a) & (img < b)] * s

        # 对于 [b, M] 区间，进行线性变换
        output[img >= b] = M

        return np.clip(output, 0, 255).astype(np.uint8)

    # 进行线性变换
    linear_img = piecewise_linear_transform(img_gray, a, b)

    # 显示原始图像和线性变换后的图像
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('原始图')

    plt.subplot(1, 3, 2)
    plt.imshow(img_gray, cmap='gray')
    plt.title('灰度图')

    plt.subplot(1, 3, 3)
    plt.imshow(linear_img, cmap='gray')
    plt.title('线性变换后的图像')

    plt.show()
