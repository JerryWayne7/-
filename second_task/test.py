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
    c_values = [30, 38, 47, 56, 65, 74, 83, 92, 101,110]

    # 分段线性变换函数
    def log_transform(img, a, b, c):
        output = np.zeros_like(img, dtype=np.float32)

        # 对于 [0, a) 区间，进行线性变换
        output[img < a] = a

        # 对于 [a, b) 区间，进行对数变换
        output[(img >= a) & (img < b)] = np.log10(1 + img[(img >= a) & (img < b)]) * c

        # 对于 [b, M] 区间，进行线性变换
        output[img >= b] = M

        return np.clip(output, 0, 255).astype(np.uint8)

    # 显示图像
    plt.figure(figsize=(20, 10))
    # 进行非线性变换
    for i, c in enumerate(c_values):
        plt.subplot(1, 3, i + 1)
        transformed_img = log_transform(img_gray, a, b, c)
        plt.imshow(transformed_img, cmap='gray')
        plt.title(f'c = {c}')

    plt.tight_layout()  # Adjusts subplot params for better spacing
    plt.show()

    # 绘制函数图像
import numpy as np
import matplotlib.pyplot as plt
a,b,M = 50,200,255
c_values = [30, 70, 110]
x = np.arange(0, 256)
def y(x, c):
    return np.piecewise(x, [x < a, (x >= a) & (x < b), x >= b], [a, lambda x: np.log10(1 + x) * c, M])

for c in c_values:
    plt.plot(x, y(x, c), linewidth=2, label=f'y = f(x), c = {c}', alpha=0.6)

plt.title('非线性变换函数')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.legend()
plt.show()