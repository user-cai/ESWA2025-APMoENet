import cv2
from matplotlib import pyplot as plt

# 细节比Canny更多，但不明显
def Sobel_demo(srcImg_path):
    img = cv2.imread(srcImg_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Sobel 算子
    x = cv2.Sobel(grayImage, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(grayImage, cv2.CV_16S, 0, 1)

    # 转换为 uint8
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    Sobel = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

    # 保存 Sobel 处理结果
    cv2.imwrite('sobel_edges.png', Sobel)

    # 显示图像
    titles = ["Original Image", "Sobel Edge Image"]
    images = [img, Sobel]
    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.imshow(images[i], "gray")
        plt.title(titles[i])
        plt.axis('off')
    plt.show()


# 处理图像
path = '../data/LOLv1/Infra/Test_00/111.png'
Sobel_demo(path)
