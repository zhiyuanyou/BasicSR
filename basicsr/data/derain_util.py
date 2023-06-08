import cv2
import numpy as np
import random


class RainGenerator:
    rain_cfg_dict = {
        "small": {
            "amount": [0.002, 0.004],
            "width": [1, 3],
            "prob": [0.5, 0.5],
            "length": [25, 35],
            "angle": [-10, 10],
        },
        "medium": {
            "amount": [0.005, 0.007],
            "width": [3, 5, 7],
            "prob": [0.25, 0.5, 0.25],
            "length": [35, 45],
            "angle": [-30, 30],
        },
        "large": {
            "amount": [0.008, 0.010],
            "width": [7, 9],
            "prob": [0.5, 0.5],
            "length": [45, 55],
            "angle": [-60, 60],
        },
        "random": {
            "amount": [0.002, 0.010],
            "width": [1, 3, 5, 7, 9],
            "prob": [0.2, 0.2, 0.2, 0.2, 0.2],
            "length": [25, 55],
            "angle": [-60, 60],
        },
    }

    def __init__(self, beta, rain_types=None, rain_cfg=None):
        self.rain_types = rain_types
        self.rain_cfg = rain_cfg
        self.beta = beta

    def __call__(self, img):
        if self.rain_types:
            assert self.rain_cfg is None, "rain_cfg must be null as rain_types are given"
            rain_type = random.choice(self.rain_types)
        else:
            assert self.rain_cfg, "rain_cfg must be given as rain_types are null"
            rain_type = None
        beta = np.random.uniform(self.beta[0], self.beta[1])
        rain, img_rain = self.add_rain(img, beta, rain_type, self.rain_cfg)
        return rain, img_rain

    def get_noise(self, img, amount):
        """
        生成噪声图像
        >>> 输入:
            img: 图像
            amount: 大小控制雨滴的多少
        >>> 输出:
            图像大小的模糊噪声图像
        """

        noise = np.random.uniform(0, 256, img.shape[0:2])
        # 控制噪声水平，取浮点数，只保留最大的一部分作为噪声
        v = 256 - amount * 256  # 百分之多少的像素是雨的中心
        noise[np.where(noise < v)] = 0
        # 噪声做初次模糊
        k = np.array([[0, 0.1, 0], [0.1, 8, 0.1], [0, 0.1, 0]])
        noise = cv2.filter2D(noise, -1, k)
        return noise

    def get_rain_blur(self, noise, length=10, angle=0, width=1):
        """
        将噪声加上运动模糊以模仿雨滴
        >>> 输入:
            noise: 输入噪声图, shape = img.shape[0:2]
            length: 对角矩阵大小, 表示雨滴的长度
            angle: 倾斜的角度, 逆时针为正
            width: 雨滴大小
        >>> 输出:
            带模糊的噪声
        """

        # 由于对角阵自带45度的倾斜，逆时针为正，所以加了-45度的误差，保证开始为正
        trans = cv2.getRotationMatrix2D((length / 2, length / 2), angle - 45, 1)
        dig = np.diag(np.ones(length))  # 生成对焦矩阵
        k = cv2.warpAffine(dig, trans, (length, length))  # 生成模糊核
        k = cv2.GaussianBlur(k, (width, width), 0)  # 高斯模糊这个旋转后的对角核，使得雨有宽度

        # k = k / length  # 是否归一化

        blurred = cv2.filter2D(noise, -1, k)  # 用刚刚得到的旋转后的核，进行滤波

        # 转换到0-255区间
        cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
        blurred = np.array(blurred, dtype=np.uint8)

        rain = np.expand_dims(blurred, 2)
        blurred = np.repeat(rain, 3, 2)

        return blurred

    def add_rain(self, img, beta, rain_type, rain_cfg):
        """
        在一张图像中加入雨
        >>> 输入:
            img: 图像
            rain_type: 雨的类型
            beta: 雨的权重
        >>> 输出:
            rain: 雨图
            img_rain: 带雨的图像
        """
        if rain_type:
            assert rain_cfg is None, "rain_cfg must be null as rain_type is given"
            rain_cfg = self.rain_cfg_dict[rain_type]
        amount = rain_cfg["amount"]
        width = rain_cfg["width"]
        prob = rain_cfg["prob"]
        length = rain_cfg["length"]
        angle = rain_cfg["angle"]

        amount = np.random.uniform(amount[0], amount[1])  # 雨线数目, [0, 1]
        width = np.random.choice(width, p=prob)  # 粗细
        length = np.random.randint(length[0], length[1] + 1)  # 长度
        angle = np.random.randint(angle[0], angle[1] + 1)  # 角度

        noise = self.get_noise(img, amount=amount)
        rain = self.get_rain_blur(noise, length=length, angle=angle, width=width)
        img_rain = cv2.addWeighted(img, 1, rain, beta=beta, gamma=0)

        return rain, img_rain


def set_val_seed(seed):
    """Set random seed for val & test

    keep added rain same for different iterations

    Args:
        seed (int): random seed
    """
    np.random.seed(seed)
    random.seed(seed)
