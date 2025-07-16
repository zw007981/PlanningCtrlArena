# -*- coding: utf-8 -*-

import os
import sys

sys.path.append(os.getcwd())

import numpy as np

from src.util.config import Config
from src.util.curvature import Curvature


def calRelError(x0, x1):
    """计算相对误差"""
    if abs(x0) < Config.ZERO_EPS:
        return abs(x1 - x0)
    else:
        return abs((x1 - x0) / x0)


def test_line_curvature_calculation():
    """
    测试一次函数的曲率计算是否准确
    对于y = ax + b，其曲率为 0
    """
    x_list = np.linspace(-4, 4, 100)
    y_list = 2 * x_list + 1
    curvatures = Curvature.CalAlongCurve(list(x_list), list(y_list))
    for curvature in curvatures:
        assert abs(curvature) < Config.ZERO_EPS


def test_parabola_curvature_calculation():
    """
    测试二次函数的曲率计算是否准确：
    对于y = ax^2 + bx + c，其曲率为 2a / (1 + (2ax + b)^2)^(3/2)
    """
    x_list = np.linspace(-4, 4, 100)
    y_list = np.empty_like(x_list)
    for i, x in enumerate(x_list):
        y_list[i] = x**2 + 2 * x + 1
    curvatures = Curvature.CalAlongCurve(list(x_list), list(y_list))
    for i, curvature in enumerate(curvatures):
        if i == 0 or i == len(curvatures) - 1:
            continue
        x = x_list[i]
        real_curvature = 2 / (1 + (2 * x + 2) ** 2) ** 1.5
        assert calRelError(real_curvature, curvature) < 0.2


def test_sin_curvature_calculation():
    """
    测试正弦函数的曲率计算是否正确
    对于y = sin(x)，其曲率为 -sin(x) / (1 + cos(x)^2)^(3/2)
    """
    x_list = np.linspace(-np.pi, np.pi, 200)
    y_list = np.sin(x_list)
    curvatures = Curvature.CalAlongCurve(list(x_list), list(y_list))
    for i, curvature in enumerate(curvatures):
        if i == 0 or i == len(curvatures) - 1:
            continue
        x = x_list[i]
        real_curvature = -np.sin(x) / (1 + np.cos(x) ** 2) ** 1.5
        assert calRelError(real_curvature, curvature) < 0.2
