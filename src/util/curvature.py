# -*- coding: utf-8 -*-

import os
import sys

sys.path.append(os.getcwd())

from typing import List

import numpy as np
from scipy import interpolate

from src.util.config import Config


class Curvature:
    """曲率计算"""

    @staticmethod
    def Cal(pd: float, pdd: float) -> float:
        """根据一阶导数和二阶导数计算曲率"""
        return pdd / ((1 + pd**2) ** 1.5)

    @staticmethod
    def CalAlongCurve(
        x_list: List[float], y_list: List[float], with_interp=False
    ) -> np.ndarray:
        """计算曲线上各点的曲率，默认不使用插值的方法计算"""
        if with_interp:
            return Curvature.__CalWithInterpolation(x_list, y_list)
        else:
            return Curvature.__CalWithoutInterpolation(x_list, y_list)

    @staticmethod
    def __CalWithInterpolation(x_list: List[float], y_list: List[float]) -> np.ndarray:
        """先基于scipy.interpolate对曲线进行插值，然后计算曲率"""
        x = np.array(x_list)
        y = np.array(y_list)
        spline_interp = interpolate.interp1d(x, y, kind="cubic")
        # 在更细的网格上评估插值函数
        x_fine = np.linspace(x.min(), x.max(), len(x) * 4)
        y_fine = spline_interp(x_fine)
        # 计算一阶和二阶导数
        dx = x_fine[1] - x_fine[0]
        pd = np.gradient(y_fine, dx)
        pdd = np.gradient(pd, dx)
        # 计算曲率
        curvature = pdd / ((1 + pd**2) ** 1.5)
        # 只保留原始点对应的曲率值（通过插值）
        return np.interp(x, x_fine, curvature)

    @staticmethod
    def __CalWithoutInterpolation(
        x_list: List[float], y_list: List[float]
    ) -> np.ndarray:
        """不插值基于差分法计算曲率"""
        curvatures = np.zeros(len(x_list))
        diff_x = np.diff(x_list)
        # 避免除0
        diff_x = np.where(np.abs(diff_x) < Config.ZERO_EPS, Config.ZERO_EPS, diff_x)
        diff_y = np.diff(y_list)
        for i in range(1, len(x_list) - 1):
            pd = diff_y[i] / diff_x[i]
            pdd = (diff_y[i] - diff_y[i - 1]) / (
                (0.5 * (diff_x[i] + diff_x[i - 1])) ** 2 + Config.ZERO_EPS
            )
            curvatures[i] = Curvature.Cal(pd, pdd)
        # 补上首尾点
        curvatures[0] = curvatures[1]
        curvatures[-1] = curvatures[-2]
        return curvatures
