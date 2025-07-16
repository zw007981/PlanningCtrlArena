# -*- coding: utf-8 -*-

import os
import sys

sys.path.append(os.getcwd())

import math
from math import factorial as fac

import numpy as np


class BezierCurve:
    @staticmethod
    def GenFromCtrlPoints(ctrl_pts: np.ndarray, sample_dist: float = 0.1) -> np.ndarray:
        """
        基于控制点生成贝塞尔曲线
        ctrl_pts: 控制点
        sample_dist: 生成的曲线上相邻采样点的距离
        """
        num = math.ceil(BezierCurve.__estimateLength(ctrl_pts) / sample_dist)
        # 贝塞尔曲线的阶数
        k = len(ctrl_pts) - 1
        # 如果控制点少于2个，直接返回控制点
        if k <= 1:
            return ctrl_pts
        t = np.linspace(0, 1, num)
        curve = np.zeros((num, 2))
        for i in range(k + 1):
            curve += np.outer(
                (1 - t) ** (k - i) * t**i * BezierCurve.__comb(k, i), ctrl_pts[i]
            )
        return curve

    @staticmethod
    def __comb(n: int, k: int) -> int:
        """
        计算组合数
        n: 总数
        k: 选取的数
        """
        return fac(n) // (fac(k) * fac(n - k))

    @staticmethod
    def __estimateLength(ctrl_pts: np.ndarray) -> float:
        """
        估算贝塞尔曲线长度，后续用于计算采样点的个数
        ctrl_pts: 控制点
        """
        length = 0.0
        for i in range(len(ctrl_pts) - 1):
            length += np.linalg.norm(ctrl_pts[i + 1] - ctrl_pts[i])
        return length  # type: ignore


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    ctrl_pts = np.array([[0, 2], [2, 2], [0, 0], [2, 0], [4, 0], [4, 2]])
    curve = BezierCurve.GenFromCtrlPoints(ctrl_pts)
    plt.plot(ctrl_pts[:, 0], ctrl_pts[:, 1], "ro-", label="Control Points")
    plt.plot(curve[:, 0], curve[:, 1], "b-", label="Bezier Curve")
    plt.legend()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Bezier Curve")
    plt.grid()
    plt.axis("equal")
    plt.show()
