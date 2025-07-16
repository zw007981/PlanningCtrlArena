# -*- coding: utf-8 -*-

import os
import sys

sys.path.append(os.getcwd())


import math
from typing import List, Tuple

import numpy as np


class BSpline:
    """B样条曲线生成类"""

    K: int = 3
    """B样条曲线的阶数"""
    INTERVAL_DIST: float = 0.1
    """相邻采样点之间的距离"""

    @staticmethod
    def SetK(k: int):
        """
        设置B样条曲线的阶数
        k: 阶数
        """
        if k >= 2:
            BSpline.K = k

    @staticmethod
    def SetIntervalDist(interval_dist: float):
        """
        设置相邻采样点之间的距离
        interval_dist: 采样点之间的距离
        """
        if interval_dist > 0.001:
            BSpline.INTERVAL_DIST = interval_dist

    @staticmethod
    def GenFromCtrlPts(ctrl_pts: np.ndarray) -> np.ndarray:
        """
        根据控制点生成B样条曲线
        ctrl_pts: 控制点
        """
        num_ctrl_pts = ctrl_pts.shape[0]
        total_num_sample_pts = (
            math.ceil(BSpline.__EstimateLen(ctrl_pts) / BSpline.INTERVAL_DIST) + 1
        )
        # 节点向量
        T = (
            [0] * BSpline.K
            + list(range(1, num_ctrl_pts - BSpline.K + 1))
            + [num_ctrl_pts - BSpline.K + 1] * BSpline.K
        )
        sample_pts_list: List[Tuple[float, float]] = []
        if num_ctrl_pts >= BSpline.K:
            num_sample_pts_each_interval = math.ceil(
                total_num_sample_pts / (num_ctrl_pts - BSpline.K + 1)
            )
            for i in range(BSpline.K - 1, num_ctrl_pts):
                for t in np.linspace(T[i], T[i + 1], num_sample_pts_each_interval):
                    try:
                        x = BSpline.__DeBoor(ctrl_pts, T, 0, BSpline.K - 1, t, i)
                        y = BSpline.__DeBoor(ctrl_pts, T, 1, BSpline.K - 1, t, i)
                        sample_pts_list.append((x, y))
                    except ZeroDivisionError:
                        return ctrl_pts
        return np.array(sample_pts_list)

    @staticmethod
    def __DeBoor(
        ctrl_pts: np.ndarray,
        T: List[int],
        coord: int,
        r: int,
        t: int,
        i: int,
    ) -> float:
        """
        递归计算de Boor
        参数:
        coord (int): 坐标索引。
        r (int): 递归深度。
        t (float): 参数值。
        i (int): 控制点索引。
        异常:
        ZeroDivisionError: 当分母为零时抛出。
        """
        if r == 0:
            return ctrl_pts[i][coord]
        else:
            denom = T[i + BSpline.K - r] - T[i]
            if denom == 0:
                raise ZeroDivisionError(
                    f"Division by zero encountered at T[{i + BSpline.K - r}] - T[{i}]"
                )
            alpha = (t - T[i]) / denom
            return (1 - alpha) * BSpline.__DeBoor(
                ctrl_pts, T, coord, r - 1, t, i - 1
            ) + alpha * BSpline.__DeBoor(ctrl_pts, T, coord, r - 1, t, i)

    @staticmethod
    def __EstimateLen(ctrl_pts: np.ndarray) -> float:
        """根据控制点的位置简单估算总长度"""
        return np.sum(np.linalg.norm(ctrl_pts[1:] - ctrl_pts[:-1], axis=1))


def BSplineDemo():
    """BSpline生成示例"""
    from matplotlib import pyplot as plt

    ctrl_pts = np.array([[0, 2], [2, 2], [0, 0], [2, 0], [4, 0], [4, 4], [2, 4]])
    sample_pts = BSpline.GenFromCtrlPts(ctrl_pts)
    plt.figure(figsize=(8, 8))
    for i in range(len(sample_pts) - 1):
        if i == 0:
            plt.plot(
                [sample_pts[i][0], sample_pts[i + 1][0]],
                [sample_pts[i][1], sample_pts[i + 1][1]],
                "r-",
                linewidth=2,
                label="{} order B-spline".format(BSpline.K),
            )
        else:
            plt.plot(
                [sample_pts[i][0], sample_pts[i + 1][0]],
                [sample_pts[i][1], sample_pts[i + 1][1]],
                "r-",
                linewidth=2,
            )
    for i in range(len(ctrl_pts)):
        if i == 0:
            plt.scatter(
                ctrl_pts[i][0], ctrl_pts[i][1], color="b", s=40, label="control points"
            )
        else:
            plt.scatter(ctrl_pts[i][0], ctrl_pts[i][1], color="b", s=40)
    for i in range(len(ctrl_pts) - 1):
        plt.plot(
            [ctrl_pts[i][0], ctrl_pts[i + 1][0]],
            [ctrl_pts[i][1], ctrl_pts[i + 1][1]],
            "b-.",
            linewidth=0.8,
        )
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    BSplineDemo()
