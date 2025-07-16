# -*- coding: utf-8 -*-

import os
import sys

sys.path.append(os.getcwd())


from typing import Callable

import casadi as ca
import numpy as np

from src.kinematic_model.base_kinematic_model import BaseKinematicModel
from src.util.config import Config
from src.util.pose import Pose


class PointMass(BaseKinematicModel):
    """
    质点模型，假设可以无视约束在一个时间步长内移动到任何位置，
    同时也不受最大速度，最大加速度等硬件条件的限制。
    """

    def __init__(self, pose: Pose, v: float = 0.0):
        """使用给定的位姿和速度初始化质点模型"""
        super().__init__(pose, v)
        self.kinematic_model_name = "POINT_MASS"
        self.setStateTransitionFunc()

    def setStateTransitionFunc(self) -> None:
        self.state_transition_func = self.__defineStateTransitionFunc()
        self.state_transition_func_mx = self.__defineStateTransitionFuncMx()

    def __defineStateTransitionFunc(self) -> Callable:
        # 假设质点匀加速运动，控制量以x方向上的移动距离和y方向上的移动距离的形式输入。
        # v_x = -v * cos(θ) + 2 * u_x / DELTA_T
        # v_y = -v * sin(θ) + 2 * u_y / DELTA_T
        return lambda x, u: np.array(
            [
                u[0] / Config.DELTA_T,
                u[1] / Config.DELTA_T,
                (
                    np.arctan2(
                        -self.v * np.sin(self.theta) + 2 * u[1] / Config.DELTA_T,
                        -self.v * np.cos(self.theta) + 2 * u[0] / Config.DELTA_T,
                    )
                    - self.theta
                )
                / Config.DELTA_T,  # 计算航向角变化率
                (
                    np.sqrt(
                        (-self.v * np.cos(self.theta) + 2 * u[0] / Config.DELTA_T) ** 2
                        + (-self.v * np.sin(self.theta) + 2 * u[1] / Config.DELTA_T)
                        ** 2
                    )
                    - self.v
                )
                / Config.DELTA_T,  # 计算速度变化率
            ]
        )

    def __defineStateTransitionFuncMx(self) -> Callable:
        # 假设质点匀加速运动，控制量以x方向上的移动距离和y方向上的移动距离的形式输入。
        return lambda x, u: ca.vertcat(
            u[0] / Config.DELTA_T,
            u[1] / Config.DELTA_T,
            (
                ca.arctan2(
                    -self.v * ca.sin(self.theta) + 2 * u[1] / Config.DELTA_T,
                    -self.v * ca.cos(self.theta) + 2 * u[0] / Config.DELTA_T,
                )
                - self.theta
            )
            / Config.DELTA_T,  # 计算航向角变化率
            (
                ca.sqrt(
                    (-self.v * ca.cos(self.theta) + 2 * u[0] / Config.DELTA_T) ** 2
                    + (-self.v * ca.sin(self.theta) + 2 * u[1] / Config.DELTA_T) ** 2
                )
                - self.v
            )
            / Config.DELTA_T,  # 计算速度变化率
        )
