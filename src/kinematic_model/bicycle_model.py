# -*- coding: utf-8 -*-

import os
import sys

sys.path.append(os.getcwd())


import casadi as ca
import numpy as np

from src.kinematic_model.base_kinematic_model import BaseKinematicModel
from src.util.config import Config
from src.util.pose import Pose


class BicycleModel(BaseKinematicModel):
    """以后轮为中心的自行车模型"""

    def __init__(self, pose: Pose, v: float = 0.0):
        """使用给定的位姿和速度初始化自行车模型"""
        super().__init__(pose, v)
        self.kinematic_model_name = "BICYCLE_MODEL"
        self.setStateTransitionFunc()

    @staticmethod
    def stateDerivative(x: ca.MX, u: ca.MX) -> ca.MX:
        # [dx/dt, dy/dt, dθ/dt] = v * [cos(θ), sin(θ), tan(δ) / L]，
        # dv/dt = a
        theta, v = x[2], x[3]
        a, delta = u[0], u[1]
        dx = ca.vertcat(
            v * ca.cos(theta),
            v * ca.sin(theta),
            v * ca.tan(delta) / Config.WHEEL_BASE,
            a,
        )
        return dx  # type: ignore

    def setStateTransitionFunc(self) -> None:
        """设置状态更新函数"""
        # [dx/dt, dy/dt, dθ/dt] = v * [cos(θ), sin(θ), tan(δ) / L]，
        # dv/dt = a
        self.state_transition_func = lambda x, u: np.array(
            [
                x[3] * np.cos(x[2]),
                x[3] * np.sin(x[2]),
                x[3] * np.tan(u[1]) / Config.WHEEL_BASE,
                u[0],
            ]
        )
        self.state_transition_func_mx = lambda x, u: ca.vertcat(
            x[3] * ca.cos(x[2]),
            x[3] * ca.sin(x[2]),
            x[3] * ca.tan(u[1]) / Config.WHEEL_BASE,
            u[0],
        )
