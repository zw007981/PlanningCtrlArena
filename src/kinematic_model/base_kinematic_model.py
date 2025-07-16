# -*- coding: utf-8 -*-

import os
import sys

sys.path.append(os.getcwd())

from abc import ABC, abstractmethod
from typing import Callable, Tuple

import casadi as ca
import numpy as np
from numpy.typing import NDArray

from src.kinematic_model.ctrl_input import CtrlInput
from src.util.config import Config
from src.util.log import Log
from src.util.pose import Pose


class BaseKinematicModel(ABC):
    """运动学模型基类"""

    def __init__(self, pose: Pose, v: float = 0.0):
        """使用给定的位姿和速度初始化运动学模型"""
        self.kinematic_model_name = "BASE"
        """运动学模型名称"""
        self.state: NDArray[np.float64] = np.array([pose.x, pose.y, pose.theta, v])
        """车辆状态"""
        self.state_transition_func: Callable
        """状态更新函数，输入输出均用numpy数组表示"""
        self.state_transition_func_mx: Callable
        """状态更新函数，输入输出均用casadi.MX表示"""

    @abstractmethod
    def setStateTransitionFunc(self) -> None:
        """设置状态更新函数"""
        pass

    @staticmethod
    def constructKinematicModel(
        kinematic_model_type: str, pose: Pose = Pose(0.0, 0.0, 0.0), v: float = 0.0
    ) -> "BaseKinematicModel":
        """输入运动学模型类型和初始化参数，返回对应的运动学模型对象"""
        from src.kinematic_model.bicycle_model import BicycleModel
        from src.kinematic_model.point_mass import PointMass

        if kinematic_model_type == "BICYCLE_MODEL":
            return BicycleModel(pose, v)
        elif kinematic_model_type == "POINT_MASS":
            return PointMass(pose, v)
        else:
            Log.error(
                "Unsupported kinematic model type: {}!!!".format(kinematic_model_type)
            )
            raise ValueError("Unsupported kinematic model type!!!")

    @staticmethod
    def stateDerivative(x: ca.MX, u: ca.MX) -> ca.MX:
        """获取车辆状态的导数"""
        Log.error("stateDerivative is not implemented!!!")
        raise NotImplementedError("stateDerivative is not implemented!!!")

    @property
    def state_dim(self) -> int:
        """状态空间维度"""
        return 4

    @property
    def ctrl_dim(self) -> int:
        """控制空间维度"""
        return 2

    @property
    def pose(self) -> Pose:
        """车辆位姿"""
        return Pose(self.state[0], self.state[1], self.state[2])

    @property
    def x(self) -> float:
        """车辆x坐标"""
        return self.state[0]

    @property
    def y(self) -> float:
        """车辆y坐标"""
        return self.state[1]

    @property
    def theta(self) -> float:
        """车辆航向角"""
        return self.state[2]

    @property
    def v(self) -> float:
        """车辆速度"""
        return self.state[3]

    def getLinearizedMatrices(self) -> Tuple[ca.Function, ca.Function]:
        """返回把系统线性化后的A，B矩阵：x_next = A * x + B * u"""
        state_sym = ca.SX.sym("state_sym", self.state_dim)  # type: ignore
        ctrl_input_sym = ca.SX.sym("ctrl_input_sym", 2)  # type: ignore
        state_dot = self.state_transition_func_mx(state_sym, ctrl_input_sym)
        state_next = state_sym + state_dot * Config.DELTA_T
        A = ca.jacobian(state_next, state_sym)
        B = ca.jacobian(state_next, ctrl_input_sym)
        A_func = ca.Function("A_func", [state_sym, ctrl_input_sym], [A])
        B_func = ca.Function("B_func", [state_sym, ctrl_input_sym], [B])
        return A_func, B_func

    def setCurState(self, pose: Pose, v: float) -> None:
        """设置当前位姿和速度"""
        self.state = np.array([pose.x, pose.y, pose.theta, v])

    def step(self, ctrl_input: CtrlInput) -> None:
        """根据控制输入更新车辆状态"""
        delta_state = self.state_transition_func(self.state, ctrl_input.val)
        self.state += Config.DELTA_T * delta_state
        self.state[2] = Pose.normalizeTheta(self.state[2])
        self.state[3] = np.clip(self.state[3], -Config.MAX_SPEED, Config.MAX_SPEED)
