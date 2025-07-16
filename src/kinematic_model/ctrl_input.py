# -*- coding: utf-8 -*-

import os
import sys

sys.path.append(os.getcwd())

import numpy as np
from numpy.typing import NDArray

from src.util.config import Config
from src.util.log import Log


class CtrlInput:
    """控制输入"""

    def __init__(self, kinematic_model_type: str) -> None:
        self.kinematic_model_type = kinematic_model_type
        """适配的运动学模型类型"""
        if self.kinematic_model_type == "BICYCLE_MODEL":
            self.a: float = 0.0
            """加速度"""
            self.delta: float = 0.0
            """前轮转向角"""
        elif self.kinematic_model_type == "POINT_MASS":
            self.u_x: float = 0.0
            """x方向上的移动距离"""
            self.u_y: float = 0.0
            """y方向上的移动距离"""
        else:
            Log.error(
                "Unsupported kinematic model type: {}!!!".format(
                    self.kinematic_model_type
                )
            )
            raise ValueError("Unsupported kinematic model type!!!")

    def __str__(self) -> str:
        if self.kinematic_model_type == "BICYCLE_MODEL":
            return "a={:.2f}, delta={:.2f}".format(self.a, self.delta)
        elif self.kinematic_model_type == "POINT_MASS":
            return "u_x={:.2f}, u_y={:.2f}".format(self.u_x, self.u_y)
        else:
            Log.error(
                "Unsupported kinematic model type: {}!!!".format(
                    self.kinematic_model_type
                )
            )
            raise ValueError("Unsupported kinematic model type!!!")

    @property
    def dim(self) -> int:
        """获取控制输入的维度"""
        if self.kinematic_model_type == "BICYCLE_MODEL":
            return 2
        elif self.kinematic_model_type == "POINT_MASS":
            return 2
        else:
            Log.error(
                "Unsupported kinematic model type: {}!!!".format(
                    self.kinematic_model_type
                )
            )
            raise ValueError("Unsupported kinematic model type!!!")

    @property
    def val(self) -> NDArray[np.float64]:
        """
        获取控制输入的值。
        对于自行车模型，返回加速度a和前轮转向角delta；
        对于质点模型，返回x方向上的移动距离u_x和y方向上的移动距离u_y。
        """
        if self.kinematic_model_type == "BICYCLE_MODEL":
            return np.array([self.a, self.delta])
        elif self.kinematic_model_type == "POINT_MASS":
            return np.array([self.u_x, self.u_y])
        else:
            Log.error(
                "Unsupported kinematic model type: {}!!!".format(
                    self.kinematic_model_type
                )
            )
            raise ValueError("Unsupported kinematic model type!!!")

    def setVal(self, *args) -> None:
        """
        设置控制输入的值。
        针对自行车模型，输入参数为加速度a和前轮转向角delta；
        针对质点模型，输入参数为x方向上的移动距离u_x和y方向上的移动距离u_y。
        """
        if self.kinematic_model_type == "BICYCLE_MODEL":
            if len(args) != 2:
                Log.error("Invalid args length: {}!!!".format(len(args)))
                raise ValueError("Invalid args length!!!")
            self.a = np.clip(args[0], -Config.MAX_ACCEL, Config.MAX_ACCEL)
            self.delta = np.clip(args[1], -Config.MAX_STEER, Config.MAX_STEER)
        elif self.kinematic_model_type == "POINT_MASS":
            if len(args) != 2:
                Log.error("Invalid args length: {}!!!".format(len(args)))
                raise ValueError("Invalid args length!!!")
            self.u_x = args[0]
            self.u_y = args[1]
        else:
            Log.error(
                "Unsupported kinematic model type: {}!!!".format(
                    self.kinematic_model_type
                )
            )
            raise ValueError("Unsupported kinematic model type!!!")
