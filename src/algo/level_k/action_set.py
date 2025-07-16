# -*- coding: utf-8 -*-

import os
import sys

sys.path.append(os.getcwd())

from enum import Enum

import casadi as ca

from src.algo.level_k.config import Config
from src.traffic_simulator.color import *
from src.util.log import Log


class ActionSet(Enum):
    """动作集枚举类，定义了所有可能的动作"""

    MAINTAIN = 0
    """保持当前状态"""
    TURN_SLIGHTLY_LEFT = 1
    """轻微向左转"""
    TURN_SLIGHTLY_RIGHT = 2
    """轻微向右转"""
    NOMINAL_ACCELERATION = 3
    """正常加速"""
    NOMINAL_DECELERATION = 4
    """正常减速"""
    MAXIMUM_ACCELERATION = 5
    """最大加速"""
    MAXIMUM_DECELERATION = 6
    """最大减速"""
    TURN_LEFT_AND_ACCELERATE = 7
    """向左转并正常加速"""
    TURN_RIGHT_AND_ACCELERATE = 8
    """向右转并正常加速"""

    @staticmethod
    def Size() -> int:
        """获取动作集的大小"""
        return len(ActionSet)

    @staticmethod
    def GetCtrlInput(action: "ActionSet") -> ca.DM:
        """根据动作获取具体的控制输入，shape: CTRL_DIM * 1"""
        controls = {
            ActionSet.MAINTAIN: (0.0, 0.0),
            ActionSet.TURN_SLIGHTLY_LEFT: (0.0, 0.2 * Config.MAX_STEER),
            ActionSet.TURN_SLIGHTLY_RIGHT: (0.0, -0.2 * Config.MAX_STEER),
            ActionSet.NOMINAL_ACCELERATION: (0.2 * Config.MAX_ACCEL, 0.0),
            ActionSet.NOMINAL_DECELERATION: (-0.2 * Config.MAX_ACCEL, 0.0),
            ActionSet.MAXIMUM_ACCELERATION: (0.6 * Config.MAX_ACCEL, 0.0),
            ActionSet.MAXIMUM_DECELERATION: (-0.6 * Config.MAX_ACCEL, 0.0),
            ActionSet.TURN_LEFT_AND_ACCELERATE: (
                0.2 * Config.MAX_ACCEL,
                0.6 * Config.MAX_STEER,
            ),
            ActionSet.TURN_RIGHT_AND_ACCELERATE: (
                0.2 * Config.MAX_ACCEL,
                -0.6 * Config.MAX_STEER,
            ),
        }
        try:
            acc, steer = controls[action]
        except KeyError:
            Log.error("Invalid action provided!!!")
            raise ValueError("Invalid action provided!!!")
        return ca.DM([[acc], [steer]])

    @staticmethod
    def GetCtrlInputFromIndex(index: int) -> ca.DM:
        """根据动作索引获取具体的控制输入，shape: CTRL_DIM * 1"""
        return ActionSet.GetCtrlInput(ActionSet(index))
