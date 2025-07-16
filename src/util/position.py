# -*- coding: utf-8 -*-

import os
import sys

sys.path.append(os.getcwd())

import math

from src.util.config import Config


class Position:
    """位置类"""

    def __init__(self, x: float = 0.0, y: float = 0.0):
        self.x = x
        """x坐标"""
        self.y = y
        """y坐标"""

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Position):
            return False
        return (
            abs(self.x - value.x) <= Config.FLOAT_EPS
            and abs(self.y - value.y) <= Config.FLOAT_EPS
        )

    def calDistance(self, pos: "Position") -> float:
        """计算当前位置与目标位置之间的距离"""
        return math.hypot(self.x - pos.x, self.y - pos.y)
