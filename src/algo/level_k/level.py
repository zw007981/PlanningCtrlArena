# -*- coding: utf-8 -*-

import os
import sys

sys.path.append(os.getcwd())

from enum import Enum
from typing import List

from src.util.log import Log


class Level(Enum):
    """level-k层级"""

    LEVEL_0 = 0
    """level-0层级"""
    LEVEL_1 = 1
    """level-1层级"""
    LEVEL_2 = 2
    """level-2层级"""
    ADAPTIVE = 3
    """自适应层级"""

    @staticmethod
    def GetLevel(level_str: str) -> "Level":
        """根据字符串获取level-k层级"""
        if level_str.lower() == "adaptive":
            return Level.ADAPTIVE
        try:
            return Level(int(level_str))
        except ValueError:
            Log.error("Invalid level_str: {}!!!".format(level_str))
            raise ValueError("level_str must be a number or 'Adaptive'")

    @staticmethod
    def GetMaxLevel(level_list: List["Level"]) -> "Level":
        """获取最大level-k层级"""
        max_level = Level.LEVEL_0
        for level in level_list:
            if level != Level.ADAPTIVE and level.value > max_level.value:
                max_level = level
        return max_level

    @staticmethod
    def MinusOne(level: "Level") -> "Level":
        """返回level-1层级"""
        if level == Level.LEVEL_0:
            return Level.LEVEL_0
        elif level == Level.LEVEL_1:
            return Level.LEVEL_0
        elif level == Level.LEVEL_2:
            return Level.LEVEL_1
        else:
            Log.error("Invalid level: {}!!!".format(level))
            raise ValueError("Invalid level!!!")

    @staticmethod
    def PlusOne(level: "Level") -> "Level":
        """返回level+1层级"""
        if level == Level.LEVEL_0:
            return Level.LEVEL_1
        elif level == Level.LEVEL_1:
            return Level.LEVEL_2
        elif level == Level.LEVEL_2:
            return Level.LEVEL_2
        else:
            Log.error("Invalid level: {}!!!".format(level))
            raise ValueError("Invalid level!!!")
