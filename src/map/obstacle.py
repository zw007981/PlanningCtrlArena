# -*- coding: utf-8 -*-

import os
import sys

sys.path.append(os.getcwd())

from enum import Enum
from typing import Any, Dict

from src.map.graph import Graph, Node
from src.util.config import Config
from src.util.log import Log


class ObstacleType(Enum):
    """障碍物类型"""

    STATIC = 0
    """静态障碍物"""
    DYNAMIC = 1
    """动态障碍物，会沿着某个轨迹运动"""
    UNCERTAIN = 2
    """不确定障碍物，只有在近距离才能精准地观察到是否存在"""


class SelectionMethod(Enum):
    """动态障碍物在面对岔路时的选择方式"""

    RANDOM = 0
    """随机选择"""
    MAX_X = 1
    """选择x坐标最大的路径"""
    MIN_X = 2
    """选择x坐标最小的路径"""
    MAX_Y = 3
    """选择y坐标最大的路径"""
    MIN_Y = 4
    """选择y坐标最小的路径"""

    @classmethod
    def from_str(cls, s: str) -> "SelectionMethod":
        """从字符串中解析出SelectionMethod"""
        mapping = {
            "random": SelectionMethod.RANDOM,
            "max_x": SelectionMethod.MAX_X,
            "min_x": SelectionMethod.MIN_X,
            "max_y": SelectionMethod.MAX_Y,
            "min_y": SelectionMethod.MIN_Y,
        }
        return mapping.get(s.lower(), SelectionMethod.RANDOM)


class Obstacle:
    """
    障碍物类，共有三种类型：
    静态障碍物：一直存在，不会消失；
    动态障碍物：会沿着某个轨迹运动；
    不确定障碍物：只有在近距离才能精准地观察到是否存在。
    """

    def __init__(self, json_obj: Dict[str, Any]):
        self.x: float
        """x坐标"""
        self.y: float
        """y坐标"""
        self.radius: float
        """半径"""
        self.obstacle_type: ObstacleType
        """障碍物类型"""
        self.route: Graph
        """动态障碍物的预设轨迹"""
        self.selection_method: SelectionMethod = SelectionMethod.RANDOM
        """动态障碍物在面对岔路时的选择方法"""
        self.will_appear: bool = False
        """是否会出现（仅用于不确定障碍物障碍物）"""
        self.has_been_observed: bool = False
        """是否已经被观察到（仅用于不确定障碍物障碍物）"""
        self.obs_threshold: float = 0.0
        """
        观测阈值（仅用于不确定障碍物障碍物），距离小于该值时才能精准观测到障碍物是否存在。
        默认为1.5倍车辆的轴距加障碍物的半径。
        """
        try:
            self.x = float(json_obj["center"]["x"])
            self.y = float(json_obj["center"]["y"])
            self.radius = float(json_obj["radius"])
            if json_obj["type"] == "static":
                self.obstacle_type = ObstacleType.STATIC
            elif json_obj["type"] == "dynamic":
                self.obstacle_type = ObstacleType.DYNAMIC
                self.route = Graph(json_obj["route"])
                if "selection_method" in json_obj:
                    self.selection_method = SelectionMethod.from_str(
                        json_obj["selection_method"]
                    )
            elif json_obj["type"] == "uncertain":
                self.obstacle_type = ObstacleType.UNCERTAIN
                self.will_appear = json_obj["will_appear"]
                self.obs_threshold = 1.5 * (Config.WHEEL_BASE + self.radius)
            else:
                raise Exception("Invalid obstacle type: {}".format(json_obj["type"]))
        except Exception as e:
            Log.error("Failed to parse obstacle {}: {}!!!".format(json_obj, e))
            raise e

    def __calManhattanDist(self, x: float = 0.0, y: float = 0.0) -> float:
        """计算车俩到障碍物的曼哈顿距离"""
        return abs(x - self.x) + abs(y - self.y)

    def __updateHasBeenObserved(self, x: float = 0.0, y: float = 0.0) -> None:
        """车辆到达(x, y)位置时更新障碍物是否被观察到"""
        if self.has_been_observed:
            return
        self.has_been_observed = self.__calManhattanDist(x, y) <= self.obs_threshold

    def getSafeRadius(self, x: float = 0.0, y: float = 0.0) -> float:
        """
        返回用于安全避障的半径。对于静态和动态障碍物，直接返回半径。
        对于不确定障碍物，如果还没有被观测到则返回半径作为估计值，
        否则根据是否会出现来返回实际半径或者0。
        """
        if (
            self.obstacle_type == ObstacleType.STATIC
            or self.obstacle_type == ObstacleType.DYNAMIC
        ):
            return self.radius
        else:
            self.__updateHasBeenObserved(x, y)
            if self.has_been_observed:
                return self.radius if self.will_appear else 0.0
            else:
                return self.radius

    def getIdealRadius(self, x: float = 0.0, y: float = 0.0) -> float:
        """
        返回用于理想避障的半径。对于静态障碍物和动态，直接返回半径。
        对于不确定障碍物，如果还没有被观测到则返回0作为估计值，
        否则根据是否会出现来返回实际半径或者0。
        """
        if (
            self.obstacle_type == ObstacleType.STATIC
            or self.obstacle_type == ObstacleType.DYNAMIC
        ):
            return self.radius
        else:
            self.__updateHasBeenObserved(x, y)
            if self.has_been_observed:
                return self.radius if self.will_appear else 0.0
            else:
                return 0.0
