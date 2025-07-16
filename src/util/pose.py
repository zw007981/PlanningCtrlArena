# -*- coding: utf-8 -*-

import os
import sys

sys.path.append(os.getcwd())

import math

from src.util.config import Config


class Pose:
    """姿态类"""

    SHARP_TURN_THRESHOLD = 0.5
    """急转弯曲率阈值，曲率大于该值则认为是急转弯"""
    GENTLE_TURN_THRESHOLD = 0.25
    """缓慢转弯曲率阈值，曲率大于该值则认为至少是缓慢转弯"""

    def __init__(self, x: float = 0.0, y: float = 0.0, theta: float = 0.0):
        self.x = x
        """x坐标"""
        self.y = y
        """y坐标"""
        self.theta = Pose.normalizeTheta(theta)
        """角度"""
        self.curvature = 0.0
        """曲率"""
        self.max_speed_rate = 1.0
        """最大速度比例，与曲率相关，用于在急转弯时降低速度"""
        self.ref_line_id: int = -1
        """所处参考线id，默认为-1"""
        self.id: int = -1
        """姿态id，默认为-1，如果在某条参考线上，则为该参考线上的点序号"""

    def __eq__(self, other: "Pose") -> bool:
        if other is None or not isinstance(other, Pose):
            return False
        return (
            abs(self.x - other.x) < Config.FLOAT_EPS
            and abs(self.y - other.y) < Config.FLOAT_EPS
            and abs(self.theta - other.theta) < Config.FLOAT_EPS
        )

    def __str__(self) -> str:
        return "({:.2f}, {:.2f}, {:.2f})".format(self.x, self.y, self.theta)

    @staticmethod
    def normalizeTheta(theta: float) -> float:
        """将角度限制在[-pi, pi)的范围内"""
        if math.isnan(theta):
            return theta
        else:
            return (theta + math.pi) % (2 * math.pi) - math.pi

    def calDistance(self, pose: "Pose") -> float:
        """计算两个位姿之间的距离"""
        return math.hypot(self.x - pose.x, self.y - pose.y)

    def isTooClose(self, other: "Pose") -> bool:
        """判断两个位姿是否过于接近"""
        return self.calDistance(other) < Config.B_SPLINE_INTERVAL_DIST

    def setCurvature(self, curvature: float):
        """设置曲率"""
        self.curvature = curvature
        if abs(curvature) > Pose.SHARP_TURN_THRESHOLD:
            self.max_speed_rate = 0.6
        elif abs(curvature) > Pose.GENTLE_TURN_THRESHOLD:
            self.max_speed_rate = 0.8
