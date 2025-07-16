# -*- coding: utf-8 -*-

import os
import sys

sys.path.append(os.getcwd())

from src.algo.iLQR import Config as BaseConfig


class Config(BaseConfig):
    """适用于level-k算法的配置"""

    def __init__(self):
        super().__init__()
        self.COLLISION_WEIGHT = 1000.0
        """碰撞惩罚权重"""
        self.COMFORT_DISTANCE = 5.0 * self.WHEEL_BASE
        """舒适距离"""
        self.COMFORT_WEIGHT = 100.0
        """舒适距离权重"""
        self.HALF_WHEEL_BASE = 0.5 * self.WHEEL_BASE
        """半轴距"""
        self.SAFE_DIST_SQUARED = self.WHEEL_BASE**2
        """安全距离的平方，如果两车中心距离的平方小于这个值，则认为发生碰撞"""
        self.COMFORT_DIST_SQUARED = self.COMFORT_DISTANCE**2
        """舒适性距离的平方，如果两车中心距离的平方小于这个值，则认为两车过于接近"""
        self.GAMMA = 0.9
        """折扣因子"""
        self.NEGATIVE_SPEED_PENALTY = 100.0
        """负速度惩罚"""

    def modifyConfig(self) -> None:
        self.N = 30
        self.X_ERROR_WEIGHT = 0.1
        self.Y_ERROR_WEIGHT = 0.1
        self.V_ERROR_WEIGHT = 1.0
        self.THETA_ERROR_WEIGHT = 2.0
        self.STEER_WEIGHT = 10000.0
        self.X_ERROR_WEIGHT_F = 0.5
        """终端x方向误差权重"""
        self.Y_ERROR_WEIGHT_F = 0.5
        """终端y方向误差权重"""
        self.THETA_ERROR_WEIGHT_F = 0.3
        """终端角度误差权重"""
        self.V_ERROR_WEIGHT_F = 0.5
