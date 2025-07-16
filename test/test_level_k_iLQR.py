# -*- coding: utf-8 -*-

import os
import sys

sys.path.append(os.getcwd())

import casadi as ca
import numpy as np
import pytest

from src.algo.level_k.config import Config
from src.algo.level_k.iLQR import iLQR


class TestILQRCostFunction:
    @pytest.fixture
    def ilqr_instance(self) -> iLQR:
        """创建一个iLQR实例用于测试"""
        # 这里需要创建一个测试用的 iLQR 实例
        config = Config()
        config.COLLISION_WEIGHT = 100.0
        """碰撞惩罚权重"""
        config.COMFORT_DISTANCE = 6.66
        """舒适距离"""
        config.COMFORT_WEIGHT = 10.0
        """舒适距离权重"""
        config.HALF_WHEEL_BASE = 0.5 * config.WHEEL_BASE
        """半轴距"""
        config.SAFE_DIST_SQUARED = config.WHEEL_BASE**2
        """安全距离的平方，如果两车中心距离的平方小于这个值，则认为发生碰撞"""
        config.COMFORT_DIST_SQUARED = config.COMFORT_DISTANCE**2
        """舒适性距离的平方，如果两车中心距离的平方小于这个值，则认为两车过于接近"""
        config.GAMMA = 0.9
        """折扣因子"""
        config.NEGATIVE_SPEED_PENALTY = 100
        """负速度惩罚"""
        return iLQR(config, num_cars=3)

    @pytest.fixture
    def ilqr_instance_without_coll_penalty(self) -> iLQR:
        """创建一个iLQR实例用于测试，不包含碰撞惩罚"""
        config = Config()
        config.COLLISION_WEIGHT = 0.0
        """碰撞惩罚权重"""
        config.COMFORT_DISTANCE = 6.66
        """舒适距离"""
        config.COMFORT_WEIGHT = 10.0
        """舒适距离权重"""
        config.HALF_WHEEL_BASE = 0.5 * config.WHEEL_BASE
        """半轴距"""
        config.SAFE_DIST_SQUARED = config.WHEEL_BASE**2
        """安全距离的平方，如果两车中心距离的平方小于这个值，则认为发生碰撞"""
        config.COMFORT_DIST_SQUARED = config.COMFORT_DISTANCE**2
        """舒适性距离的平方，如果两车中心距离的平方小于这个值，则认为两车过于接近"""
        config.GAMMA = 0.9
        """折扣因子"""
        config.NEGATIVE_SPEED_PENALTY = 100
        """负速度惩罚"""
        return iLQR(config, num_cars=3)

    @pytest.fixture
    def ilqr_instance_without_comfort_penalty(self) -> iLQR:
        """创建一个iLQR实例用于测试，不包含舒适性惩罚"""
        config = Config()
        config.COLLISION_WEIGHT = 100.0
        """碰撞惩罚权重"""
        config.COMFORT_DISTANCE = 666
        """舒适距离"""
        config.COMFORT_WEIGHT = 0.0
        """舒适距离权重"""
        config.HALF_WHEEL_BASE = 0.5 * config.WHEEL_BASE
        """半轴距"""
        config.SAFE_DIST_SQUARED = config.WHEEL_BASE**2
        """安全距离的平方，如果两车中心距离的平方小于这个值，则认为发生碰撞"""
        config.COMFORT_DIST_SQUARED = config.COMFORT_DISTANCE**2
        """舒适性距离的平方，如果两车中心距离的平方小于这个值，则认为两车过于接近"""
        config.GAMMA = 0.9
        """折扣因子"""
        config.NEGATIVE_SPEED_PENALTY = 100
        """负速度惩罚"""
        return iLQR(config, num_cars=3)

    @staticmethod
    def CalBaseCost(ilqr_instance: iLQR, X, U, X_REF):
        Q = np.diag(
            [
                ilqr_instance.config.X_ERROR_WEIGHT,
                ilqr_instance.config.Y_ERROR_WEIGHT,
                ilqr_instance.config.THETA_ERROR_WEIGHT,
                ilqr_instance.config.V_ERROR_WEIGHT,
            ]
        )
        R = np.diag(
            [ilqr_instance.config.ACC_WEIGHT, ilqr_instance.config.STEER_WEIGHT]
        )
        delta_x = X - X_REF
        return 0.5 * (delta_x.T @ Q @ delta_x + U.T @ R @ U)

    def test_base_cost(self, ilqr_instance: iLQR):
        """测试基础代价项是否正常计算"""
        # 准备测试数据
        X = ca.DM([0, 0, 0, 1])  # 状态：[x, y, theta, v]
        U = ca.DM([0.1, 0.05])  # 控制输入：[加速度, 转向角]
        X_REF = ca.DM([1, 1, 0, 2])  # 参考状态
        CAR_ID = ca.DM(0)
        LEVEL = ca.DM(1)
        T = ca.DM(0)
        CARS_X = ca.DM([0, 300, 400])
        CARS_Y = ca.DM([0, 300, 400])
        # 调用代价函数
        cost = ilqr_instance.cost_func(X, U, X_REF, CAR_ID, LEVEL, T, CARS_X, CARS_Y)
        # 验证基础代价项
        # 这里需要根据 Config 中的权重计算预期代价
        base_cost = self.CalBaseCost(ilqr_instance, X, U, X_REF)
        assert np.isclose(float(cost), float(base_cost), rtol=1e-5)

    def test_negative_speed_penalty(self, ilqr_instance: iLQR):
        """测试负速度惩罚"""
        ilqr = ilqr_instance
        X = ca.DM([0, 0, 0, -1])  # 负速度
        U = ca.DM([0.1, 0.05])
        X_REF = ca.DM([1, 1, 0, 2])
        CAR_ID = ca.DM(0)
        LEVEL = ca.DM(1)
        T = ca.DM(0)
        CARS_X = ca.DM([0, 300, 400])
        CARS_Y = ca.DM([0, 300, 400])
        cost = ilqr.cost_func(X, U, X_REF, CAR_ID, LEVEL, T, CARS_X, CARS_Y)
        base_cost = self.CalBaseCost(ilqr, X, U, X_REF)
        expected_negative_speed_cost = cost - base_cost
        negative_speed_cost = ilqr_instance.config.NEGATIVE_SPEED_PENALTY * (X[3] ** 2)
        assert np.isclose(
            float(expected_negative_speed_cost),
            float(negative_speed_cost),
            rtol=1e-5,
        )

    def test_collision_cost(self, ilqr_instance_without_comfort_penalty: iLQR):
        """测试碰撞代价"""
        ilqr = ilqr_instance_without_comfort_penalty
        X = ca.DM([2, 2, 0, 1])
        U = ca.DM([0.1, 0.05])
        X_REF = ca.DM([1, 1, 0, 2])
        CAR_ID = ca.DM(0)
        LEVEL = ca.DM(1)
        T = ca.DM(0)
        base_cost = self.CalBaseCost(ilqr, X, U, X_REF)
        # 自车几何中心
        ego_x = X[0] + ilqr.config.HALF_WHEEL_BASE * ca.cos(X[2])
        ego_y = X[1] + ilqr.config.HALF_WHEEL_BASE * ca.sin(X[2])
        # 不与任何其他车辆接近
        CARS_X = ca.DM([2, 300, 400])
        CARS_Y = ca.DM([2, 300, 400])
        cost = ilqr.cost_func(X, U, X_REF, CAR_ID, LEVEL, T, CARS_X, CARS_Y)
        assert np.isclose(float(cost), float(base_cost), rtol=1e-5)
        # 与1号车接近
        CARS_X = ca.DM([2, ego_x + 0.1, 400])
        CARS_Y = ca.DM([2, ego_y + 0.1, 400])
        cost = ilqr.cost_func(X, U, X_REF, CAR_ID, LEVEL, T, CARS_X, CARS_Y)
        min_dist_squared = (ego_x - CARS_X[1]) ** 2 + (ego_y - CARS_Y[1]) ** 2
        expected_collision_cost = (
            ilqr.config.COLLISION_WEIGHT
            * (1 - min_dist_squared / ilqr.config.SAFE_DIST_SQUARED) ** 2
        )
        assert np.isclose(
            float(cost), float(base_cost + expected_collision_cost), rtol=1e-5
        )
        # 与2号车接近
        CARS_X = ca.DM([2, 300, ego_x + 0.2])
        CARS_Y = ca.DM([2, 300, ego_y + 0.2])
        min_dist_squared = (ego_x - CARS_X[2]) ** 2 + (ego_y - CARS_Y[2]) ** 2
        cost = ilqr.cost_func(X, U, X_REF, CAR_ID, LEVEL, T, CARS_X, CARS_Y)
        expected_collision_cost = (
            ilqr.config.COLLISION_WEIGHT
            * (1 - min_dist_squared / ilqr.config.SAFE_DIST_SQUARED) ** 2
        )
        assert np.isclose(
            float(cost), float(base_cost + expected_collision_cost), rtol=1e-5
        )
        # 与1号车和2号车都接近，但是与1号车最近
        CARS_X = ca.DM([2, ego_x + 0.1, ego_x + 0.2])
        CARS_Y = ca.DM([2, ego_y + 0.1, ego_y + 0.2])
        min_dist_squared = (ego_x - CARS_X[1]) ** 2 + (ego_y - CARS_Y[1]) ** 2
        cost = ilqr.cost_func(X, U, X_REF, CAR_ID, LEVEL, T, CARS_X, CARS_Y)
        expected_collision_cost = (
            ilqr.config.COLLISION_WEIGHT
            * (1 - min_dist_squared / ilqr.config.SAFE_DIST_SQUARED) ** 2
        )
        assert np.isclose(
            float(cost), float(base_cost + expected_collision_cost), rtol=1e-5
        )

    def test_comfort_cost(self, ilqr_instance_without_coll_penalty: iLQR):
        """测试舒适距离代价"""
        ilqr = ilqr_instance_without_coll_penalty
        X = ca.DM([2, 2, 0, 1])
        U = ca.DM([0.1, 0.05])
        X_REF = ca.DM([1, 1, 0, 2])
        CAR_ID = ca.DM(0)
        LEVEL = ca.DM(1)
        T = ca.DM(0)
        base_cost = self.CalBaseCost(ilqr, X, U, X_REF)
        ego_x = X[0] + ilqr.config.HALF_WHEEL_BASE * ca.cos(X[2])
        ego_y = X[1] + ilqr.config.HALF_WHEEL_BASE * ca.sin(X[2])
        # 不与任何其他车辆接近
        CARS_X = ca.DM([2, 300, 400])
        CARS_Y = ca.DM([2, 300, 400])
        cost = ilqr.cost_func(X, U, X_REF, CAR_ID, LEVEL, T, CARS_X, CARS_Y)
        assert np.isclose(float(cost), float(base_cost), rtol=1e-5)
        # 与1号车接近
        CARS_X = ca.DM([2, ego_x + 0.1, 400])
        CARS_Y = ca.DM([2, ego_y + 0.1, 400])
        min_dist_squared = (ego_x - CARS_X[1]) ** 2 + (ego_y - CARS_Y[1]) ** 2
        expected_comfort_cost = (
            ilqr.config.COMFORT_WEIGHT
            * (1 - min_dist_squared / ilqr.config.COMFORT_DIST_SQUARED) ** 2
        )
        cost = ilqr.cost_func(X, U, X_REF, CAR_ID, LEVEL, T, CARS_X, CARS_Y)
        assert np.isclose(
            float(cost), float(base_cost + expected_comfort_cost), rtol=1e-5
        )
        # 与2号车接近
        CARS_X = ca.DM([2, 300, ego_x + 0.2])
        CARS_Y = ca.DM([2, 300, ego_y + 0.2])
        min_dist_squared = (ego_x - CARS_X[2]) ** 2 + (ego_y - CARS_Y[2]) ** 2
        expected_comfort_cost = (
            ilqr.config.COMFORT_WEIGHT
            * (1 - min_dist_squared / ilqr.config.COMFORT_DIST_SQUARED) ** 2
        )
        cost = ilqr.cost_func(X, U, X_REF, CAR_ID, LEVEL, T, CARS_X, CARS_Y)
        assert np.isclose(
            float(cost), float(base_cost + expected_comfort_cost), rtol=1e-5
        )
        # 与1号车和2号车都接近，但是与1号车最近
        CARS_X = ca.DM([2, ego_x + 0.1, ego_x + 0.2])
        CARS_Y = ca.DM([2, ego_y + 0.1, ego_y + 0.2])
        min_dist_squared = (ego_x - CARS_X[1]) ** 2 + (ego_y - CARS_Y[1]) ** 2
        expected_comfort_cost = (
            ilqr.config.COMFORT_WEIGHT
            * (1 - min_dist_squared / ilqr.config.COMFORT_DIST_SQUARED) ** 2
        )
        cost = ilqr.cost_func(X, U, X_REF, CAR_ID, LEVEL, T, CARS_X, CARS_Y)
        assert np.isclose(
            float(cost), float(base_cost + expected_comfort_cost), rtol=1e-5
        )

    def test_gamma_discount(self, ilqr_instance):
        """测试时间衰减系数"""
        # 准备测试数据，不同的时间步
        X = ca.DM([0, 0, 0, 1])
        U = ca.DM([0.1, 0.05])
        X_REF = ca.DM([1, 1, 0, 2])
        CAR_ID = ca.DM(0)
        LEVEL = ca.DM(1)
        CARS_X = ca.DM([2, 3, 4])
        CARS_Y = ca.DM([2, 3, 4])
        # 测试不同时间步的代价
        cost_t0 = ilqr_instance.cost_func(
            X, U, X_REF, CAR_ID, LEVEL, ca.DM(0), CARS_X, CARS_Y
        )
        cost_t1 = ilqr_instance.cost_func(
            X, U, X_REF, CAR_ID, LEVEL, ca.DM(1), CARS_X, CARS_Y
        )
        cost_t2 = ilqr_instance.cost_func(
            X, U, X_REF, CAR_ID, LEVEL, ca.DM(2), CARS_X, CARS_Y
        )
        # 验证代价是否随时间衰减
        assert np.isclose(
            float(ilqr_instance.config.GAMMA * cost_t0), float(cost_t1), rtol=1e-5
        )
        assert np.isclose(
            float(ilqr_instance.config.GAMMA**2 * cost_t0), float(cost_t2), rtol=1e-5
        )
