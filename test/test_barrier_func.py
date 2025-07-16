# -*- coding: utf-8 -*-

import os
import sys

sys.path.append(os.getcwd())

import numpy as np
import pytest

from src.algo.CILQR import BarrierFunc, Config


@pytest.fixture
def barrier_func(scope="session"):
    return BarrierFunc(num_obs=0)


@pytest.fixture
def barrier_func_with_obs(scope="session"):
    return BarrierFunc(num_obs=1)


@pytest.fixture
def valid_state(scope="session"):
    valid_x = 0.5 * (Config.X_MIN + Config.X_MAX)
    valid_y = 0.5 * (Config.Y_MIN + Config.Y_MAX)
    valid_theta = 0.0
    valid_v = 0.0
    return np.array([valid_x, valid_y, valid_theta, valid_v])


@pytest.fixture
def valid_ctrl(scope="session"):
    return np.array([0.0, 0.0])


def getVal(
    barrier_func: BarrierFunc,
    state: np.ndarray,
    ctrl: np.ndarray,
    obs_list: np.ndarray,
    t: float,
) -> float:
    return float(barrier_func.barrier_func(state, ctrl, obs_list, t))  # type: ignore


def isFeasible(
    barrier_func: BarrierFunc,
    state: np.ndarray,
    ctrl: np.ndarray,
    obs_list: np.ndarray,
) -> bool:
    return bool(barrier_func.is_feasible_func(state, ctrl, obs_list))


def calRelCostChange(cost0: float, cost1: float) -> float:
    """计算代价变化的相对值"""
    return (cost1 - cost0) / np.maximum(abs(cost0), Config.ZERO_EPS)


def test_compare_two_valid_states(valid_state, valid_ctrl, barrier_func):
    """测试两个有效状态之间的代价变化，理想情况下相对变化应该不大"""
    obs_list = np.array([])
    valid_state1 = np.copy(valid_state)
    valid_state1[3] = 0.2 * Config.MAX_SPEED
    cost0 = getVal(barrier_func, valid_state, valid_ctrl, obs_list, 0.5)
    cost1 = getVal(barrier_func, valid_state1, valid_ctrl, obs_list, 0.5)
    assert abs(calRelCostChange(cost0, cost1)) < 0.1
    is_feasible0 = isFeasible(barrier_func, valid_state, valid_ctrl, obs_list)
    is_feasible1 = isFeasible(barrier_func, valid_state1, valid_ctrl, obs_list)
    assert is_feasible0 and is_feasible1


def test_compare_two_t_values(valid_state, valid_ctrl, barrier_func):
    """测试相同状态和控制量下不同t值对应的代价，理想情况下t越大，代价的绝对值越小"""
    obs_list = np.array([])
    cost0 = getVal(barrier_func, valid_state, valid_ctrl, obs_list, 0.5)
    cost1 = getVal(barrier_func, valid_state, valid_ctrl, obs_list, 1.0)
    assert abs(cost0) > abs(cost1)


def test_barrier_func_with_obstacle(valid_state, valid_ctrl, barrier_func_with_obs):
    """比较车辆右边一定范围内有无障碍物的情况，理想情况下有障碍物的代价更高"""
    obs_r = 1.0
    # 此时不应该碰撞，注意车辆以后轮中心为坐标原点
    obs_list0 = np.array(
        [
            [valid_state[0] + 1.1 * (obs_r + Config.WHEEL_BASE)],
            [valid_state[1]],
            [1.0],
        ]
    )
    # 此时应该碰撞
    obs_list1 = np.array(
        [
            [valid_state[0] + 0.9 * (obs_r + Config.WHEEL_BASE)],
            [valid_state[1]],
            [1.0],
        ]
    )
    cost0 = getVal(barrier_func_with_obs, valid_state, valid_ctrl, obs_list0, 0.5)
    cost1 = getVal(barrier_func_with_obs, valid_state, valid_ctrl, obs_list1, 0.5)
    assert calRelCostChange(cost0, cost1) > 0.8
    is_feasible0 = isFeasible(barrier_func_with_obs, valid_state, valid_ctrl, obs_list0)
    is_feasible1 = isFeasible(barrier_func_with_obs, valid_state, valid_ctrl, obs_list1)
    assert is_feasible0 and not is_feasible1


def test_barrier_func_with_invalid_v(valid_state, valid_ctrl, barrier_func):
    """测试当速度小于最小值时的情况，这种非法状态的代价应该更高"""
    obs_list = np.array([])
    invalid_state = np.copy(valid_state)
    invalid_state[3] = -Config.MAX_SPEED - 0.1
    cost0 = getVal(barrier_func, valid_state, valid_ctrl, obs_list, 0.5)
    cost1 = getVal(barrier_func, invalid_state, valid_ctrl, obs_list, 0.5)
    assert calRelCostChange(cost0, cost1) > 0.8
    is_feasible0 = isFeasible(barrier_func, valid_state, valid_ctrl, obs_list)
    is_feasible1 = isFeasible(barrier_func, invalid_state, valid_ctrl, obs_list)
    assert is_feasible0 and not is_feasible1


def test_barrier_func_with_invalid_a(valid_state, valid_ctrl, barrier_func):
    """测试当加速度大于最大值时的情况"""
    obs_list = np.array([])
    invalid_ctrl = np.copy(valid_ctrl)
    invalid_ctrl[0] = Config.MAX_ACCEL + 0.1
    cost0 = getVal(barrier_func, valid_state, valid_ctrl, obs_list, 0.5)
    cost1 = getVal(barrier_func, valid_state, invalid_ctrl, obs_list, 0.5)
    assert calRelCostChange(cost0, cost1) > 0.8
    is_feasible0 = isFeasible(barrier_func, valid_state, valid_ctrl, obs_list)
    is_feasible1 = isFeasible(barrier_func, valid_state, invalid_ctrl, obs_list)
    assert is_feasible0 and not is_feasible1


def test_barrier_func_with_invalid_delta(valid_state, valid_ctrl, barrier_func):
    """测试当delta角度小于最小值时的情况"""
    obs_list = np.array([])
    invalid_ctrl = np.copy(valid_ctrl)
    invalid_ctrl[1] = -Config.MAX_STEER - 0.1
    cost0 = getVal(barrier_func, valid_state, valid_ctrl, obs_list, 0.5)
    cost1 = getVal(barrier_func, valid_state, invalid_ctrl, obs_list, 0.5)
    assert calRelCostChange(cost0, cost1) > 0.8
    is_feasible0 = isFeasible(barrier_func, valid_state, valid_ctrl, obs_list)
    is_feasible1 = isFeasible(barrier_func, valid_state, invalid_ctrl, obs_list)
    assert is_feasible0 and not is_feasible1
