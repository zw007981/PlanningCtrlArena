# -*- coding: utf-8 -*-
import os
import sys

sys.path.append(os.getcwd())

import casadi as ca
import numpy as np
import pytest

from src.algo.level_k.config import Config
from src.algo.level_k.level import Level
from src.algo.level_k.trajectory_manager import TrajectoryManager
from src.util.pose import Pose


@pytest.fixture
def traj_mgr() -> TrajectoryManager:
    """创建轨迹管理器实例"""
    config = Config()
    traj_mgr = TrajectoryManager(config)
    # 为了测试方便，设置半轴距为1.0
    traj_mgr.half_wheel_base = 1.0
    return traj_mgr


def test_trajectory_manager_init(traj_mgr: TrajectoryManager):
    """测试轨迹管理器初始化"""
    assert isinstance(traj_mgr.trajectories, dict)
    assert isinstance(traj_mgr.corrected_trajectories, dict)


def test_gen_base_trajectories(traj_mgr: TrajectoryManager):
    """测试基础轨迹生成"""
    # 创建测试位姿，对应三辆车
    poses = [
        Pose(x=1.0, y=2.0, theta=np.pi / 4),
        Pose(x=3.0, y=4.0, theta=np.pi / 2),
        Pose(x=5.0, y=6.0, theta=0),
    ]
    num_cars = len(poses)
    cars_x_seq, cars_y_seq, cars_theta_seq = traj_mgr.genBaseTrajectories(poses)
    # 验证输出维度
    assert cars_x_seq.shape == (num_cars, traj_mgr.config.N)
    assert cars_y_seq.shape == (num_cars, traj_mgr.config.N)
    assert cars_theta_seq.shape == (num_cars, traj_mgr.config.N)
    # 验证轨迹内容
    for i, pose in enumerate(poses):
        assert np.allclose(cars_x_seq[i, :].full(), pose.x)
        assert np.allclose(cars_y_seq[i, :].full(), pose.y)
        assert np.allclose(cars_theta_seq[i, :].full(), pose.theta)


def test_add_trajectory(traj_mgr: TrajectoryManager):
    """测试添加轨迹功能"""
    N = traj_mgr.config.N
    x_seq = ca.DM.ones(1, N)
    y_seq = ca.DM.ones(1, N) * 2
    theta_seq = ca.DM.ones(1, N) * np.pi / 4
    car_id = 0
    level = Level.LEVEL_0
    traj_mgr.add(level, car_id, x_seq, y_seq, theta_seq)
    # 验证原始轨迹
    assert level in traj_mgr.trajectories
    assert car_id in traj_mgr.trajectories[level]
    orig_x, orig_y, orig_theta = traj_mgr.get(level, car_id)
    assert np.allclose(orig_x.full(), x_seq.full())
    assert np.allclose(orig_y.full(), y_seq.full())
    assert np.allclose(orig_theta.full(), theta_seq.full())
    # 验证修正后的轨迹
    assert level in traj_mgr.corrected_trajectories
    assert car_id in traj_mgr.corrected_trajectories[level]
    center_x, center_y, center_theta = traj_mgr.get(level, car_id, get_corrected=True)
    # 验证几何中心轨迹的计算
    expected_delta_x = traj_mgr.half_wheel_base * ca.cos(theta_seq)
    expected_delta_y = traj_mgr.half_wheel_base * ca.sin(theta_seq)
    assert np.allclose(center_x.full(), (x_seq + expected_delta_x).full())
    assert np.allclose(center_y.full(), (y_seq + expected_delta_y).full())
    assert np.allclose(center_theta.full(), theta_seq.full())


def test_get_trajectory_error(traj_mgr: TrajectoryManager):
    """测试获取不存在轨迹时的错误处理"""
    # 尝试获取不存在的轨迹
    with pytest.raises(ValueError, match="No trajectory found"):
        traj_mgr.get(Level.LEVEL_0, 0)
    with pytest.raises(ValueError, match="No trajectory found"):
        traj_mgr.get(Level.LEVEL_0, 0, get_corrected=True)


def test_multiple_trajectories(traj_mgr: TrajectoryManager):
    """测试添加多个不同层级和车辆的轨迹"""
    trajectories_data = [
        (
            Level.LEVEL_0,
            0,
            ca.DM.ones(1, traj_mgr.config.N),
            ca.DM.ones(1, traj_mgr.config.N) * 2,
            ca.DM.ones(1, traj_mgr.config.N),
        ),
        (
            Level.LEVEL_1,
            1,
            ca.DM.ones(1, traj_mgr.config.N) * 3,
            ca.DM.ones(1, traj_mgr.config.N) * 4,
            ca.DM.ones(1, traj_mgr.config.N) * np.pi / 2,
        ),
    ]
    for level, car_id, x_seq, y_seq, theta_seq in trajectories_data:
        traj_mgr.add(level, car_id, x_seq, y_seq, theta_seq)
    for level, car_id, x_seq, y_seq, theta_seq in trajectories_data:
        orig_x, orig_y, orig_theta = traj_mgr.get(level, car_id)
        assert np.allclose(orig_x.full(), x_seq.full())
        assert np.allclose(orig_y.full(), y_seq.full())
        assert np.allclose(orig_theta.full(), theta_seq.full())


def test_get_all_trajectories(traj_mgr: TrajectoryManager):
    """测试获取所有轨迹的功能"""
    test_data = [
        (
            Level.LEVEL_0,
            0,
            ca.DM.ones(1, traj_mgr.config.N),
            ca.DM.ones(1, traj_mgr.config.N) * 2,
            ca.DM.ones(1, traj_mgr.config.N) * np.pi / 4,
        ),
        (
            Level.LEVEL_0,
            1,
            ca.DM.ones(1, traj_mgr.config.N) * 3,
            ca.DM.ones(1, traj_mgr.config.N) * 4,
            ca.DM.ones(1, traj_mgr.config.N) * np.pi / 2,
        ),
        (
            Level.LEVEL_0,
            2,
            ca.DM.ones(1, traj_mgr.config.N) * 5,
            ca.DM.ones(1, traj_mgr.config.N) * 6,
            ca.DM.ones(1, traj_mgr.config.N) * np.pi,
        ),
    ]
    for level, car_id, x_seq, y_seq, theta_seq in test_data:
        traj_mgr.add(level, car_id, x_seq, y_seq, theta_seq)
    cars_x_seq, cars_y_seq, cars_theta_seq = traj_mgr.getAll(Level.LEVEL_0)
    # 验证返回矩阵的形状
    assert cars_x_seq.shape == (3, traj_mgr.config.N)
    assert cars_y_seq.shape == (3, traj_mgr.config.N)
    assert cars_theta_seq.shape == (3, traj_mgr.config.N)
    # 验证数据内容
    expected_order = [0, 1, 2]
    for i, car_id in enumerate(expected_order):
        x_seq, y_seq, theta_seq = traj_mgr.get(Level.LEVEL_0, car_id)
        assert np.allclose(cars_x_seq[i, :].full(), x_seq.full())
        assert np.allclose(cars_y_seq[i, :].full(), y_seq.full())
        assert np.allclose(cars_theta_seq[i, :].full(), theta_seq.full())
