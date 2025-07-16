# -*- coding: utf-8 -*-
import os
import sys

sys.path.append(os.getcwd())

import casadi as ca
import numpy as np
import pytest

from src.algo.level_k.config import Config
from src.algo.level_k.control_sequence_manager import ControlSequenceManager
from src.algo.level_k.level import Level


@pytest.fixture
def ctrl_seq_mgr() -> ControlSequenceManager:
    """创建控制序列管理器实例"""
    config = Config()
    return ControlSequenceManager(config)


def test_control_sequence_manager_init(
    ctrl_seq_mgr: ControlSequenceManager,
):
    """测试控制序列管理器初始化"""
    assert isinstance(ctrl_seq_mgr.ctrl_seqs, dict)
    assert isinstance(ctrl_seq_mgr.def_ctrl_seq, ca.DM)
    # 验证默认控制序列的维度
    assert ctrl_seq_mgr.def_ctrl_seq.shape == (
        Config.CTRL_DIM,
        ctrl_seq_mgr.config.N - 1,
    )
    # 验证默认控制序列的内容（第一行为最大加速度，第二行为0）
    assert np.allclose(ctrl_seq_mgr.def_ctrl_seq[0, :].full(), Config.MAX_ACCEL)
    assert np.allclose(ctrl_seq_mgr.def_ctrl_seq[1, :].full(), 0.0)


def test_add_control_sequence(ctrl_seq_mgr: ControlSequenceManager):
    """测试添加控制序列功能"""
    # 创建测试控制序列
    ctrl_seq = ca.DM.ones(Config.CTRL_DIM, ctrl_seq_mgr.config.N - 1) * 2.0
    car_id = 0
    level = Level.LEVEL_0
    # 添加控制序列
    ctrl_seq_mgr.add(level, car_id, ctrl_seq)
    # 验证存储结构
    assert level in ctrl_seq_mgr.ctrl_seqs
    assert car_id in ctrl_seq_mgr.ctrl_seqs[level]
    assert isinstance(ctrl_seq_mgr.ctrl_seqs[level][car_id], ca.Function)


def test_get_control_sequence(ctrl_seq_mgr: ControlSequenceManager):
    """测试获取控制序列功能"""
    # 创建并添加测试控制序列
    ctrl_seq = ca.DM([[1.0, 2.0, 3.0], [0.1, 0.2, 0.3]])
    car_id = 1
    level = Level.LEVEL_1
    ctrl_seq_mgr.add(level, car_id, ctrl_seq)
    # 获取控制序列（非热启动模式）
    retrieved_seq = ctrl_seq_mgr.get(level, car_id, for_warm_start=False)
    assert np.allclose(retrieved_seq.full(), ctrl_seq.full())


def test_get_default_control_sequence(ctrl_seq_mgr: ControlSequenceManager):
    """测试获取不存在的控制序列时返回默认值"""
    default_seq = ctrl_seq_mgr.get(Level.LEVEL_0, 999)
    assert np.allclose(default_seq.full(), ctrl_seq_mgr.def_ctrl_seq.full())


def test_get_warm_start_control_sequence(
    ctrl_seq_mgr: ControlSequenceManager,
):
    """测试获取热启动控制序列功能"""
    # 创建测试控制序列
    N = 5
    ctrl_seq = ca.DM([[1.0, 2.0, 3.0, 4.0, 5.0], [0.1, 0.2, 0.3, 0.4, 0.5]])
    car_id = 2
    level = Level.LEVEL_2
    ctrl_seq_mgr.add(level, car_id, ctrl_seq)
    # 获取热启动控制序列
    warm_start_seq = ctrl_seq_mgr.get(level, car_id, for_warm_start=True)
    # 验证维度保持不变
    assert warm_start_seq.shape == ctrl_seq.shape
    # 验证前N-1个元素是原序列的后N-1个元素
    assert np.allclose(warm_start_seq[:, : N - 1].full(), ctrl_seq[:, 1:].full())
    # 验证最后一个元素是原序列的最后一个元素
    assert np.allclose(warm_start_seq[:, -1].full(), ctrl_seq[:, -1].full())


def test_get_warm_start_with_single_element(
    ctrl_seq_mgr: ControlSequenceManager,
):
    """测试单元素控制序列的热启动处理"""
    # 创建只有一个元素的控制序列
    ctrl_seq = ca.DM([[1.0], [0.1]])
    car_id = 3
    level = Level.LEVEL_0
    ctrl_seq_mgr.add(level, car_id, ctrl_seq)
    # 获取热启动控制序列（应该返回原序列）
    warm_start_seq = ctrl_seq_mgr.get(level, car_id, for_warm_start=True)
    assert np.allclose(warm_start_seq.full(), ctrl_seq.full())


def test_clear_control_sequences(ctrl_seq_mgr: ControlSequenceManager):
    """测试清空所有控制序列"""
    # 添加多个控制序列
    for level in [Level.LEVEL_0, Level.LEVEL_1]:
        for car_id in range(3):
            ctrl_seq = ca.DM.ones(Config.CTRL_DIM, ctrl_seq_mgr.config.N - 1) * (
                car_id + 1
            )
            ctrl_seq_mgr.add(level, car_id, ctrl_seq)
    assert len(ctrl_seq_mgr.ctrl_seqs) > 0
    ctrl_seq_mgr.clear()
    # 验证控制序列已清空
    assert len(ctrl_seq_mgr.ctrl_seqs) == 0
    # 验证获取控制序列时返回默认值
    default_seq = ctrl_seq_mgr.get(Level.LEVEL_0, 0)
    assert np.allclose(default_seq.full(), ctrl_seq_mgr.def_ctrl_seq.full())


def test_multiple_levels_and_cars(ctrl_seq_mgr: ControlSequenceManager):
    """测试管理多个层级和车辆的控制序列"""
    # 准备测试数据
    test_data = []
    for level in [Level.LEVEL_0, Level.LEVEL_1, Level.LEVEL_2]:
        for car_id in range(3):
            # 创建独特的控制序列
            ctrl_seq = ca.DM(
                [
                    [
                        float(level.value * 10 + car_id + i)
                        for i in range(ctrl_seq_mgr.config.N - 1)
                    ],
                    [
                        float(level.value + car_id * 0.1 + i * 0.01)
                        for i in range(ctrl_seq_mgr.config.N - 1)
                    ],
                ]
            )
            test_data.append((level, car_id, ctrl_seq))
            ctrl_seq_mgr.add(level, car_id, ctrl_seq)
    # 验证所有控制序列
    for level, car_id, expected_seq in test_data:
        retrieved_seq = ctrl_seq_mgr.get(level, car_id)
        assert np.allclose(retrieved_seq.full(), expected_seq.full())


def test_overwrite_control_sequence(ctrl_seq_mgr: ControlSequenceManager):
    """测试覆盖已存在的控制序列"""
    car_id = 0
    level = Level.LEVEL_0
    # 添加初始控制序列
    ctrl_seq1 = ca.DM.ones(Config.CTRL_DIM, ctrl_seq_mgr.config.N - 1)
    ctrl_seq_mgr.add(level, car_id, ctrl_seq1)
    # 覆盖控制序列
    ctrl_seq2 = ca.DM.ones(Config.CTRL_DIM, ctrl_seq_mgr.config.N - 1) * 2.0
    ctrl_seq_mgr.add(level, car_id, ctrl_seq2)
    # 验证获取的是新的控制序列
    retrieved_seq = ctrl_seq_mgr.get(level, car_id)
    assert np.allclose(retrieved_seq.full(), ctrl_seq2.full())


def test_warm_start_sequence_properties(
    ctrl_seq_mgr: ControlSequenceManager,
):
    """测试热启动序列的详细属性"""
    # 创建一个递增的控制序列
    N = 10
    v_seq = np.arange(1, N + 1, dtype=float)
    delta_seq = np.arange(0.1, 0.1 * (N + 1), 0.1)
    ctrl_seq = ca.DM([v_seq, delta_seq])
    car_id = 5
    level = Level.LEVEL_0
    ctrl_seq_mgr.add(level, car_id, ctrl_seq)
    # 获取热启动序列
    warm_start_seq = ctrl_seq_mgr.get(level, car_id, for_warm_start=True)
    # 详细验证热启动序列
    # 第一行（速度）：[2, 3, 4, 5, 6, 7, 8, 9, 10, 10]
    expected_v = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 10])
    assert np.allclose(warm_start_seq[0, :].full().flatten(), expected_v)
    # 第二行（转向角）：[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0]
    expected_delta = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0])
    assert np.allclose(warm_start_seq[1, :].full().flatten(), expected_delta)
