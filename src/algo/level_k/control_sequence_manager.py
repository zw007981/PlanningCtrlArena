# -*- coding: utf-8 -*-

import os
import sys

sys.path.append(os.getcwd())


from typing import Dict

import casadi as ca
import numpy as np

from src.algo.level_k.config import Config
from src.algo.level_k.level import Level


class ControlSequenceManager:
    """
    控制输入序列管理器，存储着所有车辆从level-0到level-k的控制输入序列，
    可以把上一时间步的控制输入作为这一时间步的热启动输入。
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        """配置参数"""
        def_ctrl_seq = np.zeros((Config.CTRL_DIM, self.config.N - 1))
        for i in range(self.config.N - 1):
            def_ctrl_seq[0, i] = Config.MAX_ACCEL
        self.def_ctrl_seq = ca.DM(def_ctrl_seq)
        """默认的控制输入序列，假设车辆保持最大加速度直线行驶"""
        self.ctrl_seqs: Dict[Level, Dict[int, ca.Function]] = {}
        """存储控制输入序列的字典，键为level，值为车辆ID到控制输入函数的映射"""

    def clear(self) -> None:
        """清空所有控制输入序列"""
        self.ctrl_seqs.clear()

    def add(self, level: Level, car_id: int, ctrl_seq: ca.DM) -> None:
        """添加控制输入序列"""
        if level not in self.ctrl_seqs:
            self.ctrl_seqs[level] = {}
        ctrl_func = ca.Function(
            f"ctrl_seq_{level.name}_{car_id}",
            [],
            [ctrl_seq],
            [],
            ["ctrl_seq"],
        )
        self.ctrl_seqs[level][car_id] = ctrl_func

    def get(self, level: Level, car_id: int, for_warm_start: bool = False) -> ca.DM:
        """获取对应控制输入序列，如果不存在会返回一个默认的，如果是为了热启动则会抛掉第一个控制输入"""
        if level not in self.ctrl_seqs or car_id not in self.ctrl_seqs[level]:
            return self.def_ctrl_seq
        func = self.ctrl_seqs[level][car_id]
        ctrl_seq = func()["ctrl_seq"]  # type: ignore
        if for_warm_start and ctrl_seq.shape[1] > 1:
            seq_shifted = ctrl_seq[:, 1:]
            last_term = ctrl_seq[:, -1]
            last_term = ca.reshape(last_term, Config.CTRL_DIM, 1)
            seq_warm_start = ca.horzcat(seq_shifted, last_term)
            return seq_warm_start
        else:
            return ctrl_seq
