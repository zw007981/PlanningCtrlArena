# -*- coding: utf-8 -*-

import os
import sys

sys.path.append(os.getcwd())

import time

from src.util.log import Log


class Timer:
    """计时器，提供通过上下文管理器计时和手动调用接口计时两种计时方式"""

    def __init__(self, name=""):
        """使用需要统计的任务名称初始化计时器"""
        self.name = name
        """任务名称"""
        self.time_list = []
        """历史耗时序列"""
        self.start_time = 0.0
        """开始时间"""

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end()

    def start(self):
        """开始计时"""
        self.start_time = time.perf_counter()

    def end(self):
        """结束计时"""
        elapsed_time = time.perf_counter() - self.start_time
        self.time_list.append(elapsed_time)

    def clear(self):
        """清空历史耗时"""
        self.time_list.clear()

    def printAveTime(self):
        """打印平均耗时"""
        ave_time = 0.0
        if len(self.time_list) > 0:
            ave_time = sum(self.time_list) / len(self.time_list)
        name = self.name if self.name != "" else "Task"
        if ave_time < 1e-2:
            Log.info(
                "{} average computation time: {:.2f}ms.".format(name, ave_time * 1000)
            )
        else:
            Log.info("{} average computation time: {:.4f}s.".format(name, ave_time))
