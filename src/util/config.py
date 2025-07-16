# -*- coding: utf-8 -*-

import os
import sys

sys.path.append(os.getcwd())

import json
from typing import Any, Dict

import numpy as np


class Config:
    """全局配置"""

    FLOAT_EPS: float = 1e-4
    """浮点数比较时的误差"""
    ZERO_EPS: float = 1e-6
    """如果浮点数的绝对值小于这个值，就认为它是0"""
    STATE_DIM: int = 4
    """状态空间维度"""
    CTRL_DIM: int = 2
    """控制空间维度"""
    B_SPLINE_ORDER: int = 3
    """B样条曲线的阶数"""
    B_SPLINE_INTERVAL_DIST: float = 0.1
    """B样条曲线上点的采样间隔"""
    X_MIN: float = -2.0
    """地图最小x坐标，会根据地图自动调整"""
    X_MAX: float = 10.0
    """地图最大x坐标，会根据地图自动调整"""
    Y_MIN: float = -2.0
    """地图最小y坐标，会根据地图自动调整"""
    Y_MAX: float = 10.0
    """地图最大y坐标，会根据地图自动调整"""
    WHEEL_BASE: float = 0.8
    """车辆轴距"""
    WIDTH: float = 0.48
    """车辆宽度"""
    MAX_STEER: float = np.pi / 3
    """车辆最大转角"""
    MAX_SPEED: float = 3.33
    """车辆最大速度"""
    MAX_ACCEL: float = 6.666
    """车辆最大加速度"""
    INIT_DELTA_T: float = 0.04
    """初始模拟时间间隔"""
    DELTA_T: float = INIT_DELTA_T
    """模拟时间间隔，可以按下加速或减速键调整"""
    KINEMATIC_MODEL_TYPE: str = "BICYCLE_MODEL"
    """车辆运动学模型"""
    LOG_DIR: str = "log"
    """日志文件目录"""
    FIG_DIR: str = "fig"
    """图像文件目录"""

    @staticmethod
    def GetConfigFolder(max_levels_up=3):
        """
        返回配置文件夹的路径。

        参数:
        max_levels_up (int): 向上查找的最大层数，默认为 3。

        返回:
        str: config文件夹的路径。

        异常:
        FileNotFoundError: 如果在指定层数内未找到"config"文件夹。
        """
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        # 从当前目录开始向上查找
        for level in range(max_levels_up + 1):
            # 构建当前层级的路径
            level_dir = (
                os.path.join(cur_dir, *(("..",) * level)) if level > 0 else cur_dir
            )
            config_path = os.path.join(level_dir, "config")
            if os.path.isdir(config_path):
                return config_path
        # 如果在指定层数内未找到，则抛出异常
        raise FileNotFoundError(
            f"The config folder was not found within {max_levels_up} levels up from {cur_dir}."
        )

    @staticmethod
    def GetConfigVars(config) -> Dict[str, Any]:
        """获取config类中所有可配置参数"""
        config_vars = {}
        # 1) 类属性
        cls = config if isinstance(config, type) else type(config)
        for c in cls.__mro__:
            for k, v in c.__dict__.items():
                if k.isupper() and not k.startswith("_") and not callable(v):
                    config_vars[k] = v
        # 2) 实例属性
        if not isinstance(config, type):
            for k, v in config.__dict__.items():
                if k.isupper() and not k.startswith("_") and not callable(v):
                    config_vars[k] = v
        return config_vars

    @staticmethod
    def UpdateFromJson(config, file_name: str) -> None:
        """利用json文件中的配置信息更新同名配置项"""
        from src.util.log import Log

        file_path = os.path.join(Config.GetConfigFolder(), file_name)
        config_vars = Config.GetConfigVars(config)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                Log.info("Loading config file {}.".format(file_path))
                for key, value in json.load(f).items():
                    key_upper = key.upper()
                    if key_upper in config_vars:
                        cur_val = getattr(config, key_upper)
                        if not isinstance(cur_val, type(value)):
                            continue
                        elif isinstance(cur_val, (int, float)):
                            if abs(float(cur_val) - float(value)) <= Config.FLOAT_EPS:
                                continue
                        elif cur_val == value:
                            continue
                        setattr(config, key_upper, value)
                        config_vars[key_upper] = value
                        Log.info("{} updated.".format(key))
            if hasattr(config, "init") and callable(getattr(config, "init")):
                config.init()
        except Exception as e:
            Log.warning(
                "Failed to load config file {} with error: {}!".format(file_path, e)
            )

    @staticmethod
    def PrintConfig(config) -> None:
        """打印config类中所有可配置参数"""
        from src.util.log import Log

        config_vars = Config.GetConfigVars(config)
        Log.info("Config: {}".format(json.dumps(config_vars, ensure_ascii=False)))
