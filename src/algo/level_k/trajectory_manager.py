# -*- coding: utf-8 -*-

import os
import sys

sys.path.append(os.getcwd())


from typing import Dict, List, Tuple

import casadi as ca
import numpy as np

from src.algo.level_k.config import Config
from src.algo.level_k.level import Level
from src.util.log import Log
from src.util.pose import Pose


class TrajectoryManager:
    """轨迹管理器，存储着所有车辆从level-0到level-k的轨迹"""

    def __init__(self, config: Config) -> None:
        self.config = config
        """配置参数"""
        self.trajectories: Dict[Level, Dict[int, ca.Function]] = {}
        """
        存储轨迹的字典，键为level，值为车辆ID到轨迹函数的映射。
        这里的轨迹坐标是车辆后轴中心的坐标。
        """
        self.corrected_trajectories: Dict[Level, Dict[int, ca.Function]] = {}
        """修正原始轨迹后得到以车辆几何中心为基准的轨迹，方便后续的碰撞检测等操作"""
        self.half_wheel_base: float = 0.5 * Config.WHEEL_BASE
        """车辆轴距的一半"""

    def genBaseTrajectories(
        self, poses: List[Pose], get_corrected: bool = False
    ) -> Tuple[ca.DM, ca.DM, ca.DM]:
        """
        level-k算法中生成level-0的轨迹时一般假设周围的车辆都是静态障碍物，
        把这种情况下的轨迹方案称为基础轨迹，输入当前所有车辆的位姿输出基础轨迹。
        如果get_corrected为True，则返回修正后的以车辆几何中心为基准的轨迹。
        """
        num_cars = len(poses)
        cars_x_seq = np.zeros((num_cars, self.config.N))
        cars_y_seq = np.zeros((num_cars, self.config.N))
        cars_theta_seq = np.zeros((num_cars, self.config.N))
        for car_id, pose in enumerate(poses):
            cars_x_seq[car_id, :] = pose.x
            cars_y_seq[car_id, :] = pose.y
            cars_theta_seq[car_id, :] = pose.theta
        if get_corrected:
            for i in range(num_cars):
                delta_x = 0.5 * Config.WHEEL_BASE * np.cos(cars_theta_seq[i, :])
                delta_y = 0.5 * Config.WHEEL_BASE * np.sin(cars_theta_seq[i, :])
                cars_x_seq[i, :] += delta_x
                cars_y_seq[i, :] += delta_y
        return ca.DM(cars_x_seq), ca.DM(cars_y_seq), ca.DM(cars_theta_seq)

    def clear(self) -> None:
        """清空所有轨迹"""
        self.trajectories.clear()
        self.corrected_trajectories.clear()

    def correctTrajectory(
        self, x_seq: ca.DM, y_seq: ca.DM, theta_seq: ca.DM
    ) -> Tuple[ca.DM, ca.DM, ca.DM]:
        """修正原始轨迹，使其以车辆几何中心为基准"""
        delta_x_seq = self.half_wheel_base * ca.cos(theta_seq)
        delta_y_seq = self.half_wheel_base * ca.sin(theta_seq)
        x_center_seq = x_seq + delta_x_seq
        y_center_seq = y_seq + delta_y_seq
        theta_center_seq = theta_seq
        return x_center_seq, y_center_seq, theta_center_seq

    def add(
        self,
        level: Level,
        car_id: int,
        x_seq: ca.DM,
        y_seq: ca.DM,
        theta_seq: ca.DM,
    ) -> None:
        """添加级别为level，ID为car_id的车辆轨迹"""
        if level not in self.trajectories:
            self.trajectories[level] = {}
        traj_func = ca.Function(
            f"traj_{level.name}_car_{car_id}",
            [],
            [x_seq, y_seq, theta_seq],
            [],
            ["x_seq", "y_seq", "theta_seq"],
        )
        self.trajectories[level][car_id] = traj_func
        self.__addCorrectedTrajectory(level, car_id, x_seq, y_seq, theta_seq)

    def get(
        self, level: Level, car_id: int, get_corrected: bool = False
    ) -> Tuple[ca.DM, ca.DM, ca.DM]:
        """
        获取级别为level，ID为car_id的车辆轨迹，默认会返回原始轨迹，
        如果get_corrected为True，则返回修正后的以几何中心为基准的轨迹。
        """
        dictionary = self.corrected_trajectories if get_corrected else self.trajectories
        dict_name = "corrected trajectory" if get_corrected else "trajectory"
        if level not in dictionary or car_id not in dictionary[level]:
            Log.error(
                "No {} found for car {} at level {}!!!".format(dict_name, car_id, level)
            )
            raise ValueError("No trajectory found!!!")
        func = dictionary[level][car_id]
        x_seq, y_seq, theta_seq = func()["x_seq"], func()["y_seq"], func()["theta_seq"]  # type: ignore
        return x_seq, y_seq, theta_seq

    def getAll(
        self, level: Level, get_corrected: bool = False
    ) -> Tuple[ca.DM, ca.DM, ca.DM]:
        """获取所有车辆在level级别的轨迹，在按照车辆ID排序后返回"""
        dictionary = self.corrected_trajectories if get_corrected else self.trajectories
        dict_name = "corrected trajectory" if get_corrected else "trajectory"
        if level not in dictionary or not dictionary[level]:
            Log.error("No {} found at level {}!!!".format(dict_name, level))
            raise ValueError("Empty trajectory found!!!")
        car_ids: List[int] = sorted(dictionary[level].keys())
        # 逐车调用get方法，以收集每辆车的行向量轨迹(1×N)。
        rows_x, rows_y, rows_theta = [], [], []
        for cid in car_ids:
            x_seq, y_seq, theta_seq = self.get(level, cid, get_corrected)
            rows_x.append(x_seq)
            rows_y.append(y_seq)
            rows_theta.append(theta_seq)
        return ca.vcat(rows_x), ca.vcat(rows_y), ca.vcat(rows_theta)

    def __addCorrectedTrajectory(
        self,
        level: Level,
        car_id: int,
        x_seq: ca.DM,
        y_seq: ca.DM,
        theta_seq: ca.DM,
    ) -> None:
        """添加修正后的轨迹"""
        if level not in self.corrected_trajectories:
            self.corrected_trajectories[level] = {}
        x_center_seq, y_center_seq, theta_center_seq = self.correctTrajectory(
            x_seq, y_seq, theta_seq
        )
        corrected_traj_func = ca.Function(
            f"corrected_traj_{level.name}_car_{car_id}",
            [],
            [x_center_seq, y_center_seq, theta_center_seq],
            [],
            ["x_seq", "y_seq", "theta_seq"],
        )
        self.corrected_trajectories[level][car_id] = corrected_traj_func
