# -*- coding: utf-8 -*-

import os
import sys

sys.path.append(os.getcwd())

import copy
from typing import Any, Dict, List

from matplotlib import pyplot as plt
from numpy.typing import NDArray

from src.algo.level_k.level import Level
from src.kinematic_model.base_kinematic_model import BaseKinematicModel
from src.kinematic_model.ctrl_input import CtrlInput
from src.util.config import Config
from src.util.log import Log
from src.util.pose import Pose


class Car:
    """车辆"""

    def __init__(
        self,
        car_id: str,
        pose: Pose,
        v: float = 0.0,
        kinematic_model_type: str = "BICYCLE_MODEL",
        tgt_ref_line_id: int = 0,
        level: Level = Level.LEVEL_0,
    ):
        self.id: str = car_id
        self.model = BaseKinematicModel.constructKinematicModel(
            kinematic_model_type, pose, v
        )
        """车辆动力学模型"""
        self.start_pose: Pose = copy.deepcopy(pose)
        """车辆初始位置"""
        self.start_v: float = v
        """车辆初始速度"""
        self.ref_pt: Pose = Pose()
        """对应的参考点"""
        self.time_sequence: List[float] = [0.0]
        """时间序列"""
        self.trajectory: List[Pose] = [copy.deepcopy(pose)]
        """车辆轨迹"""
        self.dist_to_ref_pts: List[float] = []
        """车辆到参考点的距离"""
        self.vel_history: List[float] = [v]
        """车辆速度历史"""
        self.ctrl_input_history: List[NDArray] = []
        """
        车辆控制输入历史，如为自行车模型第一个元素为加速度，第二个元素为前轮转角。
        如为质点模型第一个元素为x方向上的移动距离，第二个元素为y方向上的移动距离。
        """
        self.tgt_ref_line_id: int = tgt_ref_line_id
        """车辆对应的参考线ID"""
        self.level: Level = level
        """车辆的level-k层级，只用于level-k算法中"""
        Log.info(
            "Init car {} with {} model at {}.".format(
                car_id, kinematic_model_type, pose
            )
        )

    @property
    def pose(self) -> Pose:
        """车辆当前位置"""
        return self.model.pose

    @property
    def v(self) -> float:
        """车辆当前速度"""
        return self.model.v

    def updateState(self, ctrl_input: CtrlInput):
        """更新车辆状态"""
        self.model.step(ctrl_input)
        self.time_sequence.append(self.time_sequence[-1] + Config.DELTA_T)
        self.trajectory.append(copy.deepcopy(self.model.pose))
        self.dist_to_ref_pts.append(self.model.pose.calDistance(self.ref_pt))
        self.vel_history.append(self.model.v)
        self.ctrl_input_history.append(ctrl_input.val)

    def updateRefPt(self, ref_pt: Pose):
        """更新参考点"""
        self.ref_pt = copy.deepcopy(ref_pt)

    def resetState(self):
        """重置车辆状态"""
        Log.debug("Reset car {} state.".format(self.id))
        self.model.setCurState(self.start_pose, self.start_v)
        self.time_sequence = [0.0]
        self.trajectory = [copy.deepcopy(self.start_pose)]
        self.dist_to_ref_pts = []
        self.vel_history = [self.start_v]
        self.ctrl_input_history = []
        self.ref_pt = Pose()

    def genMotionData(self) -> Dict[str, Any]:
        """生成运动数据，方便后续统一保存到json文件中"""
        motion_data: Dict[str, Any] = {}
        motion_data["id"] = self.id
        motion_data["time"] = [round(t, 4) for t in self.time_sequence]
        motion_data["x"] = [round(pose.x, 4) for pose in self.trajectory]
        motion_data["y"] = [round(pose.y, 4) for pose in self.trajectory]
        motion_data["theta"] = [round(pose.theta, 4) for pose in self.trajectory]
        motion_data["v"] = [round(v, 4) for v in self.vel_history]
        motion_data["a"] = [round(ctrl[0], 4) for ctrl in self.ctrl_input_history]
        motion_data["delta"] = [round(ctrl[1], 4) for ctrl in self.ctrl_input_history]
        return motion_data
