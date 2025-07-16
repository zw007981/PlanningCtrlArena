# -*- coding: utf-8 -*-

import os
import sys

sys.path.append(os.getcwd())

from typing import List, Tuple

import casadi as ca
import numpy as np
from numpy.typing import NDArray
from PySide6 import QtCore
from PySide6.QtWidgets import QApplication

from src.algo.level_k.config import Config
from src.algo.level_k.control_sequence_manager import ControlSequenceManager
from src.algo.level_k.iLQR import iLQR
from src.algo.level_k.level import Level
from src.algo.level_k.trajectory_manager import TrajectoryManager
from src.traffic_simulator.color import *
from src.traffic_simulator.traffic_simulator import TrafficSimulator
from src.util.log import Log
from src.util.pose import Pose


class LevelK:
    """Level-k算法"""

    def __init__(self, config: Config, num_cars: int) -> None:
        self.config: Config = config
        """配置"""
        self.num_cars: int = num_cars
        """需要规划的车辆数量"""
        self.planner: iLQR = iLQR(self.config, num_cars=num_cars)
        """iLQR求解器，负责规划处于level-k的车辆的轨迹"""
        self.traj_mgr: TrajectoryManager = TrajectoryManager(self.config)
        """轨迹管理器，存储着所有车辆从level-0到level-k的轨迹"""
        self.ctrl_seq_mgr: ControlSequenceManager = ControlSequenceManager(self.config)
        """控制序列管理器，存储着所有车辆从level-0到level-k的控制序列"""
        self.ego_level: Level = Level.ADAPTIVE
        """自车的level"""
        self.max_level: Level = Level.LEVEL_0
        """需要计算的最大level"""
        self.level_list: List[Level] = []
        """车辆id到level的映射"""
        self.car_id_to_x_init: List[NDArray[np.float64]] = []
        """车辆id到起始状态的映射"""
        self.car_id_to_x_seq_ref: List[ca.DM] = []
        """车辆id到参考轨迹的映射"""

    def solve(
        self,
        init_x_list: List[NDArray[np.float64]],
        x_seq_ref_list: List[NDArray[np.float64]],
        level_list: List[Level],
    ) -> Tuple[List[NDArray], List[NDArray]]:
        """
        输入num_cars台车辆的起始状态、各自的参考轨迹和level，
        输出各车的预期轨迹和最优控制输入序列。
        """
        self.__initSolver(init_x_list, x_seq_ref_list, level_list)
        self.__planForLevel0()
        cur_level: Level = Level.LEVEL_0
        while cur_level.value < self.max_level.value:
            cur_level = Level.PlusOne(cur_level)
            self.__planForLevelK(cur_level)
        if self.ego_level == Level.ADAPTIVE:
            self.__planForAdaptiveEgoCar()
        return self.__constructFinalPlan()

    def __initSolver(
        self,
        init_x_list: List[NDArray[np.float64]],
        x_seq_ref_list: List[NDArray[np.float64]],
        level_list: List[Level],
    ) -> None:
        """根据此时需求解的问题初始化求解器"""
        self.ego_level = level_list[0]
        self.max_level = Level.GetMaxLevel(level_list)
        self.level_list = level_list
        self.traj_mgr.clear()
        self.car_id_to_x_init = init_x_list
        self.car_id_to_x_seq_ref = []
        for i in range(self.num_cars):
            self.car_id_to_x_seq_ref.append(ca.DM(x_seq_ref_list[i]))

    def __planForLevel0(self) -> None:
        """把所有车辆视为level-0，为它们规划轨迹"""
        init_pose_list = [Pose(x[0], x[1], x[2]) for x in self.car_id_to_x_init]
        base_cars_x_seq, base_cars_y_seq, base_cars_theta_seq = (
            self.traj_mgr.genBaseTrajectories(init_pose_list, get_corrected=True)
        )
        for i in range(self.num_cars):
            x_nominal, u_nominal = self.planner.plan(
                i,
                Level.LEVEL_0.value,
                self.car_id_to_x_init[i],
                self.car_id_to_x_seq_ref[i],
                self.ctrl_seq_mgr.get(Level.LEVEL_0, i, for_warm_start=True),
                base_cars_x_seq,
                base_cars_y_seq,
            )
            self.traj_mgr.add(
                Level.LEVEL_0,
                i,
                x_nominal[0, :],
                x_nominal[1, :],
                x_nominal[2, :],
            )
            self.ctrl_seq_mgr.add(Level.LEVEL_0, i, u_nominal)

    def __planForLevelK(self, k: Level) -> None:
        """把所有车辆视为level-k，为它们规划轨迹"""
        if k == Level.LEVEL_0:
            Log.error("Cannot plan for level-0!!!")
            raise ValueError("Cannot plan for level-0!!!")
        lower_plan = self.traj_mgr.getAll(Level.MinusOne(k), get_corrected=True)
        for i in range(self.num_cars):
            x_nominal, u_nominal = self.planner.plan(
                i,
                k.value,
                self.car_id_to_x_init[i],
                self.car_id_to_x_seq_ref[i],
                self.ctrl_seq_mgr.get(k, i, for_warm_start=True),
                lower_plan[0],
                lower_plan[1],
            )
            self.traj_mgr.add(
                k,
                i,
                x_nominal[0, :],
                x_nominal[1, :],
                x_nominal[2, :],
            )
            self.ctrl_seq_mgr.add(k, i, u_nominal)

    def __planForAdaptiveEgoCar(self) -> None:
        """如果自车的level是ADAPTIVE，最后规划自车轨迹"""
        # 考虑到周围车辆的level可能不固定，逐车调用get方法拼凑出它们的预期轨迹
        rows_x, rows_y = [], []
        for car_id in range(self.num_cars):
            level = Level.LEVEL_0
            if car_id != 0:
                level = self.level_list[car_id]
            x_seq, y_seq, theta_seq = self.traj_mgr.get(
                level, car_id, get_corrected=False
            )
            rows_x.append(x_seq)
            rows_y.append(y_seq)
        cars_x_seq = ca.vcat(rows_x)
        cars_y_seq = ca.vcat(rows_y)
        x_nominal, u_nominal = self.planner.plan(
            0,
            Level.ADAPTIVE.value,
            self.car_id_to_x_init[0],
            self.car_id_to_x_seq_ref[0],
            self.ctrl_seq_mgr.get(Level.ADAPTIVE, 0, for_warm_start=True),
            cars_x_seq,
            cars_y_seq,
        )
        self.traj_mgr.add(
            Level.ADAPTIVE,
            0,
            x_nominal[0, :],
            x_nominal[1, :],
            x_nominal[2, :],
        )
        self.ctrl_seq_mgr.add(Level.ADAPTIVE, 0, u_nominal)

    def __constructFinalPlan(self) -> Tuple[List[NDArray], List[NDArray]]:
        """规划完成后根据各车的level构造它们的最终轨迹和最优控制输入序列"""
        nominal_trajectories: List[NDArray] = []
        nominal_ctrl_seqs: List[NDArray] = []
        for car_id, level in enumerate(self.level_list):
            x_seq, y_seq, theta_seq = self.traj_mgr.get(
                level, car_id, get_corrected=False
            )
            trajectory = ca.vertcat(x_seq, y_seq, theta_seq)
            nominal_trajectories.append(trajectory.full())
            ctrl_seq = self.ctrl_seq_mgr.get(level, car_id, for_warm_start=False)
            nominal_ctrl_seqs.append(ctrl_seq.full())
        return nominal_trajectories, nominal_ctrl_seqs


class LevelKDemo(TrafficSimulator):
    """level-k算法演示类"""

    def __init__(self, config_name="config_level_k.json") -> None:
        super().__init__(config_name)
        self.config = Config()
        self.config.PrintConfig(self.config)
        self.solver: LevelK = LevelK(self.config, num_cars=self.data_mgr.equip_mgr.size)
        """level-k算法求解器"""
        self.x_seq_ref_list: List[NDArray] = []
        """所有车辆的参考状态序列列表"""
        self.graphic_item_mgr.addExtraCurveItem(
            "reference trajectory", BLUE, QtCore.Qt.PenStyle.DotLine
        )
        """自车参考轨迹图形项"""
        self.graphic_item_mgr.addExtraCurveItem(
            "predicted trajectory", RED, QtCore.Qt.PenStyle.DotLine
        )
        """自车预测轨迹图形项"""

    def update(self) -> None:
        car_id_list = self.data_mgr.car_id_list
        all_cars_arrived_dest = True
        for car_id in car_id_list:
            if self.getDistToDest(car_id) > 2.33 * Config.WHEEL_BASE:
                all_cars_arrived_dest = False
                break
        if all_cars_arrived_dest:
            Log.info("All cars have arrived at the destination.")
            self.finalize()
        else:
            self.__updateRefPtAndStateSeq()
            init_x_list: List[NDArray] = []
            level_list: List[Level] = []
            for car_id in car_id_list:
                car = self.data_mgr.equip_mgr.get(car_id)
                init_x_list.append(car.model.state)
                level_list.append(car.level)
            nominal_trajs, nominal_ctrl_seqs = self.solver.solve(
                init_x_list, self.x_seq_ref_list, level_list
            )
            for i, car_id in enumerate(car_id_list):
                nominal_ctrl_seq = nominal_ctrl_seqs[i]
                if self.getDistToDest(car_id) < 2.33 * Config.WHEEL_BASE:
                    # 如果快到达目的地，以最快的速度停下来
                    if (
                        self.data_mgr.equip_mgr.get(car_id).model.v
                        > 1.2 * Config.MAX_ACCEL * Config.DELTA_T
                    ):
                        self.getCtrlInput(car_id).setVal(-Config.MAX_ACCEL, 0.0)
                    else:
                        self.getCtrlInput(car_id).setVal(0.0, 0.0)
                else:
                    self.getCtrlInput(car_id).setVal(
                        nominal_ctrl_seq[0][0], nominal_ctrl_seq[1][0]
                    )
            self.__updateGraphicItems(nominal_trajs[0])

    def __genStateRefForCar(self, car_id: str) -> NDArray:
        """为指定车辆生成参考状态序列"""
        car_pose_list = [
            self.getCar(other_car_id).pose
            for other_car_id in self.data_mgr.car_id_list
            if other_car_id != car_id
        ]
        x_seq_ref = np.zeros((Config.STATE_DIM, self.config.N))
        ref_pt = self.getRefPt(car_id)
        ref_line = self.getRefLine(car_id)
        for i in range(self.config.N):
            v_ref = ref_pt.max_speed_rate * Config.MAX_SPEED
            if ref_line.isCloseToDestination(ref_pt):
                v_ref = 0.0
            is_close_to_other_car = False
            for pose in car_pose_list:
                if ref_pt.calDistance(pose) < self.config.WHEEL_BASE:
                    is_close_to_other_car = True
                    break
            if is_close_to_other_car:
                v_ref = 0.0
            dist = v_ref * Config.DELTA_T
            x_seq_ref[0, i] = ref_pt.x
            x_seq_ref[1, i] = ref_pt.y
            x_seq_ref[2, i] = ref_pt.theta
            x_seq_ref[3, i] = v_ref
            ref_pt = ref_line.getPtAfterDistance(ref_pt.id, dist)
        return x_seq_ref

    def __updateRefPtAndStateSeq(self):
        """更新参考点和参考状态序列"""
        self.x_seq_ref_list = []
        for car_id in self.data_mgr.car_id_list:
            x_seq_ref = self.__genStateRefForCar(car_id)
            self.x_seq_ref_list.append(x_seq_ref)

    def __updateGraphicItems(self, nominal_traj: NDArray) -> None:
        """更新繁杂的图像显示，分离出来避免干扰算法主逻辑"""
        state_seq_ref = self.x_seq_ref_list[0].T
        ref_pt = self.getRefPt()
        state_seq_ref[0] = [ref_pt.x, ref_pt.y, ref_pt.theta, 0.0]
        self.graphic_item_mgr.setExtraCurveData("reference trajectory", state_seq_ref)
        self.graphic_item_mgr.setExtraCurveData("predicted trajectory", nominal_traj.T)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    demo = LevelKDemo("config_level_k.json")
    demo.show()
    app.exec()
