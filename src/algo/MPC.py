# -*- coding: utf-8 -*-

import os
import sys

sys.path.append(os.getcwd())

from typing import Any, Dict, List, Tuple

import casadi as ca
import numpy as np
from PySide6 import QtCore
from PySide6.QtWidgets import QApplication

from src.kinematic_model.base_kinematic_model import BaseKinematicModel
from src.map.obstacle import Obstacle
from src.traffic_simulator.color import *
from src.traffic_simulator.traffic_simulator import TrafficSimulator
from src.util.config import Config as BaseConfig
from src.util.log import Log


class Config(BaseConfig):
    """MPC控制器配置"""

    N = 12
    """预测步数"""
    X_ERROR_WEIGHT = 1.0
    """x方向上的误差权重"""
    Y_ERROR_WEIGHT = 1.0
    """y方向上的误差权重"""
    THETA_ERROR_WEIGHT = 2.0
    """角度误差权重"""
    V_ERROR_WEIGHT = 0.888
    """速度误差权重"""
    ACC_WEIGHT = 0.2
    """加速度权重"""
    STEER_WEIGHT = 0.2
    """转向角权重"""
    CTRL_DIFF_WEIGHT = 0.1
    """控制量变化率权重"""


class MPC:
    """MPC(Model Predictive Control)控制器"""

    def __init__(
        self, kinematic_model: BaseKinematicModel, obstacles: List[Obstacle]
    ) -> None:
        """使用运动学模型和环境中的障碍物初始化MPC控制器"""
        self.kinematic_model = kinematic_model
        """运动学模型"""
        self.prob_context: ca.Opti = ca.Opti()
        """优化问题上下文"""
        self.ctrl_seq_opt: ca.MX = self.prob_context.variable(
            Config.N, self.kinematic_model.ctrl_dim
        )
        """最优控制序列"""
        self.state_seq_opt: ca.MX = self.prob_context.variable(
            Config.N + 1, self.kinematic_model.state_dim
        )
        """最优状态序列，在Multiple Shooting方法中这也是优化变量"""
        self.state_seq_ref: ca.MX = self.prob_context.parameter(
            Config.N + 1, self.kinematic_model.state_dim
        )
        """参考状态序列，为方便计算第一个状态为当前状态"""
        self.__applyKinematicConstraints()
        self.__applyValueRangeConstraints()
        self.__applyObstacleAvoidanceConstraints(obstacles)
        self.prob_context.minimize(self.__genOptimizationObj())
        self.prob_context.solver("ipopt", MPC.__genSolverOptions())

    @staticmethod
    def __genSolverOptions() -> Dict[str, Any]:
        """生成求解器配置项"""
        return {
            "ipopt.max_iter": 888,
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.acceptable_tol": 1e-6,
            "ipopt.acceptable_obj_change_tol": 1e-3,
        }

    def __applyKinematicConstraints(self):
        """添加运动学约束"""
        self.prob_context.subject_to(
            self.state_seq_opt[0, :] == self.state_seq_ref[0, :]
        )
        for k in range(Config.N):
            state_next = (
                self.state_seq_opt[k, :]
                + self.kinematic_model.state_transition_func_mx(
                    self.state_seq_opt[k, :], self.ctrl_seq_opt[k, :]
                ).T
                * Config.DELTA_T
            )
            self.prob_context.subject_to(self.state_seq_opt[k + 1, :] == state_next)

    def __applyValueRangeConstraints(self):
        """添加对状态量和控制量的取值范围约束"""
        # 默认基于自行车模型，状态量依次为x, y, theta, v，控制量依次为加速度和前轮转角
        constraints = [
            (
                self.state_seq_opt[:, 3],
                -Config.MAX_SPEED,
                Config.MAX_SPEED,
            ),
        ]
        if self.kinematic_model.kinematic_model_name == "BICYCLE_MODEL":
            constraints.extend(
                [
                    (
                        self.ctrl_seq_opt[:, 0],
                        -Config.MAX_ACCEL,
                        Config.MAX_ACCEL,
                    ),
                    (
                        self.ctrl_seq_opt[:, 1],
                        -Config.MAX_STEER,
                        Config.MAX_STEER,
                    ),
                ]
            )
        for var, lower_bound, upper_bound in constraints:
            self.prob_context.subject_to(self.prob_context.bounded(lower_bound, var, upper_bound))  # type: ignore

    def __applyObstacleAvoidanceConstraints(self, obstacles: List[Obstacle]):
        """添加避障约束"""
        if len(obstacles) == 0:
            return
        # 把车视为一个以车辆中心为圆心，半径为半轴距的圆形
        half_wheel_base = 0.5 * Config.WHEEL_BASE
        for i in range(Config.N):
            k = i + 1
            car_x = self.state_seq_opt[k, 0] + half_wheel_base * ca.cos(
                self.state_seq_opt[k, 2]
            )
            car_y = self.state_seq_opt[k, 1] + half_wheel_base * ca.sin(
                self.state_seq_opt[k, 2]
            )
            for obstacle in obstacles:
                dist_squared = (car_x - obstacle.x) ** 2 + (car_y - obstacle.y) ** 2
                # 这里本质是一个Robust MPC，对于动态障碍物不管它是否出现都会考虑
                safe_dist_squared = (obstacle.radius + half_wheel_base) ** 2
                self.prob_context.subject_to(dist_squared >= safe_dist_squared)

    def __genOptimizationObj(self) -> ca.MX:
        """生成优化目标"""
        Q = np.diag(
            [
                Config.X_ERROR_WEIGHT,
                Config.Y_ERROR_WEIGHT,
                Config.THETA_ERROR_WEIGHT,
                Config.V_ERROR_WEIGHT,
            ]
        )
        R = np.diag([Config.ACC_WEIGHT, Config.STEER_WEIGHT])
        R_d = np.diag([Config.CTRL_DIFF_WEIGHT, Config.CTRL_DIFF_WEIGHT])
        opt_obj = ca.MX(0.0)
        for k in range(Config.N):
            state_error = self.state_seq_opt[k + 1, :] - self.state_seq_ref[k + 1, :]
            opt_obj += ca.mtimes([state_error, Q, state_error.T])
            opt_obj += ca.mtimes(
                [self.ctrl_seq_opt[k, :], R, self.ctrl_seq_opt[k, :].T]
            )
        for k in range(Config.N - 1):
            ctrl_diff = self.ctrl_seq_opt[k + 1, :] - self.ctrl_seq_opt[k, :]
            opt_obj += ca.mtimes([ctrl_diff, R_d, ctrl_diff.T])
        return opt_obj

    def setStateAndCtrlTrial(self, state_trial: np.ndarray, ctrl_trial: np.ndarray):
        """设置状态序列和控制序列的初始猜测值以加快求解速度"""
        self.prob_context.set_initial(self.state_seq_opt, state_trial)
        self.prob_context.set_initial(self.ctrl_seq_opt, ctrl_trial)

    def solve(self, state_ref: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """输入参考状态序列（其中第一个状态为当前状态），返回预测状态序列和对应的最优控制序列"""
        self.prob_context.set_value(self.state_seq_ref, state_ref)
        solution = self.prob_context.solve()
        state_seq_predicted = solution.value(self.state_seq_opt)
        ctrl_seq_opt_ = solution.value(self.ctrl_seq_opt)
        # 过滤掉第一个状态和控制量，作为下一次求解的初始猜测
        self.setStateAndCtrlTrial(
            np.vstack((state_seq_predicted[1:], state_seq_predicted[-1])),
            np.vstack((ctrl_seq_opt_[1:], ctrl_seq_opt_[-1])),
        )
        return state_seq_predicted, ctrl_seq_opt_


class MPCDemo(TrafficSimulator):
    """MPC演示类"""

    def __init__(self, config_name="config.json") -> None:
        super().__init__(config_name)
        Config.UpdateFromJson(Config, config_name)
        Config.PrintConfig(Config)
        self.car = self.ego_car
        """受控车辆，本质是一个引用"""
        self.controller = MPC(self.car.model, self.data_mgr.map.obstacles)
        """MPC控制器"""
        self.state_seq_ref = np.zeros((Config.N + 1, self.car.model.state_dim))
        """用于MPC计算的参考状态序列，注意为了方便计算第一个状态为当前状态"""
        self.graphic_item_mgr.addExtraCurveItem(
            "reference trajectory", BLUE, QtCore.Qt.PenStyle.DotLine
        )
        """添加额外的参考轨迹图形项"""
        self.graphic_item_mgr.addExtraCurveItem(
            "predicted trajectory", RED, QtCore.Qt.PenStyle.DotLine
        )
        """添加额外的预测轨迹图形项"""

    def update(self) -> None:
        if self.getDistToDest() > 0.5 * Config.WHEEL_BASE:
            self.__updateRefPtAndStateSeq()
            state_seq_predicted, ctrl_seq_opt = self.controller.solve(
                self.state_seq_ref
            )
            self.getCtrlInput().setVal(ctrl_seq_opt[0][0], ctrl_seq_opt[0][1])
            self.__updateGraphicItems(state_seq_predicted)
        else:
            Log.info("Arrived at destination")
            self.finalize()

    def __updateRefPtAndStateSeq(self):
        """更新参考点和参考状态序列"""
        last_ref_pt = self.getRefPt()
        ref_line = self.getRefLine()
        # 为方便后续计算第一个参考状态为当前状态而不是参考点
        self.state_seq_ref[0] = self.car.model.state
        for i in range(Config.N):
            delta_dist = last_ref_pt.max_speed_rate * Config.MAX_SPEED * Config.DELTA_T
            ref_pt = ref_line.getPtAfterDistance(last_ref_pt.id, delta_dist)
            v_ref = ref_pt.max_speed_rate * Config.MAX_SPEED
            if ref_line.isCloseToDestination(ref_pt):
                v_ref = 0.0
            self.state_seq_ref[i + 1] = np.array(
                [ref_pt.x, ref_pt.y, ref_pt.theta, v_ref]
            )
            last_ref_pt = ref_pt

    def __updateGraphicItems(self, state_seq_predicted: np.ndarray) -> None:
        """更新繁杂的图像显示，分离出来避免干扰算法主逻辑"""
        state_seq_ref = self.state_seq_ref.copy()
        ref_pt = self.getRefPt()
        state_seq_ref[0] = [ref_pt.x, ref_pt.y, ref_pt.theta, 0.0]
        self.graphic_item_mgr.setExtraCurveData("reference trajectory", state_seq_ref)
        self.graphic_item_mgr.setExtraCurveData(
            "predicted trajectory", state_seq_predicted
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    demo = MPCDemo("config.json")
    demo.show()
    app.exec()
