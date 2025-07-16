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
    """MPC求解器配置"""

    N = 12
    """预测步长"""
    X_ERROR_WEIGHT = 1.0
    """x方向上的误差权重"""
    Y_ERROR_WEIGHT = 1.0
    """y方向上的误差权重"""
    THETA_ERROR_WEIGHT = 2.0
    """航向角误差权重"""
    V_ERROR_WEIGHT = 0.888
    """速度误差权重"""
    ACC_WEIGHT = 0.2
    """加速度权重"""
    STEER_WEIGHT = 0.2
    """转向角权重"""
    CTRL_DIFF_WEIGHT = 0.1
    """控制量变化率权重"""
    NOM_SCALING = 0.5
    """nominal trajectory权重，对应论文中的P^n"""
    CONT_SCALING = 0.5
    """contingency trajectory权重，对应论文中的P^c"""

    @staticmethod
    def Sanitize() -> None:
        """对配置进行标准化处理"""
        Config.CONT_SCALING = 1 - Config.NOM_SCALING


class CMPC:
    """CMPC（Contingency Model Predictive Control）求解器"""

    def __init__(
        self, kinematic_model: BaseKinematicModel, obstacles: List[Obstacle]
    ) -> None:
        """使用运动学模型和环境中的障碍物初始化CMPC求解器"""
        self.kinematic_model = kinematic_model
        """运动学模型"""
        self.prob_context: ca.Opti = ca.Opti()
        """优化问题上下文"""
        self.state_seq_nom: ca.MX = self.prob_context.variable(
            Config.N + 1, self.kinematic_model.state_dim
        )
        """nominal trajectory的状态序列"""
        self.state_seq_cont: ca.MX = self.prob_context.variable(
            Config.N + 1, self.kinematic_model.state_dim
        )
        """contingency trajectory的状态序列"""
        self.ctrl_seq_nom: ca.MX = self.prob_context.variable(Config.N, 2)
        """nominal trajectory的最优控制序列"""
        self.ctrl_seq_cont: ca.MX = self.prob_context.variable(Config.N, 2)
        """contingency trajectory的最优控制序列"""
        self.state_seq_ref: ca.MX = self.prob_context.parameter(
            Config.N + 1, self.kinematic_model.state_dim
        )
        """参考状态序列，为方便计算第一个状态为当前状态"""
        self.obs_radius_idea: ca.MX = self.prob_context.parameter(len(obstacles), 1)
        """各障碍物的理想半径，忽略没有观测到的动态障碍物，用于nominal trajectory的避障"""
        self.obs_radius_safe: ca.MX = self.prob_context.parameter(len(obstacles), 1)
        """各障碍物的安全半径，即使动态障碍物没有被观测到也认为它们存在，用于contingency trajectory的避障"""
        # nominal trajectory和contingency trajectory共享第一个控制量
        self.prob_context.subject_to(
            self.ctrl_seq_nom[0, :] == self.ctrl_seq_cont[0, :]
        )
        self.__applyKinematicConstraintsForTraj(self.state_seq_nom, self.ctrl_seq_nom)
        self.__applyKinematicConstraintsForTraj(self.state_seq_cont, self.ctrl_seq_cont)
        self.__applyValueRangeConstraintsForTraj(self.state_seq_nom, self.ctrl_seq_nom)
        self.__applyValueRangeConstraintsForTraj(
            self.state_seq_cont, self.ctrl_seq_cont
        )
        self.__applyObsAvoidanceConstraintsForTraj(
            self.state_seq_nom, obstacles, self.obs_radius_idea
        )
        self.__applyObsAvoidanceConstraintsForTraj(
            self.state_seq_cont, obstacles, self.obs_radius_safe
        )
        opt_obj = Config.NOM_SCALING * self.__genOptimizationObjForTraj(
            self.state_seq_nom, self.ctrl_seq_nom
        ) + Config.CONT_SCALING * self.__genOptimizationObjForTraj(
            self.state_seq_cont, self.ctrl_seq_cont
        )
        self.prob_context.minimize(opt_obj)
        self.prob_context.solver("ipopt", self.__genSolverOptions())

    @staticmethod
    def __genSolverOptions() -> Dict[str, Any]:
        """生成求解器配置项"""
        return {
            "ipopt.max_iter": 6666,
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.acceptable_tol": 1e-4,
            "ipopt.acceptable_obj_change_tol": 1e-2,
        }

    def __applyKinematicConstraintsForTraj(self, state_seq: ca.MX, ctrl_seq: ca.MX):
        """对轨迹添加运动学约束"""
        self.prob_context.subject_to(state_seq[0, :] == self.state_seq_ref[0, :])
        for k in range(Config.N):
            state_next = (
                state_seq[k, :]
                + self.kinematic_model.state_transition_func_mx(
                    state_seq[k, :], ctrl_seq[k, :]
                ).T
                * Config.DELTA_T
            )
            self.prob_context.subject_to(state_next == state_seq[k + 1, :])

    def __applyValueRangeConstraintsForTraj(self, state_seq: ca.MX, ctrl_seq: ca.MX):
        """对轨迹添加取值范围约束"""
        # 默认基于自行车模型，状态量依次为x, y, theta, v，控制量依次为加速度和前轮转角
        constraints = [
            (
                state_seq[:, 3],
                -Config.MAX_SPEED,
                Config.MAX_SPEED,
            ),
            (ctrl_seq[:, 0], -Config.MAX_ACCEL, Config.MAX_ACCEL),
            (ctrl_seq[:, 1], -Config.MAX_STEER, Config.MAX_STEER),
        ]
        for var, lower_bound, upper_bound in constraints:
            self.prob_context.subject_to(self.prob_context.bounded(lower_bound, var, upper_bound))  # type: ignore

    def __applyObsAvoidanceConstraintsForTraj(
        self,
        state_seq: ca.MX,
        obstacles: List[Obstacle],
        obs_radius: ca.MX,
    ):
        """对轨迹添加避障约束"""
        if len(obstacles) == 0:
            return
        half_wheel_base = 0.5 * Config.WHEEL_BASE
        for i in range(Config.N):
            k = i + 1
            car_x = state_seq[k, 0] + half_wheel_base * ca.cos(state_seq[k, 2])
            car_y = state_seq[k, 1] + half_wheel_base * ca.sin(state_seq[k, 2])
            for obstacle_id, obstacle in enumerate(obstacles):
                dist_squared = (car_x - obstacle.x) ** 2 + (car_y - obstacle.y) ** 2
                safe_dist_squared = (obs_radius[obstacle_id] + half_wheel_base) ** 2
                self.prob_context.subject_to(dist_squared >= safe_dist_squared)

    def __genOptimizationObjForTraj(self, state_seq: ca.MX, ctrl_seq: ca.MX) -> ca.MX:
        """为一段轨迹生成优化目标"""
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
            state_error = state_seq[k + 1, :] - self.state_seq_ref[k + 1, :]
            opt_obj += ca.mtimes([state_error, Q, state_error.T]) + ca.mtimes(
                [ctrl_seq[k, :], R, ctrl_seq[k, :].T]
            )
        for k in range(Config.N - 1):
            ctrl_diff = ctrl_seq[k + 1, :] - ctrl_seq[k, :]
            opt_obj += ca.mtimes([ctrl_diff, R_d, ctrl_diff.T])
        return opt_obj

    def setStateAndCtrlTrial(
        self,
        state_nom_trial: np.ndarray,
        state_cont_trial: np.ndarray,
        ctrl_nom_trial: np.ndarray,
        ctrl_cont_trial: np.ndarray,
    ):
        """设置状态序列和控制序列的初始猜测值以加快求解速度"""
        self.prob_context.set_initial(self.state_seq_nom, state_nom_trial)
        self.prob_context.set_initial(self.state_seq_cont, state_cont_trial)
        self.prob_context.set_initial(self.ctrl_seq_nom, ctrl_nom_trial)
        self.prob_context.set_initial(self.ctrl_seq_cont, ctrl_cont_trial)

    def updateObstacleRadius(
        self, obs_radius_idea: np.ndarray, obs_radius_safe: np.ndarray
    ):
        """更新障碍物的理想半径和安全半径信息"""
        self.prob_context.set_value(self.obs_radius_idea, obs_radius_idea)
        self.prob_context.set_value(self.obs_radius_safe, obs_radius_safe)

    def solve(
        self, state_ref: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        输入参考状态序列（其中第一个状态为当前状态），返回预测状态序列和对应的最优控制序列
        return:
            state_nom_predicted: nominal trajectory的预测状态序列
            state_cont_predicted: contingency trajectory的预测状态序列
            ctrl_nom_opt_: nominal trajectory的最优控制序列
            ctrl_cont_opt_: contingency trajectory的最优控制序列
        """
        self.prob_context.set_value(self.state_seq_ref, state_ref)
        solution = self.prob_context.solve()
        state_nom_predicted = solution.value(self.state_seq_nom)
        ctrl_nom_opt_ = solution.value(self.ctrl_seq_nom)
        state_cont_predicted = solution.value(self.state_seq_cont)
        ctrl_cont_opt_ = solution.value(self.ctrl_seq_cont)
        self.setStateAndCtrlTrial(
            np.vstack((state_nom_predicted[1:], state_nom_predicted[-1])),
            np.vstack((state_cont_predicted[1:], state_cont_predicted[-1])),
            np.vstack((ctrl_nom_opt_[1:], ctrl_nom_opt_[-1])),
            np.vstack((ctrl_cont_opt_[1:], ctrl_cont_opt_[-1])),
        )
        return state_nom_predicted, state_cont_predicted, ctrl_nom_opt_, ctrl_cont_opt_


class CMPCDemo(TrafficSimulator):
    """CMPC演示类"""

    def __init__(self, config_name="config.json") -> None:
        super().__init__(config_name)
        Config.UpdateFromJson(Config, config_name)
        Config.Sanitize()
        Config.PrintConfig(Config)
        self.car = self.ego_car
        """受控车辆，本质是一个引用"""
        self.controller = CMPC(self.car.model, self.data_mgr.map.obstacles)
        """CMPC控制器"""
        self.state_seq_ref = np.zeros((Config.N + 1, self.car.model.state_dim))
        """用于CMPC计算的参考状态序列，注意为了方便计算第一个状态为当前状态"""
        self.graphic_item_mgr.addExtraCurveItem(
            "predicted nominal trajectory", BLUE, QtCore.Qt.PenStyle.SolidLine
        )
        """添加预测nominal轨迹图像"""
        self.graphic_item_mgr.addExtraCurveItem(
            "predicted contingency trajectory", YELLOW, QtCore.Qt.PenStyle.DashLine
        )
        """添加预测contingency轨迹图像"""
        self.graphic_item_mgr.addExtraCurveItem(
            "reference trajectory", RED, QtCore.Qt.PenStyle.DotLine
        )
        """添加参考轨迹图像"""

    def update(self) -> None:
        if self.getDistToDest() >= 0.5 * Config.WHEEL_BASE:
            self.__updateRefPtAndStateSeq()
            self.__updateObstacleRadius()
            (
                state_nom_predicted,
                state_cont_predicted,
                ctrl_nom_opt,
                ctrl_cont_opt,
            ) = self.controller.solve(self.state_seq_ref)
            self.getCtrlInput().setVal(ctrl_nom_opt[0][0], ctrl_nom_opt[0][1])
            self.__updateGraphicItems(
                state_nom_predicted,
                state_cont_predicted,
            )
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
            v = ref_pt.max_speed_rate * Config.MAX_SPEED
            if ref_line.isCloseToDestination(ref_pt):
                v = 0.0
            self.state_seq_ref[i + 1] = np.array([ref_pt.x, ref_pt.y, ref_pt.theta, v])
            last_ref_pt = ref_pt

    def __updateObstacleRadius(self):
        """更新障碍物半径信息"""
        obs_radius_idea = np.zeros((len(self.data_mgr.map.obstacles), 1))
        obs_radius_safe = np.zeros((len(self.data_mgr.map.obstacles), 1))
        for i, obstacle in enumerate(self.data_mgr.map.obstacles):
            obs_radius_idea[i] = obstacle.getIdealRadius(
                self.car.pose.x, self.car.pose.y
            )
            obs_radius_safe[i] = obstacle.getSafeRadius(
                self.car.pose.x, self.car.pose.y
            )
        # 为了适配求解器避障的代码，即使障碍物不存在也不能直接输入0
        minus_half_wheel_base = -0.5 * Config.WHEEL_BASE
        for i in range(len(self.data_mgr.map.obstacles)):
            if obs_radius_idea[i] <= Config.ZERO_EPS:
                obs_radius_idea[i] = minus_half_wheel_base
            if obs_radius_safe[i] <= Config.ZERO_EPS:
                obs_radius_safe[i] = minus_half_wheel_base
        self.controller.updateObstacleRadius(obs_radius_idea, obs_radius_safe)

    def __updateGraphicItems(
        self,
        state_nom_predicted: np.ndarray,
        state_cont_predicted: np.ndarray,
    ) -> None:
        """更新繁杂的图像显示，分离出来避免干扰算法主逻辑"""
        self.graphic_item_mgr.setExtraCurveData(
            "predicted nominal trajectory", state_nom_predicted
        )
        self.graphic_item_mgr.setExtraCurveData(
            "predicted contingency trajectory", state_cont_predicted
        )
        ref_pt = self.getRefPt()
        state_seq_ref = self.state_seq_ref.copy()
        state_seq_ref[0] = [ref_pt.x, ref_pt.y, ref_pt.theta, 0.0]
        self.graphic_item_mgr.setExtraCurveData("reference trajectory", state_seq_ref)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    demo = CMPCDemo("config_CMPC.json")
    demo.show()
    app.exec()
