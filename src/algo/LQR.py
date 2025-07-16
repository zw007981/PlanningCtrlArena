# -*- coding: utf-8 -*-

import os
import sys

sys.path.append(os.getcwd())

from typing import Tuple

import casadi as ca
import numpy as np
from numpy.typing import NDArray
from PySide6 import QtCore
from PySide6.QtWidgets import QApplication

from src.kinematic_model.base_kinematic_model import BaseKinematicModel
from src.traffic_simulator.color import *
from src.traffic_simulator.traffic_simulator import TrafficSimulator
from src.util.config import Config as BaseConfig
from src.util.log import Log


class Config(BaseConfig):
    """LQR控制器配置"""

    X_ERROR_WEIGHT = 1.0
    """x方向误差权重"""
    Y_ERROR_WEIGHT = 1.0
    """y方向误差权重"""
    THETA_ERROR_WEIGHT = 1.6
    """角度误差权重"""
    V_ERROR_WEIGHT = 0.6
    """速度误差权重"""
    ACC_WEIGHT = 1.0
    """加速度权重"""
    STEER_WEIGHT = 1.0
    """转向角权重"""
    RICCATI_TOL = 0.001
    """迭代求解Riccati方程的容许误差，小于该值则认为已收敛"""
    MAX_ITER = 500
    """最大迭代求解次数"""


class LQR:
    """LQR(Linear Quadratic Regulator)控制器"""

    Q: NDArray = np.diag(
        [
            Config.X_ERROR_WEIGHT,
            Config.Y_ERROR_WEIGHT,
            Config.THETA_ERROR_WEIGHT,
            Config.V_ERROR_WEIGHT,
        ]
    )
    """状态误差权重矩阵"""
    R: NDArray = np.diag([Config.ACC_WEIGHT, Config.STEER_WEIGHT])
    """控制权重矩阵"""

    def __init__(self, kinematic_model: BaseKinematicModel) -> None:
        """使用运动学模型初始化LQR控制器"""
        self.kinematic_model = kinematic_model
        """运动学模型"""
        # 状态变量和控制变量
        x = ca.SX.sym("x", self.kinematic_model.state_dim)  # type: ignore
        u = ca.SX.sym("u", self.kinematic_model.ctrl_dim)  # type: ignore
        # 状态导数：x_dot = A * x + B * u
        delta_state = self.kinematic_model.stateDerivative(x, u)
        A = ca.jacobian(delta_state, x)
        B = ca.jacobian(delta_state, u)
        self.A_func = ca.Function("A_func", [x, u], [A])
        """状态转移矩阵函数，输入状态-控制对，输出对应的A矩阵"""
        self.B_func = ca.Function("B_func", [x, u], [B])
        """控制输入矩阵函数，输入状态-控制对，输出对应的B矩阵"""

    def solve(self, state: NDArray, state_ref: NDArray, ctrl_ref: NDArray) -> NDArray:
        """输入状态、参考状态和参考控制量，返回最优控制量"""
        x = ca.DM(state)
        x_r = ca.DM(state_ref)
        u_r = ca.DM(ctrl_ref)
        # 计算[x_r, u_r]处的A矩阵和B矩阵: x_dot = A_ * x + B_ * u
        A_ = self.A_func(x_r, u_r)
        B_ = self.B_func(x_r, u_r)
        # 将连续时间系统离散化: x[k+1] = A * x[k] + B * u[k]
        # 其中A = I + Δt * A_, B = Δt * B_
        A = ca.DM.eye(4) + Config.DELTA_T * A_  # type: ignore
        B = Config.DELTA_T * B_  # type: ignore
        # 迭代求解Riccati方程
        P = ca.DM(LQR.Q)
        for _ in range(Config.MAX_ITER):
            P_new = (
                LQR.Q
                + A.T @ P @ A
                - A.T @ P @ B @ ca.inv(LQR.R + B.T @ P @ B) @ B.T @ P @ A
            )
            if ca.mmax(ca.fabs(P_new - P)) < Config.RICCATI_TOL:
                break
            P = P_new
        # 计算反馈增益矩阵K
        K = ca.inv(LQR.R + B.T @ P @ B) @ B.T @ P @ A
        # 计算误差状态和误差控制
        x_e = x - x_r
        u_e = -K @ x_e
        # 得到最优控制量，对于自行车模型，u = [a, delta]
        u = np.array(u_e + u_r).flatten()
        u[0] = np.clip(u[0], -Config.MAX_ACCEL, Config.MAX_ACCEL)
        u[1] = np.clip(u[1], -Config.MAX_STEER, Config.MAX_STEER)
        return u


class LQRDemo(TrafficSimulator):
    """LQR演示类"""

    def __init__(self, config_name="config.json") -> None:
        super().__init__(config_name)
        Config.UpdateFromJson(Config, config_name)
        Config.PrintConfig(Config)
        self.car = self.ego_car
        """受控车辆，本质是一个引用"""
        self.controller = LQR(self.car.model)
        """LQR控制器"""
        self.graphic_item_mgr.addExtraCurveItem(
            "direction", color=RED, style=QtCore.Qt.PenStyle.DashLine, width=2
        )
        """车辆行进方向图形项"""

    def update(self) -> None:
        if self.getDistToDest() > 0.666 * Config.WHEEL_BASE:
            state_ref, ctrl_ref = self.__getRefStateAndCtrl()
            ctrl_opt = self.controller.solve(self.car.model.state, state_ref, ctrl_ref)
            self.getCtrlInput().setVal(ctrl_opt[0], ctrl_opt[1])
            self.__updateGraphicItems()
        else:
            Log.info("Arrived at destination")
            self.finalize()

    def __getRefStateAndCtrl(self) -> Tuple[NDArray, NDArray]:
        """根据车辆当前状态获取参考状态和参考控制输入"""
        ref_pt = self.getRefPt()
        ref_line = self.getRefLine()
        v_ref = ref_pt.max_speed_rate * Config.MAX_SPEED
        if ref_line.isCloseToDestination(ref_pt):
            v_ref = 0.0
        a_ref = np.clip(
            (v_ref - self.car.v) / Config.DELTA_T, -Config.MAX_ACCEL, Config.MAX_ACCEL
        )
        delta_ref = 0.0
        return (
            np.array([ref_pt.x, ref_pt.y, ref_pt.theta, v_ref]),
            np.array([a_ref, delta_ref]),
        )

    def __updateGraphicItems(self) -> None:
        """更新与算法本身无关的繁杂图形项"""
        # 生成控制量向量用于显示车辆的行进方向，起始于后轴中心，
        # 长度正比于下一时刻的速度，角度为车身角度+转向角
        l = (
            2.0
            * Config.WHEEL_BASE
            * (self.car.v + self.getCtrlInput().val[0] * Config.DELTA_T)
            / Config.MAX_SPEED
        )
        theta = self.car.pose.theta + self.getCtrlInput().val[1]
        self.graphic_item_mgr.setExtraCurveData(
            "direction",
            np.array(
                [
                    [self.car.pose.x, self.car.pose.y],
                    [
                        self.car.pose.x + l * np.cos(theta),
                        self.car.pose.y + l * np.sin(theta),
                    ],
                ]
            ),
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    demo = LQRDemo("config.json")
    demo.show()
    app.exec()
