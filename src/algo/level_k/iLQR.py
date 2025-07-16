# -*- coding: utf-8 -*-

import os
import sys

sys.path.append(os.getcwd())


from typing import Tuple

import casadi as ca
import numpy as np
from numpy.typing import NDArray

from src.algo.iLQR import Primarch
from src.algo.level_k.config import Config
from src.traffic_simulator.color import *
from src.util.log import Log


class iLQR(Primarch):
    """适用于level-k的iLQR算法实现，在代价函数中考虑了周围的车辆"""

    def __init__(self, config: Config, num_cars: int) -> None:
        """使用车辆数量初始化iLQR算法实例"""
        self.config = config
        """配置参数"""
        self.num_cars = num_cars
        """周围车辆的数量"""
        self.car_id: int = 0
        """需要规划的车辆ID，这里需要转化为int形因为后续会作为索引使用"""
        self.level_val: int = 0
        """当前规划的level值，对于level-0和level-2的车辆而言不需要考虑舒适度"""
        self.t = 0
        """当前时间步"""
        self.cars_x_seq: ca.DM = ca.DM.zeros(num_cars, self.config.N)  # type: ignore
        """所有车辆处于k-1级时规划的轨迹的x坐标序列，这里的坐标已做转换是车辆的中心而不是后轴中心"""
        self.cars_y_seq: ca.DM = ca.DM.zeros(num_cars, self.config.N)  # type: ignore
        """所有车辆处于k-1级时规划的轨迹的y坐标序列，这里的坐标已做转换是车辆的中心而不是后轴中心"""
        # 符号变量=====================================================
        self.CAR_ID: ca.MX = ca.MX.sym("car_id", 1)  # type: ignore
        """需要规划的车辆ID符号"""
        self.LEVEL: ca.MX = ca.MX.sym("level", 1)  # type: ignore
        """当前规划的level层级符号"""
        self.T: ca.MX = ca.MX.sym("t", 1)  # type: ignore
        """当前时间步符号"""
        self.CARS_X: ca.MX = ca.MX.sym("cars_x", num_cars, 1)  # type: ignore
        """k-1级别规划方案中所有车辆在T时刻中心位置的x坐标符号"""
        self.CARS_Y: ca.MX = ca.MX.sym("cars_y", num_cars, 1)  # type: ignore
        """k-1级别规划方案中所有车辆在T时刻中心位置的y坐标符号"""
        self.CARS_X_SEQ: ca.MX = ca.MX.sym("cars_x_seq", num_cars, self.config.N)  # type: ignore
        """所有车辆处于k-1级时规划的轨迹的x坐标序列符号"""
        self.CARS_Y_SEQ: ca.MX = ca.MX.sym("cars_y_seq", num_cars, self.config.N)  # type: ignore
        """所有车辆处于k-1级时规划的轨迹的y坐标序列符号"""
        super().__init__(config)

    def buildStateTransitionFunc(self) -> ca.Function:
        return self.defStateTransitionFunc

    def buildCostFunc(self) -> ca.Function:
        base_cost = self.defCostFunc(self.X, self.U, self.X_REF)
        negative_speed_cost = ca.if_else(
            self.X[3] < 0, self.config.NEGATIVE_SPEED_PENALTY, 0
        )
        # 默认车辆以后轮中心为坐标原点，所以需要额外计算出车辆的几何中心坐标
        car_x = self.X[0] + self.config.HALF_WHEEL_BASE * ca.cos(self.X[2])
        car_y = self.X[1] + self.config.HALF_WHEEL_BASE * ca.sin(self.X[2])
        # 计算当前时间步与所有车辆距离的平方，同时利用掩码把到自身的距离设为一个较大的数
        idx = ca.DM(list(range(self.num_cars)))
        coll_mask = ca.if_else(idx == self.CAR_ID, 0, 1)
        coll_dist_squared = (
            (car_x - self.CARS_X) ** 2
            + (car_y - self.CARS_Y) ** 2
            + (1 - coll_mask) * 666666
        )
        # 找出最小距离并检查是否小于安全距离和舒适距离
        min_dist_squared = ca.mmin(coll_dist_squared)
        coll_cost = ca.if_else(
            min_dist_squared < self.config.SAFE_DIST_SQUARED,
            self.config.COLLISION_WEIGHT
            * (1 - (min_dist_squared / self.config.SAFE_DIST_SQUARED)) ** 2,
            0,
        )
        comfort_cost = ca.if_else(
            min_dist_squared < self.config.COMFORT_DIST_SQUARED,
            self.config.COMFORT_WEIGHT
            * (1 - (min_dist_squared / self.config.COMFORT_DIST_SQUARED)) ** 2,
            0,
        )
        # 激进的车辆不用考虑舒适度
        comfort_cost = ca.if_else(
            ca.logic_or(self.LEVEL == 0, self.LEVEL == 2), 0, comfort_cost
        )
        cost = (self.config.GAMMA**self.T) * (
            base_cost + negative_speed_cost + coll_cost + comfort_cost
        )
        return Primarch.BuildFunc(
            [
                self.X,
                self.U,
                self.X_REF,
                self.CAR_ID,
                self.LEVEL,
                self.T,
                self.CARS_X,
                self.CARS_Y,
            ],
            [cost],
        )

    def buildFinalCostFunc(self) -> ca.Function:
        return self.defFinalCostFunc

    def buildBatchCostDerivativesFunc(self) -> ca.Function:
        # 先构建计算单步代价函数的偏导的函数
        l = self.cost_func(
            self.X,
            self.U,
            self.X_REF,
            self.CAR_ID,
            self.LEVEL,
            self.T,
            self.CARS_X,
            self.CARS_Y,
        )
        l_x = ca.gradient(l, self.X)
        l_u = ca.gradient(l, self.U)
        l_xx = ca.hessian(l, self.X)[0]
        l_uu = ca.hessian(l, self.U)[0]
        l_ux = ca.jacobian(l_u, self.X)
        cost_derivatives_func = Primarch.BuildFunc(
            [
                self.X,
                self.U,
                self.X_REF,
                self.CAR_ID,
                self.LEVEL,
                self.T,
                self.CARS_X,
                self.CARS_Y,
            ],
            [l_x, l_u, l_xx, l_uu, l_ux],
        )
        # 使用map函数批量计算
        func_map = cost_derivatives_func.map(self.config.N - 1)
        time_steps = ca.DM(range(self.config.N - 1))
        car_id_seq = ca.repmat(self.CAR_ID, 1, self.config.N - 1)
        level_seq = ca.repmat(self.LEVEL, 1, self.config.N - 1)
        cars_x_seq = self.CARS_X_SEQ[:, : self.config.N - 1]
        cars_y_seq = self.CARS_Y_SEQ[:, : self.config.N - 1]
        lx_seq, lu_seq, lxx_seq, luu_seq, lux_seq = func_map(
            self.X_SEQ[:, : self.config.N - 1],
            self.U_SEQ,
            self.X_SEQ_REF[:, : self.config.N - 1],
            car_id_seq,
            level_seq,
            time_steps,
            cars_x_seq,
            cars_y_seq,
        )
        return Primarch.BuildFunc(
            [
                self.X_SEQ,
                self.U_SEQ,
                self.X_SEQ_REF,
                self.CAR_ID,
                self.LEVEL,
                self.CARS_X_SEQ,
                self.CARS_Y_SEQ,
            ],
            [lx_seq, lu_seq, lxx_seq, luu_seq, lux_seq],
        )

    def buildFinalCostDerivativesFunc(self) -> ca.Function:
        lf = self.final_cost_func(self.X, self.X_REF)
        lf_x = ca.jacobian(lf, self.X)
        lf_xx = ca.hessian(lf, self.X)[0]
        return Primarch.BuildFunc([self.X, self.X_REF], [lf_x, lf_xx])

    def buildTrajCostFunc(self) -> ca.Function:
        stage_costs = self.cost_func.map(self.config.N - 1)(
            self.X_SEQ[:, : self.config.N - 1],
            self.U_SEQ,
            self.X_SEQ_REF[:, : self.config.N - 1],
            ca.repmat(self.CAR_ID, 1, self.config.N - 1),
            ca.repmat(self.LEVEL, 1, self.config.N - 1),
            ca.DM(range(self.config.N - 1)),
            self.CARS_X_SEQ[:, : self.config.N - 1],
            self.CARS_Y_SEQ[:, : self.config.N - 1],
        )
        final_cost = self.final_cost_func(self.X_SEQ[:, -1], self.X_SEQ_REF[:, -1])
        return Primarch.BuildFunc(
            [
                self.X_SEQ,
                self.U_SEQ,
                self.X_SEQ_REF,
                self.CAR_ID,
                self.LEVEL,
                self.CARS_X_SEQ,
                self.CARS_Y_SEQ,
            ],
            [ca.sum2(stage_costs) + final_cost],
        )

    def precomputeDerivatives(self):
        self.lf_x, self.lf_xx = self.final_cost_derivatives_func(
            self.x_seq[:, -1], self.x_seq_ref[:, -1]  # type: ignore
        )
        self.A_seq, self.B_seq = self.batch_state_transition_derivatives_func(
            self.x_seq, self.u_seq  # type: ignore
        )
        (
            self.lx_seq,
            self.lu_seq,
            self.lxx_seq,
            self.luu_seq,
            self.lux_seq,
        ) = self.batch_cost_derivatives_func(
            self.x_seq, self.u_seq, self.x_seq_ref, self.car_id, self.level_val, self.cars_x_seq, self.cars_y_seq  # type: ignore
        )

    def plan(
        self,
        car_id: int,
        level_val: int,
        x_init: NDArray,
        x_seq_ref: ca.DM,
        u_seq_ref: ca.DM,
        cars_x_seq: ca.DM,
        cars_y_seq: ca.DM,
    ) -> Tuple[ca.DM, ca.DM]:
        """
        输入int型的车辆ID、所处层级、初始状态、参考状态序列、参考控制输入序列以及k-1层级时所有车辆的规划路径，
        为处于level-k的车辆规划轨迹并返回预测状态序列和控制输入序列。
        由于与基类的入参大不相同所以没有再重载基类中的solve方法。
        """
        self.__initSolver(
            car_id, level_val, x_init, x_seq_ref, u_seq_ref, cars_x_seq, cars_y_seq
        )
        self.J = float(
            self.traj_cost_func(self.x_seq, self.u_seq, self.x_seq_ref, self.car_id, self.level_val, self.cars_x_seq, self.cars_y_seq)  # type: ignore
        )
        delta_J = np.inf
        for _ in range(self.config.MAX_ITER):
            self.backwardPass()
            for alpha in self.config.ALPHA_LIST:
                self.forwardPass(alpha)
                J_new = float(
                    self.traj_cost_func(self.x_seq_temp, self.u_seq_temp, self.x_seq_ref, self.car_id, self.level_val, self.cars_x_seq, self.cars_y_seq)  # type: ignore
                )
                delta_J = self.J - J_new
                if delta_J > self.config.FLOAT_EPS:
                    # 接受更新并降低正则化系数
                    self.acceptNewSol(self.x_seq_temp, self.u_seq_temp, J_new)
                    self.updateReg(0.5)
                    break
                else:
                    self.updateReg(2.0)
                self.reg = np.clip(self.reg, self.config.REG_MIN, self.config.REG_MAX)
            if abs(delta_J) < self.config.TOL:
                break
        return self.x_seq, self.u_seq

    def solve(self, x_init: NDArray, x_seq_ref: NDArray) -> Tuple[NDArray, NDArray]:
        Log.error("iLQR.solve() is not implemented. Use plan() instead!!!")
        raise NotImplementedError(
            "This method is not implemented. Use plan() instead!!!"
        )

    def __initSolver(
        self,
        car_id: int,
        level_val: int,
        x_init: NDArray,
        x_seq_ref: ca.DM,
        u_seq_ref: ca.DM,
        cars_x_seq: ca.DM,
        cars_y_seq: ca.DM,
    ) -> None:
        """初始化求解器"""
        self.car_id = car_id
        self.level_val = level_val
        self.x_seq[0, 0] = x_init[0]
        self.x_seq[1, 0] = x_init[1]
        self.x_seq[2, 0] = x_init[2]
        self.x_seq[3, 0] = x_init[3]
        self.x_seq_ref = x_seq_ref
        # 后续会更改控制输入，复制参考控制输入序列。
        self.u_seq = ca.DM(u_seq_ref)
        self.cars_x_seq = cars_x_seq
        self.cars_y_seq = cars_y_seq
        self.rollOut()
        self.reg = self.config.REG_INIT
