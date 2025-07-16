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

from src.algo.iLQR import Config as BaseConfig
from src.algo.iLQR import Primarch
from src.traffic_simulator.color import *
from src.traffic_simulator.traffic_simulator import TrafficSimulator
from src.util.log import Log


class Config(BaseConfig):
    """Constrained iLQR控制器配置"""

    def __init__(self) -> None:
        super().__init__()
        self.T_LIST = [0.5, 2.0, 5.0]
        """Barrier函数的系数列表"""

    def modifyConfig(self) -> None:
        # 适当降低迭代次数以提高性能
        self.MAX_ITER = 10
        self.ALPHA_LIST = [0.666**i for i in range(4)]


class BarrierFunc:
    """
    Barrier函数，以CasADi函数的形式封装。
    先把约束条件以变量的形式汇总到conditions列表，列表中的每个变量都对应着一个约束条件：ca.MX < 0。
    最后得到barrier_func用于计算Barrier函数值，另外得到is_feasible_func用于判断解是否可行。
    """

    def __init__(self, config: Config, num_obs: int = 0) -> None:
        self.config = config
        """配置"""
        self.X = ca.MX.sym("x", Config.STATE_DIM)  # type: ignore
        """状态符号变量"""
        self.U = ca.MX.sym("u", Config.CTRL_DIM)  # type: ignore
        """控制符号变量"""
        self.X_SEQ = ca.MX.sym("x_seq", Config.STATE_DIM, self.config.N)  # type: ignore
        """状态序列符号变量"""
        self.U_SEQ = ca.MX.sym("u_seq", Config.CTRL_DIM, self.config.N - 1)  # type: ignore
        """控制序列符号变量"""
        self.OBS_LIST = ca.MX.sym("obs_list", 3, num_obs)  # type: ignore
        """障碍物列表符号变量"""
        self.T = ca.MX.sym("t")  # type: ignore
        """Barrier函数的超参数t符号变量"""

        self.conditions: List[ca.MX] = []
        """约束条件列表，要求每个元素都小于0"""
        self.barrier_func: ca.Function = self.__buildBarrierFunc(num_obs)
        """Barrier函数，输入状态、控制输入、障碍物列表和超参数t，返回Barrier函数值"""
        self.is_feasible_func: ca.Function = self.__buildIsFeasibleFunc()
        """判断解是否可行的函数，输入状态、控制输入和障碍物列表，返回此单步解是否可行"""
        self.is_traj_feasible_func = self.__buildIsTrajFeasibleFunc()
        """判断轨迹是否可行的函数，输入状态序列、控制序列和障碍物列表，返回整个轨迹是否可行"""

    @staticmethod
    def LogarithmicBarrier(param: ca.MX) -> ca.MX:
        """
        计算对数Barrier函数：b = -log(-param)，
        注意为了计算效率这里没有除以超参数t而是放在后面统一处理。
        """
        return -ca.log(ca.fmax(-param, Config.ZERO_EPS))

    @staticmethod
    def ChainConditions(conditions: List[ca.MX], index: int = 0) -> ca.MX:
        """递归构造嵌套if_else表达式，满足所有条件时返回1，否则返回0"""
        if index == len(conditions):
            return ca.MX(1)
        else:
            return ca.if_else(
                conditions[index] < 0.0,
                BarrierFunc.ChainConditions(conditions, index + 1),
                ca.MX(0),
            )

    def __addValRangeConstraint(self, param: ca.MX, lower: float, upper: float):
        """添加值域约束：lower - param < 0, param - upper < 0"""
        self.conditions.append(lower - param)
        self.conditions.append(param - upper)

    def __addObsAvoidanceConstraint(
        self,
        state: ca.MX,
        half_wheel_base: float,
        obs: ca.MX,
    ):
        """添加障碍物避障约束：safe_dist_squared - dist_squared < 0"""
        safe_dist_squared = (obs[2] + half_wheel_base) ** 2
        # 默认车辆以后轮中心为坐标原点，所以需要额外计算出车辆的几何中心坐标
        car_x = state[0] + half_wheel_base * ca.cos(state[2])
        car_y = state[1] + half_wheel_base * ca.sin(state[2])
        dist_squared = (car_x - obs[0]) ** 2 + (car_y - obs[1]) ** 2
        self.conditions.append(safe_dist_squared - dist_squared)

    def __buildBarrierFunc(self, num_obs):
        """输入障碍物数量构建默认的Barrier函数"""
        # 添加值域约束
        for param, lower, upper in [
            (self.X[3], -Config.MAX_SPEED, Config.MAX_SPEED),  # v
            (self.U[0], -Config.MAX_ACCEL, Config.MAX_ACCEL),  # a
            (self.U[1], -Config.MAX_STEER, Config.MAX_STEER),  # steer
        ]:
            self.__addValRangeConstraint(param, lower, upper)
        # 添加障碍物避障约束
        if num_obs > 0:
            half_wheel_base = 0.5 * Config.WHEEL_BASE
            for i in range(num_obs):
                self.__addObsAvoidanceConstraint(
                    self.X, half_wheel_base, self.OBS_LIST[:, i]
                )
        barrier_terms = ca.vertcat(
            *[BarrierFunc.LogarithmicBarrier(c) for c in self.conditions]
        )
        return Primarch.BuildFunc(
            [self.X, self.U, self.OBS_LIST, self.T],
            [ca.sum1(barrier_terms) / self.T],
        )

    def __buildIsFeasibleFunc(self):
        """构建判断解是否可行的函数"""
        return Primarch.BuildFunc(
            [self.X, self.U, self.OBS_LIST],
            [BarrierFunc.ChainConditions(self.conditions)],
        )

    def __buildIsTrajFeasibleFunc(self):
        # 内部递归函数：对从index开始调用is_feasible_func判断是否可行
        def recursiveCheck(index):
            if index == self.config.N - 1:
                return ca.MX(1)
            else:
                return ca.if_else(
                    self.is_feasible_func(
                        self.X_SEQ[:, index], self.U_SEQ[:, index], self.OBS_LIST
                    ),
                    recursiveCheck(index + 1),
                    ca.MX(0),
                )

        return Primarch.BuildFunc(
            [self.X_SEQ, self.U_SEQ, self.OBS_LIST],
            [recursiveCheck(0)],
        )


class CILQR(Primarch):
    """CILQR(Constrained iLQR)控制器"""

    def __init__(
        self,
        config: Config,
        num_obs: int,
    ):
        """使用障碍物数量，状态转移函数，代价函数和终端代价函数初始化CILQR控制器"""
        self.config = config
        """配置"""
        self.T = ca.MX.sym("t")  # type: ignore
        """Barrier函数的超参数t符号变量"""
        self.barrier_func = BarrierFunc(self.config, num_obs)
        """Barrier函数"""
        super().__init__(self.config, num_obs)

        self.t = self.config.T_LIST[0]
        """Barrier函数的超参数t"""

    def buildStateTransitionFunc(self) -> ca.Function:
        return self.defStateTransitionFunc

    def buildCostFunc(self) -> ca.Function:
        cost = self.defCostFunc(self.X, self.U, self.X_REF)
        cost += self.barrier_func.barrier_func(
            self.X, self.U, self.OBS_LIST, self.T
        )  # type: ignore
        return Primarch.BuildFunc(
            [self.X, self.U, self.X_REF, self.OBS_LIST, self.T],
            [cost],
        )

    def buildFinalCostFunc(self) -> ca.Function:
        # 为了简便在终端处不考虑barrier函数。
        return self.defFinalCostFunc

    def buildBatchCostDerivativesFunc(self) -> ca.Function:
        # 先构建计算单步代价函数的偏导的函数
        l = self.cost_func(self.X, self.U, self.X_REF, self.OBS_LIST, self.T)
        l_x = ca.gradient(l, self.X)
        l_u = ca.gradient(l, self.U)
        l_xx = ca.hessian(l, self.X)[0]
        l_uu = ca.hessian(l, self.U)[0]
        l_ux = ca.jacobian(l_u, self.X)
        cost_derivatives_func = Primarch.BuildFunc(
            [self.X, self.U, self.X_REF, self.OBS_LIST, self.T],
            [l_x, l_u, l_xx, l_uu, l_ux],
        )
        func_map = cost_derivatives_func.map(self.config.N - 1)
        lx_seq, lu_seq, lxx_seq, luu_seq, lux_seq = func_map(
            self.X_SEQ[:, : self.config.N - 1],
            self.U_SEQ,
            self.X_SEQ_REF[:, : self.config.N - 1],
            self.OBS_LIST,
            self.T,
        )
        return Primarch.BuildFunc(
            [self.X_SEQ, self.U_SEQ, self.X_SEQ_REF, self.OBS_LIST, self.T],
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
            self.OBS_LIST,
            self.T,
        )
        final_cost = self.final_cost_func(self.X_SEQ[:, -1], self.X_SEQ_REF[:, -1])
        return Primarch.BuildFunc(
            [self.X_SEQ, self.U_SEQ, self.X_SEQ_REF, self.OBS_LIST, self.T],
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
            self.x_seq, self.u_seq, self.x_seq_ref, self.obs_list, self.t  # type: ignore
        )

    def solve(self, x_init: NDArray, x_seq_ref: NDArray) -> Tuple[NDArray, NDArray]:
        self.initSolver(x_init, x_seq_ref)
        self.J = self.__getTrajCost(self.x_seq, self.u_seq)
        for t in self.config.T_LIST:
            self.t = t
            is_traj_updated = self.__innerLoop()
            if not is_traj_updated:
                Log.info("No feasible trajectory found, t = {:.3f}".format(self.t))
                break
        x_seq_np = self.x_seq.full()
        u_seq_np = self.u_seq.full()
        self.setCtrlSeqTrialForNextIter()
        return x_seq_np, u_seq_np

    def __isTrajFeasible(self, x_seq: ca.DM, u_seq: ca.DM) -> bool:
        """判断轨迹是否可行"""
        return bool(
            self.barrier_func.is_traj_feasible_func(x_seq, u_seq, self.obs_list)
        )

    def __getTrajCost(self, x_seq: ca.DM, u_seq: ca.DM) -> float:
        """计算轨迹代价"""
        return float(
            self.traj_cost_func(
                x_seq, u_seq, self.x_seq_ref, self.obs_list, self.t
            )  # type: ignore
        )

    def __innerLoop(self) -> bool:
        """
        对应论文中的inner loop，即在固定t的情况下使用基础iLQR算法求解。
        最后会返回是否更新了轨迹，可以据此判断是否该终止外层的循环。
        """
        # 需要根据当前的t更新现有轨迹的代价
        self.J = self.__getTrajCost(self.x_seq, self.u_seq)
        delta_J = np.inf
        is_traj_updated = False
        self.reg = self.config.REG_INIT
        for _ in range(self.config.MAX_ITER):
            self.backwardPass()
            for alpha in self.config.ALPHA_LIST:
                self.forwardPass(alpha)
                is_reg_decreased = False
                if self.__isTrajFeasible(self.x_seq_temp, self.u_seq_temp):
                    J_new = self.__getTrajCost(self.x_seq_temp, self.u_seq_temp)
                    delta_J = self.J - J_new
                    if delta_J > Config.FLOAT_EPS:
                        # 接受更新并降低正则化系数
                        self.acceptNewSol(self.x_seq_temp, self.u_seq_temp, J_new)
                        self.updateReg(0.5)
                        is_traj_updated = True
                        is_reg_decreased = True
                        break
                if not is_reg_decreased:
                    self.updateReg(2.0)
            if abs(delta_J) < self.config.TOL:
                break
        return is_traj_updated


class CILQRDemo(TrafficSimulator):
    """CILQR演示类"""

    def __init__(self, config_name="config.json") -> None:
        super().__init__(config_name)
        self.config = Config()
        self.config.PrintConfig(self.config)
        self.car = self.ego_car
        """受控车辆，本质是一个引用"""
        self.controller = CILQR(self.config, num_obs=len(self.data_mgr.map.obstacles))
        """CILQR控制器"""
        self.x_seq_ref = np.zeros((self.config.N, self.car.model.state_dim))
        """用于CILQR计算的参考状态序列"""
        self.graphic_item_mgr.addExtraCurveItem(
            "reference trajectory", BLUE, QtCore.Qt.PenStyle.DotLine
        )
        """添加额外的参考轨迹图形项"""
        self.graphic_item_mgr.addExtraCurveItem(
            "predicted trajectory", RED, QtCore.Qt.PenStyle.DotLine
        )
        """添加额外的预测轨迹图形项"""

    def update(self) -> None:
        if self.getDistToDest() > 1.0 * Config.WHEEL_BASE:
            self.__updateRefPtAndStateSeq()
            self.__updateObstacleInfo()
            state_seq_predicted, ctrl_seq_opt = self.controller.solve(
                self.car.model.state, self.x_seq_ref.T
            )
            self.getCtrlInput().setVal(ctrl_seq_opt[0][0], ctrl_seq_opt[1][0])
            self.__updateGraphicItems(state_seq_predicted.T)
        else:
            Log.info("Arrived at destination")
            self.finalize()

    def __updateObstacleInfo(self):
        """更新障碍物信息"""
        for i, obs in enumerate(self.data_mgr.map.obstacles):
            self.controller.obs_list[:, i] = np.array([obs.x, obs.y, obs.radius])

    def __updateRefPtAndStateSeq(self):
        """更新参考点和参考状态序列"""
        ref_pt = self.getRefPt()
        ref_line = self.getRefLine()
        for i in range(self.config.N):
            v_ref = ref_pt.max_speed_rate * Config.MAX_SPEED
            dist = v_ref * Config.DELTA_T
            if ref_line.isCloseToDestination(ref_pt):
                v_ref = 0.0
            self.x_seq_ref[i, 0] = ref_pt.x
            self.x_seq_ref[i, 1] = ref_pt.y
            self.x_seq_ref[i, 2] = ref_pt.theta
            self.x_seq_ref[i, 3] = v_ref
            ref_pt = ref_line.getPtAfterDistance(ref_pt.id, dist)

    def __updateGraphicItems(self, state_seq_predicted: NDArray) -> None:
        """更新繁杂的图像显示，分离出来避免干扰算法主逻辑"""
        state_seq_ref = self.x_seq_ref.copy()
        ref_pt = self.getRefPt()
        state_seq_ref[0] = [ref_pt.x, ref_pt.y, ref_pt.theta, 0.0]
        self.graphic_item_mgr.setExtraCurveData("reference trajectory", state_seq_ref)
        self.graphic_item_mgr.setExtraCurveData(
            "predicted trajectory", state_seq_predicted
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    demo = CILQRDemo("config1.json")
    demo.show()
    app.exec()
