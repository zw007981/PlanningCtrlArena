# -*- coding: utf-8 -*-

import os
import sys

sys.path.append(os.getcwd())


from abc import ABC, abstractmethod
from typing import List, Tuple

import casadi as ca
import numpy as np
from numpy.typing import NDArray
from PySide6 import QtCore
from PySide6.QtWidgets import QApplication

from src.traffic_simulator.color import *
from src.traffic_simulator.traffic_simulator import TrafficSimulator
from src.util.config import Config as BaseConfig
from src.util.log import Log


class Config(BaseConfig):
    """iLQR控制器配置"""

    def __init__(self) -> None:
        self.N = 25
        """预测步数"""
        self.TOL = 0.01
        """收敛容许误差，小于该值则认为已收敛"""
        self.MAX_ITER = 40
        """最大迭代求解次数"""
        self.REG_INIT = 1.234
        """初始正则化系数"""
        self.REG_MIN = 1e-3
        """正则化系数最小值"""
        self.REG_MAX = 1e3
        """正则化系数最大值"""
        self.ALPHA_LIST = [0.666**i for i in range(6)]
        """线搜索步长列表"""
        self.X_ERROR_WEIGHT = 1.0
        """x方向误差权重"""
        self.Y_ERROR_WEIGHT = 1.0
        """y方向误差权重"""
        self.THETA_ERROR_WEIGHT = 0.8
        """角度误差权重"""
        self.V_ERROR_WEIGHT = 0.6
        """速度误差权重"""
        self.X_ERROR_WEIGHT_F = 1.2
        """终端x方向误差权重"""
        self.Y_ERROR_WEIGHT_F = 1.2
        """终端y方向误差权重"""
        self.THETA_ERROR_WEIGHT_F = 0.8
        """终端角度误差权重"""
        self.V_ERROR_WEIGHT_F = 1.0
        """终端速度误差权重"""
        self.ACC_WEIGHT = 0.4
        """加速度权重"""
        self.STEER_WEIGHT = 0.8
        """转向角权重"""
        self.modifyConfig()

    def modifyConfig(self) -> None:
        """根据需要修改配置"""
        pass


class Primarch(ABC):
    """iLQR控制器基类，提取出公共部分以简化后续的iLQR，CILQR和AL-iLQR算法开发"""

    FUNC_COUNT: int = 0
    """CasADi函数计数器，用于生成唯一的函数名"""

    def __init__(self, config: Config, num_obs: int = 0) -> None:
        """使用环境中障碍物的数量初始化iLQR控制器基类"""
        # 参数========================================================
        self.config = config
        """iLQR控制器配置"""
        self.num_obs = num_obs
        """障碍物数量"""
        self.reg = self.config.REG_INIT
        """正则化系数"""
        self.J = 0.0
        """最优轨迹代价"""
        N, STATE_DIM, CTRL_DIM = (
            self.config.N,
            self.config.STATE_DIM,
            self.config.CTRL_DIM,
        )
        # 数据=========================================================
        self.x_seq: ca.DM = ca.DM.zeros(STATE_DIM, N)  # type: ignore
        """最优状态序列，其中元素的维度：STATE_DIM"""
        self.u_seq: ca.DM = ca.DM.ones(CTRL_DIM, N - 1)  # type: ignore
        """最优控制序列，其中元素的维度：CTRL_DIM"""
        self.x_seq_ref: ca.DM = ca.DM.zeros(STATE_DIM, N)  # type: ignore
        """参考状态序列，其中元素的维度：STATE_DIM"""
        self.x_seq_temp: ca.DM = ca.DM.zeros(STATE_DIM, N)  # type: ignore
        """
        临时状态序列，其中元素的维度：STATE_DIM。主要用在计算出新轨迹但还没有更新
        到x_seq时，初始化时分配好内存尽量避免运行中对堆内存的频繁申请和释放。
        """
        self.u_seq_temp: ca.DM = ca.DM.zeros(CTRL_DIM, N - 1)  # type: ignore
        """临时控制序列，其中元素的维度：CTRL_DIM"""
        self.K_seq: ca.DM = ca.DM.zeros(CTRL_DIM, STATE_DIM * (N - 1))  # type: ignore
        """反馈增益序列，其中元素的维度：CTRL_DIM * STATE_DIM"""
        self.d_seq: ca.DM = ca.DM.zeros(CTRL_DIM, N - 1)  # type: ignore
        """前馈控制量序列，其中元素的维度：CTRL_DIM"""
        self.A_seq: ca.DM = ca.DM.zeros(STATE_DIM, STATE_DIM * (N - 1))  # type: ignore
        """状态转移函数对状态的偏导序列，其中元素的维度：STATE_DIM * STATE_DIM"""
        self.B_seq: ca.DM = ca.DM.zeros(STATE_DIM, CTRL_DIM * (N - 1))  # type: ignore
        """状态转移函数对控制量的偏导序列，其中元素的维度：STATE_DIM * CTRL_DIM"""
        self.lx_seq: ca.DM = ca.DM.zeros(STATE_DIM, N - 1)  # type: ignore
        """代价函数对状态的偏导序列，其中元素的维度：STATE_DIM"""
        self.lu_seq: ca.DM = ca.DM.zeros(CTRL_DIM, N - 1)  # type: ignore
        """代价函数对控制量的偏导序列，其中元素的维度：CTRL_DIM"""
        self.lxx_seq: ca.DM = ca.DM.zeros(STATE_DIM, STATE_DIM * (N - 1))  # type: ignore
        """代价函数对状态的二阶偏导序列，其中元素的维度：STATE_DIM * STATE_DIM"""
        self.luu_seq: ca.DM = ca.DM.zeros(CTRL_DIM, CTRL_DIM * (N - 1))  # type: ignore
        """代价函数对控制量的二阶偏导序列，其中元素的维度：CTRL_DIM * CTRL_DIM"""
        self.lux_seq: ca.DM = ca.DM.zeros(CTRL_DIM, STATE_DIM * (N - 1))  # type: ignore
        """代价函数对控制量和状态的偏导序列，其中元素的维度：CTRL_DIM * STATE_DIM"""
        self.lf_x: ca.DM = ca.DM.zeros(1, STATE_DIM)  # type: ignore
        """终端代价函数对状态的偏导，维度：1*STATE_DIM"""
        self.lf_xx: ca.DM = ca.DM.zeros(STATE_DIM, STATE_DIM)  # type: ignore
        """终端代价函数对状态的二阶偏导，维度：STATE_DIM*STATE_DIM"""
        self.obs_list = ca.DM.zeros(3, num_obs)  # type: ignore
        """
        障碍物列表，第一行是x坐标，第二行是y坐标，第三行是半径，
        在基础iLQR中无用，用于后面带约束的版本
        """
        # 符号变量=====================================================
        self.X: ca.MX = ca.MX.sym("x", STATE_DIM, 1)  # type: ignore
        """状态符号"""
        self.U: ca.MX = ca.MX.sym("u", CTRL_DIM, 1)  # type: ignore
        """控制量符号"""
        self.X_REF: ca.MX = ca.MX.sym("x_ref", STATE_DIM, 1)  # type: ignore
        """参考状态符号"""
        self.X_SEQ: ca.MX = ca.MX.sym("x_seq", STATE_DIM, N)  # type: ignore
        """状态序列符号"""
        self.U_SEQ: ca.MX = ca.MX.sym("u_seq", CTRL_DIM, N - 1)  # type: ignore
        """控制序列符号"""
        self.X_SEQ_REF: ca.MX = ca.MX.sym("x_seq_ref", STATE_DIM, N)  # type: ignore
        """参考状态序列符号"""
        self.A_SEQ: ca.MX = ca.MX.sym("A_seq", STATE_DIM, STATE_DIM * (N - 1))  # type: ignore
        """状态转移函数对状态的偏导序列符号"""
        self.B_SEQ: ca.MX = ca.MX.sym("B_seq", STATE_DIM, CTRL_DIM * (N - 1))  # type: ignore
        """状态转移函数对控制量的偏导序列符号"""
        self.LX_SEQ: ca.MX = ca.MX.sym("lx_seq", STATE_DIM, N - 1)  # type: ignore
        """代价函数对状态的偏导序列符号"""
        self.LU_SEQ: ca.MX = ca.MX.sym("lu_seq", CTRL_DIM, N - 1)  # type: ignore
        """代价函数对控制量的偏导序列符号"""
        self.LXX_SEQ: ca.MX = ca.MX.sym("lxx_seq", STATE_DIM, STATE_DIM * (N - 1))  # type: ignore
        """代价函数对状态的二阶偏导序列符号"""
        self.LUU_SEQ: ca.MX = ca.MX.sym("luu_seq", CTRL_DIM, CTRL_DIM * (N - 1))  # type: ignore
        """代价函数对控制量的二阶偏导序列符号"""
        self.LUX_SEQ: ca.MX = ca.MX.sym("lux_seq", CTRL_DIM, STATE_DIM * (N - 1))  # type: ignore
        """代价函数对控制量和状态的偏导序列符号"""
        self.OBS_LIST = ca.MX.sym("obs_list", 3, num_obs)  # type: ignore
        """障碍物列表符号"""
        # 函数=========================================================
        self.state_transition_func: ca.Function = self.buildStateTransitionFunc()
        """状态转移函数，默认输入状态和控制量，返回下一个状态"""
        self.cost_func: ca.Function = self.buildCostFunc()
        """代价函数，默认输入状态，控制量和参考状态，返回代价"""
        self.final_cost_func: ca.Function = self.buildFinalCostFunc()
        """终端代价函数，默认输入状态和参考状态，返回代价"""
        self.batch_state_transition_derivatives_func: ca.Function = (
            self.buildBatchStateTransitionDerivativesFunc()
        )
        """用于批量计算状态转移函数的偏导"""
        self.batch_cost_derivatives_func: ca.Function = (
            self.buildBatchCostDerivativesFunc()
        )
        """用于批量计算代价函数的偏导"""
        self.final_cost_derivatives_func: ca.Function = (
            self.buildFinalCostDerivativesFunc()
        )
        """终端代价函数的偏导数函数，默认输入状态和参考状态，返回lf_x, lf_xx"""
        self.traj_cost_func: ca.Function = self.buildTrajCostFunc()
        """轨迹代价函数，默认输入状态序列，控制量序列和参考状态序列，返回轨迹代价"""
        self.roll_out_func: ca.Function = self.buildRollOutFunc()
        """滚动预测函数，默认输入现有状态序列和控制序列，返回预测状态序列"""
        self.forward_pass_func: ca.Function = self.buildForwardPassFunc()
        """
        前向传播函数，默认输入现有状态序列，控制序列，反馈增益序列，前馈控制量序列和线搜索步长，
        返回更新后的状态序列和控制序列
        """
        self.backward_pass_func: ca.Function = self.buildBackwardPassFunc()
        """
        后向传播函数，默认输入A_seq, B_seq, lx_seq, lu_seq, lxx_seq, luu_seq, lux_seq, lf_x, lf_xx和正则化系数，
        返回反馈增益序列和前馈控制量序列。
        """

    @staticmethod
    def BuildFunc(inputs: List, outputs: List, name: str = "") -> ca.Function:
        """
        根据输入、输出和函数名构建CasADi函数并返回。
        参数:
        inputs: 输入变量列表。
        outputs: 输出变量列表。
        name: 函数名，默认为空字符串。
        返回:
        ca.Function: 构建的CasADi函数。
        """
        name = name if name else "Primarch_func{:d}".format(Primarch.FUNC_COUNT)
        Primarch.FUNC_COUNT += 1
        try:
            input_names = [input_.name() for input_ in inputs]
            if len(set(input_names)) != len(input_names):
                raise ValueError("Input names must be unique!!!")
            output_names = ["output{:d}".format(i) for i in range(len(outputs))]
            return ca.Function(
                name, inputs, outputs, input_names, output_names
            ).expand()
        except Exception as e:
            Log.warning(
                "Failed to build complete function, using default settings: {}!".format(
                    e
                )
            )
            return ca.Function(name, inputs, outputs)

    @property
    def defStateTransitionFunc(self) -> ca.Function:
        """默认状态转移函数，输入状态和控制量，返回下一个状态"""
        from src.kinematic_model.bicycle_model import BicycleModel

        delta_x = Config.DELTA_T * BicycleModel.stateDerivative(self.X, self.U)
        return Primarch.BuildFunc([self.X, self.U], [self.X + delta_x])

    @property
    def defCostFunc(self) -> ca.Function:
        """默认代价函数，输入状态，控制量和参考状态，返回代价"""
        delta_x = self.X - self.X_REF
        Q = ca.diag(
            [
                self.config.X_ERROR_WEIGHT,
                self.config.Y_ERROR_WEIGHT,
                self.config.THETA_ERROR_WEIGHT,
                self.config.V_ERROR_WEIGHT,
            ]
        )
        R = ca.diag([self.config.ACC_WEIGHT, self.config.STEER_WEIGHT])
        cost = 0.5 * (delta_x.T @ Q @ delta_x + self.U.T @ R @ self.U)
        return Primarch.BuildFunc([self.X, self.U, self.X_REF], [cost])

    @property
    def defFinalCostFunc(self) -> ca.Function:
        """默认终端代价函数，输入状态和参考状态，返回代价"""
        delta_x = self.X - self.X_REF
        Qf = ca.diag(
            [
                self.config.X_ERROR_WEIGHT_F,
                self.config.Y_ERROR_WEIGHT_F,
                self.config.THETA_ERROR_WEIGHT_F,
                self.config.V_ERROR_WEIGHT_F,
            ]
        )
        cost = 0.5 * delta_x.T @ Qf @ delta_x
        return Primarch.BuildFunc([self.X, self.X_REF], [cost])

    @abstractmethod
    def buildStateTransitionFunc(self) -> ca.Function:
        """构造状态转移函数"""
        pass

    @abstractmethod
    def buildCostFunc(self) -> ca.Function:
        """构造代价函数"""
        pass

    @abstractmethod
    def buildFinalCostFunc(self) -> ca.Function:
        """构造终端代价函数"""
        pass

    @abstractmethod
    def buildBatchCostDerivativesFunc(self) -> ca.Function:
        """构造用于批量计算代价函数的偏导的函数"""
        pass

    @abstractmethod
    def buildFinalCostDerivativesFunc(self) -> ca.Function:
        """构造用于计算终端代价函数的偏导的函数"""
        pass

    @abstractmethod
    def buildTrajCostFunc(self) -> ca.Function:
        """构造轨迹代价函数"""
        pass

    @abstractmethod
    def precomputeDerivatives(self):
        """预计算各个偏导数"""
        pass

    @abstractmethod
    def solve(self, x_init: NDArray, x_seq_ref: NDArray) -> Tuple[NDArray, NDArray]:
        """
        输入初始状态和参考轨迹，返回最优状态和控制序列。
        注意在这里状态和控制量都是列向量。
        """
        pass

    def initSolver(self, x_init: NDArray, x_seq_ref: NDArray):
        """
        输入机器人的初始状态和参考状态序列以初始化求解器，
        包含推演初始轨迹和初始化正则化系数两部分。
        """
        for i in range(Config.STATE_DIM):
            self.x_seq[i, 0] = x_init[i]
        self.x_seq_ref = ca.DM(x_seq_ref)
        self.rollOut()
        self.reg = self.config.REG_INIT

    def acceptNewSol(self, x_seq_new: ca.DM, u_seq_new: ca.DM, J_new: float):
        """接受新解"""
        # 由于Python的GC机制这种写法应该是安全的。
        self.x_seq, x_seq_new = x_seq_new, self.x_seq
        self.u_seq, u_seq_new = u_seq_new, self.u_seq
        self.J = J_new

    def updateReg(self, coef: float):
        """输入乘数以更新正则化系数"""
        self.reg = np.clip(coef * self.reg, self.config.REG_MIN, self.config.REG_MAX)

    def setCtrlSeqTrialForNextIter(self):
        """
        假设会执行第一个控制量并会在下一个时间步重新调用iLQR算法，
        因此把过滤掉第一个控制量后的控制序列作为下一次求解的初始猜测。
        """
        self.u_seq[:, :-1] = self.u_seq[:, 1:]

    def buildBatchStateTransitionDerivativesFunc(self) -> ca.Function:
        """构造用于批量计算状态转移函数的偏导的函数"""
        # 先构建计算单步状态转移函数的偏导的函数
        x_next = self.state_transition_func(self.X, self.U)
        A = ca.jacobian(x_next, self.X)
        B = ca.jacobian(x_next, self.U)
        state_transition_derivatives_func = Primarch.BuildFunc([self.X, self.U], [A, B])
        func_map = state_transition_derivatives_func.map(self.config.N - 1)
        A_seq, B_seq = func_map(self.X_SEQ[:, : self.config.N - 1], self.U_SEQ)
        return Primarch.BuildFunc([self.X_SEQ, self.U_SEQ], [A_seq, B_seq])

    def buildRollOutFunc(self) -> ca.Function:
        """构造用于滚动预测的函数"""
        x_seq_new = ca.MX.zeros(Config.STATE_DIM, self.config.N)  # type: ignore
        x_seq_new[:, 0] = self.X_SEQ[:, 0]
        for k in range(self.config.N - 1):
            x_seq_new[:, k + 1] = self.state_transition_func(
                x_seq_new[:, k], self.U_SEQ[:, k]
            )
        return Primarch.BuildFunc([self.X_SEQ, self.U_SEQ], [x_seq_new])

    def rollOut(self):
        """基于初始状态和控制序列，更新状态序列"""
        self.x_seq = self.roll_out_func(self.x_seq, self.u_seq)  # type: ignore

    def buildForwardPassFunc(self) -> ca.Function:
        """构造前向传播函数"""
        K_SEQ = ca.MX.sym("K_seq", Config.CTRL_DIM, Config.STATE_DIM * (self.config.N - 1))  # type: ignore
        D_SEQ = ca.MX.sym("d_seq", Config.CTRL_DIM, self.config.N - 1)  # type: ignore
        ALPHA = ca.MX.sym("alpha")  # type: ignore
        x_seq_new = ca.MX.zeros(Config.STATE_DIM, self.config.N)  # type: ignore
        u_seq_new = ca.MX.zeros(Config.CTRL_DIM, self.config.N - 1)  # type: ignore
        x_seq_new[:, 0] = self.X_SEQ[:, 0]
        for k in range(self.config.N - 1):
            K = K_SEQ[:, k * Config.STATE_DIM : (k + 1) * Config.STATE_DIM]
            delta_x = x_seq_new[:, k] - self.X_SEQ[:, k]
            delta_u = ALPHA * D_SEQ[:, k] + K @ delta_x
            u_seq_new[:, k] = self.U_SEQ[:, k] + delta_u
            x_seq_new[:, k + 1] = self.state_transition_func(
                x_seq_new[:, k], u_seq_new[:, k]
            )
        return Primarch.BuildFunc(
            [self.X_SEQ, self.U_SEQ, K_SEQ, D_SEQ, ALPHA],
            [x_seq_new, u_seq_new],
            name="forward_pass_func",
        )

    def forwardPass(self, alpha: float = 1.0):
        """前向传播，更新 self.x_seq_temp 和 self.u_seq_temp"""
        self.x_seq_temp, self.u_seq_temp = self.forward_pass_func(
            self.x_seq,
            self.u_seq,
            self.K_seq,
            self.d_seq,
            alpha,
        )  # type: ignore

    def buildBackwardPassFunc(self) -> ca.Function:
        """构造后向传播函数"""
        REG = ca.MX.sym("reg")  # type: ignore
        LF_X = ca.MX.sym("lf_x", 1, Config.STATE_DIM)  # type: ignore
        LF_XX = ca.MX.sym("lf_xx", Config.STATE_DIM, Config.STATE_DIM)  # type: ignore
        K_seq = ca.MX.zeros(Config.CTRL_DIM, Config.STATE_DIM * (self.config.N - 1))  # type: ignore
        d_seq = ca.MX.zeros(Config.CTRL_DIM, self.config.N - 1)  # type: ignore
        V_x = LF_X.T
        V_xx = LF_XX
        I = ca.MX.eye(Config.CTRL_DIM)  # type: ignore
        for k in range(self.config.N - 2, -1, -1):
            A = self.A_SEQ[:, k * Config.STATE_DIM : (k + 1) * Config.STATE_DIM]
            B = self.B_SEQ[:, k * Config.CTRL_DIM : (k + 1) * Config.CTRL_DIM]
            l_x = self.LX_SEQ[:, k]
            l_u = self.LU_SEQ[:, k]
            l_xx = self.LXX_SEQ[:, k * Config.STATE_DIM : (k + 1) * Config.STATE_DIM]
            l_uu = self.LUU_SEQ[:, k * Config.CTRL_DIM : (k + 1) * Config.CTRL_DIM]
            l_ux = self.LUX_SEQ[:, k * Config.STATE_DIM : (k + 1) * Config.STATE_DIM]
            Q_x = l_x + A.T @ V_x
            Q_u = l_u + B.T @ V_x
            Q_xx = l_xx + A.T @ V_xx @ A
            Q_uu = l_uu + B.T @ V_xx @ B
            Q_ux = l_ux + B.T @ V_xx @ A
            # 正则化
            Q_uu_reg = Q_uu + REG * B.T @ B
            Q_ux_reg = Q_ux + REG * B.T @ A
            # 计算反馈增益和前馈控制量并写入
            inv_Q_uu_reg = ca.solve(Q_uu_reg, I)
            K = -inv_Q_uu_reg @ Q_ux_reg
            d = -inv_Q_uu_reg @ Q_u
            K_seq[:, k * Config.STATE_DIM : (k + 1) * Config.STATE_DIM] = K
            d_seq[:, k] = d
            # 更新值函数
            V_x = Q_x + K.T @ Q_uu @ d + K.T @ Q_u + Q_ux.T @ d
            V_xx = Q_xx + K.T @ Q_uu @ K + K.T @ Q_ux + Q_ux.T @ K
        return Primarch.BuildFunc(
            [
                self.A_SEQ,
                self.B_SEQ,
                self.LX_SEQ,
                self.LU_SEQ,
                self.LXX_SEQ,
                self.LUU_SEQ,
                self.LUX_SEQ,
                LF_X,
                LF_XX,
                REG,
            ],
            [K_seq, d_seq],
            name="backward_pass_func",
        )

    def backwardPass(self):
        """后向传播，计算最优控制序列"""
        self.precomputeDerivatives()
        self.K_seq, self.d_seq = self.backward_pass_func(
            self.A_seq,
            self.B_seq,
            self.lx_seq,
            self.lu_seq,
            self.lxx_seq,
            self.luu_seq,
            self.lux_seq,
            self.lf_x,
            self.lf_xx,
            self.reg,
        )  # type: ignore


class iLQR(Primarch):
    """iLQR(Iterative Linear Quadratic Regulator)控制器"""

    def __init__(self, config: Config) -> None:
        super().__init__(config)

    def buildStateTransitionFunc(self) -> ca.Function:
        return self.defStateTransitionFunc

    def buildCostFunc(self) -> ca.Function:
        return self.defCostFunc

    def buildFinalCostFunc(self) -> ca.Function:
        return self.defFinalCostFunc

    def buildBatchCostDerivativesFunc(self) -> ca.Function:
        # 先构建计算单步代价函数的偏导的函数
        l = self.cost_func(self.X, self.U, self.X_REF)
        l_x = ca.gradient(l, self.X)
        l_u = ca.gradient(l, self.U)
        l_xx = ca.hessian(l, self.X)[0]
        l_uu = ca.hessian(l, self.U)[0]
        l_ux = ca.jacobian(l_u, self.X)
        cost_derivatives_func = Primarch.BuildFunc(
            [self.X, self.U, self.X_REF],
            [l_x, l_u, l_xx, l_uu, l_ux],
        )
        func_map = cost_derivatives_func.map(self.config.N - 1)
        lx_seq, lu_seq, lxx_seq, luu_seq, lux_seq = func_map(
            self.X_SEQ[:, : self.config.N - 1],
            self.U_SEQ,
            self.X_SEQ_REF[:, : self.config.N - 1],
        )
        return Primarch.BuildFunc(
            [self.X_SEQ, self.U_SEQ, self.X_SEQ_REF],
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
        )
        final_cost = self.final_cost_func(self.X_SEQ[:, -1], self.X_SEQ_REF[:, -1])
        return Primarch.BuildFunc(
            [self.X_SEQ, self.U_SEQ, self.X_SEQ_REF],
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
            self.x_seq, self.u_seq, self.x_seq_ref  # type: ignore
        )

    def solve(self, x_init: NDArray, x_seq_ref: NDArray) -> Tuple[NDArray, NDArray]:
        self.initSolver(x_init, x_seq_ref)
        self.J = float(
            self.traj_cost_func(self.x_seq, self.u_seq, self.x_seq_ref)  # type: ignore
        )
        delta_J = np.inf
        for _ in range(self.config.MAX_ITER):
            self.backwardPass()
            for alpha in self.config.ALPHA_LIST:
                self.forwardPass(alpha)
                J_new = float(
                    self.traj_cost_func(self.x_seq_temp, self.u_seq_temp, self.x_seq_ref)  # type: ignore
                )
                delta_J = self.J - J_new
                if delta_J > Config.FLOAT_EPS:
                    # 接受更新并降低正则化系数
                    self.acceptNewSol(self.x_seq_temp, self.u_seq_temp, J_new)
                    self.updateReg(0.5)
                    break
                else:
                    self.updateReg(2.0)
                    # Log.warning("Increase regularization coefficient!")
                self.reg = np.clip(self.reg, self.config.REG_MIN, self.config.REG_MAX)
            if abs(delta_J) < self.config.TOL:
                break
        x_seq_np = self.x_seq.full()
        u_seq_np = self.u_seq.full()
        self.setCtrlSeqTrialForNextIter()
        return x_seq_np, u_seq_np


class iLQRDemo(TrafficSimulator):
    """iLQR演示类"""

    def __init__(self, config_name="config.json") -> None:
        super().__init__(config_name)
        self.config = Config()
        self.config.PrintConfig(self.config)
        self.car = self.ego_car
        """受控车辆，本质是一个引用"""
        self.controller = iLQR(self.config)
        """iLQR控制器"""
        self.x_seq_ref = np.zeros((self.config.N, self.car.model.state_dim))
        """用于iLQR计算的参考状态序列"""
        self.graphic_item_mgr.addExtraCurveItem(
            "reference trajectory", BLUE, QtCore.Qt.PenStyle.DotLine
        )
        """添加额外的参考轨迹图形项"""
        self.graphic_item_mgr.addExtraCurveItem(
            "predicted trajectory", RED, QtCore.Qt.PenStyle.DotLine
        )
        """添加额外的预测轨迹图形项"""

    def update(self) -> None:
        if self.getDistToDest() > 1.0 * self.config.WHEEL_BASE:
            self.__updateRefPtAndStateSeq()
            state_seq_predicted, ctrl_seq_opt = self.controller.solve(
                self.car.model.state, self.x_seq_ref.T
            )
            self.getCtrlInput().setVal(ctrl_seq_opt[0][0], ctrl_seq_opt[1][0])
            self.__updateGraphicItems(state_seq_predicted.T)
        else:
            Log.info("Arrived at destination.")
            self.finalize()

    def __updateRefPtAndStateSeq(self):
        """更新参考点和参考状态序列"""
        ref_pt = self.getRefPt()
        ref_line = self.getRefLine()
        for i in range(self.config.N):
            v_ref = ref_pt.max_speed_rate * self.config.MAX_SPEED
            dist = v_ref * self.config.DELTA_T
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
    demo = iLQRDemo("config.json")
    demo.show()
    app.exec()
