# -*- coding: utf-8 -*-
# 论文中的动作集是离散的，一开始尝试通过MCTS方法来生成最优控制输入序列。
# 但是相较于梯度下降，蒙特卡洛方法的收敛速度较慢，所以最终未使用。

import os
import sys

sys.path.append(os.getcwd())

from typing import Callable, List, Optional, Tuple

import casadi as ca
import numpy as np
from numpy.typing import NDArray

from src.algo.iLQR import Primarch
from src.algo.level_k.action_set import ActionSet
from src.algo.level_k.config import Config
from src.algo.level_k.MCTS import MCTS
from src.kinematic_model.bicycle_model import BicycleModel
from src.traffic_simulator.color import *
from src.util.log import Log


class Node:
    """MCTS树节点，存储车辆在depth步选择的动作"""

    def __init__(self, node_id: int, depth: int, action: int, parent: int = -1) -> None:
        self.node_id: int = node_id
        """节点ID"""
        self.depth: int = depth
        """节点深度"""
        self.action: int = action
        """执行第几个动作"""
        self.val: float = 0.0
        """节点价值"""
        self.num_visits: int = 0
        """访问次数"""
        self.parent: int = parent
        """父节点ID"""
        self.children: List[int] = []
        """子节点列表"""

    def addChild(self, child_node_id: int) -> None:
        """添加子节点"""
        self.children.append(child_node_id)


class MCTS:
    """蒙特卡洛树搜索算法，用于通过蒙特卡洛方法选择最佳控制输入序列"""

    CONTROLS: List[Tuple[float, float]] = [
        (0.0, 0.0),  # MAINTAIN
        (0.0, 0.2 * Config.MAX_STEER),  # TURN_SLIGHTLY_LEFT
        (0.0, -0.2 * Config.MAX_STEER),  # TURN_SLIGHTLY_RIGHT
        (0.2 * Config.MAX_ACCEL, 0.0),  # NOMINAL_ACCELERATION
        (-0.2 * Config.MAX_ACCEL, 0.0),  # NOMINAL_DECELERATION
        (0.6 * Config.MAX_ACCEL, 0.0),  # MAXIMUM_ACCELERATION
        (-0.6 * Config.MAX_ACCEL, 0.0),  # MAXIMUM_DECELERATION
        (0.2 * Config.MAX_ACCEL, 0.6 * Config.MAX_STEER),  # TURN_LEFT_AND_ACCELERATE
        (0.2 * Config.MAX_ACCEL, -0.6 * Config.MAX_STEER),  # TURN_RIGHT_AND_ACCELERATE
    ]
    """从动作到具体控制输入的映射"""

    def __init__(self) -> None:
        self.nodes: List[Node] = []
        """MCTS树节点列表"""
        self.root: Node = Node(self.new_node_id, -1, -1)
        """根节点"""
        self.addNodeToTree(self.root)

    @property
    def new_node_id(self) -> int:
        """新节点的ID"""
        return len(self.nodes)

    def reset(self) -> None:
        """重置MCTS树"""
        self.nodes = []
        self.root = Node(self.new_node_id, -1, -1)
        self.addNodeToTree(self.root)

    def getBestCtrlInputSeq(self, calCostMethod: Callable[[ca.DM], float]) -> ca.DM:
        """返回最佳控制输入序列"""
        for _ in range(Config.MAX_NUM_TRIALS):
            self.simulate(calCostMethod)
        node = self.root
        while node.depth < Config.N - 2:
            if not node.children:
                self.expand(node)
            node = self.selChild(node, use_exploration=False)
        return self.constructCtrlInputSeq(node)

    def addNodeToTree(self, node: Node) -> None:
        """将节点添加到MCTS树中"""
        self.nodes.append(node)

    def expand(self, node: Node) -> None:
        """扩展叶子节点"""
        for action in range(ActionSet.Size()):
            child_node = Node(
                self.new_node_id, node.depth + 1, action, parent=node.node_id
            )
            node.addChild(child_node.node_id)
            self.addNodeToTree(child_node)

    def selChild(self, node: Node, use_exploration: bool = True) -> Node:
        """
        从node的子节点中基于UCB1值选择一个进行扩展，
        如果use_exploration为False则单纯选择价值最高的子节点。
        """
        max_ucb = float("-inf")
        best_child: Optional[Node] = None
        for child_id in node.children:
            child_node = self.nodes[child_id]
            if use_exploration and child_node.num_visits == 0:
                return child_node
            ucb = child_node.val
            if use_exploration:
                ucb += Config.EXPLORATION_COEFFICIENT * np.sqrt(
                    np.log(node.num_visits) / child_node.num_visits
                )
            if ucb > max_ucb + Config.FLOAT_EPS:
                max_ucb = ucb
                best_child = child_node
        if best_child is None:
            Log.error("Failed to select a child node!!!")
            raise ValueError("Failed to select a child node!!!")
        return best_child

    def constructCtrlInputSeq(self, node: Node) -> ca.DM:
        """从根节点回溯构建完整的控制输入序列"""
        ctrl_input_seq: ca.DM = ca.DM.zeros(Config.CTRL_DIM, Config.N - 1)  # type: ignore
        while node.depth >= 0:
            ctrl_input = MCTS.CONTROLS[node.action]
            ctrl_input_seq[0, node.depth] = ctrl_input[0]
            ctrl_input_seq[1, node.depth] = ctrl_input[1]
            node = self.nodes[node.parent]
        return ctrl_input_seq

    def simulate(self, calCostMethod: Callable[[ca.DM], float]) -> None:
        """从根节点开始一路向前选择直到最大深度，模拟得到价值后再回溯更新"""
        node = self.root
        while node.depth < Config.N - 2:
            if not node.children:
                self.expand(node)
            node = self.selChild(node)
        ctrl_input_seq = self.constructCtrlInputSeq(node)
        cost = calCostMethod(ctrl_input_seq)
        while True:
            node.num_visits += 1
            discount = Config.GAMMA ** (Config.N - 1 - node.depth)
            reward = cost * discount
            node.val += (reward - node.val) / node.num_visits
            if node.parent == -1:
                break
            else:
                node = self.nodes[node.parent]


class Planner:
    """基于采样的路径规划求解器，用于level-k算法"""

    # constexpr参数
    HALF_WHEEL_BASE: float = 0.5 * Config.WHEEL_BASE
    """半轴距"""
    SAFE_DIST_SQUARED: float = Config.WHEEL_BASE**2
    """安全距离的平方，如果两车中心距离的平方小于这个值，则认为发生碰撞"""
    COMFORT_DIST_SQUARED: float = Config.COMFORT_DISTANCE**2
    """舒适性距离的平方，如果两车中心距离的平方小于这个值，则认为两车过于接近"""

    def __init__(self, num_cars: int, N: int) -> None:
        """使用环境中车辆的数量和时间步数初始化求解器"""
        self.num_cars = num_cars
        """环境中车辆的数量"""
        self.car_id: int = 0
        """需要规划的车辆ID，这里需要转化为int形因为后续会作为索引使用"""
        self.state_seq: ca.DM = ca.DM.zeros(Config.STATE_DIM, N)  # type: ignore
        """此车的状态序列"""
        self.ref_state_seq: ca.DM = ca.DM.zeros(Config.STATE_DIM, N)  # type: ignore
        """此车的参考状态序列"""
        self.cars_x_seq: ca.DM = ca.DM.zeros(num_cars, N)  # type: ignore
        """所有车辆处于k-1级时规划的轨迹的x坐标序列，这里的坐标已做转换是车辆的中心而不是后轴中心"""
        self.cars_y_seq: ca.DM = ca.DM.zeros(num_cars, N)  # type: ignore
        """所有车辆处于k-1级时规划的轨迹的y坐标序列，这里的坐标已做转换是车辆的中心而不是后轴中心"""
        self.tree: MCTS = MCTS()
        """MCTS树，用于寻找最佳控制输入序列"""
        # 符号变量=====================================================
        self.CAR_ID: ca.MX = ca.MX.sym("car_id", 1)  # type: ignore
        """需要规划的车辆ID符号"""
        self.T: ca.MX = ca.MX.sym("t", 1)  # type: ignore
        """当前时间步符号"""
        self.STATE: ca.MX = ca.MX.sym("state", Config.STATE_DIM, 1)  # type: ignore
        """T时刻车辆的状态符号"""
        self.U: ca.MX = ca.MX.sym("u", Config.CTRL_DIM, 1)  # type: ignore
        """T时刻车辆的输入符号"""
        self.REF_STATE: ca.MX = ca.MX.sym("ref_state", Config.STATE_DIM, 1)  # type: ignore
        """T时刻车辆的参考状态符号"""
        self.STATE_SEQ: ca.MX = ca.MX.sym("state_seq", Config.STATE_DIM, N)  # type: ignore
        """此车轨迹的状态序列符号"""
        self.U_SEQ: ca.MX = ca.MX.sym("u_seq", Config.CTRL_DIM, N - 1)  # type: ignore
        """此车轨迹的输入序列符号"""
        self.REF_STATE_SEQ: ca.MX = ca.MX.sym("ref_state_seq", Config.STATE_DIM, N)  # type: ignore
        """此车轨迹的参考状态序列符号"""
        self.CARS_X: ca.MX = ca.MX.sym("cars_x", num_cars, 1)  # type: ignore
        """k-1级别规划方案中所有车辆在T时刻中心位置的x坐标符号"""
        self.CARS_Y: ca.MX = ca.MX.sym("cars_y", num_cars, 1)  # type: ignore
        """k-1级别规划方案中所有车辆在T时刻中心位置的y坐标符号"""
        self.CARS_X_SEQ: ca.MX = ca.MX.sym("cars_x_seq", num_cars, N)  # type: ignore
        """所有车辆处于k-1级时规划的轨迹的x坐标序列符号"""
        self.CARS_Y_SEQ: ca.MX = ca.MX.sym("cars_y_seq", num_cars, N)  # type: ignore
        """所有车辆处于k-1级时规划的轨迹的y坐标序列符号"""
        # 函数=====================================================
        self.cost_func: ca.Function = self.buildCostFunc()
        """单步代价函数"""
        self.traj_cost_func: ca.Function = self.buildTrajCostFunc()
        """轨迹代价函数"""
        self.state_transition_func: ca.Function = self.buildStateTransitionFunc()
        """状态转移函数"""
        self.rollout_func: ca.Function = self.buildRolloutFunc()
        """rollout函数"""
        self.eval_ctrl_input_seq_func: ca.Function = self.buildEvalCtrlInputSeqFunc()
        """评估控制输入序列函数"""

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

    def buildCostFunc(self) -> ca.Function:
        """构建单步代价函数"""
        # 与参考状态的基础偏差
        delta_state = self.STATE - self.REF_STATE
        Q = ca.diag(
            [
                Config.X_ERROR_WEIGHT,
                Config.Y_ERROR_WEIGHT,
                Config.THETA_ERROR_WEIGHT,
                Config.V_ERROR_WEIGHT,
            ]
        )
        R = ca.diag([Config.ACC_WEIGHT, Config.STEER_WEIGHT])
        base_cost = 0.5 * (delta_state.T @ Q @ delta_state + self.U.T @ R @ self.U)
        # 默认车辆以后轮中心为坐标原点，所以需要额外计算出车辆的几何中心坐标
        car_x = self.STATE[0] + self.HALF_WHEEL_BASE * ca.cos(self.STATE[2])
        car_y = self.STATE[1] + self.HALF_WHEEL_BASE * ca.sin(self.STATE[2])
        # 计算当前时间步与所有车辆距离的平方，同时利用掩码把到自身的距离设为一个较大的数
        idx = ca.DM(list(range(self.num_cars)))
        mask = ca.if_else(idx == self.CAR_ID, 0, 1)
        dist_squared = (
            (car_x - self.CARS_X) ** 2
            + (car_y - self.CARS_Y) ** 2
            + (1 - mask) * 666666
        )
        # 找出最小距离并检查是否小于安全距离和舒适距离
        min_dist_squared = ca.mmin(dist_squared)
        coll_cost = ca.if_else(
            min_dist_squared < Planner.SAFE_DIST_SQUARED,
            Config.COLLISION_WEIGHT,
            0,
        )
        comfort_cost = ca.if_else(
            min_dist_squared < Planner.COMFORT_DIST_SQUARED,
            Config.COMFORT_WEIGHT,
            0,
        )
        cost = (Config.GAMMA**self.T) * (base_cost + coll_cost + comfort_cost)
        return Planner.BuildFunc(
            [
                self.STATE,
                self.U,
                self.REF_STATE,
                self.CAR_ID,
                self.T,
                self.CARS_X,
                self.CARS_Y,
            ],
            [cost],
        )

    def buildTrajCostFunc(self) -> ca.Function:
        """构建轨迹代价函数"""
        stage_costs = self.cost_func.map(Config.N - 1)(
            self.STATE_SEQ[:, : Config.N - 1],
            self.U_SEQ,
            self.REF_STATE_SEQ[:, : Config.N - 1],
            ca.repmat(self.CAR_ID, 1, Config.N - 1),
            ca.DM(range(Config.N - 1)),
            self.CARS_X_SEQ[:, : Config.N - 1],
            self.CARS_Y_SEQ[:, : Config.N - 1],
        )
        return Planner.BuildFunc(
            [
                self.STATE_SEQ,
                self.U_SEQ,
                self.REF_STATE_SEQ,
                self.CAR_ID,
                self.CARS_X_SEQ,
                self.CARS_Y_SEQ,
            ],
            [ca.sum2(stage_costs)],
        )

    def buildStateTransitionFunc(self) -> ca.Function:
        """默认状态转移函数，输入状态和控制量，返回下一个状态"""
        delta_x = Config.DELTA_T * BicycleModel.stateDerivative(self.STATE, self.U)
        return Primarch.BuildFunc([self.STATE, self.U], [self.STATE + delta_x])

    def buildRolloutFunc(self) -> ca.Function:
        """构建rollout函数"""
        state_seq_new = ca.MX.zeros(Config.STATE_DIM, Config.N)  # type: ignore
        state_seq_new[:, 0] = self.STATE_SEQ[:, 0]
        for i in range(Config.N - 1):
            state_seq_new[:, i + 1] = self.state_transition_func(
                state_seq_new[:, i], self.U_SEQ[:, i]
            )
        return Primarch.BuildFunc([self.STATE_SEQ, self.U_SEQ], [state_seq_new])

    def buildEvalCtrlInputSeqFunc(self) -> ca.Function:
        """构建评估控制输入序列的函数"""
        state_seq_new = self.rollout_func(self.STATE_SEQ, self.U_SEQ)
        cost = self.traj_cost_func(
            state_seq_new,
            self.U_SEQ,
            self.REF_STATE_SEQ,
            self.CAR_ID,
            self.CARS_X_SEQ,
            self.CARS_Y_SEQ,
        )
        return Primarch.BuildFunc(
            [
                self.STATE_SEQ,
                self.U_SEQ,
                self.REF_STATE_SEQ,
                self.CAR_ID,
                self.CARS_X_SEQ,
                self.CARS_Y_SEQ,
            ],
            [cost],
        )

    def evalCtrlInputSeq(self, ctrl_input_seq: ca.DM) -> float:
        """评估给定的控制输入序列的代价"""
        cost = self.eval_ctrl_input_seq_func(
            self.state_seq,
            ctrl_input_seq,
            self.ref_state_seq,
            self.car_id,
            self.cars_x_seq,
            self.cars_y_seq,
        )
        return float(cost)

    def findPath(
        self,
        car_id: int,
        state_init: NDArray,
        ref_state_seq: ca.DM,
        cars_x_seq: ca.DM,
        cars_y_seq: ca.DM,
    ) -> Tuple[ca.DM, ca.DM]:
        """
        输入车辆ID、初始状态、参考状态序列以及k-1层级时所有车辆的规划路径，
        为处于level-k的车辆规划轨迹并返回预测状态序列和控制输入序列。
        """
        self.__initSolver(car_id, state_init, ref_state_seq, cars_x_seq, cars_y_seq)
        ctrl_seq = self.tree.getBestCtrlInputSeq(self.evalCtrlInputSeq)
        state_seq = self.rollout_func(self.state_seq, ctrl_seq)
        return state_seq, ctrl_seq  # type: ignore

    def __initSolver(
        self,
        car_id: int,
        x_init: NDArray,
        ref_state_seq: ca.DM,
        cars_x_seq: ca.DM,
        cars_y_seq: ca.DM,
    ) -> None:
        """初始化求解器"""
        self.car_id = car_id
        self.state_seq[0, 0] = x_init[0]
        self.state_seq[1, 0] = x_init[1]
        self.state_seq[2, 0] = x_init[2]
        self.state_seq[3, 0] = x_init[3]
        self.ref_state_seq = ref_state_seq
        self.cars_x_seq = cars_x_seq
        self.cars_y_seq = cars_y_seq
        self.tree.reset()
