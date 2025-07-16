# -*- coding: utf-8 -*-

import os
import sys

sys.path.append(os.getcwd())

import copy
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from src.util.config import Config as BaseConfig
from src.util.log import Log
from src.util.random_num_generator import RandomNumGenerator
from src.util.tiger import (
    Action,
    Belief,
    History,
    Observation,
    ParticleFilter,
    Reward,
    State,
    Tiger,
)


class Config(BaseConfig):
    MAX_DEPTH = 10
    """最大搜索深度，同时也控制scenario序列长度"""
    NUM_PARTICLES = 1234
    """粒子数量"""
    NUM_TRAILS = 888
    """构建树时需要进行多少次探索"""
    NUM_DEF_POLICY_TRIALS = 3
    """默认策略尝试次数"""
    GAMMA = 0.1
    """折扣系数，这里选了一个较小的值以专注于即时奖励"""
    K = 100
    """scenario数量"""
    EPSILON = 0.95
    """计算WEU时的系数"""
    LAMBDA = 0.01
    """正则化系数"""
    RAND_NUM_GEN = RandomNumGenerator(888)
    """用于DESPOT算法的随机数生成器"""


class Scenario:
    """场景类，包含初始状态和参数phi序列"""

    def __init__(self, id: int, state: State, size: int) -> None:
        """使用场景ID，初始状态和长度构造场景"""
        self.id = id
        """场景ID"""
        self.state = state
        """初始状态"""
        self.phi_list: NDArray = np.array(
            [Config.RAND_NUM_GEN.getRandomFloat(0.0, 1.0) for _ in range(size)]
        )
        """参数phi序列"""

    @property
    def size(self) -> int:
        """场景长度"""
        return self.phi_list.size

    def getPhi(self, depth: int) -> float:
        """获取指定深度的参数phi"""
        if depth < 0 or depth >= self.size:
            Log.error("Depth out of range!!!")
            raise ValueError("Depth out of range!!!")
        return self.phi_list[depth]


class DefaultPolicy:
    """默认策略类，用于构筑节点下界"""

    @staticmethod
    def Run(scenario_id_to_state: Dict[int, State], env: Tiger) -> Tuple[Action, float]:
        """
        通过默认策略为节点估算价值下界。具体过程如下：从传入的场景ID与状态映射中
        随机选取一个初始状态，第一步固定执行LISTEN动作，后续动作随机，进行多轮模拟并累
        计折扣奖励，最后将平均折扣奖励作为下界估计，返回首次动作LISTEN和此平均收益。
        """
        val = 0.0
        for _ in range(Config.NUM_DEF_POLICY_TRIALS):
            state = Config.RAND_NUM_GEN.choice(list(scenario_id_to_state.values()), 1)[
                0
            ]
            action = Action.LISTEN
            for i in range(Config.MAX_DEPTH):
                reward = env.getReward(state, action)
                val += (Config.GAMMA**i) * reward.value
                next_state, _ = env.sampleNextStateAndObs(state, action)
                state = next_state
                action = Config.RAND_NUM_GEN.getRandomEnum(Action)
        return Action.LISTEN, val / Config.NUM_DEF_POLICY_TRIALS


class Tree:
    """DESPOT树"""

    def __init__(
        self, init_history: History, env: Tiger, particle_filter: ParticleFilter
    ) -> None:
        """使用动作-观测历史，环境和粒子滤波器初始化树"""
        self.env = env
        """环境"""
        self.scenarios: List[Scenario] = [
            Scenario(i, particle_filter.sample(), Config.MAX_DEPTH)
            for i in range(Config.K)
        ]
        """场景列表"""
        self.root: Node = self.__constructRoot(init_history)
        """根节点"""
        self.history_to_node: Dict[History, Node] = {}
        """历史记录到默认节点的映射"""
        self.history_to_q_node: Dict[History, QNode] = {}
        """历史记录到动作节点的映射"""
        self.addNode(init_history, self.root)
        for _ in range(Config.NUM_TRAILS):
            bottom_node = self.__runTrail(self.root)
            self.__backup(bottom_node)

    def getScenario(self, scenario_id: int) -> Scenario:
        """获取场景"""
        if scenario_id < 0 or scenario_id >= len(self.scenarios):
            Log.error("Scenario ID out of range!!!")
            raise ValueError("Scenario ID out of range!!!")
        return self.scenarios[scenario_id]

    def addNode(self, history: History, node) -> None:
        """将节点添加到树中"""
        if type(node) == Node:
            if history in self.history_to_node:
                Log.warning("Node with history {} already exists!".format(history))
                return
            self.history_to_node[history] = node
        elif type(node) == QNode:
            if history in self.history_to_q_node:
                Log.warning("QNode with history {} already exists!".format(history))
                return
            self.history_to_q_node[history] = node
        else:
            Log.error("Node type not supported!!!")
            raise ValueError("Node type not supported!!!")

    def __constructRoot(self, init_history: History) -> "Node":
        """构建根节点"""
        scenario_id_to_state = {
            scenario.id: scenario.state for scenario in self.scenarios
        }
        return Node(init_history, 0, scenario_id_to_state, self)

    def __runTrail(self, node: "Node") -> "Node":
        """从node开始往下面探索，返回最终停止探索的节点"""
        if node.depth != 0:
            # 在某些极端情况下默认的策略给了一个很接近上界的下界，这个时候至少要保证
            # 对根节点进行了一次扩展。
            if node.depth >= Config.MAX_DEPTH - 1 or node.WEU < Config.ZERO_EPS:
                return node
        if node.is_leaf:
            node.expand(self)
        # 选择上界最高的动作子节点
        next_q_node = node.selChild()[0]
        # 选择动作子节点下WEU最高的默认节点
        next_node = next_q_node.selChild()[0]
        return self.__runTrail(next_node)

    def __backup(self, node: "Node") -> None:
        """从node开始往上更新节点价值的上下界"""
        node.updateBounds()
        parent_q_node = node.parent_node
        if parent_q_node is not None:
            parent_q_node.updateBounds()
            self.__backup(parent_q_node.parent_node)


class BaseNode(ABC):
    """DESPOT树节点基类"""

    def __init__(self, history: History) -> None:
        """使用历史记录初始化节点基类"""
        self.history = history
        """历史记录"""
        self.upper_bound: float = -np.inf
        """价值上界"""
        self.lower_bound: float = np.inf
        """价值下界"""

    @abstractmethod
    def expand(self, tree: Tree) -> None:
        """扩展节点"""
        pass

    @abstractmethod
    def updateBounds(self) -> None:
        """更新价值上界和下界"""
        pass


class Node(BaseNode):
    """默认节点类，其中观测历史长度与动作历史长度相同"""

    def __init__(
        self,
        history: History,
        depth: int,
        scenario_id_to_state: Dict[int, State],
        tree: Tree,
        parent_node: Optional["QNode"] = None,
    ) -> None:
        """使用历史记录，节点深度，场景ID到状态的映射，树和父节点初始化节点"""
        super().__init__(history)
        self.depth = depth
        """节点深度"""
        self.parent_node: Optional["QNode"] = parent_node
        """父节点"""
        self.scenario_id_to_state: Dict[int, State] = scenario_id_to_state
        """从经过这里的场景ID到这些场景下在此处状态的映射"""
        self.action_to_child: Dict[Action, QNode] = {}
        """从执行的动作到到达的子节点的映射"""
        # 对于每一个新树节点都需要基于默认策略进行一次更新，这里存储
        # 默认策略在此处的动作和默认策略下的价值。
        self.def_move: Action = Action.LISTEN
        """默认动作"""
        self.def_val: float = 0.0
        """默认动作的价值"""
        self.__updateDefMoveAndVal(tree.env)
        self.upper_bound = -Config.LAMBDA + Reward.MaxVal() / (1 - Config.GAMMA)
        self.lower_bound = self.def_val

    @property
    def is_leaf(self) -> bool:
        """是否为叶子节点"""
        return len(self.action_to_child) == 0

    @property
    def WEU(self) -> float:
        """weighted excess uncertainty值"""
        last_term = Config.EPSILON * (Config.GAMMA ** (-self.depth))
        return (self.upper_bound - self.lower_bound) - last_term

    def expand(self, tree: Tree) -> None:
        # 对于每一个可能动作都生成一个动作节点作为子节点
        for action in Action:
            self.action_to_child[action] = QNode(self, action, tree)
            tree.addNode(
                self.action_to_child[action].history, self.action_to_child[action]
            )

    def updateBounds(self) -> None:
        if self.is_leaf:
            return
        lower_child_bound = max(
            child.lower_bound for child in self.action_to_child.values()
        )
        upper_child_bound = max(
            child.upper_bound for child in self.action_to_child.values()
        )
        # 比较执行默认动作和执行子节点对应的动作哪一个好
        self.lower_bound = max(self.def_val, lower_child_bound)
        self.upper_bound = max(self.def_val, upper_child_bound)
        if (
            abs(self.lower_bound - self.def_val) < Config.FLOAT_EPS
            or self.lower_bound > self.upper_bound
        ):
            # 考虑数值误差保证上界不小于下界
            self.upper_bound = self.lower_bound

    def selChild(self, using_upper_bound: bool = True) -> Tuple["QNode", Action]:
        """
        选择一个子节点，返回选中的子节点和它的动作，
        默认选择上界最大的子节点，也可以选择下界最大的子节点
        """
        if not self.action_to_child:
            Log.error("No child node found!!!")
            raise ValueError("No child node found!!!")
        best_action, best_node = max(
            self.action_to_child.items(),
            key=lambda x: x[1].upper_bound if using_upper_bound else x[1].lower_bound,
        )
        return best_node, best_action

    def __updateDefMoveAndVal(self, env: Tiger) -> None:
        """使用默认策略更新默认动作和价值"""
        first_action, val = DefaultPolicy.Run(self.scenario_id_to_state, env)
        self.def_move = first_action
        self.def_val = val - Config.LAMBDA


class QNode(BaseNode):
    """动作节点，是默认节点Node执行一个动作后到达的节点"""

    def __init__(self, parent_node: Node, action: Action, tree: Tree) -> None:
        """使用父节点和在父节点执行的动作构造动作节点，注意会自动往下扩展一层"""
        super().__init__(parent_node.history.augment(new_action=action))
        self.parent_node = parent_node
        """父节点"""
        self.action = action
        """在父节点执行的动作"""
        self.scenario_id_to_child: Dict[int, Node] = {}
        """从经过这里的场景ID到这些场景下在此处状态的映射，注意这里不同的场景可能会对应同一个节点"""
        self.reward: float = self.__calReward(tree)
        """在父节点执行对应的动作并减去正则项后的奖励"""
        self.expand(tree)
        self.updateBounds()

    @property
    def depth(self) -> int:
        """节点深度"""
        return self.parent_node.depth

    @property
    def num_scenarios(self) -> int:
        """经过此节点的场景数量"""
        return len(self.parent_node.scenario_id_to_state)

    def expand(self, tree: Tree) -> None:
        # 对于每一个scenario生成一个默认节点作为子节点，不同的scenario可能对应同一个节点，注意去重
        obs_to_scenario_id_to_state: Dict[Observation, Dict[int, State]] = {}
        for scenario_id, state in self.parent_node.scenario_id_to_state.items():
            phi = tree.getScenario(scenario_id).getPhi(self.depth)
            next_state, next_obs = tree.env.sampleNextStateAndObs(
                state, self.action, phi
            )
            obs_to_scenario_id_to_state.setdefault(next_obs, {})[
                scenario_id
            ] = next_state
        for obs, scenario_id_to_state in obs_to_scenario_id_to_state.items():
            history = self.history.augment(new_obs=obs)
            node = Node(
                history, self.depth + 1, scenario_id_to_state, tree, parent_node=self
            )
            tree.addNode(history, node)
            for scenario_id, _ in scenario_id_to_state.items():
                self.scenario_id_to_child[scenario_id] = node

    def updateBounds(self) -> None:
        lower_bound_sum = sum(
            child.lower_bound for child in self.scenario_id_to_child.values()
        )
        upper_bound_sum = sum(
            child.upper_bound for child in self.scenario_id_to_child.values()
        )
        factor = Config.GAMMA / self.num_scenarios
        self.lower_bound = self.reward + factor * lower_bound_sum
        self.upper_bound = self.reward + factor * upper_bound_sum

    def selChild(self) -> Tuple["Node", float]:
        """选择一个WEU值最大的子节点，返回选中的子节点和它WEU值"""
        children = list(self.scenario_id_to_child.values())
        if not children:
            Log.error("No child node found!!!")
            raise ValueError("No child node found!!!")
        best_node = max(children, key=lambda c: c.WEU)
        return best_node, best_node.WEU

    def __calReward(self, tree: Tree) -> float:
        """计算减去正则项后的即时奖励"""
        total_reward = sum(
            tree.env.getReward(state, self.action).value
            for state in self.parent_node.scenario_id_to_state.values()
        )
        return total_reward / self.num_scenarios - Config.LAMBDA


class DESPOT:
    """DESPOT算法"""

    def __init__(self, env: Tiger) -> None:
        """使用环境初始化DESPOT算法"""
        self.history: History = History.ConstructFromList([], [])
        """动作-观测历史"""
        self.env = env
        """所处环境"""
        self.particle_filter = ParticleFilter(env, Config.NUM_PARTICLES)
        """粒子滤波器"""
        self.tree: Tree
        """DESPOT树"""

    def search(
        self, prev_action: Optional[Action] = None, obs: Optional[Observation] = None
    ) -> Action:
        """搜索，输入上一步执行的动作和这一步的观测，返回这一步应该执行的动作"""
        if prev_action is not None and obs is not None:
            self.history = self.history.augment(prev_action, obs)
            self.particle_filter.update(prev_action, obs)
        self.tree = Tree(self.history, self.env, self.particle_filter)
        best_action, best_val = self.tree.root.def_move, self.tree.root.def_val
        child, child_action = self.tree.root.selChild(using_upper_bound=False)
        if child.lower_bound > best_val + Config.FLOAT_EPS:
            return child_action
        else:
            return best_action

    @staticmethod
    def Demo() -> None:
        """DESPOT算法演示"""
        env = Tiger()
        solver = DESPOT(env)
        prev_action: Optional[Action] = None
        obs: Optional[Observation] = None
        total_reward = 0.0
        state_history: List[State] = []
        action_history: List[Action] = []
        belief_history: List[Belief] = []
        for i in range(20):
            Log.info(
                "=========================Episode {}=========================".format(i)
            )
            cur_state = copy.deepcopy(env.state)
            action = solver.search(prev_action, obs)
            belief = solver.particle_filter.getBelief()
            Log.info("Current State: {}, Belief: {}".format(cur_state.name, belief))
            result = env.step(action)
            Log.info(
                "Action: {}, Reward: {}, Observation: {}".format(
                    action.name, result[0].value, result[1].name
                )
            )
            prev_action = action
            obs = result[1]
            total_reward += result[0].value
            state_history.append(cur_state)
            action_history.append(action)
            belief_history.append(belief)
        Log.info("Simulation finished, total reward: {}.".format(total_reward))
        Tiger.PlotHistory(
            state_history,
            action_history,
            belief_history=belief_history,
            title="DESPOT",
        )


if __name__ == "__main__":
    DESPOT.Demo()
