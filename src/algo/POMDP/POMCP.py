# -*- coding: utf-8 -*-

import os
import sys

sys.path.append(os.getcwd())

import copy
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.util.config import Config as BaseConfig
from src.util.log import Log
from src.util.random_num_generator import RandomNumGenerator
from src.util.tiger import (
    Action,
    Belief,
    History,
    Observation,
    ParticleFilter,
    State,
    Tiger,
)


class Config(BaseConfig):
    MAX_DEPTH = 10
    """最大搜索深度"""
    MAX_NUM_SIMULATIONS = 200
    """最大模拟次数"""
    NUM_PARTICLES = 1234
    """粒子数量"""
    GAMMA = 0.1
    """
    折扣系数，这里选了一个较小的折扣系数更专注于及时奖励。
    在调试的过程中发现即使第一步打开正确的门，在重置后计算未来奖励的过程中，
    由于rollout policy并不好很容易打开后面是老虎的门带来一个很大的惩罚导致未
    来奖励非常小，加在一起后不能体现出第一步打开正确的门的优势。
    """
    EXPLORE_COEF = 6.66
    """探索系数"""


class Node:
    """POMCP树节点"""

    def __init__(
        self,
        history: History,
        total_reward: float,
        parent: Optional["Node"] = None,
    ) -> None:
        """使用历史记录，总奖励，子节点和父节点构造节点"""
        self.history = history
        """历史记录"""
        self.total_reward = total_reward
        """总奖励"""
        self.num_visits = 0
        """访问次数"""
        self.history_to_child: Dict[History, "Node"] = {}
        """子节点，存储为从历史记录到节点的映射"""
        self.parent: Optional["Node"] = parent
        """父节点"""
        if self.parent is not None and self.history not in self.parent.history_to_child:
            self.parent.history_to_child[self.history] = self

    @property
    def val(self) -> float:
        """计算当前节点的值"""
        if self.num_visits == 0:
            return 0.0
        return self.total_reward / self.num_visits

    @property
    def action(self) -> Action:
        """当前节点的动作"""
        return self.history.action

    def calUCB1(self, num_parent_visits: int) -> float:
        """输入父节点的访问次数，计算当前节点的UCB1值"""
        if self.num_visits == 0:
            return float("inf")
        return self.val + Config.EXPLORE_COEF * np.sqrt(
            np.log(num_parent_visits) / self.num_visits
        )

    def selChild(self, enable_explore: bool = True) -> Tuple["Node", Action]:
        """
        选择一个子节点并返回它和它的动作，如果enable_explore为True，
        使用UCB1选择子节点，否则直接选择价值最高的子节点
        """
        max_val = float("-inf")
        best_node = None
        for child in self.history_to_child.values():
            if enable_explore:
                val = child.calUCB1(max(self.num_visits, 1))
            else:
                val = child.val
            if val > max_val + Config.FLOAT_EPS:
                max_val = val
                best_node = child
        if best_node is None:
            Log.error("No child node found!!!")
            raise ValueError("No child node found!!!")
        return best_node, best_node.action

    def update(self, reward: float) -> None:
        """更新当前节点的总奖励和访问次数"""
        self.total_reward += reward
        self.num_visits += 1


class Tree:
    """蒙特卡洛树"""

    def __init__(self) -> None:
        """构造一个空的蒙特卡洛树"""
        self.history_to_node: Dict[History, Node] = {}
        """存储为从历史记录到节点的映射"""

    def contains(self, history: History) -> bool:
        """判断树中是否包含某个历史记录"""
        return history in self.history_to_node

    def at(self, history: History) -> Node:
        """获取树中某个历史记录对应的节点"""
        if not self.contains(history):
            Log.error("Node with history {} not found!!!".format(history))
            raise ValueError("Node not found!!!")
        return self.history_to_node[history]

    def add(self, node: Node) -> None:
        """将节点添加到树中"""
        if node.history in self.history_to_node:
            Log.warning("Node with history {} already exists!".format(node.history))
            return
        self.history_to_node[node.history] = node


class POMCP:
    """Partially Observable Monte Carlo Planning求解器"""

    def __init__(self, env: Tiger) -> None:
        self.env = env
        """所处环境"""
        self.rand_num_gen = RandomNumGenerator(233)
        """随机数生成器"""
        self.tree = Tree()
        """蒙特卡洛搜索树"""
        self.history: History = History.ConstructFromList([], [])
        """历史记录"""
        self.node = Node(self.history, 0.0)
        """当前节点"""
        self.particle_filter = ParticleFilter(env, Config.NUM_PARTICLES)
        """粒子滤波器"""

    def __rollOutPolicy(self, history: History) -> Action:
        """
        前向模拟策略，如果最近三次都选择倾听且老虎位于同一侧，则打开反方向的门，
        否则随机选择一个动作。
        """
        if history.action_vals.size >= 3 and all(
            history.action_vals[-i] == Action.LISTEN.value for i in range(1, 4)
        ):
            if history.obs_vals.size >= 3 and all(
                history.obs_vals[-i] == history.obs_vals[-1] for i in range(1, 4)
            ):
                return (
                    Action.OPEN_RIGHT
                    if history.obs_vals[-1] == Observation.LEFT.value
                    else Action.OPEN_LEFT
                )
        return self.rand_num_gen.getRandomEnum(Action)

    def rollOut(self, state: State, history: History, depth: int) -> float:
        """前向模拟，返回从当前状态开始的模拟折扣回报"""
        if depth > Config.MAX_DEPTH:
            return 0.0
        action = self.__rollOutPolicy(history)
        next_state, next_obs = self.env.sampleNextStateAndObs(state, action)
        reward = self.env.getReward(state, action)
        future_reward = self.rollOut(
            next_state, history.augment(action, next_obs), depth + 1
        )
        return reward.value + Config.GAMMA * future_reward

    def simulate(self, state: State, history: History, depth: int) -> float:
        """模拟，返回此轮模拟中节点所获得的奖励"""
        if depth > Config.MAX_DEPTH:
            return 0.0
        if not self.tree.contains(history):
            node = Node(history, 0.0)
            self.tree.add(node)
            for action in Action:
                new_history = history.augment(new_action=action)
                new_node = Node(new_history, 0.0, parent=node)
                self.tree.add(new_node)
            return self.rollOut(state, node.history, depth)
        node = self.tree.at(history)
        # 只是中间节点，非用来进行后续模拟的节点
        action_node, best_action = node.selChild(enable_explore=True)
        immediate_reward = self.env.getReward(state, best_action)
        next_state, next_obs = self.env.sampleNextStateAndObs(state, best_action)
        new_history = history.augment(best_action, next_obs)
        # 生成新的节点并开始迭代模拟
        if not self.tree.contains(new_history):
            new_node = Node(new_history, 0.0, parent=action_node)
        new_node = action_node.history_to_child[new_history]
        r = immediate_reward.value + Config.GAMMA * self.simulate(
            next_state, new_history, depth + 1
        )
        for node_to_update in [node, action_node, new_node]:
            node_to_update.update(r)
        return r

    def search(
        self, prev_action: Optional[Action] = None, obs: Optional[Observation] = None
    ) -> Action:
        """搜索，输入上一步执行的动作和这一步的观测，返回这一步应该执行的动作"""
        if prev_action is not None and obs is not None:
            self.history = self.history.augment(prev_action, obs)
            self.particle_filter.update(prev_action, obs)
        for _ in range(Config.MAX_NUM_SIMULATIONS):
            if self.history.empty:
                state = self.rand_num_gen.getRandomEnum(State)
            else:
                state = self.particle_filter.sample()
            r = self.simulate(state, self.history, 0)
        _, best_action = self.tree.at(self.history).selChild(enable_explore=False)
        return best_action

    @staticmethod
    def Demo() -> None:
        """POMCP求解器演示"""
        env = Tiger()
        solver = POMCP(env)
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
            title="POMCP",
        )


if __name__ == "__main__":
    POMCP.Demo()
